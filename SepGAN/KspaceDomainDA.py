
"""
This is a train function for gan.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import os
import torch 
import shutil
import logging
from torchvision import transforms
import yaml
from  tqdm import tqdm
from types import SimpleNamespace

# from eval import eval_net
from torch.utils.data import DataLoader, random_split
from IXIDataSet import IXIDataSetComplex, IXIDatasetMultiSlices
from models import *
from Eval import psycho_eval_net
from NetLoss import KspaceNetLoss,set_grad
from utils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



def train(args):
    ## Init the  paths including png path
    if not os.path.exists(args.rec_png_path):
        os.makedirs(args.rec_png_path)

    if not os.path.exists(args.full_png_path):
        os.makedirs(args.full_png_path)
    
    ##Init the Generator Model
    G_model = SepWnet(args)
    logging.info('Using SepWnet {} as Generator'.format(args.netV))

    G_optimizer = torch.optim.Adam(G_model.parameters(), lr = args.lr, betas = (0.5,0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    G_model.to(device= args.device)

    #Init the Discriminator Model
    D_model = SepDiscriminatorV2(inchannels=1, ndf=64, isFc=True, num_layers=4)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr = 2*args.lr, betas=(0.5, 0.999))  ##discriminator 的学习率为Generator的两倍
    D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    D_model.to(device=args.device)

    ## define the transfor for data augmentation
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
        ])

    #init dataloader
    # dataset = IXIDataSetComplex(args.imageDataDir, args, args.isTrain, transform = None)
    train_dataset = IXIDatasetMultiSlices(args.imageDataDir, args, True, transform = None)
    val_dataset = IXIDatasetMultiSlices(args.imageDataDir, args, False, transform = None)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True
                            , num_workers=args.train_num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True
                            , num_workers= args.val_num_workers, pin_memory=True, drop_last=True)
    #Init earlystop
    early_stopping = EarlyStopping(patience=args.stopPatience, savepath=args.output_dir ,cpname = "CP_early_stop",verbose=True)


    #Init tensor board writer
    if args.tb_write_loss or args.tb_write_image:
        writer = SummaryWriter(log_dir=args.tb_dir+'/tensorboard')

    #Init loss object
    loss = KspaceNetLoss(args)

    #load check point
    if args.load_cp:
        checkpoint = torch.load(args.load_cp, map_location=args.device)
        G_model.load_state_dict(checkpoint['G_model_state_dict'])
        D_model.load_state_dict(checkpoint['D_model_state_dict'])
        if args.resume_training:
            start_epoch = int(checkpoint['epoch'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
            G_scheduler.load_state_dict(checkpoint['G_scheduler_state_dict'])
            logging.info(f'Models, optimizer and scheduler loaded from {args.load_cp}')
        else:
            logging.info(f'Models only load from {args.load_cp}')
    else:
        start_epoch = 0

    #init the min val loss 
    minValLoss = 100000.0  
    maxpsnr = 0
    #Start training

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')
    try:
        for epoch in range(start_epoch, args.epochs):
            G_model.train()   #注意train的调用位置，防止在train的过程中进入eval模式中。
            D_model.train()
            progress = 0  
            # for calculate the evaluation result
            epoch_D_Loss = 0
            epoch_G_Loss = 0
            epoch_ImgL1_Loss = 0
            epoch_ImgL2_Loss = 0
            epoch_K_Loss = 0
            epoch_Full_Loss = 0
            epoch_SSIM = 0
            epoch_PSNR = 0
            epoch_NMSE = 0
            epoch_MID_SSIM = 0
            epoch_Gradient_Loss = 0
            ###
            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                ##Train loop
                for step,batch in enumerate(train_loader):

                    masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                    target_k = batch['target_K'].to(device = args.device, dtype = torch.float32)
                    target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
           
                    #forward G
                    rec_img, rec_k, rec_mid_img = G_model(masked_k)

                    # get the D out
                    real_D_example = target_img.detach()  #不需要梯度
                    fake_D_example = rec_img
                    D_real_pred = D_model(real_D_example)
                    D_fake_pred = D_model(fake_D_example)

                    # calculate the D Loss
                    D_fake_detach = D_model(fake_D_example.detach()) #stop backprop to G by detaching
                    D_real_loss, D_fake_loss, DLoss = loss.calc_disc_loss(D_real_pred, D_fake_detach)

                    #calculate G loss
                    FullLoss, ImL2, ImL1, kL2, advLoss , ssimLoss, mid_ssimLoss, gradientLoss = \
                                        loss.calc_gen_loss(rec_img, rec_mid_img,rec_k, target_img, target_k, D_fake_pred, calcuMid=False)

                    #Optimize parameters
                    #update D
                    D_optimizer.zero_grad()
                    DLoss.backward()

                    #update G
                    G_optimizer.zero_grad()
                    FullLoss.backward()

                    D_optimizer.step()
                    G_optimizer.step()

                    ## calculate the psnr
                    target_img_detach = target_img.detach().cpu().numpy()
                    rec_img_detach = rec_img.detach().cpu().numpy()
                    psnr = calculatePSNR(np.squeeze(target_img_detach), np.squeeze(rec_img_detach), True)
                    nmse = calcuNMSE(np.squeeze(target_img_detach), np.squeeze(rec_img_detach), True)
                    psnr = psnr/target_img_detach.shape[0]
                    nmse = nmse/target_img_detach.shape[0]

                    # update fullLoss
                    epoch_Full_Loss += FullLoss.item()
                    epoch_D_Loss += DLoss.item()
                    epoch_G_Loss += advLoss.item()
                    epoch_ImgL1_Loss += ImL1.item()
                    epoch_ImgL2_Loss += ImL2.item()
                    epoch_K_Loss += kL2.item()
                    epoch_SSIM += ssimLoss.item()
                    epoch_MID_SSIM += mid_ssimLoss.item()
                    epoch_Gradient_Loss += gradientLoss.item()
                    epoch_PSNR += psnr
                    epoch_NMSE += nmse
                    #Update progress bar
                    progress += 100*target_k.shape[0]/len(train_dataset)
                    if args.GAN_training:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(), 'SSIM': ssimLoss.item(),"SSIM_MID":mid_ssimLoss.item()
                                            ,"GradientLoss":gradientLoss.item(), 'KspaceL2': kL2.item(),'Adv G': advLoss.item(),'Adv D - Real' : D_real_loss.item(),'PSNR: ' : psnr, 'NMSE: ':nmse,
                                            'Adv D - Fake' : D_fake_loss.item(), 'Prctg of train set': progress})
                    else:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                            'KspaceL2': kL2.item(), 'Prctg of train set': progress})
                        
                    pbar.update(target_k.shape[0])# current batch size
                    writer.add_scalar('train/G_AdvLoss_per_step', advLoss.item(), epoch*(len(train_loader))+step)
                    writer.add_scalar('train/D_AdvLoss_per_step', DLoss.item(), epoch*(len(train_loader))+step)
                    writer.add_scalar('train/D_AdvLoss_real', D_real_loss.item(), epoch*(len(train_loader))+step)
                    writer.add_scalar('train/D_AdvLoss_fake', D_fake_loss.item(), epoch*(len(train_loader))+step)

###
                    if (step % args.save_png_step) == (args.save_png_step-1):
                        png_saver(rec_img, args.rec_png_path+'recon'+str(step)+'.png')
                        png_saver(rec_mid_img, args.rec_png_path+'recon_mid'+str(step)+'.png')
                        png_saver(target_img, args.full_png_path+'full'+str(step)+'.png')
###
            # On epoch end
            # Validation
            val_rec_img, val_full_img, val_rec_mid_image, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR, val_ssim,val_mid_ssim, val_gradient =\
                psycho_eval_net(G_model, val_loader, loss, args.device)
            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}, SSIM: {}, SSIM_MID: {},  GradientLoss:  {}'
                         .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR, val_ssim, val_mid_ssim, val_gradient))
            # Schedular update
            G_scheduler.step(val_FullLoss)
            D_scheduler.step(val_FullLoss)
            # check if need early stop or not
            # early_stopping(val_FullLoss , G_model)
            # if early_stopping.early_stop == True:
            #     print("Early stopping!")
            #     break

            # save the best checkPoint
            
            if minValLoss > val_FullLoss:
                minValLoss = val_FullLoss
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': G_model.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': D_model.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                }, args.output_dir + f'Best_CP_epoch.pth')
                logging.info(f'Best Checkpoint saved !')

            if maxpsnr < val_PSNR:
                maxpsnr = val_PSNR
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': G_model.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': D_model.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                }, args.output_dir + f'Best_PSNR_epoch.pth')
                logging.info(f'Best PSNR Checkpoint saved !')

            #Write to TB:
            if args.tb_write_loss:
                writer.add_scalar('train/FullLoss', epoch_Full_Loss / len(train_loader), epoch)
                writer.add_scalar('train/ImL2',epoch_ImgL2_Loss / len(train_loader), epoch)
                writer.add_scalar('train/ImL1', epoch_ImgL1_Loss / len(train_loader), epoch)
                writer.add_scalar('train/KspaceL2', epoch_K_Loss / len(train_loader), epoch)
                writer.add_scalar('train/SSIM', epoch_SSIM / len(train_loader), epoch)
                writer.add_scalar('train/SSIM MID', epoch_MID_SSIM / len(train_loader), epoch)
                writer.add_scalar('train/GradientLoss', epoch_Gradient_Loss / len(train_loader), epoch)
                writer.add_scalar('train/PSNR', epoch_PSNR / len(train_loader), epoch)
                writer.add_scalar('train/NMSE', epoch_NMSE / len(train_loader), epoch)
                writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                if args.GAN_training:
                    writer.add_scalar('train/G_AdvLoss', epoch_G_Loss / len(train_loader), epoch)
                    writer.add_scalar('train/D_AdvLoss', epoch_D_Loss / len(train_loader), epoch)

                writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
                writer.add_scalar('validation/ImL2', val_ImL2, epoch)
                writer.add_scalar('validation/ImL1', val_ImL1, epoch)
                writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
                writer.add_scalar('validation/ssimLoss', val_ssim, epoch)
                writer.add_scalar('validation/ssimmidLoss', val_mid_ssim, epoch)
                writer.add_scalar('validation/PSNR', val_PSNR, epoch)
                writer.add_scalar('validation/GradientLoss', val_gradient, epoch)


            if args.tb_write_image:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/Rec_images', rec_img, epoch)
                writer.add_images('train/Rec_mid_image', rec_mid_img, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/Rec_images', val_rec_img, epoch)
                writer.add_images('validation/Rec_mid_image', val_rec_mid_image, epoch)

            #Save Checkpoint per 5 epoch
            if (epoch % args.save_step) == (args.save_step-1):
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': G_model.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': D_model.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                    'D_scheduler_state_dict': D_scheduler.state_dict()
                }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

            #Save the image as png
            if (epoch % args.save_png_step) == (args.save_png_step-1):
                png_saver(val_rec_img, args.rec_png_path+'recon'+str(epoch)+'.png')
                png_saver(rec_mid_img, args.rec_png_path+'recon_mid'+str(epoch)+'.png')
                png_saver(val_full_img, args.full_png_path+'full'+str(epoch)+'.png')

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'G_model_state_dict': G_model.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'G_scheduler_state_dict': G_scheduler.state_dict(),
            'D_model_state_dict': D_model.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
            'D_scheduler_state_dict': D_scheduler.state_dict()
        }, args.output_dir + f'CP_epoch{epoch + 1}_INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    writer.close()

def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

     # Create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Copy configuration file to output directory
    shutil.copyfile('config.yaml',os.path.join(args.output_dir,'config.yaml'))
    logging.info(f'Using device {args.device}')

    train(args)
if __name__ == "__main__":
    main()


