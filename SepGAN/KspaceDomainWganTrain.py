'''
    training strategy like the da gan
'''
import sys
import os
from numpy import mod
import torch
import logging
from torchvision import transforms
import yaml
from tqdm import tqdm
import shutil

from torch.utils.data import DataLoader, random_split
from IXIDataSet import IXIDataSetComplex, IXIDatasetMultiSlices
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from NetLoss import *
from models import *
from Eval import *
from utils import png_saver, EarlyStopping, calcu_gradient_penalty
from KINet import *

def weight_init(m):
    classname = m.__class__.__name__    ##获取当前结构名称
    if classname.find('Conv2d') != -1:
        nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
    elif classname.find('BatchNorm') != -1:
        nn.init.uniform_(m.weight, a=-0.05, b=0.05)
        nn.init.constant_(m.bias, 0)

def train(args):
    # init the save folder

    if not os.path.exists(args.rec_png_path):
        os.makedirs(args.rec_png_path)

    if not os.path.exists(args.full_png_path):
        os.makedirs(args.full_png_path)

    # Init the generator model
    G_model = SepWnet(args)
    # G_model.apply(weight_init)
    logging.info('Generator NetWork: Using {} attention module'.format(args.attName))   
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr = args.lr, betas = (0.5,0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    G_model.to(device= args.device)

    # Init Discriminator model
    # D_model = patchGAN(1,64)
    D_model = SepDiscriminatorV2(1, 64,False,num_layers=4)
    # D_model = SepPatchGan(1,64,3,False)
    # D_model.apply(weight_init)
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
    mse = nn.MSELoss().to(args.device)
    
    #Start training

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
        ''')

    try:
        for epoch in range(start_epoch, args.epochs):
            progress = 0  
            epoch_Full_Loss = 0
            epoch_D_Loss = 0
            epoch_G_Loss = 0
            epoch_ImgL1_Loss = 0
            epoch_ImgL2_Loss = 0
            epoch_gradientLoss = 0
            epoch_K_Loss = 0
            epoch_SSIM = 0
            epoch_SSIM_Mid = 0
            G_model.train()
            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                for step, batch in enumerate(train_loader):

                    masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                    target_k = batch['target_K'].to(device = args.device, dtype = torch.float32)
                    target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)

                    ## train the Discriminator
                    for p in D_model.parameters():
                        p.requires_grad = True
                    
                    rec_img, rec_k, rec_mid_img = G_model(masked_k)

                    # rec = rec_img.detach().cpu().numpy()
                    # rec = np.squeeze(rec)
                    # plt.figure()
                    # plt.imshow(rec[0,:,:], plt.cm.gray)
                    # plt.show()
                    # print(rec_img[0,0,:,:], target_img[0,0,:,:],(rec_img[0,0,:,:]-target_img[0,0,:,:]) ,"diff")

                    d_real = D_model(target_img) 
                    with torch.no_grad():
                        rec_img_clone = rec_img.detach().clone()
                    d_fake = D_model(rec_img_clone)
                    d_loss = torch.mean(d_fake - d_real)
                    epoch_D_Loss = epoch_D_Loss + d_loss.item()

                    D_optimizer.zero_grad()
                    d_loss.backward()
                    D_optimizer.step()

                    for p in D_model.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)
                                        
                 ############################################# 1.2 update generator #############################################
                    for p in D_model.parameters():  # reset requires_grad
                        p.requires_grad = False  # they are set to False below in netG update

                    dis_fake = D_model(rec_img)

                    FullLoss, ImL2, ImL1, kL2, g_loss, ssimLoss, ssimlossmid, gradientLoss = \
                                        loss.calc_gen_loss_wgan(rec_img, rec_mid_img,rec_k, target_img, target_k, dis_fake, args.is_mid_loss)

                    G_optimizer.zero_grad()
                    FullLoss.backward()
                    G_optimizer.step()

                    epoch_G_Loss = epoch_G_Loss + g_loss.item()
                    epoch_Full_Loss = epoch_Full_Loss + FullLoss.item()
                    epoch_ImgL1_Loss = epoch_ImgL1_Loss + ImL1.item()
                    epoch_ImgL2_Loss = epoch_ImgL2_Loss + ImL2.item()
                    epoch_SSIM = epoch_SSIM + ssimLoss.item()
                    epoch_K_Loss = epoch_K_Loss + kL2.item()
                    epoch_gradientLoss = epoch_gradientLoss + gradientLoss.item()
                    epoch_SSIM_Mid = epoch_SSIM_Mid + ssimlossmid.item()

                    pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(), 'ssim': ssimLoss.item(), 'mid ssim': ssimlossmid.item(),
                                            'KspaceL2': kL2.item(),'Adv G': g_loss.item(),'DLoss:':d_loss.item(),'Prctg of train set': progress})

                    pbar.update(target_k.shape[0])# current batch size
            # On epoch end
            # Validation
            val_rec_img, val_full_img, val_F_rec_mid_img, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR, val_ssim, val_ssim_mid, val_gradient =\
                psycho_eval_net_wgan(G_model, val_loader, loss, args)

            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}'
                .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR))
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
                    'D_scheduler_state_dict': G_scheduler.state_dict()
                }, args.output_dir + f'Best_CP_epoch.pth')
                logging.info(f'Best Checkpoint saved !')

             #Write to TB:
            if args.tb_write_loss:
                writer.add_scalar('train/FullLoss', epoch_Full_Loss / len(train_loader), epoch)
                writer.add_scalar('train/ImL2',epoch_ImgL2_Loss / len(train_loader), epoch)
                writer.add_scalar('train/ImL1', epoch_ImgL1_Loss / len(train_loader), epoch)
                writer.add_scalar('train/KspaceL2', epoch_K_Loss / len(train_loader), epoch)
                writer.add_scalar('train/SSIM', epoch_SSIM / len(train_loader), epoch)
                writer.add_scalar('train/Mid SSIM', epoch_SSIM_Mid / len(train_loader), epoch)

                writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                if args.GAN_training:
                    writer.add_scalar('train/G_AdvLoss', epoch_G_Loss / len(train_loader), epoch)
                    writer.add_scalar('train/D_AdvLoss', epoch_D_Loss / len(train_loader), epoch)

                writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
                writer.add_scalar('validation/ImL2', val_ImL2, epoch)
                writer.add_scalar('validation/ImL1', val_ImL1, epoch)
                writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
                writer.add_scalar('validation/SsimLoss', val_ssim, epoch)
                writer.add_scalar('validation/Mid ssimLoss', val_ssim_mid, epoch)
                writer.add_scalar('validation/PSNR', val_PSNR, epoch)
                writer.add_scalar('validation/Gradient', val_gradient, epoch)


            if args.tb_write_image:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/rec_images', rec_img, epoch)
                writer.add_images('train/Kspace_rec_images', rec_mid_img, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/rec_images', val_rec_img, epoch)
                writer.add_images('validation/Kspace_rec_images', val_F_rec_mid_img, epoch)
            


            #Save Checkpoint per 5 epoch
            if (epoch % args.save_step) == (args.save_step-1):
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': G_model.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': D_model.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                    'D_scheduler_state_dict': G_scheduler.state_dict()
                }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
                if maxpsnr < val_PSNR:
                    maxpsnr = val_PSNR
                    torch.save({
                        'epoch': epoch,
                        'G_model_state_dict': G_model.state_dict(),
                        'G_optimizer_state_dict': G_optimizer.state_dict(),
                        'G_scheduler_state_dict': G_scheduler.state_dict(),
                        'D_model_state_dict': D_model.state_dict(),
                        'D_optimizer_state_dict': D_optimizer.state_dict(),
                        'D_scheduler_state_dict': G_scheduler.state_dict()
                    }, args.output_dir + f'Best_PSNR_epoch.pth')
                    logging.info(f'Best PSNR Checkpoint saved !')

            #Save the image as png
            if (epoch % args.save_png_step) == (args.save_png_step-1):
                png_saver(val_rec_img, args.rec_png_path+'recon'+str(epoch)+'.png')
                png_saver(val_full_img, args.full_png_path+'full'+str(epoch)+'.png')


    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'G_model_state_dict': G_model.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'G_scheduler_state_dict': G_scheduler.state_dict(),
            'D_model_state_dict': D_model.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
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
