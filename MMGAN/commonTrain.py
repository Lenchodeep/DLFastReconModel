'''
Training  file
'''
import sys
import os
import torch 
import shutil
import yaml

import logging
from torchvision import transforms
from  tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''my modules'''
from Models import Generator, Discriminator
from MyDataset import FastMRIDataset, BraTSDataset
from loss import *
from utils import *
# from MANet import Generator
from torch.utils.tensorboard import SummaryWriter

def train(args):
    '''Init the model'''
    ## generator
    net_G = Generator(args)
    # net_G = Generator()           

    G_optimizer = torch.optim.Adam(net_G.parameters(), lr = args.lr, betas = (0.5,0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    net_G.to(device= args.device)

    ## discriminator
    net_D = Discriminator(inchannels=1, ndf=32, isFc=False, num_layers=4)
    D_optimizer = torch.optim.Adam(net_D.parameters(), lr = 2*args.lr, betas=(0.5, 0.999))  ##discriminator 的学习率为Generator的两倍
    D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    net_D.to(device=args.device)

    ## define the transfor for data augmentation
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
        ])

    #init dataloader

    train_dataset = FastMRIDataset(args, True, None)
    val_dataset = FastMRIDataset(args, False, None)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True
                            , num_workers=args.train_num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True
                            , num_workers= args.val_num_workers, pin_memory=True, drop_last=True)

    #Init tensor board writer
    if args.tb_write_loss or args.tb_write_image:
        writer = SummaryWriter(log_dir=args.tb_dir+'/tensorboard')

    #Init loss object

    ## loss return : fullLoss, imgloss_T1, imgloss_T2, kloss_T1, kloss_T2, ssimloss_T1, ssimloss_T2, advLoss

    loss = Loss(args)

    #load check point
    if args.load_cp:
        checkpoint = torch.load(args.load_cp, map_location=args.device)
        net_G.load_state_dict(checkpoint['G_model_state_dict'])
        net_D.load_state_dict(checkpoint['D_model_state_dict'])
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

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')
    
    minValLoss = 100000.0  
    maxpsnr = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            net_G.train()   #注意train的调用位置，防止在train的过程中进入eval模式中。
            net_D.train()
            progress = 0 

            ## train metrics
            epoch_Full_Loss = 0
            epoch_D_Loss = 0
            epoch_G_Loss = 0
            epoch_Img_Loss_T1 = 0
            epoch_Img_Loss_T2 = 0
            epoch_K_Loss_T1 = 0
            epoch_K_Loss_T2 = 0

            epoch_SSIM = 0
            epoch_PSNR = 0
            epoch_NMSE = 0

            ## val metrics
            val_FullLoss = 0
            val_Img_Loss_T1 = 0
            val_Img_Loss_T2 = 0
            val_K_Loss_T1 = 0
            val_K_Loss_T2 = 0

            val_SSIM = 0
            val_PSNR = 0
            val_NMSE = 0

            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                '''Trainging stage'''
                for step,batch in enumerate(train_loader):
                    ## BATCH DATA
                    T2_masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                    T2_target_k = batch['target_K'].to(device = args.device, dtype = torch.float32)
                    T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                    T1_masked_k = batch['refer_masked_K'].to(device = args.device, dtype = torch.float32)
                    T1_target_k = batch['refer_K'].to(device = args.device, dtype = torch.float32)
                    T1_target_img = batch['refer_img'].to(device = args.device, dtype = torch.float32)



                    #forward G
                    rec_img_T1,rec_k_T1, rec_img_T2, rec_k_T2 = net_G(T1_masked_k, T2_masked_k)


                    # T2t = T2_masked_k.detach().cpu().numpy()
                    # T1t = rec_img_T2.detach().cpu().numpy()
                    # k_t1 = rec_k_T1.detach().cpu().numpy()
                    # print(k_t1.shape)
                    # plt.figure()
                    # plt.imshow(np.log(abs(T2t[0,0,:,:]+T2t[0,1,:,:]*1j)), plt.cm.gray)
                    # plt.figure()
                    # plt.imshow(T1t[0,0,:,:], plt.cm.gray)
                    # plt.show()


                   
                    #calculate G loss
                    FullLoss, IMloss_T1, IMloss_T2, KLoss_T1, KLoss_T2, SSIMLoss_T1, SSIMLoss_T2, _\
                     = loss.calc_gen_loss(T1_target_img, T2_target_img, rec_img_T1, rec_img_T2,\
                                        T1_target_k, T2_target_k, rec_k_T1,rec_k_T2)

                    #Optimize parameters
                    G_optimizer.zero_grad()
                    FullLoss.backward()
                    G_optimizer.step()

                    ## calculate the psnr
                    target_img_detach = T2_target_img.detach().cpu().numpy()
                    rec_img_detach = rec_img_T2.detach().cpu().numpy()

                    psnr = calculatePSNR(np.squeeze(target_img_detach), np.squeeze(rec_img_detach), True)
                    nmse = calcuNMSE(np.squeeze(target_img_detach), np.squeeze(rec_img_detach), True)
                    ssim = calculateSSIM(np.squeeze(target_img_detach), np.squeeze(rec_img_detach), True)

                    psnr = psnr/target_img_detach.shape[0]
                    nmse = nmse/target_img_detach.shape[0]
                    ssim = ssim/target_img_detach.shape[0]

                    # update fullLoss
                    epoch_Full_Loss += FullLoss.item()
                    epoch_Img_Loss_T1 += IMloss_T1.item()
                    epoch_Img_Loss_T2 += IMloss_T2.item()
                    epoch_K_Loss_T1 += KLoss_T1.item()
                    epoch_K_Loss_T2 += KLoss_T2.item()

                    epoch_PSNR += psnr
                    epoch_NMSE += nmse
                    epoch_SSIM += ssim

                    progress += 100*T2_target_k.shape[0]/len(train_dataset)
                    
                    pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImT1': IMloss_T1.item(), 'ImT2': IMloss_T2.item(),
                                    "KLoss_T1 ":KLoss_T1.item(), 'KLoss_T2': KLoss_T2.item(),
                                    'PSNR: ' : psnr, 'NMSE: ':nmse, 'SSIM: ':ssim,  'Prctg of train set': progress})

                    pbar.update(T2_target_k.shape[0])# current batch size


                '''valdation stage'''
                net_G.eval()
                net_D.eval()
                n_val = len(val_loader)  # number of batch
                for batch in tqdm(val_loader):
                    val_T2_masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                    val_T2_target_k = batch['target_K'].to(device = args.device, dtype = torch.float32)
                    val_T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                    val_T1_masked_k = batch['refer_masked_K'].to(device = args.device, dtype = torch.float32)
                    val_T1_target_k = batch['refer_K'].to(device = args.device, dtype = torch.float32)
                    val_T1_target_img = batch['refer_img'].to(device = args.device, dtype = torch.float32)

                    with torch.no_grad():
                        val_rec_img_T1,val_rec_k_T1, val_rec_img_T2, val_rec_k_T2 = net_G(val_T1_masked_k, val_T2_masked_k)

                    val_FullLoss, val_IMloss_T1, val_IMloss_T2, val_KLoss_T1, val_KLoss_T2, val_SSIMLoss_T1, val_SSIMLoss_T2, val_advLoss\
                     = loss.calc_gen_loss(val_T1_target_img, val_T2_target_img, val_rec_img_T1, val_rec_img_T2,\
                                        val_T1_target_k, val_T2_target_k, val_rec_k_T1,val_rec_k_T2, None)

                    val_target_img_detach = val_T2_target_img.detach().cpu().numpy()
                    val_rec_img_detach = val_rec_img_T2.detach().cpu().numpy()

                    val_psnr = calculatePSNR(np.squeeze(val_target_img_detach), np.squeeze(val_rec_img_detach), True)
                    val_nmse = calcuNMSE(np.squeeze(val_target_img_detach), np.squeeze(val_rec_img_detach), True)
                    val_ssim = calculateSSIM(np.squeeze(val_target_img_detach), np.squeeze(val_rec_img_detach), True)

                    val_psnr = val_psnr/val_target_img_detach.shape[0]
                    val_nmse = val_nmse/val_target_img_detach.shape[0]
                    val_ssim = val_ssim/val_target_img_detach.shape[0]

                    val_FullLoss += val_FullLoss.item()
                    val_Img_Loss_T1 += val_IMloss_T1.item()
                    val_Img_Loss_T2 += val_IMloss_T2.item()
                    val_K_Loss_T1 += val_KLoss_T1.item() 
                    val_K_Loss_T2 +=  val_KLoss_T2.item()

                    val_PSNR += val_psnr
                    val_NMSE += val_nmse
                    val_SSIM += val_ssim

                logging.info('Validation full score: {}, IMT1Loss: {}. IMT2Loss: {}, KLossT1: {}, KLossT2: {}, PSNR: {}, SSIM: {}, NMSE: {},  '
                         .format(val_FullLoss/n_val, val_Img_Loss_T1/n_val, val_Img_Loss_T2/n_val, val_K_Loss_T1/ n_val, val_K_Loss_T2/ n_val, 
                                val_PSNR/ n_val, val_SSIM/n_val, val_NMSE/n_val))
                # Schedular update
                G_scheduler.step(val_FullLoss)
                D_scheduler.step(val_FullLoss)
                net_G.train()
                net_D.train()

            if minValLoss > val_FullLoss:
                minValLoss = val_FullLoss
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': net_G.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': net_D.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                    'D_scheduler_state_dict': D_scheduler.state_dict(),

                }, args.output_dir + f'Best_CP_epoch.pth')
                logging.info(f'Best Checkpoint saved !')

            if maxpsnr < val_PSNR:
                maxpsnr = val_PSNR
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': net_G.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': net_D.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                    'D_scheduler_state_dict': D_scheduler.state_dict()
                }, args.output_dir + f'Best_PSNR_epoch.pth')
                logging.info(f'Best PSNR Checkpoint saved !')

                        #Write to TB:
            if args.tb_write_loss:
                writer.add_scalar('train/FullLoss', epoch_Full_Loss / len(train_loader), epoch)
                writer.add_scalar('train/ImLossT1',epoch_Img_Loss_T1 / len(train_loader), epoch)
                writer.add_scalar('train/ImLossT2', epoch_Img_Loss_T2 / len(train_loader), epoch)
                writer.add_scalar('train/KspaceT1', epoch_K_Loss_T1 / len(train_loader), epoch)
                writer.add_scalar('train/KspaceT2', epoch_K_Loss_T2 / len(train_loader), epoch)
                writer.add_scalar('train/SSIM', epoch_SSIM / len(train_loader), epoch)
                writer.add_scalar('train/PSNR', epoch_PSNR / len(train_loader), epoch)
                writer.add_scalar('train/NMSE', epoch_NMSE / len(train_loader), epoch)
                writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('train/G_AdvLoss', epoch_G_Loss / len(train_loader), epoch)
                writer.add_scalar('train/D_AdvLoss', epoch_D_Loss / len(train_loader), epoch)

                writer.add_scalar('validation/FullLoss', val_FullLoss / len(val_loader), epoch)
                writer.add_scalar('validation/ImLossT1', val_Img_Loss_T1 / len(val_loader), epoch)
                writer.add_scalar('validation/ImLossT2', val_Img_Loss_T2 / len(val_loader), epoch)
                writer.add_scalar('validation/KspaceT1', val_K_Loss_T1 / len(val_loader), epoch)
                writer.add_scalar('validation/KspaceT2', val_K_Loss_T2 / len(val_loader), epoch)
                writer.add_scalar('validation/SSIM', val_SSIM / len(val_loader), epoch)
                writer.add_scalar('validation/PSNR', val_PSNR / len(val_loader), epoch)
                writer.add_scalar('validation/NMSE', val_nmse / len(val_loader), epoch)


            if args.tb_write_image:
                writer.add_images('train/T1_target', T1_target_img, epoch)
                writer.add_images('train/T2_target', T2_target_img, epoch)
                writer.add_images('train/T1_rec', rec_img_T1, epoch)
                writer.add_images('train/T2_rec', rec_img_T2, epoch)

                writer.add_images('validation/T1_target', val_T1_target_img, epoch)
                writer.add_images('validation/T2_target', val_T2_target_img, epoch)
                writer.add_images('validation/T1_rec', val_rec_img_T1, epoch)
                writer.add_images('validation/T2_rec', val_rec_img_T2, epoch)


                  #Save Checkpoint per 5 epoch
            if (epoch % args.save_step) == (args.save_step-1):
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': net_G.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': net_D.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                    'D_scheduler_state_dict': D_scheduler.state_dict()
                }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'G_model_state_dict': net_G.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'G_scheduler_state_dict': G_scheduler.state_dict(),
            'D_model_state_dict': net_D.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
            'D_scheduler_state_dict': D_scheduler.state_dict()
        }, args.output_dir + f'CP_epoch{epoch + 1}_INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



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
    # pass
