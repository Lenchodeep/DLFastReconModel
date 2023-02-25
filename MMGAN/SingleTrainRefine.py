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
# from Models import Generator, Discriminator
from MyDataset import FastMRIDataset, BraTSDataset, FastMRIDatasetRefineGan
from loss import *
from utils import *
from SingleModel  import *
# from RefineGAN import *
# from lightRefineGAN import *
# from SingleModel import *
from torch.utils.tensorboard import SummaryWriter

def output2complex2(im_tensor, isZeroToOne = False):
    '''
    param: im_tensor : [B, 2, W, H]
    return : [B,W,H] complex value
    '''
    ############## revert each channel to [0,1.] range
    if not isZeroToOne:
        im_tensor = revert_scale(im_tensor)
    # 2 channel to complex
    im_tensor = torch.abs(torch.complex(im_tensor[:,0:1,:,:], im_tensor[:,1:2,:,:]))
    
    return im_tensor

def revert_scale(im_tensor, a=2., b=-1.):
    '''
    param: im_tensor : [B, 2, W, H]
    '''
    b = b * torch.ones_like(im_tensor)
    im = (im_tensor - b) / a

    return im

def train(args):
    '''Init the model'''
    #init dataloader

    train_dataset = FastMRIDatasetRefineGan(args, True, None)
    val_dataset = FastMRIDatasetRefineGan(args, False, None)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True
                            , num_workers=args.train_num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True
                            , num_workers= args.val_num_workers, pin_memory=True, drop_last=True)

    ## generator
    net_G = SingleGenerator(args)
    # net_G = Refine_G(args)
    # net_G = Refine_G_light(args)
    
    G_optimizer = torch.optim.Adam(net_G.parameters(), lr = args.lr, betas = (0.5,0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    net_G.to(device= args.device)

    ## discriminator
    net_D = Discriminator1(ndf=64)
    # net_D = Discriminator()
    D_optimizer = torch.optim.Adam(net_D.parameters(), lr = args.lr, betas=(0.5, 0.999))  ##discriminator 的学习率为Generator的两倍
    D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    net_D.to(device=args.device)

    ## define the transfor for data augmentation
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
        ])


    #Init tensor board writer
    if args.tb_write_loss or args.tb_write_image:
        writer = SummaryWriter(log_dir=args.tb_dir+'/tensorboard')

    #Init loss object

    ## loss return : fullLoss, imgloss_T1, imgloss_T2, kloss_T1, kloss_T2, ssimloss_T1, ssimloss_T2, advLoss

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
    
    maxpsnr = 0
    net_D.train()

    try:
        for epoch in range(start_epoch, args.epochs):
            progress = 0 

            ## train metrics
            epoch_Full_Loss = 0
            epoch_D_Loss = 0
            epoch_G_Loss = 0
            epoch_Img_Loss_T2 = 0
            epoch_Error_Loss_T2 = 0
            epoch_K_Loss_T2 = 0
            epoch_SSIM = 0
            epoch_PSNR = 0
            epoch_NMSE = 0
            ## val metrics
            val_SSIM = 0
            val_PSNR = 0
            val_NMSE = 0
            val_PSNR_ZF = 0
            net_G.train()   #注意train的调用位置，防止在train的过程中进入eval模式中。

            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                '''Trainging stage'''
                for step,batch in enumerate(train_loader):
                    ## BATCH DATA
                    T2_masked_img = batch['target_masked_img'].to(device = args.device, dtype = torch.float32)
                    T2_masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                    T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                    mask = batch['mask'].to(device = args.device, dtype = torch.float32)

                    for p in net_D.parameters():
                        p.requires_grad = True
                    
                    #forward G
                    rec_mid_T2_pre, rec_mid_T2, rec_img_T2_pre, rec_img_T2 = net_G(T2_masked_img, T2_masked_k, mask)
                   
                    # get the D out
                    D_real_pred = net_D(T2_target_img)
                    ############################# update D #################################
                    with torch.no_grad():
                        rec_clone = rec_img_T2.detach().clone()

                    D_fake_pred = net_D(rec_clone)

                    DLoss = cal_loss(T2_target_img, T2_masked_k, mask, rec_mid_T2_pre, rec_mid_T2,\
                                     rec_img_T2_pre, rec_img_T2, D_real_pred, D_fake_pred, False)

                    epoch_D_Loss += DLoss.item()

                    D_optimizer.zero_grad()
                    DLoss.backward()
                    D_optimizer.step()

                    for p in net_D.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)

############################################# update G ######################################
                    for p in net_D.parameters():
                        p.requires_grad = False

                    D_fake_pred_update = net_D(rec_img_T2)


                    #calculate G loss
                    FullLoss, frqloss_T2, IMloss_T2, errorloss_T2, Gloss\
                    = cal_loss(T2_target_img, T2_masked_k, mask, rec_mid_T2_pre, rec_mid_T2,\
                                     rec_img_T2_pre, rec_img_T2, D_real_pred, D_fake_pred_update, True)


                    #Optimize parameters
                    G_optimizer.zero_grad()
                    FullLoss.backward()
                    G_optimizer.step()

                    ## calculate the psnr
                    T2_target_img = output2complex2(T2_target_img, args.isZeroToOne)
                    rec_img_T2 = output2complex2(rec_img_T2, args.isZeroToOne)

                    psnr = calculatePSNR(T2_target_img.detach().cpu().numpy().squeeze(), rec_img_T2.detach().cpu().numpy().squeeze(), True)
                    nmse = calcuNMSE(T2_target_img.detach().cpu().numpy().squeeze(), rec_img_T2.detach().cpu().numpy().squeeze(), True)
                    ssim = calculateSSIM(T2_target_img.detach().cpu().numpy().squeeze(), rec_img_T2.detach().cpu().numpy().squeeze(), True)

                    psnr = psnr/T2_target_img.shape[0]
                    nmse = nmse/T2_target_img.shape[0]
                    ssim = ssim/T2_target_img.shape[0]

                    # update fullLoss
                    epoch_Full_Loss += FullLoss.item()
                    epoch_Img_Loss_T2 += IMloss_T2.item()
                    epoch_Error_Loss_T2 += errorloss_T2.item()
                    epoch_K_Loss_T2 += frqloss_T2.item()
                    epoch_G_Loss += Gloss.item()
                    epoch_PSNR += psnr
                    epoch_NMSE += nmse
                    epoch_SSIM += ssim

                    progress += 100*T2_target_img.shape[0]/len(train_dataset)
                    
                    pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'Gadvloss: ': Gloss.item(), 'Dadvloss: ': DLoss.item(),'ImT2': IMloss_T2.item(),
                                    'ImerrorT2': errorloss_T2.item(),'KT2loss': frqloss_T2.item(),'PSNR: ' : psnr, 'NMSE: ':nmse, 
                                    'SSIM: ':ssim,  'Prctg of train set': progress})

                    pbar.update(T2_target_img.shape[0])# current batch size


                '''valdation stage'''
                net_G.eval()
                net_D.eval()
                n_val = len(val_loader)  # number of batch
                with torch.no_grad():

                    for batch in tqdm(val_loader):
                        val_T2_masked_img = batch['target_masked_img'].to(device = args.device, dtype = torch.float32)
                        val_T2_masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                        # val_T2_target_k = batch['target_K'].to(device = args.device, dtype = torch.float32)
                        val_T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                        val_mask = batch['mask'].to(device = args.device, dtype = torch.float32)

                        val_rec_mid_T2_pre, val_rec_mid_T2, val_rec_img_T2_pre, val_rec_img_T2,  = net_G(val_T2_masked_img, val_T2_masked_k, val_mask)

                        # val_FullLoss, val_IMloss_T2, val_IMloss_T2_mid, val_SSIMLoss_T2,val_SSIMLoss_T2_mid\
                        # = loss.calc_gen_loss_single(val_T2_target_img, val_rec_img_T2,\
                        #                         val_rec_mid_T2, None)

                        val_T2_target = output2complex2(val_T2_target_img, args.isZeroToOne)
                        val_T2_rec = output2complex2(val_rec_img_T2, args.isZeroToOne)
                        val_T2_rec_mid = output2complex2(val_rec_mid_T2, args.isZeroToOne)
                        val_ZF_img = output2complex2(val_T2_masked_img, args.isZeroToOne)

                        val_psnr = calculatePSNR(val_T2_target.detach().cpu().numpy().squeeze(), val_T2_rec.detach().cpu().numpy().squeeze(), True)
                        val_nmse = calcuNMSE(val_T2_target.detach().cpu().numpy().squeeze(), val_T2_rec.detach().cpu().numpy().squeeze(), True)
                        val_ssim = calculateSSIM(val_T2_target.detach().cpu().numpy().squeeze(), val_T2_rec.detach().cpu().numpy().squeeze(), True)
                        val_zf_psnr = calculatePSNR(val_T2_target.detach().cpu().numpy().squeeze(), val_ZF_img.detach().cpu().numpy().squeeze(), True)

                        val_psnr = val_psnr/val_T2_target.shape[0]
                        val_nmse = val_nmse/val_T2_target.shape[0]
                        val_ssim = val_ssim/val_T2_target.shape[0]
                        val_zf_psnr = val_zf_psnr/val_T2_target.shape[0]

                        val_PSNR += val_psnr
                        val_NMSE += val_nmse
                        val_SSIM += val_ssim
                        val_PSNR_ZF += val_zf_psnr
                        logging.info("Step: {}  , RecPSNR: {:5}, RecSSIM: {:5}, RecNMSE: {:5}, ZFPSNR: {:5}".format(
                            step + 1, val_psnr, val_ssim, val_nmse, val_zf_psnr)) 
                # Schedular update
                # G_scheduler.step(val_FullLoss)
                # D_scheduler.step(val_FullLoss)
        

                logging.info("Epoch: {}  , RecPSNR: {:5}, RecSSIM: {:5}, RecNMSE: {:5}, ZFPSNR: {:5}".format(
                            epoch + 1, val_psnr, val_ssim, val_nmse, val_zf_psnr)) 

                net_G.train()
                net_D.train()


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
                writer.add_scalar('train/ImLossT2', epoch_Img_Loss_T2 / len(train_loader), epoch)
                writer.add_scalar('train/ImLossT2_error', epoch_Error_Loss_T2 / len(train_loader), epoch)
                writer.add_scalar('train/FreqLossT2', epoch_K_Loss_T2 / len(train_loader), epoch)
                writer.add_scalar('train/SSIM', epoch_SSIM / len(train_loader), epoch)
                writer.add_scalar('train/PSNR', epoch_PSNR / len(train_loader), epoch)
                writer.add_scalar('train/NMSE', epoch_NMSE / len(train_loader), epoch)
                # writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('train/G_AdvLoss', epoch_G_Loss / len(train_loader), epoch)
                writer.add_scalar('train/D_AdvLoss', epoch_D_Loss / len(train_loader), epoch)

                writer.add_scalar('validation/SSIM', val_SSIM / len(val_loader), epoch)
                writer.add_scalar('validation/PSNR', val_PSNR / len(val_loader), epoch)
                writer.add_scalar('validation/NMSE', val_NMSE / len(val_loader), epoch)
                writer.add_scalar('validation/ZFPSNR', val_zf_psnr / len(val_loader), epoch)


            if args.tb_write_image:
                writer.add_images('train/T2_target',T2_target_img, epoch)
                writer.add_images('train/T2_rec', rec_img_T2, epoch)

                writer.add_images('validation/val_target_img', val_T2_target, epoch)
                writer.add_images('validation/T2_rec', val_T2_rec, epoch)
                writer.add_images('validation/T2_rec_mid', val_T2_rec_mid, epoch)
                writer.add_images('validation/ZF', val_ZF_img, epoch)


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
