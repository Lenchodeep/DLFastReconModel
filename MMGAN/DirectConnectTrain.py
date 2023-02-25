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
from lightRefineGAN  import *
from DirictModel import DirectG
from SingleModel import Discriminator1
from MMGAN import MMGenerator
# from MMNET import *
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

def total_variant(images):
    '''
    :param images:  [B,C,W,H]
    :return: total_variant
    '''
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]

    tot_var = torch.abs(pixel_dif1).sum([1, 2, 3]) + torch.abs(pixel_dif2).sum([1, 2, 3])

    return tot_var

def cal_loss(tar_T1, T1_k_un, tar_T2, T2_k_un, mask, rec_T1_mid_0, rec_T1_mid, rec_T1_0, rec_T1, \
            rec_T2_mid_0, rec_T2_mid, rec_T2_0, rec_T2, T2_dis_real, T2_dis_fake, T1_dis_real, T1_dis_fake, \
            cal_G=True, args = None):

    G_loss_T2, D_loss_T2 = build_loss(T2_dis_real, T2_dis_fake)
    G_loss_T1, D_loss_T1 = build_loss(T1_dis_real, T1_dis_fake)

    if cal_G:
        ## freq loss
        frq_loss_T1_mid = torch.mean(torch.abs(T1_k_un - RF(rec_T1_mid_0, mask)))
        frq_loss_T1 = torch.mean(torch.abs(T1_k_un - RF(rec_T1_0, mask)))
        frq_loss_T2_mid = torch.mean(torch.abs(T2_k_un - RF(rec_T2_mid_0, mask)))
        frq_loss_T2 = torch.mean(torch.abs(T2_k_un - RF(rec_T2_0, mask)))
        frq_loss = args.alpha * (frq_loss_T2_mid + frq_loss_T2) + (1-args.alpha) * (frq_loss_T1_mid + frq_loss_T1)
        ## img loss
        img_loss_T1_mid = torch.mean(torch.abs(tar_T1 - rec_T1_mid))
        img_loss_T1_mid_0 = torch.mean(torch.abs(tar_T1 - rec_T1_mid_0))

        img_loss_T2_mid = torch.mean(torch.abs(tar_T2 - rec_T2_mid))
        img_loss_T2_mid_0 = torch.mean(torch.abs(tar_T2 - rec_T2_mid_0))

        img_loss_T1 = torch.mean(torch.abs(tar_T1 - rec_T1))
        img_loss_T1_0 = torch.mean(torch.abs(tar_T1 - rec_T1_0))

        img_loss_T2 = torch.mean(torch.abs(tar_T2 - rec_T2))
        img_loss_T2_0 = torch.mean(torch.abs(tar_T2 - rec_T2_0))

        smooth_T1 = torch.mean(total_variant(rec_T1))
        smooth_T2 = torch.mean(total_variant(rec_T2))


        rec_img_loss = args.alpha * (img_loss_T2_mid + img_loss_T2) + (1-args.alpha) * (img_loss_T1_mid+img_loss_T1)
        error_img_loss = args.alpha *(img_loss_T2_mid_0 + img_loss_T2_0) + (1-args.alpha) * (img_loss_T1_mid_0 + img_loss_T1_0)
        smooth_loss =  args.alpha * smooth_T1+ (1-args.alpha) * smooth_T2


        g_loss =(args.alpha* G_loss_T2 + (1-args.alpha) * G_loss_T1) + rec_img_loss*args.rate \
                    +error_img_loss*args.rate\
                    +frq_loss*1 +  smooth_loss*1e-3

        return g_loss, frq_loss, rec_img_loss, error_img_loss, smooth_loss, G_loss_T2, G_loss_T1
    else:
        return D_loss_T1, D_loss_T2, (args.alpha* D_loss_T2 + (1-args.alpha) * D_loss_T1) 




def RF(x_rec, mask, norm='ortho'):
    '''
    RF means R*F(input), F is fft, R is applying mask;
    return the masked k-space of x_rec,
    '''
    x_rec = x_rec.permute(0, 2, 3, 1)
    mask = mask.permute(0, 2, 3, 1)
    k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=norm)
    k_rec = torch.view_as_real(k_rec)
    k_rec *= mask
    k_rec = k_rec.permute(0, 3, 1, 2)
    return k_rec


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
    # net_G = Refine_G_DirectConnet(args)
    # net_G = DirectG(args)    
    # net_G = Generator(args)    
    net_G = MMGenerator(args)    

    G_optimizer = torch.optim.Adam(net_G.parameters(), lr= args.lr, betas=(0.5, 0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    net_G.to(device= args.device)

    ## discriminator
    net_D = Discriminator1(ndf = 32)
    D_optimizer = torch.optim.Adam(net_D.parameters(), lr= args.lr, betas=(0.5, 0.999))  ##discriminator 的学习率为Generator的两倍
    D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    net_D.to(device=args.device)

    # net_D_T1 = Discriminator()
    net_D_T1 = Discriminator1(ndf=32)
    D_optimizer_T1 = torch.optim.Adam(net_D.parameters(), lr= args.lr, betas=(0.5, 0.999))  ##discriminator 的学习率为Generator的两倍
    D_scheduler_T1 = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率

    net_D_T1.to(device=args.device)
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
            epoch_smooth_Loss = 0
            epoch_SSIM = 0
            epoch_PSNR = 0
            epoch_NMSE = 0
            ## val metrics
            val_SSIM = 0
            val_PSNR = 0
            val_NMSE = 0
            val_PSNR_ZF = 0
            val_SSIM_T1 = 0
            val_PSNR_T1 = 0
            val_NMSE_T1 = 0

            net_G.train()   #注意train的调用位置，防止在train的过程中进入eval模式中。

            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                '''Trainging stage'''
                for step,batch in enumerate(train_loader):
                    ## BATCH DATA
                    T2_masked_img = batch['target_masked_img'].to(device = args.device, dtype = torch.float32)
                    T2_masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                    T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)

                    T1_masked_img = batch['refer_masked_img'].to(device = args.device, dtype = torch.float32)
                    T1_masked_k = batch['refer_masked_K'].to(device = args.device, dtype = torch.float32)
                    T1_target_img = batch['refer_img'].to(device = args.device, dtype = torch.float32)

                    mask = batch['mask'].to(device = args.device, dtype = torch.float32)

                    for p in net_D.parameters():
                        p.requires_grad = True

                    for p in net_D_T1.parameters():
                        p.requires_grad = True
                    
                    #forward G
                    rec_mid_T1_pre, rec_mid_T2_pre, rec_mid_T1, rec_mid_T2, \
                        rec_img_T1_pre, rec_img_T2_pre, rec_img_T1, rec_img_T2\
                        = net_G(T1_masked_img, T1_masked_k,T2_masked_img, T2_masked_k, mask)
                   
                    # get the D out
                    D_real_pred = net_D(T2_target_img)
                    D_real_pred_T1 = net_D_T1(T1_target_img)

                    ############################# update D #################################
                    rec_clone = rec_img_T2.detach().clone()
                    rec_clone_T1 = rec_img_T1.detach().clone()

                    D_fake_pred = net_D(rec_clone)
                    D_fake_pred_T1 = net_D_T1(rec_clone_T1)

                    DLossT1, DLossT2, Dloss = cal_loss(T1_target_img, T1_masked_k, T2_target_img, T2_masked_k, mask, rec_mid_T1_pre,\
                            rec_mid_T1, rec_img_T1_pre, rec_img_T1, rec_mid_T2_pre, rec_mid_T2,\
                            rec_img_T2_pre, rec_img_T2, D_real_pred, D_fake_pred, D_real_pred_T1, D_fake_pred_T1, False, args)

                    epoch_D_Loss += Dloss.item()

                    D_optimizer.zero_grad()
                    DLossT2.backward()
                    D_optimizer.step()

                    D_optimizer_T1.zero_grad()
                    DLossT1.backward()
                    D_optimizer_T1.step()

                    for p in net_D.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)
                    for p in net_D_T1.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)
############################################# update G ######################################
                    if step % args.critic ==0:

                        for p in net_D.parameters():
                            p.requires_grad = False
                        for p in net_D_T1.parameters():
                            p.requires_grad = False
                        D_fake_pred_update = net_D(rec_img_T2)
                        D_fake_pred_T1_update = net_D_T1(rec_img_T1)


                        #calculate G loss
                        FullLoss, frqloss_T2, IMloss_T2, errorloss_T2, smooth_loss, Gloss, GlossT1\
                        = cal_loss(T1_target_img, T1_masked_k, T2_target_img, T2_masked_k, mask, rec_mid_T1_pre,\
                                    rec_mid_T1, rec_img_T1_pre, rec_img_T1, rec_mid_T2_pre, rec_mid_T2,\
                                    rec_img_T2_pre, rec_img_T2, D_real_pred, D_fake_pred_update, D_real_pred_T1, D_fake_pred_T1_update,\
                                    True, args)


                        #Optimize parameters
                        G_optimizer.zero_grad()
                        FullLoss.backward()
                        G_optimizer.step()

                    ## calculate the psnr
                    T2_target_img = output2complex2(T2_target_img, args.isZeroToOne)
                    rec_img_T2 = output2complex2(rec_img_T2, args.isZeroToOne)
                    T1_target_img = output2complex2(T1_target_img, args.isZeroToOne)
                    rec_img_T1 = output2complex2(rec_img_T1, args.isZeroToOne)

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
                    epoch_smooth_Loss += smooth_loss.item()
                    epoch_G_Loss += Gloss.item()
                    epoch_PSNR += psnr
                    epoch_NMSE += nmse
                    epoch_SSIM += ssim

                    progress += 100*T2_target_img.shape[0]/len(train_dataset)
                    
                    pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'Gadvloss: ': Gloss.item(), 'GadvlossT1: ': GlossT1.item(), 'Dadvloss: ': Dloss.item(),'ImT2': IMloss_T2.item(),
                                    'ImerrorT2': errorloss_T2.item(),'KT2loss': frqloss_T2.item(),'TVloss': smooth_loss.item(), 'PSNR: ' : psnr, 'NMSE: ':nmse, 
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
                        val_T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                        
                        val_T1_masked_img = batch['refer_masked_img'].to(device = args.device, dtype = torch.float32)
                        val_T1_masked_k = batch['refer_masked_K'].to(device = args.device, dtype = torch.float32)
                        val_T1_target_img = batch['refer_img'].to(device = args.device, dtype = torch.float32)

                        val_mask = batch['mask'].to(device = args.device, dtype = torch.float32)

                        val_rec_mid_T1_pre, val_rec_mid_T2_pre, val_rec_mid_T1, val_rec_mid_T2,\
                             val_rec_img_T1_pre, val_rec_img_T2_pre, val_rec_img_T1, val_rec_img_T2, \
                                 = net_G(val_T1_masked_img,val_T1_masked_k, val_T2_masked_img, val_T2_masked_k, val_mask)

                        ## T2 image visualization
                        val_T2_target = output2complex2(val_T2_target_img, args.isZeroToOne)
                        val_T2_rec = output2complex2(val_rec_img_T2, args.isZeroToOne)
                        val_T2_rec_mid = output2complex2(val_rec_mid_T2, args.isZeroToOne)
                        val_ZF_img = output2complex2(val_T2_masked_img, args.isZeroToOne)
                        ## T1 image visualization
                        val_T1_target = output2complex2(val_T1_target_img, args.isZeroToOne)
                        val_T1_rec = output2complex2(val_rec_img_T1, args.isZeroToOne)

                        val_psnr = calculatePSNR(val_T2_target.detach().cpu().numpy().squeeze(), val_T2_rec.detach().cpu().numpy().squeeze(), True)
                        val_nmse = calcuNMSE(val_T2_target.detach().cpu().numpy().squeeze(), val_T2_rec.detach().cpu().numpy().squeeze(), True)
                        val_ssim = calculateSSIM(val_T2_target.detach().cpu().numpy().squeeze(), val_T2_rec.detach().cpu().numpy().squeeze(), True)
                        val_zf_psnr = calculatePSNR(val_T2_target.detach().cpu().numpy().squeeze(), val_ZF_img.detach().cpu().numpy().squeeze(), True)

                        val_psnr_T1 = calculatePSNR(val_T1_target.detach().cpu().numpy().squeeze(), val_T1_rec.detach().cpu().numpy().squeeze(), True)
                        val_nmse_T1 = calcuNMSE(val_T1_target.detach().cpu().numpy().squeeze(), val_T1_rec.detach().cpu().numpy().squeeze(), True)
                        val_ssim_T1 = calculateSSIM(val_T1_target.detach().cpu().numpy().squeeze(), val_T1_rec.detach().cpu().numpy().squeeze(), True)

                        val_psnr = val_psnr/val_T2_target.shape[0]
                        val_nmse = val_nmse/val_T2_target.shape[0]
                        val_ssim = val_ssim/val_T2_target.shape[0]
                        val_zf_psnr = val_zf_psnr/val_T2_target.shape[0]
                        val_psnr_T1 = val_psnr_T1/val_T2_target.shape[0]
                        val_nmse_T1 = val_nmse_T1/val_T2_target.shape[0]
                        val_ssim_T1 = val_ssim_T1/val_T2_target.shape[0]

                        val_PSNR += val_psnr
                        val_NMSE += val_nmse
                        val_SSIM += val_ssim
                        val_PSNR_ZF += val_zf_psnr
                        val_PSNR_T1 += val_psnr_T1
                        val_SSIM_T1 += val_ssim_T1
                        val_NMSE_T1 += val_nmse_T1

                        logging.info("Step: {}  , RecPSNR: {:5}, RecSSIM: {:5}, RecNMSE: {:5}, ZFPSNR: {:5}".format(
                            step + 1, val_psnr, val_ssim, val_nmse, val_zf_psnr)) 
                # Schedular update
                G_scheduler.step(-val_PSNR)
                D_scheduler.step(-val_PSNR)
                D_scheduler_T1.step(-val_PSNR)


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
                writer.add_scalar('train/TVloss', epoch_smooth_Loss / len(train_loader), epoch)
                writer.add_scalar('train/SSIM', epoch_SSIM / len(train_loader), epoch)
                writer.add_scalar('train/PSNR', epoch_PSNR / len(train_loader), epoch)
                writer.add_scalar('train/NMSE', epoch_NMSE / len(train_loader), epoch)
                # writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('train/G_AdvLoss', epoch_G_Loss / len(train_loader), epoch)
                writer.add_scalar('train/D_AdvLoss', epoch_D_Loss / len(train_loader), epoch)

                writer.add_scalar('validation/SSIM', val_SSIM / len(val_loader), epoch)
                writer.add_scalar('validation/PSNR', val_PSNR / len(val_loader), epoch)
                writer.add_scalar('validation/NMSE', val_NMSE / len(val_loader), epoch)
                writer.add_scalar('validation/T1SSIM', val_SSIM_T1 / len(val_loader), epoch)
                writer.add_scalar('validation/T1PSNR', val_PSNR_T1 / len(val_loader), epoch)
                writer.add_scalar('validation/T1NMSE', val_NMSE_T1 / len(val_loader), epoch)
                writer.add_scalar('validation/ZFPSNR', val_zf_psnr / len(val_loader), epoch)


            if args.tb_write_image:
                writer.add_images('train/T2_target',T2_target_img, epoch)
                writer.add_images('train/T2_rec', rec_img_T2, epoch)
                writer.add_images('train/T1_target',T1_target_img, epoch)
                writer.add_images('train/T1_rec', rec_img_T1, epoch)

                writer.add_images('validation/T2_target', val_T2_target, epoch)
                writer.add_images('validation/T2_rec', val_T2_rec, epoch)
                # writer.add_images('validation/T2_rec_mid', val_T2_rec_mid, epoch)
                # writer.add_images('validation/ZF', val_ZF_img, epoch)
                writer.add_images('validation/T1_target', val_T1_target, epoch)
                writer.add_images('validation/T1_rec', val_T1_rec, epoch)

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
