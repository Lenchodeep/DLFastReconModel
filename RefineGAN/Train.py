import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from os.path import join
import torch
import logging
import yaml
import shutil
import torch.optim as optim
from model import *
from dataset import IXIDatasetRefinrGan
from FastMRIDataset import FastMRIDatasetRefinrGan
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm
from utils import cal_psnr, calculatePSNR, output2complex, output2complex2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from NetLoss import *
from torch.optim.lr_scheduler import ExponentialLR

def train(args):
    if not os.path.exists(args.savePngPath):
        os.makedirs(args.savePngPath)  
    ## INIT TENSORBOARD
    writer = SummaryWriter(log_dir=args.tb_dir+'/tensorboard')
    ## INIT THE DATAS
    trainDataset = FastMRIDatasetRefinrGan(args.data_dir, args, isTrainData=True, transform=None)
    valDataset = FastMRIDatasetRefinrGan(args.data_dir, args, isTrainData=False, transform=None)

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True
                                , num_workers=args.train_num_workers, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=True
                                , num_workers=args.val_num_workers, pin_memory=True, drop_last=True)
    
    # INIT THE MODELS AND OPTIMIZERS
    G_model = Refine_G(args)
    G_optimizer = optim.Adam(G_model.parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-3, weight_decay=1e-10)
    scheduler_lr_G = optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.8, last_epoch=-1)
    G_model = G_model.to(args.device)
    
    D_model = Discriminator()
    D_optimizer = optim.Adam(D_model.parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-3, weight_decay=1e-10)
    scheduler_lr_D = optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.8, last_epoch=-1)
    D_model = D_model.to(args.device)

    best_val_psnr = 0
    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')
    
    global GLOBAL_STEP
    GLOBAL_STEP = 0

    D_model.train()

    for epoch in range(0, args.epochs):
        progress = 0
        total_recon_img_loss_train = 0
        total_error_img_loss_train = 0
        total_gradient_loss_train = 0
        total_frq_loss_train = 0
        total_d_loss_train = 0
        total_g_loss_train = 0
        total_full_loss_train = 0
        G_model.train()
        
        with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
            for data in trainLoader:
                tar_im_A = data['im_A'].to(args.device,  dtype = torch.float32)
                tar_im_B = data['im_B'].to(args.device,  dtype = torch.float32)
                ZF_im_A = data['im_A_und'].to(args.device,  dtype = torch.float32)
                ZF_im_B = data['im_B_und'].to(args.device,  dtype = torch.float32)
                k_A_und = data['k_A_und'].to(args.device,  dtype = torch.float32)               
                k_B_und = data['k_B_und'].to(args.device,  dtype = torch.float32)
                mask = data['mask'].to(args.device, dtype = torch.float32)


                for p in D_model.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                Sp1, S1, Tp1, T1 = G_model(ZF_im_A, k_A_und, mask)
                Sp2, S2, Tp2, T2 = G_model(ZF_im_B, k_B_und, mask)


                S1_dis_real = D_model(tar_im_A)
                S2_dis_real = D_model(tar_im_B)

                ############################################# 1.1 update discriminator #############################################
                with torch.no_grad():
                    S1_clone = S1.detach().clone()
                    T1_clone = T1.detach().clone()
                    S2_clone = S2.detach().clone()
                    T2_clone = T2.detach().clone() 

                S1_dis_fake = D_model(S1_clone)
                T1_dis_fake = D_model(T1_clone)
                S2_dis_fake = D_model(S2_clone)
                T2_dis_fake = D_model(T2_clone)
                # print('test', S1_dis_real, S1_dis_fake)
                loss_d,d_loss_list =\
                    cal_loss(tar_im_A, k_A_und, tar_im_B, k_B_und, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2,
                            S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake, T2_dis_fake, cal_G=False)

                total_d_loss_train = total_d_loss_train+loss_d.item()

                D_optimizer.zero_grad()
                loss_d.backward()
                D_optimizer.step()

                for p in D_model.parameters():
                    p.data.clamp_(-args.weight_clip, args.weight_clip)
                
                 ############################################# 1.2 update generator #############################################
                for p in D_model.parameters():  # reset requires_grad
                    p.requires_grad = False  # they are set to False below in netG update

                S1_dis_fake = D_model(S1)
                T1_dis_fake = D_model(T1)
                S2_dis_fake = D_model(S2)
                T2_dis_fake = D_model(T2)

                loss_g,recon_img_loss,error_img_loss,gradient_loss,frq_loss,G_loss, g_loss_list = \
                    cal_loss(tar_im_A, k_A_und, tar_im_B, k_B_und, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2,
                             S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake, T2_dis_fake, cal_G=True)

                G_optimizer.zero_grad()
                loss_g.backward()
                G_optimizer.step()

                total_recon_img_loss_train = total_recon_img_loss_train+recon_img_loss.item()
                total_error_img_loss_train = total_error_img_loss_train+error_img_loss.item()
                total_gradient_loss_train = total_gradient_loss_train+gradient_loss.item()
                total_frq_loss_train = total_frq_loss_train+frq_loss.item()
                total_g_loss_train = total_g_loss_train+G_loss.item()
                total_full_loss_train = total_full_loss_train+loss_g.item()
                
                progress += 100*args.batch_size/len(trainDataset)
                pbar.set_postfix(**{'FullLoss ': loss_g.item(),'RECON IM Loss ':recon_img_loss.item(),'ERROR IM Loss ': error_img_loss.item(),
                                    'Gradient Loss ' : gradient_loss.item(), 'K space Loss ': frq_loss.item(),
                                    'Adv G ' : G_loss.item(),'Adv D ': loss_d.item(), 'Prctg of train set': progress})
                pbar.update(args.batch_size)# current batch size
     
        if args.tb_write_image:
            writer.add_images('train/Fully_sampled_images', output2complex2(tar_im_A, args.isZeroToOne), epoch)
            writer.add_images('train/zf_img', output2complex2(ZF_im_A, args.isZeroToOne), epoch)
            writer.add_images('train/recon_img', output2complex2(T1, args.isZeroToOne), epoch)
            writer.add_images('train/recon_mid_img', output2complex2(S1, args.isZeroToOne), epoch)
        # record training eval
        if args.tb_write_loss:
            writer.add_scalar('Training/FullLoss', total_full_loss_train/len(trainLoader), global_step=epoch)
            writer.add_scalar('Training/Recon IM Loss', total_recon_img_loss_train/len(trainLoader), global_step=epoch)
            writer.add_scalar('Training/Error IM Loss', total_error_img_loss_train/len(trainLoader), global_step=epoch)
            writer.add_scalar('Training/Gradient Loss', total_gradient_loss_train/len(trainLoader), global_step=epoch)
            writer.add_scalar('Training/K Loss', total_frq_loss_train/len(trainLoader), global_step=epoch)

        # scheduler_lr_G.step()
        # scheduler_lr_D.step()    

        ####################### 2. validate #######################
        base_psnr = 0
        test_psnr = 0
        G_model.eval()
        with torch.no_grad():
            
            for step,data in enumerate(valLoader):
                val_tar_im_A = data['im_A'].to(args.device,  dtype = torch.float32)
                val_ZF_im_A = data['im_A_und'].to(args.device,  dtype = torch.float32)
                val_k_A_und = data['k_A_und'].to(args.device,  dtype = torch.float32)               
                val_mask = data['mask'].to(args.device, dtype = torch.float32)



                val_Sp1, val_S1, val_Tp1, val_T1 = G_model(val_ZF_im_A, val_k_A_und, val_mask)
                # tar_im_A_cpu = val_T1.detach().cpu().numpy()
                # print(tar_im_A_cpu.shape,'popop')
                # plt.figure()
                # plt.imshow(tar_im_A_cpu[0,0,...], plt.cm.gray)
                # plt.show()
                val_T1 = output2complex2(val_T1, args.isZeroToOne)
                val_tar_im_A = output2complex2(val_tar_im_A, args.isZeroToOne)
                val_ZF_im_A = output2complex2(val_ZF_im_A, args.isZeroToOne)
                val_S1 = output2complex2(val_S1, args.isZeroToOne)
                # print(val_T1.max(), val_T1.min(),'uuuuuuuuuuuuuuuuuu', val_tar_im_A.max(), val_tar_im_A.min())
                # base_psnr += cal_psnr(val_tar_im_A, val_ZF_im_A, maxp=1.)
                # test_psnr += cal_psnr(val_tar_im_A, val_T1, maxp=1.)
                base_psnr += calculatePSNR(val_tar_im_A.detach().cpu().numpy().squeeze(), val_ZF_im_A.detach().cpu().numpy().squeeze(), True)
                test_psnr += calculatePSNR(val_tar_im_A.detach().cpu().numpy().squeeze(), val_T1.detach().cpu().numpy().squeeze(),True)

                logging.info("Step: {}  base PSNR: {:8}, test PSNR: {:8}".format(
                    step + 1, base_psnr, test_psnr)) 

        base_psnr /= len(valDataset)
        test_psnr /= len(valDataset)
        logging.info("Epoch: {}  base PSNR: {:8}, test PSNR: {:8}".format(
                    epoch + 1, base_psnr, test_psnr)) 
        G_model.train()

        if best_val_psnr < test_psnr:
            best_val_psnr = test_psnr
            torch.save({
            'epoch': epoch,
            'G_model_state_dict': G_model.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'G_scheduler_state_dict': scheduler_lr_G.state_dict(),
            'D_model_state_dict': D_model.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
            'D_scheduler_state_dict':  scheduler_lr_D.state_dict()
            }, args.output_dir + f'Best_CP_epoch.pth')

        if (epoch + 1) % args.save_step == 0:
            torch.save({
            'epoch': epoch,
            'G_model_state_dict': G_model.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'G_scheduler_state_dict': scheduler_lr_G.state_dict(),
            'D_model_state_dict': D_model.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
            'D_scheduler_state_dict':  scheduler_lr_D.state_dict()
            }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        if args.tb_write_loss:
            writer.add_scalar("loss/G_loss_total", loss_g, epoch)
            writer.add_scalar("loss/D_loss_total", loss_d, epoch)
            writer.add_scalar("loss/G_loss_AA", g_loss_list[0], epoch)
            writer.add_scalar("loss/G_loss_Aa", g_loss_list[1], epoch)
            writer.add_scalar("loss/recon_img_AA", g_loss_list[2], epoch)
            writer.add_scalar("loss/recon_img_Aa", g_loss_list[3], epoch)
            writer.add_scalar("loss/error_img_AA", g_loss_list[4], epoch)
            writer.add_scalar("loss/error_img_Aa", g_loss_list[5], epoch)
            writer.add_scalar("loss/recon_frq_AA", g_loss_list[6], epoch)
            writer.add_scalar("loss/recon_frq_Aa", g_loss_list[7], epoch)       
            writer.add_scalar("loss/smoothness_AA", g_loss_list[8], epoch)
            writer.add_scalar("loss/smoothness_Aa", g_loss_list[9], epoch)
            writer.add_scalar("loss/D_loss_AA", d_loss_list[0], epoch)
            writer.add_scalar("loss/D_loss_Aa", d_loss_list[1], epoch)
            writer.add_scalar("loss/D_loss_AB", d_loss_list[2], epoch)
            writer.add_scalar("loss/D_loss_Ab", d_loss_list[3], epoch)

            writer.add_scalar("val/base_psnr", base_psnr, epoch)
            writer.add_scalar("val/test_psnr", test_psnr, epoch)
            writer.add_scalar('Training/LR', G_optimizer.param_groups[0]['lr'], global_step=epoch)

            writer.add_images('val/Fully_sampled_images', val_tar_im_A, epoch)
            writer.add_images('val/zf_img', val_ZF_im_A, epoch)
            writer.add_images('val/recon_img', val_T1, epoch)
            writer.add_images('val/recon_mid_img', val_S1, epoch)




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

    shutil.copyfile('config.yaml',os.path.join(args.output_dir,'config.yaml'))
    logging.info(f'Using device {args.device}')
    train(args)
if __name__ == "__main__":
    main()