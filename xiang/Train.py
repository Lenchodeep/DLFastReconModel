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
from Dataset import FastMRIDataset
from utils import *
from Model import *
from torch.utils.tensorboard import SummaryWriter

def train(args):
    '''Init the model'''
    #init dataloader

    train_dataset = FastMRIDataset(args, True, None)
    val_dataset = FastMRIDataset(args, False, None)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True
                            , num_workers=args.train_num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True
                            , num_workers= args.val_num_workers, pin_memory=True, drop_last=True)
   
    model = Xnet()    

    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    model.to(device= args.device)


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



    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')
    
    maxpsnr = 0

    try:
        for epoch in range(0, args.epochs):
            progress = 0 

            ## train metrics
            epoch_Img_Loss_T2 = 0
            epoch_SSIM = 0
            epoch_PSNR = 0
            epoch_NMSE = 0
            ## val metrics
            val_SSIM = 0
            val_PSNR = 0
            val_NMSE = 0
            val_PSNR_ZF = 0
  

            model.train()   #注意train的调用位置，防止在train的过程中进入eval模式中。

            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                '''Trainging stage'''
                for step,batch in enumerate(train_loader):
                    ## BATCH DATA
                    T2_masked_img = batch['target_masked_img'].to(device = args.device, dtype = torch.float32)
                    T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)

                    T1_masked_img = batch['refer_masked_img'].to(device = args.device, dtype = torch.float32)
                    T1_target_img = batch['refer_img'].to(device = args.device, dtype = torch.float32)

                    rec_img_T2 = model(T1_masked_img, T2_masked_img)
          
                    imgloss = F.mse_loss(rec_img_T2,T2_target_img)*1000
                    
                    optimizer.zero_grad()
                    imgloss.backward()
                    optimizer.step()

                    optimizer

                    psnr = calculatePSNR(T2_target_img.detach().cpu().numpy().squeeze(), rec_img_T2.detach().cpu().numpy().squeeze(), True)
                    nmse = calcuNMSE(T2_target_img.detach().cpu().numpy().squeeze(), rec_img_T2.detach().cpu().numpy().squeeze(), True)
                    ssim = calculateSSIM(T2_target_img.detach().cpu().numpy().squeeze(), rec_img_T2.detach().cpu().numpy().squeeze(), True)

                    psnr = psnr/T2_target_img.shape[0]
                    nmse = nmse/T2_target_img.shape[0]
                    ssim = ssim/T2_target_img.shape[0]

                    # update fullLoss
                    epoch_Img_Loss_T2 += imgloss.item()
                    epoch_PSNR += psnr
                    epoch_NMSE += nmse
                    epoch_SSIM += ssim

                    progress += 100*T2_target_img.shape[0]/len(train_dataset)
                    
                    pbar.set_postfix(**{'ImT2': imgloss.item(),'PSNR: ' : psnr, 'NMSE: ':nmse, 
                                    'SSIM: ':ssim,  'Prctg of train set': progress})

                    pbar.update(T2_target_img.shape[0])# current batch size


                '''valdation stage'''
                model.eval()
                n_val = len(val_loader)  # number of batch
                with torch.no_grad():

                    for batch in tqdm(val_loader):
                        val_T2_masked_img = batch['target_masked_img'].to(device = args.device, dtype = torch.float32)
                        val_T2_target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                        
                        val_T1_masked_img = batch['refer_masked_img'].to(device = args.device, dtype = torch.float32)
                        val_T1_target_img = batch['refer_img'].to(device = args.device, dtype = torch.float32)

                        val_rec_img_T2 =  model(val_T1_masked_img, val_T2_masked_img)



                        val_psnr = calculatePSNR(val_T2_target_img.detach().cpu().numpy().squeeze(), val_rec_img_T2.detach().cpu().numpy().squeeze(), True)
                        val_nmse = calcuNMSE(val_T2_target_img.detach().cpu().numpy().squeeze(), val_rec_img_T2.detach().cpu().numpy().squeeze(), True)
                        val_ssim = calculateSSIM(val_T2_target_img.detach().cpu().numpy().squeeze(), val_rec_img_T2.detach().cpu().numpy().squeeze(), True)
                        val_zf_psnr = calculatePSNR(val_T2_target_img.detach().cpu().numpy().squeeze(), val_T2_masked_img.detach().cpu().numpy().squeeze(), True)

                        val_psnr = val_psnr/val_T2_target_img.shape[0]
                        val_nmse = val_nmse/val_T2_target_img.shape[0]
                        val_ssim = val_ssim/val_T2_target_img.shape[0]
                        val_zf_psnr = val_zf_psnr/val_T2_target_img.shape[0]
      

                        val_PSNR += val_psnr
                        val_NMSE += val_nmse
                        val_SSIM += val_ssim
                        val_PSNR_ZF += val_zf_psnr
       

                        logging.info("Step: {}  , RecPSNR: {:5}, RecSSIM: {:5}, RecNMSE: {:5}, ZFPSNR: {:5}".format(
                            step + 1, val_psnr, val_ssim, val_nmse, val_zf_psnr)) 


                logging.info("Epoch: {}  , RecPSNR: {:5}, RecSSIM: {:5}, RecNMSE: {:5}, ZFPSNR: {:5}".format(
                            epoch + 1, val_PSNR/n_val, val_SSIM/n_val, val_NMSE/n_val, val_PSNR_ZF/n_val)) 
                scheduler.step(-val_PSNR/n_val)
                model.train()


            if maxpsnr < val_PSNR:
                maxpsnr = val_PSNR
                torch.save({
                    'epoch': epoch,
                    'moel_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, args.output_dir + f'Best_PSNR_epoch.pth')
                logging.info(f'Best PSNR Checkpoint saved !')

                        #Write to TB:
            if args.tb_write_loss:
                writer.add_scalar('train/ImLossT2', epoch_Img_Loss_T2 / len(train_loader), epoch)
                writer.add_scalar('train/SSIM', epoch_SSIM / len(train_loader), epoch)
                writer.add_scalar('train/PSNR', epoch_PSNR / len(train_loader), epoch)
                writer.add_scalar('train/NMSE', epoch_NMSE / len(train_loader), epoch)
                # writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)

                writer.add_scalar('validation/SSIM', val_SSIM / len(val_loader), epoch)
                writer.add_scalar('validation/PSNR', val_PSNR / len(val_loader), epoch)
                writer.add_scalar('validation/NMSE', val_NMSE / len(val_loader), epoch)
                writer.add_scalar('validation/ZFPSNR', val_zf_psnr / len(val_loader), epoch)


            if args.tb_write_image:
                writer.add_images('train/T2_target',T2_target_img, epoch)
                writer.add_images('train/T2_rec', rec_img_T2, epoch)

                writer.add_images('validation/T2_target', val_T2_target_img, epoch)
                writer.add_images('validation/T2_rec', val_rec_img_T2, epoch)



                  #Save Checkpoint per 5 epoch
            if (epoch % args.save_step) == (args.save_step-1):
                torch.save({
                    'epoch': epoch,
                    'moel_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    except KeyboardInterrupt:
        torch.save({
                    'epoch': epoch,
                    'moel_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
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
