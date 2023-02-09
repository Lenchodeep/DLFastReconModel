import os
import numpy as np
import torch
from tqdm import tqdm
import shutil
import yaml
import sys
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, dataset, random_split
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

from model import *
from IXIDataSet import *
from myEarlystop import EarlyStopping
from NetLoss import *
from eval import eval_net

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
        nn.init.zeros_(m.bias)

def train(args):
    G_model = Generator(args.input_channel,args.output_channel,args.num_filters)
    # G_model.apply(weight_init)
    G_model.to(device=args.device)
    G_optimizer = optim.Adam(G_model.parameters(), lr =args.lr, betas=[0.9,0.999])
    G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=args.lrStep, gamma=args.lrgamma) 

    D_model = Discriminator(args.input_channel, args.output_channel, args.num_filters)
    D_model.apply(weight_init)
    D_optimizer = optim.Adam(D_model.parameters(), lr =args.lr, betas=[0.9,0.999])
    D_model.to(device= args.device)

    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
    ])

    dataset = IXIDataSet(args.imageDataDir, args, args.isTrain, transform = transform)
    ##split datas to trainning data and val data
    n_train = int(len(dataset)* (1-args.val_percent))
    n_val = int(len(dataset) - n_train)
    trainData , valData = random_split(dataset , [n_train , n_val])

    train_loader = DataLoader(trainData, batch_size = args.batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True)
    val_loader = DataLoader(valData, batch_size = args.batch_size, shuffle = True, num_workers= args.val_num_workers, pin_memory=True, drop_last=True)

    early_stopping = EarlyStopping(patience=args.stopPatience, savepath=args.output_dir ,cpname = "CP_early_stop",verbose=True)
    

    if args.tb_write_loss or args.tb_write_image:
        writer = SummaryWriter(log_dir=args.tb_dir+'/tensorboard')

    loss = NetLoss(args)

    minValLoss = 100000.0
    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')

    try:
        for epoch in range(0, args.epochs):
            G_model.train()
            D_model.train()
            progress = 0
            epoch_Full_Loss = 0
            epoch_D_Loss = 0
            epoch_G_Loss = 0
            epoch_Img_Loss = 0
            epoch_ssim = 0
            epoch_gradientLoss = 0
            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit='imgs') as pbar:
                for batch in train_loader:
                    fold_img = batch['masked_img'].to(device = args.device, dtype = torch.float32)
                    target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                    recon_img = G_model(fold_img)

                    if args.GAN_training:
                        real_D_example = target_img.detach()  #不需要梯度
                        fake_D_example = recon_img
                        D_real_pred = D_model(real_D_example)
                        D_fake_pred = D_model(fake_D_example)
                    else:
                        D_fake_pred = None
                    

                    FullLoss, adv_Loss, ImL1, ms_ssim, gradient_Loss, _= loss.calc_gen_loss(recon_img, target_img, D_fake_pred)

                    if args.GAN_training:
                        D_fake_detach = D_model(fake_D_example.detach())
                        D_real_loss, D_fake_loss, DLoss = loss.calc_disc_loss(D_real_pred, D_fake_detach)

                        train_D = adv_Loss.item()<D_real_loss.item()*1.5

                     #Optimize parameters
                    #update G
                    if args.GAN_training:
                        set_grad(D_model, False) #No D update

                    G_optimizer.zero_grad()
                    FullLoss.backward()
                    G_optimizer.step()

                    #update D
                    if args.GAN_training:
                        set_grad(D_model, True)  # enable backprop for D
                        if train_D:
                            D_optimizer.zero_grad()
                            DLoss.backward()
                            D_optimizer.step()


                    epoch_Full_Loss += FullLoss.item()
                    epoch_G_Loss += adv_Loss.item()
                    epoch_D_Loss += DLoss.item()
                    epoch_Img_Loss += ImL1.item()
                    epoch_ssim += ms_ssim.item()
                    epoch_gradientLoss += gradient_Loss.item()

                    #Update progress bar
                    progress += 100*target_img.shape[0]/len(trainData)   
                    if args.GAN_training:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'ImL1': ImL1.item(), 'ms-ssim': ms_ssim.item(), 'gradient_Loss': gradient_Loss.item(),
                                            'Adv G': adv_Loss.item(), 'Adv D' : DLoss.item(),'Adv D - Real' : D_real_loss.item(),
                                            'Adv D - Fake' : D_fake_loss.item(),'Train D': train_D, 'Prctg of train set': progress})
                    else:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'ImL1': ImL1.item(),
                                            'Prctg of train set': progress})
                    pbar.update(target_img.shape[0])# current batch size

            # G_model.eval()
            val_rec_img, val_full_img, val_FullLoss, val_ImL1, val_msssim, val_psnr,\
                 val_gradientLoss, val_mse = eval_net(G_model, val_loader, loss, args.device)
            # G_model.train()

            logging.info('Validation full score: {}, ImL1: {}, MSE: {}, PSNR: {}, ssim: {}'
                         .format(val_FullLoss,  val_ImL1, val_mse, val_psnr, val_msssim))
            G_scheduler.step(val_FullLoss)

            #using mse loss as judge standard
            early_stopping(val_mse , G_model)
            if early_stopping.early_stop == True:
                print("Early stopping!")
                break

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

            #Write to TB:
            if args.tb_write_loss:
                writer.add_scalar('train/FullLoss', epoch_Full_Loss / len(train_loader), epoch)
                writer.add_scalar('train/ImL1',  epoch_Img_Loss / len(train_loader), epoch)
                writer.add_scalar('train/SSIM',  epoch_ssim / len(train_loader), epoch)
                writer.add_scalar('train/GradientLoss',  epoch_gradientLoss / len(train_loader), epoch)
                writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                if args.GAN_training:
                    writer.add_scalar('train/G_AdvLoss',  epoch_G_Loss / len(train_loader), epoch)
                    writer.add_scalar('train/D_AdvLoss',epoch_D_Loss / len(train_loader), epoch)

                writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
                writer.add_scalar('validation/ImL1', val_ImL1, epoch)
                writer.add_scalar('validation/GradientLoss', val_gradientLoss, epoch)
                writer.add_scalar('validation/SSIM', val_msssim, epoch)
                writer.add_scalar('validation/PSNR', val_psnr, epoch)

            if args.tb_write_image:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/rec_images', recon_img, epoch)
                # writer.add_images('train/Kspace_rec_images', rec_mid_img, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/rec_images', val_rec_img, epoch)
                # writer.add_images('validation/Kspace_rec_images', val_F_rec_Kspace, epoch)

            #Save Checkpoint per 5 epoch
            if (epoch % args.save_step) == (args.save_step-1):
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': G_model.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'G_scheduler_state_dict': G_scheduler.state_dict(),
                    'D_model_state_dict': D_model.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
        
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
    try:
        os.mkdir(args.output_dir)
        logging.info('Created checkpoint directory')
    except OSError:
        pass

    # Copy configuration file to output directory
    shutil.copyfile('config.yaml',os.path.join(args.output_dir,'config.yaml'))
    logging.info(f'Using device {args.device}')

    train(args)
if __name__ == "__main__":
    main()