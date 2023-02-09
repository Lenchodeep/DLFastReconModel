"""
This is a train function for gan.
"""
import sys
import os
import torch 
import shutil
import logging
from torch.nn.functional import fold
from torchvision import transforms
import yaml
from  tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from IXIDataSet import IXIDatasetImage
from utils import png_saver, EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from NetLoss import *
from models import *
from Eval import *


def train(args):
    #Init the Generator Model
    # G_model = PsychoGenerator(args.inchannels, args.outchannels, args.nb_filters, args.attName)
    G_model = SepWnet(args)
    logging.info('Using SepWnet {} as Generator'.format(args.netV))
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr = args.lr, betas = (0.5,0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience = 5, factor=0.1) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    G_model.to(device= args.device)

    #Init the Discriminator Model
    D_model = SepDiscriminatorV2(inchannels=1, ndf=64, isFc=True, num_layers=4)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr = args.lr, betas=(0.5, 0.999))  ##discriminator 的学习率为Generator的两倍
    D_model.to(device=args.device)

    ## define the transfor for data augmentation
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
    ])

    #init dataloader
    dataset = IXIDatasetImage(args.imageDataDir, args, args.isTrain, transform= transform)
        ##split datas to trainning data and val data
    n_train = int(len(dataset)* (1-args.val_percent))
    n_val = int(len(dataset) - n_train)
    trainData , valData = random_split(dataset , [n_train , n_val])

    train_loader = DataLoader(trainData, batch_size = args.batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True)
    val_loader = DataLoader(valData, batch_size = args.batch_size, shuffle = True, num_workers= args.val_num_workers, pin_memory=True, drop_last=True)
    #Init earlystop
    early_stopping = EarlyStopping(patience=args.stopPatience, savepath=args.output_dir ,cpname = "CP_early_stop",verbose=True)


    #Init tensor board writer
    if args.tb_write_loss or args.tb_write_image:
        writer = SummaryWriter(log_dir=args.tb_dir+'/tensorboard')

    #Init loss object
    loss = NetLoss(args)

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
            tot_ImL2 = 0
            tot_ImL1 = 0
            tot_KL2 = 0
            tot_adverD = 0
            tot_adverG = 0
            tot_fullLoss = 0
            tot_SSIM = 0
            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                ##Train loop
                for batch in train_loader:

                    fold_img = batch['fold_img'].to(device = args.device, dtype = torch.float32)
                    target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)

                    #forward G
                    rec_img, rec_mid_img = G_model(fold_img)

                    D_fake_pred = D_model(rec_img)
                    D_real_pred = D_model(target_img)

                    #calculate G loss
                    FullLoss, ImL2, ImL1, kL2, g_adv , ssimValue = loss.calc_gen_loss(rec_img, target_img, D_fake_pred)
                    D_real_loss, D_fake_loss, DLoss = loss.calc_disc_loss(D_real_pred, D_fake_pred)


                    D_optimizer.zero_grad()
                    DLoss.backward(retain_graph=True)

                    
                    G_optimizer.zero_grad()
                    FullLoss.backward()

                    D_optimizer.step()

                    G_optimizer.step()
                    train_D = 1

                    tot_ImL1 += ImL1.item()
                    tot_ImL2 += ImL2.item()
                    tot_KL2 += kL2.item()
                    tot_adverD += DLoss.item()
                    tot_adverG += g_adv.item()
                    tot_fullLoss += FullLoss.item()
                    tot_SSIM +=ssimValue.item()
                    #Update progress bar
                    progress += 100*target_img.shape[0]/len(trainData)
                    if args.GAN_training:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(), 'ssim': ssimValue.item(),
                                            'KspaceL2': kL2.item(),'Adv G': g_adv.item(),'Adv D - Real' : D_real_loss.item(),
                                            'Adv D - Fake' : D_fake_loss.item(),'Train D': train_D, 'Prctg of train set': progress})
                    else:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                            'KspaceL2': kL2.item(), 'Prctg of train set': progress})
                    pbar.update(target_img.shape[0])# current batch size

            # On epoch end
            # Validation
            val_rec_img, val_rec_mid_img, val_full_img, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR, val_SSIM =\
                eval_net(G_model, val_loader, loss, args.device)
            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}'
                         .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR, val_SSIM))
            # Schedular update
            G_scheduler.step(val_FullLoss)

            # check if need early stop or not
            early_stopping(val_FullLoss , G_model)
            if early_stopping.early_stop == True:
                print("Early stopping!")
                break

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


            #Write to TB:
            if args.tb_write_loss:
                writer.add_scalar('train/FullLoss', tot_fullLoss/len(train_loader), epoch)
                writer.add_scalar('train/ImL2', tot_ImL2/len(train_loader), epoch)
                writer.add_scalar('train/ImL1', tot_ImL1/len(train_loader), epoch)
                writer.add_scalar('train/KspaceL2', tot_KL2/len(train_loader), epoch)
                writer.add_scalar('train/ssimLoss', tot_SSIM/len(train_loader), epoch)
                writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                if args.GAN_training:
                    writer.add_scalar('train/G_AdvLoss', tot_adverG/len(train_loader), epoch)
                    writer.add_scalar('train/D_AdvLoss', tot_adverD/len(train_loader), epoch)

                writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
                writer.add_scalar('validation/ImL2', val_ImL2, epoch)
                writer.add_scalar('validation/ImL1', val_ImL1, epoch)
                writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
                writer.add_scalar('validation/ssimLoss', val_SSIM, epoch)
                writer.add_scalar('validation/PSNR', val_PSNR, epoch)

            if args.tb_write_image:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/rec_mid_img', rec_mid_img, epoch)
                writer.add_images('train/rec_images', rec_img, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/rec_mid_img', val_rec_mid_img, epoch)
                writer.add_images('validation/rec_images', val_rec_img, epoch)
                

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


