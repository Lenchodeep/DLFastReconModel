import os
import numpy as np
import torch
from torch.nn.modules import module
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
from eval import eval_net, eval_net1

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
        nn.init.zeros_(m.bias)

def train(args):
    model = Unet(1,1)
    model.to(device=args.device)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=[0.9,0.999])

    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
    ])   

    dataset = IXIDataSet(args.imageDataDir, args, args.isTrain, transform = transform)
    n_train = int(len(dataset)* (1-args.val_percent))
    n_val = int(len(dataset) - n_train) 
    trainData , valData = random_split(dataset , [n_train , n_val])

    train_loader = DataLoader(trainData, batch_size = args.batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True)
    val_loader = DataLoader(valData, batch_size = args.batch_size, shuffle = False, num_workers= args.val_num_workers, pin_memory=True, drop_last=True)
    
    early_stopping = EarlyStopping(patience=args.stopPatience, savepath=args.output_dir ,cpname = "CP_early_stop",verbose=True)

    if args.tb_write_loss or args.tb_write_image:
        writer = SummaryWriter(log_dir=args.tb_dir+'/tensorboard')

    Loss = nn.MSELoss()

    minValLoss = 100000.0
    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')

    try:
        for epoch in range(0, args.epochs):
            model.train()
            progress = 0
            epoch_loss = 0
            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit='imgs') as pbar:
                for batch in train_loader:
                    fold_img = batch['masked_img'].to(device = args.device, dtype = torch.float32)
                    target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)
                    recon_img = model(fold_img)

                    loss = Loss(recon_img, target_img)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss +=loss.item()
                    
                    progress += 100*target_img.shape[0]/len(trainData)   
                    pbar.set_postfix(**{'FullLoss': loss.item(), 
                                            'Prctg of train set': progress})
                    pbar.update(target_img.shape[0])# current batch size

            val_rec_img, val_full_img, val_ImL2 = eval_net1(model, val_loader, Loss, args.device)
            logging.info('Validation full score: {}'
                .format(val_ImL2))
            # G_scheduler.step(val_FullLoss)
            early_stopping(val_ImL2 , model)
            if early_stopping.early_stop == True:
                print("Early stopping!")
                break

            if minValLoss > val_ImL2:
                minValLoss = val_ImL2
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': model.state_dict(),
                    'G_optimizer_state_dict': optimizer.state_dict()

                }, args.output_dir + f'Best_CP_epoch.pth')
            logging.info(f'Best Checkpoint saved !')

            if args.tb_write_loss:
                writer.add_scalar('train/ImL1',  epoch_loss / len(train_loader), epoch)


                writer.add_scalar('validation/FullLoss', val_ImL2, epoch)


            if args.tb_write_image:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/rec_images', recon_img, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/rec_images', val_rec_img, epoch)

            #Save Checkpoint per 5 epoch
            if (epoch % args.save_step) == (args.save_step-1):
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': model.state_dict(),
                    'G_optimizer_state_dict': optimizer.state_dict()

                }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'G_model_state_dict': model.state_dict(),
            'G_optimizer_state_dict': optimizer.state_dict(),
   
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