"""
This is a train function for gan.
"""
from six import viewvalues
from torch.optim import optimizer
from NetLoss import netLoss, set_grad
import sys
import os
import torch.nn as nn
import torch 
import shutil
import logging
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
import yaml
from  tqdm import tqdm
from torch import autograd
from MyDataSet import *
from Generator import *
from discriminator import *
from eval import eval_net
from torch.utils.data import DataLoader, dataset, random_split
from myEarlystop import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def calcu_gradient_penalty(D, batch_size, real, fake, device = "cpu"):
    """"To calculate gradient penalty"""
    alpha = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    alpha = alpha.expand(batch_size, real.size(1), real.size(2), real.size(3))
    alpha = alpha.to(device)

    interpolates = alpha * real + ((1-alpha) * fake)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty



def train(args):
    #Init the Generator Model
    G_model = WNet(args)
    logging.info(f'Generator NetWork:\n'
                f'\t{"Bilinear" if G_model.bilinear else "TransPose conv"} upscaling'
                    )
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr = args.lr, betas = (0.5,0.999))
    # G_optimizer = torch.optim.RMSprop(G_model.parameters(), lr = args.lr)  ## for wgan strategy
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience = 5, factor=0.5) ## min 表示当监控的量不在下降的时候，降低学习率， patience表示当性能在5个epoch后仍然不下降，降低学习率
    G_model.to(device= args.device)

    #Init the Discriminator Model
    D_model = patchGAN(1, ndf = args.ndf)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr = 2*args.lr, betas=(0.5, 0.999))  ##discriminator 的学习率为Generator的两倍
    # D_optimizer = torch.optim.RMSprop(D_model.parameters(), lr = args.lr)   ##for wgan strategy
    D_model.to(device=args.device)

    ## define the transfor for data augmentation
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
    ])

    #init dataloader
    dataset = MyDataSet(args.isTrain, args, transform= transform)
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
    loss = netLoss(args)

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
            D_epochloss = 0
            G_epochloss = 0
            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                ##Train loop
                for i,batch in enumerate(train_loader, 0):
                    
                    if i %4 == 0:
                        train_G = True
                    else:
                        train_G = False

                    masked_k = batch['masked_K'].to(device = args.device, dtype = torch.float32)
                    target_k = batch['target_K'].to(device = args.device, dtype = torch.float32)
                    target_img = batch['target_img'].to(device = args.device, dtype = torch.float32)

                    set_grad(D_model, True)
                    # set_grad(G_model, False)
                    D_model.zero_grad()

                    rec_img, rec_k, rec_mid_img = G_model(masked_k)

                    real_D_example = target_img
                    fake_D_example = rec_img

                    D_real_loss = D_model(real_D_example).mean()
                    D_fake_loss = D_model(fake_D_example).mean()

                    gradient_penalty = calcu_gradient_penalty(D_model, args.batch_size, real_D_example.data, fake_D_example.data,args.device)

                    D_loss =-(D_real_loss - D_fake_loss) + gradient_penalty
                    
                    D_loss.backward(retain_graph = True)
                    D_epochloss += D_loss.item()
                    D_optimizer.step()


                    G_loss = -D_model(fake_D_example).mean()
                    FullLoss, ImL2, ImL1, kL2, advLoss = loss.calc_gen_loss_wgan( rec_img, rec_k, target_img, target_k, G_loss)
                    
                    if train_G:
                    ##train G
                        # set_grad(G_model, True)
                        set_grad(D_model, False)
                        G_model.zero_grad()
                        FullLoss.backward()
                        G_epochloss += FullLoss.item()
                        G_optimizer.step()

                    #Update progress bar
                    progress += 100*target_k.shape[0]/len(trainData)
                    if args.GAN_training:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                            'KspaceL2': kL2.item(),'Adv G': advLoss.item(), 'Adv D ': D_loss.item(), 'Adv D - Real' : D_real_loss.item(),
                                            'Adv D - Fake' : D_fake_loss.item(),'Train G': train_G, 'Prctg of train set': progress})
                    else:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                            'KspaceL2': kL2.item(), 'Prctg of train set': progress})
                    pbar.update(target_k.shape[0])# current batch size

            # On epoch end
            # Validation
            val_rec_img, val_full_img, val_F_rec_Kspace, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR =\
                eval_net(G_model, val_loader, loss, args.device)
            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}'
                         .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR))
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
                writer.add_scalar('train/FullLoss', FullLoss.item(), epoch)
                writer.add_scalar('train/ImL2', ImL2.item(), epoch)
                writer.add_scalar('train/ImL1', ImL1.item(), epoch)
                writer.add_scalar('train/KspaceL2', kL2.item(), epoch)
                writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                if args.GAN_training:
                    writer.add_scalar('train/G_AdvLoss', advLoss.item(), epoch)
                    writer.add_scalar('train/D_AdvLoss', D_loss.item(), epoch)

                writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
                writer.add_scalar('validation/ImL2', val_ImL2, epoch)
                writer.add_scalar('validation/ImL1', val_ImL1, epoch)
                writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
                writer.add_scalar('validation/PSNR', val_PSNR, epoch)

            if args.tb_write_image:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/rec_images', rec_img, epoch)
                writer.add_images('train/Kspace_rec_images', rec_mid_img, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/rec_images', val_rec_img, epoch)
                writer.add_images('validation/Kspace_rec_images', val_F_rec_Kspace, epoch)

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
    with open('config.yml') as f:
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
    shutil.copyfile('config.yml',os.path.join(args.output_dir,'config.yaml'))
    logging.info(f'Using device {args.device}')

    train(args)

if __name__ == "__main__":
    main()


