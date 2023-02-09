from pickle import load
from time import localtime, strftime, time
from tqdm import tqdm
import torch.optim as optim
import torchvision
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import cv2

from dataset import IXIDataSetImage
from config import config, log_config
from model import *
from utils import *


def main_train(args):
    print('[*] Run Basic Configs ... ')
    # setup log
    tb_dir = args.tb_path
    isExists = os.path.exists(tb_dir)
    if not isExists:
        os.makedirs(tb_dir)

    # tensorbordX log
    logger_tensorboard = SummaryWriter(os.path.join(tb_dir,'tensorboard'))


    ################################## prepare thr data ###############################################

    trainData = IXIDataSetImage(args.training_data_path, args, True)
    valData = IXIDataSetImage(args.val_data_path, args, False)

    print('[*] Loading Network ... ')
    # data loader
    dataloader = DataLoader(trainData, batch_size=args.batch_size, num_workers=0, pin_memory=True, timeout=0,
                                             shuffle=True)
    dataloader_val = DataLoader(valData, batch_size=args.batch_size, num_workers=0, pin_memory=True, timeout=0,
                                                 shuffle=True)

    # early stopping
    early_stopping = EarlyStopping(args.early_stopping_num, savepath=args.checkpoint_path ,cpname = "CP_early_stop",verbose=True)
    # pre-processing for vgg
    preprocessing = PREPROCESS()

    # pre-processing for vgg
    vgg_pre = VGG_PRE(args.device)

    # load vgg
    vgg16_cnn = VGG_CNN()
    vgg16_cnn = vgg16_cnn.to(args.device)

    # load unet
    generator = UNet()
    generator = generator.to(args.device)

    # load discriminator
    discriminator = Discriminator()
    discriminator = discriminator.to(args.device)

    # load loss function
    bce = nn.BCELoss(reduction='mean').to(args.device)
    mse = nn.MSELoss(reduction='mean').to(args.device)

    # real and fake label
    real = 1.
    fake = 0.

    # optimizer
    g_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    g_lr_scheduler = optim.lr_scheduler.StepLR(g_optim, args.lr_decay_every * len(dataloader), gamma=args.lr_decay, last_epoch=-1)

    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    d_lr_scheduler = optim.lr_scheduler.StepLR(d_optim, args.lr_decay_every * len(dataloader), gamma=args.lr_decay, last_epoch=-1)

    print('[*] Training  ... ')
    # initialize global step
    global GLOBAL_STEP
    GLOBAL_STEP = 0
    minValue = 999999

    for epoch in range(0, args.n_epoch):
        generator.train()   #注意train的调用位置，防止在train的过程中进入eval模式中。
        # initialize training
        total_nmse_training = 0
        total_ssim_training = 0
        total_psnr_training = 0
        num_training_temp = 0
        progress = 0  

        with tqdm(desc=f'Epoch {epoch + 1}/{args.n_epoch}', unit=' imgs') as pbar:

        # training
            for step, data in enumerate(dataloader):

                # update step
                GLOBAL_STEP = GLOBAL_STEP + 1

                # get learning rate
                lr_g = g_lr_scheduler.get_last_lr()[0]
                lr_d = d_lr_scheduler.get_last_lr()[0]

                # pre-processing for unet
                X_good = data['target_img'].to(args.device, dtype=torch.float32)
                X_good_plot = X_good
                X_bad = data['fold_img'].to(args.device, dtype=torch.float32)

                X_generated = generator(X_bad, is_refine=True)
                X_generated_plot = X_generated

                # discriminator
                logits_fake = discriminator(X_generated)
                logits_real = discriminator(X_good)

                # vgg
                X_good_244 = vgg_pre(X_good)
                net_vgg_conv4_good = vgg16_cnn(X_good_244)
                X_generated_244 = vgg_pre(X_generated)
                net_vgg_conv4_gen = vgg16_cnn(X_generated_244)

                # discriminator loss
                d_loss_real = bce(logits_real, torch.full((logits_real.size()), real).to(args.device))
                d_loss_fake = bce(logits_fake, torch.full((logits_fake.size()), fake).to(args.device))

                d_loss = d_loss_real + d_loss_fake

                # generator loss (adversarial)
                g_adversarial = bce(logits_fake, torch.full((logits_fake.size()), real).to(args.device))

                # generator loss (perceptual)
                g_perceptual = mse(net_vgg_conv4_gen, net_vgg_conv4_good)

                # generator loss (pixel-wise)
                g_nmse_a = mse(X_generated, X_good)
                g_nmse_b = mse(X_good, torch.zeros_like(X_good).to(args.device))
                g_nmse = torch.div(g_nmse_a, g_nmse_b)

                # generator loss (frequency)
                g_fft = mse(fft_abs_for_map_fn(X_generated), fft_abs_for_map_fn(X_good))

                # generator loss (total)
                g_loss = args.g_adv * g_adversarial + args.g_alpha * g_nmse + args.g_gamma * g_perceptual + args.g_beta * g_fft

                # clear gradient (discriminator)
                d_optim.zero_grad()
                # back propagation (discriminator)
                d_loss.backward(retain_graph=True)

                # clear gradient (generator)
                g_optim.zero_grad()
                # back propagation (generator)
                g_loss.backward()

                # update weight
                d_optim.step()
                g_optim.step()

                # update learning rate
                d_lr_scheduler.step()
                g_lr_scheduler.step()
                #### update tqdm
                progress += 100*X_good.shape[0]/len(trainData)
                    
                pbar.set_postfix(**{'FullLoss': g_loss.item(),'NMSE': g_nmse.item(),"perceptual": g_perceptual.item(),
                                    'KspaceL2': g_fft.item(),'Adv G': g_adversarial.item(), 'Adv D' : d_loss.item(),
                                     'Prctg of train set': progress})
                pbar.update(X_good.shape[0])# current batch size

                with torch.no_grad():
                    # record the learning rate
                    logger_tensorboard.add_scalar('Learning Rate', lr_g,
                                                global_step=GLOBAL_STEP)
                    # record train loss
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_LOSS', g_loss.item(),
                                                global_step=GLOBAL_STEP)
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_adversarial', g_adversarial.item(),
                                                global_step=GLOBAL_STEP)
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_perceptual', g_perceptual.item(),
                                                global_step=GLOBAL_STEP)
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_nmse', g_nmse.item(),
                                                global_step=GLOBAL_STEP)
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_fft', g_fft.item(),
                                                global_step=GLOBAL_STEP)
                    logger_tensorboard.add_scalar('TRAIN Discriminator LOSS/D_LOSS', d_loss.item(),
                                                global_step=GLOBAL_STEP)
                    logger_tensorboard.add_scalar('TRAIN Discriminator LOSS/d_loss_real', d_loss_real.item(),
                                                global_step=GLOBAL_STEP)
                    logger_tensorboard.add_scalar('TRAIN Discriminator LOSS/d_loss_fake', d_loss_fake.item(),
                                                global_step=GLOBAL_STEP)


                    # gpu --> cpu
                    X_good = X_good.cpu()
                    X_bad = X_bad.cpu()
                    X_generated = X_generated.cpu()

                    # (-1,1)-->(0,1)
                    X_good_0_1 = torch.div(torch.add(X_good, torch.ones_like(X_good)), 2)
                    X_bad_0_1 = torch.div(torch.add(X_bad, torch.ones_like(X_bad)), 2)
                    X_generated_0_1 = torch.div(torch.add(X_generated, torch.ones_like(X_generated)), 2)

                    # eval for training
                    nmse_a = mse(X_generated_0_1, X_good_0_1)
                    nmse_b = mse(X_good_0_1, torch.zeros_like(X_good_0_1))
                    nmse_res = torch.div(nmse_a, nmse_b).numpy()
                    ssim_res = calculateSSIM(X_generated_0_1.numpy(), X_good_0_1.numpy(), isBatch=True)
                    psnr_res = calculatePSNR(X_generated_0_1.numpy(), X_good_0_1.numpy(), isBatch=True)

                    total_nmse_training = total_nmse_training + nmse_res
                    total_ssim_training = total_ssim_training + ssim_res
                    total_psnr_training = total_psnr_training + psnr_res

                    num_training_temp = num_training_temp +args.batch_size


        total_nmse_training = total_nmse_training / num_training_temp
        total_ssim_training = total_ssim_training / num_training_temp
        total_psnr_training = total_psnr_training / num_training_temp

        # record training eval
        logger_tensorboard.add_scalar('Training/NMSE', total_nmse_training, global_step=epoch)
        logger_tensorboard.add_scalar('Training/SSIM', total_ssim_training, global_step=epoch)
        logger_tensorboard.add_scalar('Training/PSNR', total_psnr_training, global_step=epoch)

        logger_tensorboard.add_images('Training/Rec_img', to_real_image(X_generated_plot), global_step=epoch)
        logger_tensorboard.add_images('Training/Tar_img', to_real_image(X_good_plot), global_step=epoch)


        logging.info('Training NMSE score: {}, Training SSIM score: {}. Training PSNR score: {}'
                         .format(total_nmse_training, total_ssim_training, total_psnr_training))


        # initialize validation
        total_nmse_val = 0
        total_ssim_val = 0
        total_psnr_val = 0
        num_val_temp = 0

        with torch.no_grad():
            # validation
            for step_val, data in enumerate(dataloader_val):

                # pre-processing for unet
                X_good = data['target_img'].to(args.device, dtype=torch.float32)
                X_good_plot_val = X_good
                # good-->bad
                X_bad = data['fold_img'].to(args.device, dtype=torch.float32)

                X_generated = generator(X_bad, is_refine=True)
                X_generated_plot_val = X_generated
                # discriminator
                logits_fake = discriminator(X_generated)
                logits_real = discriminator(X_good)

                # vgg
                X_good_244 = vgg_pre(X_good)
                net_vgg_conv4_good = vgg16_cnn(X_good_244)
                X_generated_244 = vgg_pre(X_generated)
                net_vgg_conv4_gen = vgg16_cnn(X_generated_244)

                # discriminator loss
                d_loss_real = bce(logits_real, torch.full((logits_real.size()), real).to(args.device))
                d_loss_fake = bce(logits_fake, torch.full((logits_fake.size()), fake).to(args.device))

                d_loss = d_loss_real + d_loss_fake

                # generator loss (adversarial)
                g_adversarial = bce(logits_fake, torch.full((logits_fake.size()), real).to(args.device))

                # generator loss (perceptual)
                g_perceptual = mse(net_vgg_conv4_gen, net_vgg_conv4_good)

                # generator loss (pixel-wise)
                g_nmse_a = mse(X_generated, X_good)
                g_nmse_b = mse(X_good, torch.zeros_like(X_good).to(args.device))
                g_nmse = torch.div(g_nmse_a, g_nmse_b)

                # generator loss (frequency)
                g_fft = mse(fft_abs_for_map_fn(X_generated), fft_abs_for_map_fn(X_good))

                # generator loss (total)
                g_loss = args.g_adv * g_adversarial + args.g_alpha * g_nmse + args.g_gamma * g_perceptual + args.g_beta * g_fft

                # record validation loss
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/G_LOSS', g_loss.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_adversarial', g_adversarial.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_perceptual', g_perceptual.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_nmse', g_nmse.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_fft', g_fft.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Discriminator LOSS/D_LOSS', d_loss.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Discriminator LOSS/d_loss_real', d_loss_real.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Discriminator LOSS/d_loss_fake', d_loss_fake.item(),
                                              global_step=GLOBAL_STEP)

                # gpu --> cpu
                X_good = X_good.cpu()
                X_bad = X_bad.cpu()
                X_generated = X_generated.cpu()

                # (-1,1)-->(0,1)
                X_good_0_1 = torch.div(torch.add(X_good, torch.ones_like(X_good)), 2)
                X_bad_0_1 = torch.div(torch.add(X_bad, torch.ones_like(X_bad)), 2)
                X_generated_0_1 = torch.div(torch.add(X_generated, torch.ones_like(X_generated)), 2)

                # eval for validation
                nmse_a = mse(X_generated_0_1, X_good_0_1)
                nmse_b = mse(X_good_0_1, torch.zeros_like(X_good_0_1))
                nmse_res = torch.div(nmse_a, nmse_b).numpy()
                ssim_res = calculateSSIM(X_generated_0_1.numpy(), X_good_0_1.numpy(), isBatch=True)
                psnr_res = calculatePSNR(X_generated_0_1.numpy(), X_good_0_1.numpy(), isBatch=True)

                total_nmse_val = total_nmse_val + nmse_res
                total_ssim_val = total_ssim_val + ssim_res
                total_psnr_val = total_psnr_val + psnr_res

                num_val_temp = num_val_temp + args.batch_size

            total_nmse_val = total_nmse_val / num_val_temp
            total_ssim_val = total_ssim_val / num_val_temp
            total_psnr_val = total_psnr_val / num_val_temp
            
            # record validation eval
            logger_tensorboard.add_scalar('Validation/NMSE', total_nmse_val, global_step=epoch)
            logger_tensorboard.add_scalar('Validation/SSIM', total_ssim_val, global_step=epoch)
            logger_tensorboard.add_scalar('Validation/PSNR', total_psnr_val, global_step=epoch)

            logger_tensorboard.add_images('Training/Rec_img', to_real_image(X_generated_plot_val), global_step=epoch)
            logger_tensorboard.add_images('Training/Tar_img', to_real_image(X_good_plot_val), global_step=epoch)

            logging.info("Epoch: {}  NMSE val: {:8}, SSIM val: {:8}, PSNR val: {:8}".format(
                epoch, total_nmse_val, total_ssim_val, total_psnr_val))
            
            if minValue > total_nmse_val:
                minValue = total_nmse_val
                torch.save({
                    'epoch': epoch,
                    'G_model_state_dict': generator.state_dict(),
                    'G_optimizer_state_dict': g_optim.state_dict(),
                    'G_scheduler_state_dict': g_lr_scheduler.state_dict(),
                    'D_model_state_dict': discriminator.state_dict(),
                    'D_optimizer_state_dict': d_optim.state_dict(),
                    'D_scheduler_state_dict': d_lr_scheduler.state_dict(),
                }, args.checkpoint_path + f'Best_CP_epoch.pth')
                logging.info(f'Best Checkpoint saved !')

            torch.save({
                'epoch': epoch,
                'G_model_state_dict': generator.state_dict(),
                'G_optimizer_state_dict': g_optim.state_dict(),
                'G_scheduler_state_dict': g_lr_scheduler.state_dict(),
                'D_model_state_dict': discriminator.state_dict(),
                'D_optimizer_state_dict': d_optim.state_dict(),
                'D_scheduler_state_dict': d_lr_scheduler.state_dict(),
            }, args.checkpoint_path + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

            # early stopping
            if early_stopping:
                early_stopping(total_nmse_val , generator)
                if early_stopping.early_stop:
                    print("Early stopping!")
                    break

import yaml
import shutil
from types import SimpleNamespace
def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    # setup checkpoint dir
    checkpoint_dir = args.checkpoint_path
    isExists = os.path.exists(checkpoint_dir)
    if not isExists:
        os.makedirs(os.path.join(checkpoint_dir))

    shutil.copyfile('config.yaml',os.path.join(args.checkpoint_path,'config.yaml'))
    logging.info(f'Using device {args.device}')
    main_train(args)
