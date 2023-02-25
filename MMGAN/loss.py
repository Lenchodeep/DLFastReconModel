import torch
import torch.nn as nn
from pytorch_msssim import  MS_SSIM, SSIM

import matplotlib.pyplot as plt
import pickle


def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad


class Loss():
    def __init__(self, args):
        self.args = args
        mask_path = args.mask_path                  

        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1']==1, device=self.args.device)
        self.maskNot = self.mask == 0
        
        self.alpha = args.alpha
        self.beta = args.beta

        self.Adv_weight = args.advWeight

        self.ImL1Loss_T1 = nn.L1Loss()
        self.ImL1Loss_T2 = nn.L1Loss()

        self.ImL1Loss_T1_mid = nn.L1Loss()
        self.ImL1Loss_T2_mid = nn.L1Loss()

        self.KL1Loss_T1 = nn.L1Loss()
        self.KL1Loss_T2 = nn.L1Loss()
        
        self.ssimLoss_T1 = SSIM(data_range=1.0, size_average=True, channel=1)
        self.ssimLoss_T2 = SSIM(data_range=1.0, size_average=True, channel=1)

        self.ssimLoss_T1_mid = SSIM(data_range=1.0, size_average=True, channel=1)
        self.ssimLoss_T2_mid = SSIM(data_range=1.0, size_average=True, channel=1)

        self.AdverLoss = nn.BCEWithLogitsLoss() 

    def gen_adver_loss(self,D_fake):      
        """generator loss"""
        real_ = torch.tensor(1.0).expand_as(D_fake).to(self.args.device)     ###图像全部设置为1
        return self.AdverLoss(D_fake, real_)

    def disc_adver_loss(self, D_real, D_fake):
        """Discriminator loss"""
        real_ = torch.tensor(1.0).expand_as(D_real).to(self.args.device)
        fake_ = torch.tensor(0.0).expand_as(D_fake).to(self.args.device)
        real_loss = self.AdverLoss(D_real,real_)
        fake_loss = self.AdverLoss(D_fake,fake_)

        return real_loss,fake_loss

    def calc_disc_loss(self,D_real,D_fake):
        real_loss,fake_loss = self.disc_adver_loss(D_real,D_fake)
        return real_loss,fake_loss, 0.5*(real_loss + fake_loss)

    def calc_gen_loss_single(self, T2img_tar, T2img_pred, T2img_mid, D_fake=None):

        ssimloss_T2 = self.ssimLoss_T2(T2img_tar, T2img_pred)
        ssimloss_T2_mid = self.ssimLoss_T2_mid(T2img_tar, T2img_mid)

        imgloss_T2 = self.ImL1Loss_T2(T2img_tar, T2img_pred)
        imgloss_T2_mid = self.ImL1Loss_T2_mid(T2img_tar, T2img_mid)

        # if D_fake is not None:  
        #     advLoss = self.gen_adver_loss(D_fake)
        # else:
        #     advLoss = 0
        fullLoss = self.alpha*(10*imgloss_T2_mid+ssimloss_T2_mid)\
                        +(1-self.alpha)*(10*imgloss_T2 + ssimloss_T2)
                        # + self.Adv_weight* advLoss
        return fullLoss, imgloss_T2, imgloss_T2_mid, ssimloss_T2, ssimloss_T2_mid


    def calc_gen_loss(self, T1img_tar, T2img_tar, T1img_pred, T2img_pred, 
                                    T1K_tar, T2K_tar,T1K_pred, T2K_pred, D_fake=None):

        ssimloss_T1 = self.ssimLoss_T1(T1img_tar, T1img_pred)
        ssimloss_T2 = self.ssimLoss_T2(T2img_tar, T2img_pred)

        imgloss_T1 = self.ImL1Loss_T1(T1img_tar, T1img_pred)
        imgloss_T2 = self.ImL1Loss_T2(T2img_tar, T2img_pred)

        kloss_T1 = self.KL1Loss_T1(T1K_tar, T1K_pred)
        kloss_T2 = self.KL1Loss_T2(T2K_tar, T2K_pred)

        ## gan loss
        if D_fake is not None:  
            advLoss = self.gen_adver_loss(D_fake)
        else:
            advLoss = 0

        # fullLoss = self.alpha*((1-self.beta)* kloss_T1 + self.beta * kloss_T2)\
        #         +(1-self.alpha)*((1-self.beta)* imgloss_T1 + self.beta * imgloss_T2)\
        #         +((1-self.beta)* ssimloss_T1 + self.beta * ssimloss_T2)\
        #         +self.Adv_weight*advLoss
        fullLoss = self.alpha*( kloss_T1 + kloss_T2)\
                +(1-self.alpha)*(imgloss_T1*1000 +  1000* imgloss_T2)
        return fullLoss, imgloss_T1, imgloss_T2, kloss_T1, kloss_T2, ssimloss_T1, ssimloss_T2, advLoss

        
    def calc_gen_loss_D(self, T1img_tar, T2img_tar, T1img_pred, T2img_pred, 
                                        T1K_tar, T2K_tar,T1K_pred, T2K_pred, D_fake=None, D_fake_T1 = None):

            ssimloss_T1 = self.ssimLoss_T1(T1img_tar, T1img_pred)
            ssimloss_T2 = self.ssimLoss_T2(T2img_tar, T2img_pred)

            imgloss_T1 = self.ImL1Loss_T1(T1img_tar, T1img_pred)
            imgloss_T2 = self.ImL1Loss_T2(T2img_tar, T2img_pred)

            kloss_T1 = self.KL1Loss_T1(T1K_tar, T1K_pred)
            kloss_T2 = self.KL1Loss_T2(T2K_tar, T2K_pred)

            ## gan loss
            if D_fake is not None:  
                advLoss = self.gen_adver_loss(D_fake)
            else:
                advLoss = 0

            if D_fake_T1 is not None:  
                advLoss_T1 = self.gen_adver_loss(D_fake_T1)
            else:
                advLoss_T1 = 0


            fullLoss = self.alpha*( (1-self.beta)*kloss_T1 + self.beta * kloss_T2)\
                +(1-self.alpha)*((1-self.beta)*imgloss_T1*100 + self.beta *100* imgloss_T2)\
                +self.Adv_weight*((1-self.beta)*advLoss+self.beta*advLoss_T1)+((1-self.beta)* ssimloss_T1 + self.beta * ssimloss_T2)
            return fullLoss, imgloss_T1, imgloss_T2, kloss_T1, kloss_T2, ssimloss_T1, ssimloss_T2, advLoss, advLoss_T1

    def calc_gen_loss_D_img(self, T1img_tar, T2img_tar, T1img_pred, T2img_pred, 
                                        T1img_mid, T2img_mid, D_fake=None, D_fake_T1 = None):

            ssimloss_T1 = self.ssimLoss_T1(T1img_tar, T1img_pred)
            ssimloss_T2 = self.ssimLoss_T2(T2img_tar, T2img_pred)

            imgloss_T1 = self.ImL1Loss_T1(T1img_tar, T1img_pred)
            imgloss_T2 = self.ImL1Loss_T2(T2img_tar, T2img_pred)

            imglossT1_mid = self.ImL1Loss_T1(T1img_tar, T1img_mid)
            imglossT2_mid = self.ImL1Loss_T1(T1img_tar, T2img_mid)

            ssimloss_T1_mid = self.ssimLoss_T1(T1img_tar, T1img_mid)
            ssimloss_T2_mid = self.ssimLoss_T2(T2img_tar, T2img_mid)

            ## gan loss
            if D_fake is not None:  
                advLoss = self.gen_adver_loss(D_fake)
            else:
                advLoss = 0

            if D_fake_T1 is not None:  
                advLoss_T1 = self.gen_adver_loss(D_fake_T1)
            else:
                advLoss_T1 = 0

            fullLoss = self.alpha*((1-self.beta)*(10*imglossT1_mid+ssimloss_T1_mid)+self.beta*(10*imglossT2_mid+ssimloss_T2_mid))\
                        +(1-self.alpha)*((1-self.beta)*(10*imgloss_T1+ssimloss_T1)+self.beta*(10*imgloss_T2+ssimloss_T2))\
                        +(1-self.beta)*0*advLoss_T1+self.beta*self.Adv_weight*advLoss
            return fullLoss, imgloss_T1, imgloss_T2, imglossT1_mid, imglossT2_mid, ssimloss_T1, ssimloss_T2, \
                        ssimloss_T1_mid, ssimloss_T2_mid, advLoss, advLoss_T1

    def cal_loss(self, tar_T1, T1_k_un, tar_T2, T2_k_un, mask, rec_T1_mid_0, rec_T1_mid, rec_T1_0, rec_T1, \
            rec_T2_mid_0, rec_T2_mid, rec_T2_0, rec_T2, T2_dis_real, T2_dis_fake, T1_dis_real, T1_dis_fake, \
            cal_G=True, args = None):
        G_loss_T1 = self.gen_adver_loss(T1_dis_fake)
        G_loss_T2 = self.gen_adver_loss(T2_dis_fake)

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


        g_loss =(args.alpha* G_loss_T2 + (1-args.alpha) * G_loss_T1) + rec_img_loss*100 \
                    +error_img_loss*args.rate\
                    +frq_loss*1 +  smooth_loss*1e-3

        return g_loss, frq_loss, rec_img_loss, error_img_loss, smooth_loss, G_loss_T2, G_loss_T1

def total_variant(images):
    '''
    :param images:  [B,C,W,H]
    :return: total_variant
    '''
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]

    tot_var = torch.abs(pixel_dif1).sum([1, 2, 3]) + torch.abs(pixel_dif2).sum([1, 2, 3])

    return tot_var

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


def build_loss(dis_real, dis_fake):
    '''
    calculate WGAN loss
    '''
    d_loss = torch.mean(dis_fake - dis_real)
    g_loss = -torch.mean(dis_fake)
    return g_loss, d_loss