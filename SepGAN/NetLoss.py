import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import SSIM, ssim, MS_SSIM
import matplotlib.pyplot as plt
import math
from focal_frequency_loss import FocalFrequencyLoss as FFL
import pickle

def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad


class KspaceNetLoss():

    def __init__(self, args, masked_kspace_flag = True):
        self.args = args
        mask_path = args.mask_path                  
        
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)

        self.masked_kspace_flag = masked_kspace_flag
        self.mask = torch.tensor(masks['mask1']==1, device=self.args.device)
        self.maskNot = self.mask == 0
        self.useFacolLoss = args.isFacolLoss

        self.ImL2_weight = args.loss_weights[0]
        self.ImL1_weight = args.loss_weights[1]
        if self.useFacolLoss:
            print("Using focal Loss with weight {}".format(args.loss_weights[5]))
            self.kspaceLoss_weight = args.loss_weights[3]
        else:
            print("Using KspaceL2 Loss with weight {}".format(args.loss_weights[2]))                      
            self.kspaceLoss_weight = args.loss_weights[2]

        self.AdverLoss_weight = args.loss_weights[4]
        self.ssim_Loss_weight = args.loss_weights[5]
        self.gradient_loss_weight = args.loss_weights[6]
        self.ffl_loss = FFL()
        self.ImL2Loss = nn.MSELoss()
        self.ImL1Loss = nn.SmoothL1Loss()

        self.AdverLoss = nn.BCEWithLogitsLoss() ### BCEWithLogitsLoss就是把Sigmoid-BCELoss
        self.ssimLoss = SSIM(data_range=1.0, size_average=True, channel=1)

        if self.masked_kspace_flag:
            self.KspaceL2Loss = nn.MSELoss(reduction='sum')
        else:
            self.KspaceL2Loss = nn.MSELoss() 

    def img_space_loss(self, pred_img, tar_img):
        return self.ImL1Loss(pred_img, tar_img), self.ImL2Loss(pred_img, tar_img)

    def k_space_loss(self, pred_k, tar_k):
        if self.masked_kspace_flag:
            return self.KspaceL2Loss(pred_k, tar_k)/(torch.sum(self.maskNot)*tar_k.max())   ###为什么要初一mask的和
        else:
            return self.KspaceL2Loss(pred_k, tar_k)

    def calc_focal_frequency_loss(self, pred_img, tar_img):
        return self.ffl_loss(pred_img, tar_img)

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

    def calc_gen_loss_wgan(self, pred_Im,mid_pred_Im, pred_K, tar_Im, tar_K,D_fake=None, calcuMid = False):
        '''
            The generator loss for wgan strategy.
        '''
        ssimloss = self.ssimLoss(pred_Im, tar_Im)
        ssimloss_mid = self.ssimLoss(mid_pred_Im, tar_Im)

        ImL1,ImL2 = self.img_space_loss(pred_Im, tar_Im)
        gradientLoss = self.calcGradientLoss(pred_Im, tar_Im)

        if calcuMid:
            mid_gradientLoss = self.calcGradientLoss(mid_pred_Im, tar_Im)
            mid_Im1, mid_Im2 = self.img_space_loss(pred_Im, mid_pred_Im)

        if self.useFacolLoss:
            KspaceLoss = self.calc_focal_frequency_loss(mid_pred_Im, tar_Im)
        else:
            KspaceLoss = self.k_space_loss(pred_K, tar_K)
        if D_fake is not None:  
            advLoss = -torch.mean(D_fake)
        else:
            advLoss = 0
        
        if calcuMid:
            fullLoss = self.ImL2_weight*(ImL2 + mid_Im2 )+ self.ImL1_weight*(ImL1 + mid_Im1) + self.kspaceLoss_weight*KspaceLoss \
                +self.AdverLoss_weight*advLoss + self.ssim_Loss_weight *((1-ssimloss)+(1-ssimloss_mid)) + self.gradient_loss_weight * (gradientLoss+mid_gradientLoss)
            return fullLoss, (ImL2+mid_Im2), (ImL1+mid_Im1), KspaceLoss, advLoss , ssimloss, ssimloss_mid, (gradientLoss+mid_gradientLoss)
        else:
            fullLoss = self.ImL2_weight*ImL2 + self.ImL1_weight*ImL1 + self.kspaceLoss_weight*KspaceLoss \
                +self.AdverLoss_weight*advLoss + self.ssim_Loss_weight *(1-ssimloss) + self.gradient_loss_weight * gradientLoss
            return fullLoss, ImL2, ImL1, KspaceLoss, advLoss , ssimloss, ssimloss_mid, gradientLoss


    def calc_gen_loss(self, pred_Im, mid_pred_Im,  pred_K, tar_Im, tar_K,D_fake=None,calcuMid = False):
    
        ssimloss = self.ssimLoss(pred_Im, tar_Im)
        ssimloss_mid = self.ssimLoss(mid_pred_Im, tar_Im)
        gradientLoss = self.calcGradientLoss(pred_Im, tar_Im)


        ImL1,ImL2 = self.img_space_loss(pred_Im, tar_Im)
        
        if calcuMid:
            mid_gradientLoss = self.calcGradientLoss(mid_pred_Im, tar_Im)
            mid_Im1, mid_Im2 = self.img_space_loss(pred_Im, mid_pred_Im)

        if self.useFacolLoss:
            KspaceLoss = self.calc_focal_frequency_loss(mid_pred_Im, tar_Im)
        else:
            KspaceLoss = self.k_space_loss(pred_K, tar_K)
        if D_fake is not None:  
            advLoss = self.gen_adver_loss(D_fake)
        else:
            advLoss = 0
        if calcuMid:
            fullLoss = self.ImL2_weight*(ImL2 + mid_Im2 )+ self.ImL1_weight*(ImL1 + mid_Im1) + self.kspaceLoss_weight*KspaceLoss \
                +self.AdverLoss_weight*advLoss + self.ssim_Loss_weight *((1-ssimloss)+(1-ssimloss_mid)) + self.gradient_loss_weight * (gradientLoss+mid_gradientLoss)
            return fullLoss, (ImL2+mid_Im2), (ImL1+mid_Im1), KspaceLoss, advLoss , ssimloss, ssimloss_mid, (gradientLoss+mid_gradientLoss)
        else:
            fullLoss = self.ImL2_weight*ImL2 + self.ImL1_weight*ImL1 + self.kspaceLoss_weight*KspaceLoss \
                +self.AdverLoss_weight*advLoss + self.ssim_Loss_weight *(1-ssimloss) + self.gradient_loss_weight * gradientLoss
            return fullLoss, ImL2, ImL1, KspaceLoss, advLoss , ssimloss, ssimloss_mid, gradientLoss

        
    def calc_disc_loss(self,D_real,D_fake):
        real_loss,fake_loss = self.disc_adver_loss(D_real,D_fake)
        return real_loss,fake_loss, 0.5*(real_loss + fake_loss)

    def  calcGradientLoss(self,rec_img, label): 
        good_dif1 = label[:,:,1:,:] - label[:,:,:-1,:]
        good_dif2 = label[:,:,:,1:] - label[:,:,:,:-1]

        rec_dif1 = rec_img[:,:,1:,:] - rec_img[:,:,:-1,:]
        rec_dif2 = rec_img[:,:,:,1:] - rec_img[:,:,:,:-1]

        dif1 = torch.mean(torch.mean((good_dif1 - rec_dif1)**2,dim=[2,3]))
        dif2 = torch.mean(torch.mean((good_dif2 - rec_dif2)**2,dim=[2,3]))  
        img_grad = dif1+dif2
        return img_grad