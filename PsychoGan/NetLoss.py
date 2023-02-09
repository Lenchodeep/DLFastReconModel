import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np

# from pytorch_ssim import SSIM, ssim
from pytorch_msssim import SSIM, ssim

class netLoss():

    def __init__(self, args, masked_kspace_flag = True):
        self.args = args
        mask_path = args.mask_path
        mask = np.load(mask_path)

        self.masked_kspace_flag = masked_kspace_flag
        self.mask = torch.tensor(mask==1, device = self.args.device)

        self.maskNot = self.mask==0

        self.ImL2_weight = args.loss_weights[0]
        self.ImL1_weight = args.loss_weights[1]
        self.kspaceL2_weight = args.loss_weights[2]
        self.AdverLoss_weight = args.loss_weights[3]
        self.ssimLoss_weight = args.loss_weights[4]

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

    def calc_gen_loss_wgan(self, pred_Im, pred_k,tar_Im, tar_k, Gloss = None):
        ImL1,ImL2 = self.img_space_loss(pred_Im, tar_Im)
        KspaceL2 = self.k_space_loss(pred_k, tar_k)
        if Gloss is not None:
            advLoss = Gloss
        else:
            advLoss = 0
        fullLoss = self.ImL2_weight*ImL2 + self.ImL1_weight*ImL1 + self.kspaceL2_weight*KspaceL2 + self.AdverLoss_weight*advLoss
        return fullLoss, ImL2, ImL1, KspaceL2 ,advLoss

    def calc_gen_loss(self, pred_Im, pred_K, tar_Im, tar_K,D_fake=None):

        ssimloss = self.ssimLoss(pred_Im, tar_Im)
        ImL1,ImL2 = self.img_space_loss(pred_Im, tar_Im)

        KspaceL2 = self.k_space_loss(pred_K, tar_K)

        if D_fake is not None:  
            advLoss = self.gen_adver_loss(D_fake)
        else:
            advLoss = 0
        fullLoss = self.ImL2_weight*ImL2 + self.ImL1_weight*ImL1 + self.kspaceL2_weight*KspaceL2 +self.AdverLoss_weight*advLoss + self.ssimLoss_weight * (1-ssimloss)
        return fullLoss, ImL2, ImL1, KspaceL2, advLoss , ssimloss

    def calc_disc_loss(self,D_real,D_fake):
        real_loss,fake_loss = self.disc_adver_loss(D_real,D_fake)
        return real_loss,fake_loss, 0.5*(real_loss + fake_loss)

def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad

    