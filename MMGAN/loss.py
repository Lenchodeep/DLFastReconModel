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

        self.KL1Loss_T1 = nn.L1Loss()
        self.KL1Loss_T2 = nn.L1Loss()
        
        self.ssimLoss_T1 = SSIM(data_range=1.0, size_average=True, channel=1)
        self.ssimLoss_T2 = SSIM(data_range=1.0, size_average=True, channel=1)

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

    def calc_gen_loss_single(self, T2img_tar, T2img_pred, T2K_tar, T2K_pred, D_fake=None):
        ssimloss_T2 = self.ssimLoss_T2(T2img_tar, T2img_pred)
        imgloss_T2 = self.ImL1Loss_T2(T2img_tar, T2img_pred)
        kloss_T2 = self.KL1Loss_T2(T2K_tar, T2K_pred)
        if D_fake is not None:  
            advLoss = self.gen_adver_loss(D_fake)
        else:
            advLoss = 0
        fullLoss = self.alpha*(kloss_T2)\
                +(1-self.alpha)*(1000* imgloss_T2) + self.Adv_weight*advLoss + ssimloss_T2
        return fullLoss, imgloss_T2, kloss_T2, ssimloss_T2, advLoss


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


            # fullLoss = self.alpha*((1-self.beta)* kloss_T1 + self.beta * kloss_T2)\
            #         +(1-self.alpha)*((1-self.beta)* imgloss_T1 + self.beta * imgloss_T2)\
            #         +((1-self.beta)* ssimloss_T1 + self.beta * ssimloss_T2)\
            #         +self.Adv_weight*advLoss
            fullLoss = self.alpha*( (1-self.beta)*kloss_T1 + self.beta * kloss_T2)\
                +(1-self.alpha)*((1-self.beta)*imgloss_T1*100 + self.beta *100* imgloss_T2)\
                +self.Adv_weight*((1-self.beta)*advLoss+self.beta*advLoss_T1)+((1-self.beta)* ssimloss_T1 + self.beta * ssimloss_T2)
            return fullLoss, imgloss_T1, imgloss_T2, kloss_T1, kloss_T2, ssimloss_T1, ssimloss_T2, advLoss, advLoss_T1
