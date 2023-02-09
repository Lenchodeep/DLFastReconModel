import os
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np
from AttentionModule import *
from unet_parts import *

import yaml as yaml
from types import SimpleNamespace
import pickle



class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class WNet(nn.Module):

    def __init__(self, args, masked_kspace=True):
        super(WNet, self).__init__()

        self.bilinear = args.bilinear
        self.args = args
        self.masked_kspace = masked_kspace

        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
        self.maskNot = self.mask == 0

        self.kspace_Unet = UNet(n_channels_in=args.num_input_slices*2, n_channels_out=2, bilinear=self.bilinear)
        self.img_UNet = UNet(n_channels_in=1, n_channels_out=1, bilinear=self.bilinear)

    def fftshift(self, img):

        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2

    def inverseFT(self, Kspace):
        """The input Kspace has two channels(real and img)"""
        Kspace = Kspace.permute(0, 2, 3, 1)
        img_cmplx = torch.fft.ifft2(Kspace, dim=(1,2))
        img = torch.absolute(img_cmplx[:,:,:,0]+ 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img

    def forward(self, Kspace):

        rec_all_Kspace = self.kspace_Unet(Kspace)
        if self.masked_kspace:
            rec_Kspace = self.mask*Kspace[:, int(Kspace.shape[1]/2)-1:int(Kspace.shape[1]/2)+1, :, :] +\
                         self.maskNot*rec_all_Kspace
            rec_mid_img = self.inverseFT(rec_Kspace)
        else:
            rec_Kspace = rec_all_Kspace
            rec_mid_img = self.fftshift(self.inverseFT(rec_Kspace))
        refine_Img = self.img_UNet(rec_mid_img)
        rec_img = torch.tanh(refine_Img + rec_mid_img)
        rec_img = torch.clamp(rec_img, 0, 1)
        # if self.train():
        return rec_img, rec_Kspace, rec_mid_img

################################# patch gan discriminator ##############################################
class patchGAN(nn.Module):
    """
        Parameters:
            input_channels (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            crop_center      -- None ot the size of the center patch to be cropped
            FC_bottleneck      -- If True use global average pooling and output a one-dimension prediction
    """
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)
            
    def __init__(self, input_channels, ndf, n_layers = 3, crop_center = None, FC_bottleneck = False):
        super(patchGAN, self).__init__()

        self.crop_center = crop_center
        ks = 3
        padw=1
        sequence = [
            nn.Conv2d(input_channels, ndf, kernel_size=ks, stride=1, padding=padw), 
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf, ndf, kernel_size=ks, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace= True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            print(nf_mult, 'nf_mult')

            sequence+=[
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks,stride=1, padding=padw),  ###BN之前的卷积层中的偏置b不会起到作用
                nn.BatchNorm2d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf*nf_mult, ndf*nf_mult, kernel_size=ks, stride=2, padding=padw),  ###BN之前的卷积层中的偏置b不会起到作用
                nn.BatchNorm2d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence +=[
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks, stride = 1, padding=padw),
            nn.BatchNorm2d(ndf*nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        if FC_bottleneck:
            sequence += [
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(ndf*nf_mult, 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128,1) 
            ]
        else:
            sequence+=[
                nn.Conv2d(ndf*nf_mult, 1,kernel_size=ks, stride=1, padding=padw)
            ]
        
        self.model = nn.Sequential(*sequence).apply(self.weight_init)   ##apply函数将函数递归传递到网络模型的每一个子模型中，主要用于参数的初始化过程中

    def forward(self, input):
        if self.crop_center is not None:
            _,_,h,w = input.shape
            x0 = (h-self.crop_center) //2
            y0 = (w-self.crop_center) //2
            input = input[:,:,x0:x0+self.crop_center,y0:y0+self.crop_center]
        return self.model(input)

def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args


if __name__=='__main__':
    args = get_args()
    netG = WNet(args)
    netD = patchGAN(1,64)

    print('    Total params of G: %.2fM' % (sum(p.numel() for p in netG.parameters()) / 1000000)) 
    print('    Total params of D: %.2fM' % (sum(p.numel() for p in netD.parameters()) / 1000000))
    print('    Total params : %.2fM' %((sum(p.numel() for p in netG.parameters()) / 1000000)+(sum(p.numel() for p in netD.parameters()) / 1000000)))