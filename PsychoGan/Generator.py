"""This is a basic Wnet(cascade Unet) model"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle ##对对象进行序列化的包
import numpy as np
import matplotlib.pyplot as plt
from Unet import *
sys.path.append("./")


class WNet(nn.Module):
    def __init__(self, args, masked_kspace = True):
        super(WNet, self).__init__()

        self.args = args
        self.bilinear = args.bilinear
        self.ispool = args.ispool
        self.masked_kspace = masked_kspace

        mask_path = args.mask_path
        mask = np.load(mask_path)
        self.mask = torch.tensor(mask == 1, device = self.args.device)
        self.maskNot = self.mask == 0

        self.Kspace_Unet = Unet(n_channels_in=args.num_input_slices, n_channels_out=1, bilinear=self.bilinear, ispooling=self.ispool)
        self.Ispace_Unet = Unet(n_channels_in = 1, n_channels_out = 1, bilinear=self.bilinear, ispooling= self.ispool)

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
        img = torch.absolute(img_cmplx[:, :, :, 0] + 1j*img_cmplx[:, :, :, 1])
        img = img[:, None, :, :]
        return img

    def forward(self, kspace):
        recon_all_kspace = self.Kspace_Unet(kspace)
        if self.masked_kspace:
            rec_Kspace = self.mask*kspace+self.maskNot*recon_all_kspace
            rec_mid_img = self.inverseFT(self.fftshift(rec_Kspace))
        else:
            rec_Kspace = recon_all_kspace
            rec_mid_img = self.inverseFT(self.fftshift(rec_Kspace))

        refine_img = self.Ispace_Unet(rec_mid_img)
        rec_img = torch.clamp(torch.tanh(refine_img+ rec_mid_img),0,1)

        return rec_img, rec_Kspace, rec_mid_img
