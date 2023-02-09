import os
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np
from AttentionModule import *
from unet_parts import *

import yaml as yaml
from types import SimpleNamespace


##################################### Modified Wnet ###################################################
class RIRB(nn.Module):
    def __init__(self, channel_num , stride = 1):
        super(RIRB, self).__init__()

        self.channel_num = channel_num
        self.stride = stride


        self.conv_1 = nn.Conv2d(self.channel_num, self.channel_num // 2, kernel_size=3, stride =1, padding=1)
        self.conv_2 = nn.Conv2d(self.channel_num //2, self.channel_num //2, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(self.channel_num //2, self.channel_num //2, kernel_size=1, stride=1)
        self.conv_4 = nn.Conv2d(self.channel_num //2, self.channel_num, kernel_size=3, stride=1, padding=1)


        self.bn = nn.BatchNorm2d(self.channel_num)
        self.bn_half = nn.BatchNorm2d(self.channel_num // 2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.activation(self.bn_half(self.conv_1(x)))
        out2 = self.activation(self.bn_half(self.conv_2(out1)))
        out3 = self.activation(self.bn_half(self.conv_3(out2)))
        res1 = torch.add(out1, out3)
        out4 = self.activation(self.bn(self.conv_4(res1)))  
        out = torch.add(x, out4)
        return out


class PsychoDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PsychoDown, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,True),
            DoubleConv(out_channels, out_channels)
        )
    def forward(self,x):
        return self.down(x)

class PsychoUP(nn.Module):
    def __init__(self, in_channels, out_channels, attName=''):
        super(PsychoUP, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )
        self.convBlock = DoubleConv(in_channels, out_channels)
        if attName == 'PSA':
            print('psa')
            self.att = PSAModule(out_channels)
        elif attName == 'SA':
            print("sa")
            self.att = ShuffleAttention(out_channels) 
        else:
            print("here")
            self.att = None

    def forward(self, x):
        att_in = self.convBlock(x)

        if self.att is not None:
            att_out = self.att(att_in)
        else:
            att_out = att_in
        out = self.up(att_out)

        return out

class PsychoUnet(nn.Module):
    def __init__(self, in_channels, out_channels, nb_filters=64, attName = ''):
        super(PsychoUnet,self).__init__()
        self.inc = DoubleConv(in_channels,nb_filters)
        self.down1 = PsychoDown(nb_filters, nb_filters*2)
        self.down2 = PsychoDown(nb_filters*2, nb_filters*4)
        self.down3 = PsychoDown(nb_filters*4, nb_filters*8)
        self.down4 = PsychoDown(nb_filters*8, nb_filters*16)

        self.rirb0 = RIRB(nb_filters)
        self.rirb1 = RIRB(nb_filters*2)
        self.rirb2 = RIRB(nb_filters*4)
        self.rirb3 = RIRB(nb_filters*8)

        self.up4 = PsychoUP(nb_filters*16, nb_filters*8, attName)
        self.up3 = PsychoUP(nb_filters*8*2, nb_filters*4, attName)
        self.up2 = PsychoUP(nb_filters*4*2, nb_filters*2, attName)
        self.up1 = PsychoUP(nb_filters*2*2, nb_filters, attName)

        self.outc = nn.Conv2d(nb_filters*2, out_channels, kernel_size=1)

    def forward(self, x):
        input = x

        incOut = self.inc(input)
        down1 = self.down1(incOut)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        rirb_en_0 = self.rirb0(incOut)
        rirb_en_1 = self.rirb1(down1)
        rirb_en_2 = self.rirb2(down2)
        rirb_en_3 = self.rirb3(down3)

        up3 = self.up4(down4)
        residual3 = torch.cat((up3,rirb_en_3), dim = 1)

        up2 = self.up3(residual3)
        redsidual2 = torch.cat((up2,rirb_en_2), dim = 1)

        up1 = self.up2(redsidual2)
        residual1 = torch.cat((up1, rirb_en_1), dim = 1)

        up0 = self.up1(residual1)
        residual0 = torch.cat((up0, rirb_en_0), dim = 1)    

        out_tmp = self.outc(residual0)

        # out = torch.tanh(x+out_tmp)
        # out = torch.clamp(out, 0,1)
        out = out_tmp
        return out

class PsychoWNet(nn.Module):
    def __init__(self, args):
        super(PsychoWNet, self).__init__()

        self.args = args
        mask_path = args.mask_path
        mask = np.load(mask_path)
        self.mask = torch.tensor(mask == 1, device = self.args.device)
        self.maskNot = torch.tensor(mask == 0, device = self.args.device)


        self.kspace_Unet = PsychoUnet(in_channels= args.num_input_slices, out_channels=2, nb_filters=64, attName=args.attName)
        self.ispace_Unet = PsychoUnet(in_channels = 1, out_channels = 1, nb_filters=64, attName=args.attName)

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

    def forward(self, kspace):
        recon_all_kspace = self.kspace_Unet(kspace)

        rec_Kspace = self.mask*kspace+self.maskNot*recon_all_kspace
        rec_mid_img = (self.inverseFT(rec_Kspace))

        refine_img = self.ispace_Unet(rec_mid_img)
        rec_img = torch.clamp(torch.tanh(refine_img+ rec_mid_img), 0,1)

        return rec_img, rec_Kspace,rec_mid_img

class psychoDiscriminator(nn.Module):

    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)
            

    def __init__(self, n_channel, nb_filters = 64):
        super(psychoDiscriminator, self).__init__()
            
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=n_channel, out_channels=nb_filters, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=nb_filters*2, out_channels=nb_filters*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_filters*2, out_channels=nb_filters*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=nb_filters*4, out_channels=nb_filters*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=nb_filters*4, out_channels=nb_filters*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=nb_filters*8, out_channels=nb_filters*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=nb_filters*8, out_channels=nb_filters*16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=nb_filters*16, out_channels=nb_filters*16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nb_filters*16),
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=nb_filters*16, out_channels=nb_filters*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nb_filters*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=nb_filters*16, out_channels=nb_filters*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nb_filters*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv6 = nn.Conv2d(nb_filters*16, 1,kernel_size=3, stride=1, padding=1)
        sequence = []
        sequence += [self.conv0, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]

        self.model = nn.Sequential(*sequence).apply(self.weight_init)   ##apply函数将函数递归传递到网络模型的每一个子模型中，主要用于参数的初始化过程中

    def forward(self,input):
        return self.model(input)

