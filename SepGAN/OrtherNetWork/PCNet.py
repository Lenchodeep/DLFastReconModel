
import os
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np
from AttentionModule import *
from unet_parts import *

import yaml as yaml
from types import SimpleNamespace

##################################### Partial conv part ###############################################

class PartialConv(nn.Module):
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0,dilation = 1, group=1, bias = False ):
        super(PartialConv, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, group, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, group, False)                                
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        self.input_conv.apply(self.weight_init)

        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, input, mask):
        output = self.input_conv(input*mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1,-1,1,1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActive(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, activ='leaky', conv_bias = False):
        super(PCBActive, self).__init__()

        self.conv = PartialConv(in_channels, out_channels, 3, stride=stride, padding=1, bias= conv_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        
    def forward(self, input, mask):
        out, out_mask = self.conv(input, mask)
        out = self.bn(out)
        out = self.activation(out)

        return out, out_mask

class DoublePConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride = 1, active='leaky', conv_bias = False):
        super(DoublePConv, self).__init__()
        self.pconv1 = PCBActive(inchannels, outchannels, stride=stride, activ= active, conv_bias=conv_bias) 
        self.pconv2 = PCBActive(outchannels, outchannels, stride=stride, activ= active, conv_bias=conv_bias) 

    def forward(self, input, mask):
        out1, mask1 = self.pconv1(input, mask)
        out, out_mask = self.pconv2(out1, mask1)

        return out, out_mask

class PConvUnet(nn.Module):
    def __init__(self, in_channels, out_channels,  args):
        super(PConvUnet, self).__init__()
        # self.upmode = upmode
        # self.layers = layers
        self.inchannels = in_channels
        self.outchannels = out_channels
        self.fMapNum = args.fMapNum

        self.enc_PC1 = DoublePConv(self.inchannels, self.fMapNum)
        self.enc_PC2 = DoublePConv(self.fMapNum,self.fMapNum*2)
        self.enc_PC3 = DoublePConv(self.fMapNum*2, self.fMapNum*4)
        self.enc_PC4 = DoublePConv(self.fMapNum*4, self.fMapNum*8)

        self.bottle_neck = DoublePConv(self.fMapNum*8, self.fMapNum*16)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec_PC1 = DoublePConv(self.fMapNum*(16+8), self.fMapNum*8)
        self.dec_PC2 = DoublePConv(self.fMapNum*(8+4), self.fMapNum*4)
        self.dec_PC3 = DoublePConv(self.fMapNum*(4+2), self.fMapNum*2)
        self.dec_PC4 = DoublePConv(self.fMapNum*(2+1), self.fMapNum*1)

        self.outLayer = PartialConv(self.fMapNum, self.outchannels, kernel_size=3, padding=1)

    def forward(self, input, mask):
        x1, msk1 = self.enc_PC1(input, mask)
        x1_d = self.down(x1)
        msk1_d = self.down(msk1)
        
        x2, msk2 = self.enc_PC2(x1_d, msk1_d)
        x2_d = self.down(x2)
        msk2_d = self.down(msk2)

        x3, msk3 = self.enc_PC3(x2_d, msk2_d)
        x3_d = self.down(x3)
        msk3_d = self.down(msk3)

        x4, msk4 = self.enc_PC4(x3_d, msk3_d)
        x4_d = self.down(x4)
        msk4_d = self.down(msk4)


        bottle, msk_bottle = self.bottle_neck(x4_d, msk4_d)

        bottle_u = self.up(bottle)
        msk_bottle_u = self.up(msk_bottle)
        bottle_u = torch.cat([bottle_u, x4], dim=1)
        msk_bottle_u = torch.cat([msk_bottle_u, msk4], dim = 1)

        x5 , msk5 = self.dec_PC1(bottle_u, msk_bottle_u)

        x5_u = self.up(x5)
        msk5_u = self.up(msk5)
        x5_u = torch.cat([x5_u, x3], dim=1)
        msk5_u = torch.cat([msk5_u, msk3], dim = 1)

        x6, msk6 = self.dec_PC2(x5_u, msk5_u)

        x6_u = self.up(x6)
        msk6_u = self.up(msk6)
        x6_u = torch.cat([x6_u, x2], dim=1)
        msk6_u = torch.cat([msk6_u, msk2], dim=1)

        x7, msk7 = self.dec_PC3(x6_u, msk6_u)

        x7_u = self.up(x7)
        msk7_u = self.up(msk7)
        x7_u = torch.cat([x7_u, x1], dim=1)
        msk7_u = torch.cat([msk7_u, msk1], dim=1)

        x8, msk8 = self.dec_PC4(x7_u, msk7_u)

        out, msk_out = self.outLayer(x8, msk8)

        return out, msk_out

class PCWNet(nn.Module):
    def __init__(self, args, masked_kspace = True , use_CBAM = False):
        super(PCWNet, self).__init__()

        self.args = args
        self.bilinear = args.bilinear
        self.masked_kspace = masked_kspace

        mask_path = args.mask_path
        npmask = np.load(mask_path)

        self.mask = torch.tensor(npmask == 1, device = self.args.device)

        ###input mask of the PCUnet, we need to expand it's dim to the input num
        self.input_mask = torch.from_numpy(npmask.astype(np.float32))
        self.input_mask = torch.Tensor.repeat(torch.unsqueeze(self.mask , dim=0),(args.num_input_slices,1,1))
        self.input_mask = torch.unsqueeze(self.input_mask, dim = 0)
        self.input_mask = self.input_mask.repeat(args.batch_size,1,1,1)
        self.input_mask = self.input_mask.type(torch.float32)
        ###

        self.maskNot = self.mask == 0

        self.kspace_Unet = PConvUnet(in_channels=args.num_input_slices, out_channels=1, args = self.args)
        if use_CBAM:
            self.ispace_Unet = UnetCBAM(n_channels_in = 1, n_channels_out = 1, bilinear=self.bilinear)
        else:
            self.ispace_Unet = UNet(n_channels_in = 1, n_channels_out = 1, bilinear=self.bilinear)

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
        img = torch.absolute(img_cmplx[:,:,:,0] + 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img

    def forward(self, kspace):
        recon_all_kspace, _ = self.kspace_Unet(kspace, self.input_mask)
        if self.masked_kspace:
            rec_Kspace = self.mask*kspace+self.maskNot*recon_all_kspace
            rec_mid_img = (self.inverseFT(rec_Kspace))
        else:
            rec_Kspace = recon_all_kspace
            rec_mid_img = (self.inverseFT(rec_Kspace))

        refine_img = self.ispace_Unet(rec_mid_img)
        rec_img = torch.clamp(torch.tanh(refine_img+ rec_mid_img), 0,1)

        return rec_img, rec_Kspace, rec_mid_img


##################################### CBAM Unet #######################################################
class UnetCBAM(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear = True, ispooling = True):
        super(UnetCBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.ispooling = ispooling

        self.inconv = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64, self.ispooling)
        self.down2 = Down(64, 128, self.ispooling)
        self.down3 = Down(128, 256, self.ispooling)
        if self.bilinear:
            factor = 2
        else:
            factor = 1
        self.down4 = Down(256, 512//factor, self.ispooling)
        self.up1 = UpCBAM(512, 256//factor, self.bilinear)
        self.up2 = UpCBAM(256, 128//factor, self.bilinear)
        self.up3 = UpCBAM(128, 64//factor, self.bilinear)
        self.up4 = UpCBAM(64, 32, self.bilinear)
        self.outconv = nn.Conv2d(in_channels=32, out_channels=n_channels_out, kernel_size=1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outconv(x)
        return out
