import torch.nn as nn
import torch 
import os
from AttentionModule import *
import yaml as yaml
from types import SimpleNamespace
import numpy as np



#################################### RIRBWnet #########################################################

class RIR_block(nn.Module):
    def __init__(self, channels):
        super(RIR_block, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels //2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels//2)
        self.relu1 = nn.LeakyReLU(0.2, True)
        
        self.conv2 = nn.Conv2d(channels//2, channels//2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels//2)
        self.relu2 = nn.LeakyReLU(0.2, True)

        self.conv3 = nn.Conv2d(channels//2, channels//2, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels//2)
        self.relu3 = nn.LeakyReLU(0.2, True)

        self.conv4 = nn.Conv2d(channels//2, channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(channels)
        self.relu4 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        input = x
        
        conv_1 = self.relu1(self.bn1(self.conv1(input)))

        conv_2 = self.relu2(self.bn2(self.conv2(conv_1)))

        conv_3 = self.relu3(self.bn3(self.conv3(conv_2)))

        res = torch.add(conv_1, conv_3) 

        conv_4 = self.relu4(self.bn4(self.conv4(res)))

        rir_out = torch.add(input, conv_4)

        return rir_out

class RIR_encoder_block(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(RIR_encoder_block, self).__init__()
        self.conva = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1)
        self.bna = nn.BatchNorm2d(outchannels)
        self.relua = nn.LeakyReLU(0.2, True)

        self.rirblock = RIR_block(outchannels)

        self.convb = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1)
        self.bnb = nn.BatchNorm2d(outchannels)        
        self.relub = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        input = x
        conv_a = self.relua(self.bna(self.conva(input)))
        rirbout = self.rirblock(conv_a)
        conv_b = self.relub(self.bnb(self.convb(rirbout)))

        return conv_b

class RIR_decoder_block(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(RIR_decoder_block, self).__init__()
        self.deconva = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1)
        self.bna = nn.BatchNorm2d(outchannels)
        self.relua = nn.LeakyReLU(0.2, True)

        self.de_rirbock = RIR_block(outchannels)

        self.deconvb = nn.ConvTranspose2d(outchannels, outchannels, kernel_size=3, stride=2, padding=1,output_padding=1)   ## how to select the kernel sizes
        self.bnb = nn.BatchNorm2d(outchannels)
        self.relub = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        input = x

        deconva = self.relua(self.bna(self.deconva(input)))

        derirbout = self.de_rirbock(deconva)

        deconvb = self.relub(self.bnb(self.deconvb(derirbout)))

        return deconvb


class RIRDiscriminator(nn.Module):
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)

    def __init__(self, inchannels, nb_filters, isFC = False):
        super(RIRDiscriminator, self).__init__()

        sequence = []
        sequence+=[
            nn.Conv2d(inchannels, nb_filters, kernel_size=3,stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            RIR_encoder_block(nb_filters, nb_filters),
            RIR_encoder_block(nb_filters, nb_filters*2),
            RIR_encoder_block(nb_filters*2, nb_filters*4),
            RIR_encoder_block(nb_filters*4, nb_filters*8),
        ]
        if isFC:
            sequence+=[
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(nb_filters*8, 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128,1)
            ]
        else:
            sequence+=[
                nn.Conv2d(nb_filters*8, 1, kernel_size=3,padding=1, stride=1)
            ]
        self.model = nn.Sequential(*sequence).apply(self.weight_init)

    def forward(self,x):
        return self.model(x)


class RIRGenerator(nn.Module):
    def __init__(self, args):
        super(RIRGenerator, self).__init__() 

        inchannels = args.inchannels
        outchannels = args.outchannels
        nb_filters = args.nb_filters

        self.isKDc = args.isKDC
        mask_path = args.mask_path
        mask = np.load(mask_path)
 
        self.mask = torch.tensor(mask == 1, device = args.device)
        self.maskNot = torch.tensor(mask == 0, device = args.device)
        ############################ KUnet part ################################
        self.conv1_0 = nn.Conv2d(inchannels, nb_filters, kernel_size=3, padding=1)
        self.bn1_0 = nn.BatchNorm2d(nb_filters)
        self.relu1_0 = nn.LeakyReLU(0.2, True)

        self.conv1_1 = RIR_encoder_block(nb_filters, nb_filters)
        self.conv1_2 = RIR_encoder_block(nb_filters, nb_filters*2)
        self.conv1_3 = RIR_encoder_block(nb_filters*2, nb_filters*4)
        self.conv1_4 = RIR_encoder_block(nb_filters*4, nb_filters*8)

        self.deconv1_4 = RIR_decoder_block(nb_filters*8, nb_filters*4)
        self.deconv1_3 = RIR_decoder_block(nb_filters*4, nb_filters*2)
        self.deconv1_2 = RIR_decoder_block(nb_filters*2, nb_filters*1)
        self.deconv1_1 = RIR_decoder_block(nb_filters, nb_filters)

        self.soam1 = SOAM(nb_filters)

        self.output1 = nn.Conv2d(nb_filters, 2, kernel_size=3, padding=1)

        ############################ IUnet part ################################

        self.conv2_0 = nn.Conv2d(1, nb_filters, kernel_size=3, padding=1)
        self.bn2_0 = nn.BatchNorm2d(nb_filters)
        self.relu2_0 = nn.LeakyReLU(0.2, True)

        self.conv2_1 = RIR_encoder_block(nb_filters, nb_filters)
        self.conv2_2 = RIR_encoder_block(nb_filters, nb_filters*2)
        self.conv2_3 = RIR_encoder_block(nb_filters*2, nb_filters*4)
        self.conv2_4 = RIR_encoder_block(nb_filters*4, nb_filters*8)

        self.deconv2_4 = RIR_decoder_block(nb_filters*8, nb_filters*4)
        self.deconv2_3 = RIR_decoder_block(nb_filters*4, nb_filters*2)
        self.deconv2_2 = RIR_decoder_block(nb_filters*2, nb_filters)
        self.deconv2_1 = RIR_decoder_block(nb_filters, nb_filters)

        self.soam2 = SOAM(nb_filters)

        self.output2 = nn.Conv2d(nb_filters, outchannels, kernel_size=3, padding=1)

        self.tanh = nn.Tanh()

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
        Kspace = self.fftshift(Kspace)
        Kspace = Kspace.permute(0, 2, 3, 1)
        img_cmplx = torch.fft.ifft2(Kspace, dim=(1,2))
        img = torch.absolute(img_cmplx[:,:,:,0]+ 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img

    def forward(self, kspace):
        input = kspace
        en1_0 = self.relu1_0(self.bn1_0(self.conv1_0(input)))
        en1_1 = self.conv1_1(en1_0)
        en1_2 = self.conv1_2(en1_1)
        en1_3 = self.conv1_3(en1_2)
        en1_4 = self.conv1_4(en1_3)

        dec1_4 = self.deconv1_4(en1_4)
        residual1_4 = torch.add(en1_3, dec1_4)

        dec1_3 = self.deconv1_3(residual1_4)
        residual1_3 = torch.add(en1_2, dec1_3)

        dec1_2 = self.deconv1_2(residual1_3)
        residual1_2 = torch.add(en1_1, dec1_2)

        dec1_1 = self.deconv1_1(residual1_2)
        residual1_1 = torch.add(en1_0, dec1_1)
        ##using soam attention part
        soam_1, att_HH_1, att_WW_1 = self.soam1(residual1_1)
        out1 = self.output1(soam_1)  ## as the k space unet output ,weathr needs activation function

        if self.isKDc:
            rec_Kspace = self.mask*kspace+self.maskNot*out1
            rec_mid_img = self.inverseFT(rec_Kspace)
        else:
            rec_Kspace = out1
            rec_mid_img = self.inverseFT(rec_Kspace)

        en2_0 = self.relu2_0(self.bn2_0(self.conv2_0(rec_mid_img)))
        en2_1 = self.conv2_1(en2_0)
        en2_2 = self.conv2_2(en2_1)
        en2_3 = self.conv2_3(en2_2)
        en2_4 = self.conv2_4(en2_3)

        dec2_4 = self.deconv2_4(en2_4)
        residual2_4 = torch.add(en2_3, dec2_4)

        dec2_3 = self.deconv2_3(residual2_4)
        residual2_3 = torch.add(en2_2, dec2_3)

        dec2_2 = self.deconv2_2(residual2_3)
        residual2_2 = torch.add(en2_1, dec2_2)

        dec2_1 = self.deconv2_1(residual2_2)
        residual2_1 = torch.add(en2_0, dec2_1)

        soam_2, att_HH_2, att_WW_2 = self.soam2(residual2_1)

        out2 = self.output2(soam_2)
        refineImg = torch.add(rec_mid_img, out2)

        rec_img = torch.clamp(self.tanh(refineImg),0,1)

        return rec_img, rec_Kspace, rec_mid_img
