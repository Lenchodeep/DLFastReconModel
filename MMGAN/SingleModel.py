import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

from CommonPart import *

class SingleUnet(nn.Module):
    def __init__(self, in_chan, out_chan, filters):
        super(SingleUnet, self).__init__()
        self.inconv = nn.Conv2d(in_chan,int(filters/2),1)

        self.d_1 = BasicBlockDown(int(filters/2), filters)
        self.down_1 = DownSample()
        self.d_2 = BasicBlockDown(filters, filters*2)
        self.down_2 = DownSample()
        self.d_3 = BasicBlockDown(filters*2, filters*4)
        self.down_3 = DownSample()
        self.d_4 = BasicBlockDown(filters*4, filters*8)
        self.down_4 = DownSample()

        self.bottom = DRDB(filters*8)

        self.up4 = UpSample_T(filters*8)
        self.u4 = BasicBlockUp(filters*4)
        self.up3 = UpSample_T(filters*4)
        self.u3 = BasicBlockUp(filters*2)
        self.up2 = UpSample_T(filters*2)
        self.u2 = BasicBlockUp(filters)
        self.up1 = UpSample_T(filters)
        self.u1 = BasicBlockUp(int(filters/2))   

        self.outconv = nn.Conv2d(int(filters/2),out_chan, 1)

    def forward(self, T2):

        T2_x1 = self.inconv(T2)
        T2_x2,T2_y1 = self.down_1(self.d_1(T2_x1))
        T2_x3,T2_y2 = self.down_2(self.d_2(T2_x2))
        T2_x4,T2_y3 = self.down_3(self.d_3(T2_x3))
        T2_x5,T2_y4 = self.down_4(self.d_4(T2_x4))

        T2_x = self.bottom(T2_x5)

        T2_1x = self.u4(self.up4(T2_x,T2_y4))
        T2_2x = self.u3(self.up3(T2_1x,T2_y3))
        T2_3x = self.u2(self.up2(T2_2x,T2_y2))
        T2_4x = self.u1(self.up1(T2_3x,T2_y1))

        out = self.outconv(T2_4x)

        return out

class SingleGenerator(nn.Module):
    def __init__(self, args):
        super(SingleGenerator,self).__init__()

        self.filters = args.filters
        self.device = args.device

        ## init the mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1'] == 1, device=args.device)
        self.maskNot = self.mask == 0

        self.KUnet = SingleUnet(2,2,self.filters)
        self.IUnet = SingleUnet(1,1,self.filters)

    def fftshift(self, img):
        '''
            4d tensor FFT operation
        '''
        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2
    
    def FT(self, image):
        '''
            Fourier operation 
        '''

        kspace_cplx = self.fftshift(torch.fft.fft2(image, dim=(2,3)))
        return kspace_cplx

    def inverseFT(self, Kspace):
        """The input Kspace has two channels(real and img)"""
        Kspace = Kspace.permute(0, 2, 3, 1)
        img_cmplx = torch.fft.ifft2(Kspace, dim=(1,2))
        img = torch.absolute(img_cmplx[:,:,:,0]+ 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img

    def KDC_layer(self, rec_K, und_K):
        '''
            K data consistency layer
        '''
        rec_Kspace = (self.mask*torch.complex(und_K[:, 0, :, :], und_K[:, 1, :, :]) + self.maskNot*torch.complex(rec_K[:, 0, :, :], rec_K[:, 1, :, :]))[:, None, :, :]
        final_rec =torch.absolute(torch.fft.ifft2(self.fftshift(rec_Kspace),dim=(2,3)))

        return final_rec, rec_Kspace
    
    def IDC_layer(self, rec_Img, und_K):
        und_k_cplx = torch.complex(und_K[:,0,:,:], und_K[:,1,:,:])[:,None,:,:] 
        rec_k = self.FT(rec_Img)
        final_k = (torch.mul(self.mask, und_k_cplx) + torch.mul(self.maskNot, rec_k))
        final_rec  =  torch.absolute(torch.fft.ifft2(self.fftshift(final_k), dim=(2,3)))

        return final_rec

    def forward(self, Kspace):
        reconK = self.KUnet(Kspace)

        rec_mid_img , rec_mid_k = self.KDC_layer(reconK, Kspace)

        rec_img = self.IUnet(rec_mid_img)

        final_rec_img = self.IDC_layer(rec_img, Kspace)

        return final_rec_img, reconK

################################ discriminator ####################################
class SepDown(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SepDown, self).__init__()
        
        self.sepConv = nn.Sequential(
            DeepWiseConv(inchannels,outchannels,kernel_size=3,padding=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2,True),
            DeepWiseConv(outchannels, outchannels,kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2,True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self,input):
        down = self.pool(input)
        out = self.sepConv(down)
        return out

class Discriminator(nn.Module):
    '''
        A modified sepDiscriminator, which make the discriminator more deeper.
    '''
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)

    def __init__(self,inchannels, ndf = 64, isFc = True, num_layers=5):
        super(Discriminator, self).__init__()
        sequence = []
        kw = 3
        padw = 1
        sequence+=[
            nn.Conv2d(inchannels, ndf, kernel_size=kw, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        ## reduce the feature map's size by half
        ndf_num = 1
        ndf_num_pre = 1
        for n in range(1, num_layers):
            ndf_num_pre = ndf_num
            ndf_num = min(2**n, 8)
            sequence+=[
                SepDown(ndf*ndf_num_pre, ndf*ndf_num)
            ]
        
        ndf_num_pre = ndf_num
        ndf_num = min(2**num_layers, 8)

        sequence+=[
            DeepWiseConv(ndf*ndf_num_pre, ndf*ndf_num, kernel_size=kw, padding=padw),
            nn.BatchNorm2d(ndf*ndf_num),
            nn.LeakyReLU(0.2, True),
        ]
        ### reduce the feature maps
        ndf_num_pre = ndf_num
        ndf_num = ndf_num//2
        sequence +=[
            DeepWiseConv(ndf*ndf_num_pre, ndf*ndf_num, kernel_size=kw, padding=padw),
            nn.BatchNorm2d(ndf*ndf_num),
            nn.LeakyReLU(0.2, True)
        ]
        if isFc:
            sequence+=[
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(ndf*ndf_num, 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128,1)
            ]
        else:
            sequence+=[
                DeepWiseConv(ndf*ndf_num, 1, kernel_size=kw, padding=padw)
            ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


