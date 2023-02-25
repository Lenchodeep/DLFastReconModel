import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

from CommonPart import *

class Dataconsistency(nn.Module):
    def __init__(self):
        super(Dataconsistency, self).__init__()

    def forward(self, x_rec, mask, k_un, norm='ortho'):
        x_rec = x_rec.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        k_un = k_un.permute(0, 2, 3, 1)
        k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()),dim=(1,2))
        k_rec = torch.view_as_real(k_rec)
        k_out = k_rec + (k_un - k_rec) * mask
        k_out = torch.view_as_complex(k_out)
        x_out = torch.view_as_real(torch.fft.ifft2(k_out,dim=(1,2)))
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out




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
        self.ac = nn.Tanh()
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

        out = self.ac(self.outconv(T2_4x))

        return out +T2

class SingleGenerator(nn.Module):
    def __init__(self, args):
        super(SingleGenerator,self).__init__()

        self.filters = args.filters
        self.device = args.device
        self.args = args
        ## init the mask
        self.dc = Dataconsistency()
        self.g0 = SingleUnet(2,2,self.filters)
        self.g1 = SingleUnet(2,2,self.filters)

    def forward(self, T2masked, Kspace, mask):
        reconimg = self.g0(T2masked)

        rec_mid_img = self.dc(reconimg, mask, Kspace)

        rec_img = self.g1(rec_mid_img)

        final_rec_img = self.dc(rec_img, mask, Kspace)

        if self.args.isZeroToOne:
            final_rec_img = torch.clamp(final_rec_img,0,1)
        else:
            final_rec_img = torch.clamp(final_rec_img,-1,1)
        return reconimg, rec_mid_img, rec_img, final_rec_img
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


class Discriminator1(nn.Module):
    
    def __init__(self, ndf):
        super(Discriminator1, self).__init__()
        filters = ndf
        self.inconv = nn.Conv2d(2,int(filters/2),1)

        self.d_1 = BasicBlockDown(int(filters/2), filters)
        self.down_1 = DownSample()
        self.d_2 = BasicBlockDown(filters, filters*2)
        self.down_2 = DownSample()
        self.d_3 = BasicBlockDown(filters*2, filters*4)
        self.down_3 = DownSample()
        self.d_4 = BasicBlockDown(filters*4, filters*8)
        self.down_4 = DownSample()
        self.bottom = DenseBlock(filters*8)
        self.ret = nn.Conv2d(filters*8, 1, kernel_size=4, stride=1, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)

        
    def forward(self, x):
        input = self.inconv(x)
        x_e1,_ = self.down_1(self.d_1(input))
        x_e2,_ = self.down_2(self.d_2(x_e1))
        x_e3,_ = self.down_3(self.d_3(x_e2))
        x_e4,_ = self.down_4(self.d_4(x_e3))
        x_e4 = self.bottom(x_e4)
        ret = self.ret(x_e4)

        return ret