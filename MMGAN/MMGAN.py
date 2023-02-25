import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

from CommonPart import *

class MMUnet(nn.Module):
    def __init__(self, in_chan,out_chan, filters):
        super(MMUnet,self).__init__()
        self.conv1T1 = nn.Conv2d(in_chan,int(filters/2),1)
        self.conv1T2 = nn.Conv2d(in_chan*2,int(filters/2),1)

        self.dT1_1 = BasicBlockDown(int(filters/2), filters)
        self.downT1_1 = DownSample()
        self.dT1_2 = BasicBlockDown(filters, filters*2)
        self.downT1_2 = DownSample()
        self.dT1_3 = BasicBlockDown(filters*2, filters*4)
        self.downT1_3 = DownSample()
        self.dT1_4 = BasicBlockDown(filters*4, filters*8)
        self.downT1_4 = DownSample()


        self.dT2_1 = BasicBlockDown(int(filters/2), filters)
        self.downT2_1 = DownSample()
        self.dT2_2 = BasicBlockDown(filters, filters*2)
        self.downT2_2 = DownSample()
        self.dT2_3 = BasicBlockDown(filters*2, filters*4)
        self.downT2_3 = DownSample()
        self.dT2_4 = BasicBlockDown(filters*4, filters*8)
        self.downT2_4 = DownSample()


        self.bottomT1 = DenseBlock(filters*8)
        self.bottomT2 = DenseBlock(filters*8)

        self.up4_T1 = UpSample_R(filters*8)
        self.u4_T1 = BasicBlockUp(filters*4)
        self.up3_T1 = UpSample_R(filters*4)
        self.u3_T1 = BasicBlockUp(filters*2)
        self.up2_T1 = UpSample_R(filters*2)
        self.u2_T1 = BasicBlockUp(filters)
        self.up1_T1 = UpSample_R(filters)
        self.u1_T1 = BasicBlockUp(int(filters/2))


        self.up4_T2 = UpSample_T(filters*8)
        self.u4_T2 = BasicBlockUp(filters*4)
        self.up3_T2 = UpSample_T(filters*4)
        self.u3_T2 = BasicBlockUp(filters*2)
        self.up2_T2 = UpSample_T(filters*2)
        self.u2_T2 = BasicBlockUp(filters)
        self.up1_T2 = UpSample_T(filters)
        self.u1_T2 = BasicBlockUp(int(filters/2))
        
        self.outconvT1 = nn.Conv2d(int(filters/2),out_chan, 1)
        self.outconvT2 = nn.Conv2d(int(filters/2),out_chan, 1)

        self.msca1=BlockMSCA(filters)
        self.msca2=BlockMSCA(filters*2)
        self.msca3=BlockMSCA(filters*4)
        self.msca4=BlockMSCA(filters*8)

        self.bottom=BlockMSCA(filters*8)

        self.mscaU4=BlockMSCA(filters*4)
        self.mscaU3=BlockMSCA(filters*2)
        self.mscaU2=BlockMSCA(filters*1)
        self.mscaU1=BlockMSCA(int(filters/2))

        self.acT1 = nn.Tanh()
        self.acT2 = nn.Tanh()

    def forward(self, T1, T2):
        ## input feature extract
        T1_x1 = self.conv1T1(T1)
        T2cat = torch.cat([T1,T2], dim = 1)
        T2_x1 = self.conv1T2(T2cat)

        #encoder part
        T1_x2,T1_y1 = self.downT1_1(self.dT1_1(T1_x1))
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        temp = T1_x2.mul(self.msca1(T1_x2))
        T12_x2 = T2_x2.mul(temp)+T2_x2

        T1_x3,T1_y2 = self.downT1_2(self.dT1_2(T1_x2))
        T2_x3,T2_y2 = self.downT2_2(self.dT2_2(T12_x2))
        temp = T1_x3.mul(self.msca2(T1_x3))
        T12_x3 = T2_x3.mul(temp)+T2_x3

        T1_x4,T1_y3 = self.downT1_3(self.dT1_3(T1_x3))
        T2_x4,T2_y3 = self.downT2_3(self.dT2_3(T12_x3))
        temp = T1_x4.mul(self.msca3(T1_x4))
        T12_x4 = T2_x4.mul(temp)+T2_x4        

        T1_x5,T1_y4 = self.downT1_4(self.dT1_4(T1_x4))
        T2_x5,T2_y4 = self.downT2_4(self.dT2_4(T12_x4))
        temp = T1_x5.mul(self.msca4(T1_x5))
        T12_x5 = T2_x5.mul(temp)+T2_x5   

        ##bootleneck part
        T1_x = self.bottomT1(T1_x5)
        T2_x = self.bottomT2(T12_x5)

        ##decoder part
        T1_1x = self.u4_T1(self.up4_T1(T1_x))
        T2_1x = self.u4_T2(self.up4_T2(T2_x,T2_y4))
        temp = T1_1x.mul(self.mscaU4(T1_1x))
        T12_x = T2_1x.mul(temp)+T2_1x   

        T1_2x = self.u3_T1(self.up3_T1(T1_1x))
        T2_2x = self.u3_T2(self.up3_T2(T12_x,T2_y3))
        temp = T1_2x.mul(self.mscaU3(T1_2x))
        T12_x = T2_2x.mul(temp)+T2_2x   

        T1_3x = self.u2_T1(self.up2_T1(T1_2x))
        T2_3x = self.u2_T2(self.up2_T2(T12_x,T2_y2))
        temp = T1_3x.mul(self.mscaU2(T1_3x))
        T12_x = T2_3x.mul(temp)+T2_3x    

        T1_4x = self.u1_T1(self.up1_T1(T1_3x))
        T2_4x = self.u1_T2(self.up1_T2(T12_x,T2_y1))
        temp = T1_4x.mul(self.mscaU1(T1_4x))
        T12_x = T2_4x.mul(temp)+T2_4x   
        ##output part
        outT1 = self.acT1(self.outconvT1(T1_4x))
        outT2 = self.acT2(self.outconvT2(T12_x))

        outT1 = outT1+T1
        outT2 = outT2+T2

        return outT1,outT2

class MMGenerator(nn.Module):
    def __init__(self, args):
        super(MMGenerator,self).__init__()
        self.filters = args.filters
        self.device = args.device
        self.g0 = MMUnet(2,2,self.filters)
        self.g1 = MMUnet(2,2,self.filters)
        self.dc = Dataconsistency()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_un_T1, k_un_T1, x_un_T2, k_un_T2, mask):
        ## first stage kspace restoration
        x_rec0_T1, x_rec0_T2 = self.g0(x_un_T1, x_un_T2)
        ## DC operation
        x_dc0_T1 = self.dc(x_rec0_T1, mask, k_un_T1)
        x_dc0_T2 = self.dc(x_rec0_T2, mask, k_un_T2)

        x_rec1_T1, x_rec1_T2 = self.g1(x_dc0_T1, x_dc0_T2)
        x_dc1_T1 = self.dc(x_rec1_T1, mask, k_un_T1)
        x_dc1_T2 = self.dc(x_rec1_T2, mask, k_un_T2)

        x_dc1_T1 =  torch.clamp(x_dc1_T1, 0, 1)
        x_dc1_T2 =  torch.clamp(x_dc1_T2, 0, 1)

        return x_rec0_T1, x_rec0_T2, x_dc0_T1, x_dc0_T2, x_rec1_T1, x_rec1_T2, x_dc1_T1, x_dc1_T2
