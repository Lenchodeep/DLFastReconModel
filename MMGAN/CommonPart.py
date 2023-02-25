import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

## depthwise seperable convolution
class DeepWiseConv(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(DeepWiseConv, self).__init__()
        self.deepwiseConv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=inchannels),
            nn.Conv2d(inchannels, outchannels, kernel_size=1)
        )  

    def forward(self,x):
        return self.deepwiseConv(x)

## common blocks
class DownSample(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2):
        super(DownSample,self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        out = self.downsample(x)
        return out, x

class UpSample_T(nn.Module):
    def __init__(self, filters):
        super(UpSample_T,self).__init__()
        # self.upsample = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DeepWiseConv(filters*2, int(filters/2), kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(int(filters/2))

    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat([x,y],dim=1)
        x = F.leaky_relu(self.bn(self.conv(x)),0.2)
        return x

class UpSample_R(nn.Module):
    def __init__(self, filters):
        super(UpSample_R,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DeepWiseConv(filters, int(filters/2), kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(int(filters/2))

    def forward(self, x):
        x = self.upsample(x)
        x = F.leaky_relu(self.bn(self.conv(x)),0.2)
        return x

class BasicBlockDown(nn.Module):
    def __init__(self, infilters, outfilters):
        super(BasicBlockDown, self).__init__()
        self.conv1 = DeepWiseConv(infilters, outfilters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(outfilters)
        
        self.conv2 = DeepWiseConv(outfilters, outfilters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outfilters)

        self.conv3 = DeepWiseConv(outfilters, outfilters, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(outfilters)

    def forward(self, x):
        ##first conv adjust the channel nums
        out1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)

        ##second conv
        out2 = F.leaky_relu(self.bn2(self.conv2(out1)), 0.2)

        ##third conv

        out3 = F.leaky_relu(self.bn3(self.conv3(out2)), 0.2)
        ## output including a residual connection
        return out3+out1

class BasicBlockUp(nn.Module):
    def __init__(self, filters):
        super(BasicBlockUp, self).__init__()
        self.conv1 = DeepWiseConv(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = DeepWiseConv(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self,x):
        ##first conv
        out1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        ##second conv
        out2 = F.leaky_relu(self.bn2(self.conv2(out1)), 0.2)
        ## output including a residual connection
        return x+out2

class DRDB(nn.Module):
    def __init__(self, channels,dilateSet = None, isSeparable = True):
        super(DRDB, self).__init__()
        if dilateSet is None:
            dilateSet = [1,2,4,4]

        self.conv = nn.Sequential(
            DRDB_Conv(channels*1, channels, k_size=3, dilate=dilateSet[0], isSep = isSeparable),
            DRDB_Conv(channels*2, channels, k_size=3, dilate=dilateSet[1], isSep = isSeparable),
            DRDB_Conv(channels*3, channels, k_size=3, dilate=dilateSet[2], isSep = isSeparable),
            DRDB_Conv(channels*4, channels, k_size=3, dilate=dilateSet[3], isSep = isSeparable)
        )

        self.outConv = nn.Conv2d(channels*5, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        
        fusion_out = self.conv(input)
        out = self.relu(self.bn(self.outConv(fusion_out)))

        return out


class DRDB_Conv(nn.Module):
    def __init__(self,inchannels, outchannels, k_size = 3, dilate=1, stride = 1, isSep=True):
        super(DRDB_Conv, self).__init__()
        if isSep:
            self.conv = DeepWiseConv(inchannels, outchannels, kernel_size=k_size, padding=dilate*(k_size-1)//2
                            , dilation=dilate, stride=stride)
        else:
            self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=k_size, padding=dilate*(k_size-1)//2
                            , dilation=dilate, stride=stride),
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat((input, out), dim=1)        

class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()

        self.conv = nn.Sequential(
            DenseConv(channels*1, channels, k_size=3),
            DenseConv(channels*2, channels, k_size=3),
            DenseConv(channels*3, channels, k_size=3),
            DenseConv(channels*4, channels, k_size=3)
        )

        self.outConv = nn.Conv2d(channels*5, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        
        fusion_out = self.conv(input)
        out = self.relu(self.bn(self.outConv(fusion_out)))

        return out
        
class DenseConv(nn.Module):
    def __init__(self,inchannels, outchannels, k_size = 3, dilate=1, stride = 1):
        super(DenseConv, self).__init__()
   
        self.conv = DeepWiseConv(inchannels, outchannels, kernel_size=k_size, padding=dilate*(k_size-1)//2
                        , dilation=dilate, stride=stride)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat((input, out), dim=1)   
############################ segnext part ################################################
class MSCA(nn.Module):

    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) 
        # split into multipats of multiscale attention
        self.conv13_0 = nn.Conv2d(dim, dim, (1,3), padding=(0, 1), groups=dim)
        self.conv13_1 = nn.Conv2d(dim, dim, (3,1), padding=(1, 0), groups=dim)

        self.conv17_0 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1,11), padding=(0, 5), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (11,1), padding=(5, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1) # channel mixer

    def forward(self, x):
        
        skip = x.clone()

        c55 = self.conv55(x)
        c17 = self.conv13_0(x)
        c17 = self.conv13_1(c17)
        c111 = self.conv17_0(x)
        c111 = self.conv17_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

        return op


class BlockMSCA(nn.Module):
    def __init__(self, dim, ls_init_val=1e-2, drop_path=0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):

        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)
        out = self.sig(x)

        return out


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