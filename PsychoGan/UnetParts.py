"""The Unet parts"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d 

from AttentionModule import *

class DoubleConv(nn.Module):
    """conv->BN->LRelu*2"""
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )
    def forward(self, x):
        return self.doubleConv(x)

class Down(nn.Module):
    """maxPooling->doubleConv"""
    def __init__(self, in_channels, out_channels, ispool = True):
        super().__init__()
        if ispool:
            self.down = nn.MaxPool2d(2)
        else:
            self.down = nn.Conv2d(in_channels,in_channels, kernel_size=2,padding=1,stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self,x):
        x = self.down(x)
        x = self.conv(x)
        return x


class UP(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels,in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        ## if the 2 input are not in the same size, input is BCHW

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2,x1], dim = 1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)

##################################################################################
##residual unet part

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, stride=2):
        super(ResidualConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride = stride),   ##using stride conv to down sample
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x1 = self.conv_block(x)
        x2 = self.conv_skip(x)
        return x1+x2


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.resConv = ResidualConv(output_channels*2, output_channels,stride = 1)
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1,x2], dim=1) 
        out = self.resConv(x)
        return out


class UpCBAM(nn.Module):
    def __init__(self, input_channels, output_channels, bilinear = True):
        super(UpCBAM, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(input_channels, output_channels, input_channels//2)
        else:
            self.up = nn.ConvTranspose2d(input_channels, input_channels //2, kernel_size=2,stride=2)
            self.conv = DoubleConv(input_channels, output_channels)
        self.cbam = CBAM(output_channels, ratio=8, kernel_size=7 )
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation =  nn.LeakyReLU(0.2,inplace=True)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2,x1], dim = 1)
        
        out = self.conv(x)
        out = self.cbam(out)
        out = self.bn(out)
        return self.activation(out) 