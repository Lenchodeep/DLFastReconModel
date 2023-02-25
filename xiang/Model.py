import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

class DenseBlock(nn.Module):
    def __init__(self, channels=16):
        super(DenseBlock, self).__init__()

        self.conv = nn.Sequential(
            DenseConv(channels*1, channels, k_size=3),
            DenseConv(channels*2, channels, k_size=3),
            DenseConv(channels*3, channels, k_size=3),
            DenseConv(channels*4, channels, k_size=3)
        )
    def forward(self, input):
        fusion_out = self.conv(input)
        return fusion_out
        
class DenseConv(nn.Module):
    def __init__(self,inchannels, outchannels, k_size = 3, padding = 1):
        super(DenseConv, self).__init__()
   
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=k_size, padding=padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat((input, out), dim=1)   
    
class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x),x
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.denConv = DenseBlock()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.down = DownSample()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.denConv(out1)
        out3 = self.conv2(out2)
        out,_ = self.down(out3)
        return out, out3

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=80, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.denseConv = DenseBlock()

    def forward(self, x, y):
        up_x = self.up(x)
        out1 = self.conv1(torch.cat([up_x, y], dim=1))
        out = self.denseConv(out1)

        return out


class Xnet(nn.Module):
    def __init__(self):
        super(Xnet,self).__init__()

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.en1 = Encoder()
        self.en2 = Encoder()

        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            DenseBlock()
        )

        self.de1 = Decoder()
        self.de2 = Decoder()

        self.out = nn.Conv2d(in_channels=80, out_channels=1, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)
    def forward(self, T1, T2):
        input = torch.cat([T1, T2], dim = 1)
        input_o = self.inconv(input)

        en_1, y1 = self.en1(input_o)
        en_2, y2 = self.en2(en_1)

        bottom = self.bottom(en_2)

        de_2 = self.de2(bottom,y2) 
        de_1 = self.de1(de_2, y1)

        out = self.out(de_1)
        torch.clamp(out, 0, 1)
        return out


if __name__ == '__main__':
    # tensor1 = torch.rand((1,1,256,256)).to('cuda')
    # model = Xnet().to('cuda')
    # out = model(tensor1, tensor1)
    pass

