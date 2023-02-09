""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from AttentionModule import *

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ##########################################################################################
#residual Unet part
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


##################################################  RIRB UNET PART  ###################################################
class RIRBBlock (nn.Module):
    def __init__(self, ndf):
        super(RIRBBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ndf, out_channels=ndf//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=ndf//2, out_channels=ndf//2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=ndf//2, out_channels=ndf//2, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=ndf//2, out_channels=ndf, kernel_size=3, padding=1)

        self.relu = nn.LeakyReLU(0.2, True)
        self.bn1 = nn.BatchNorm2d(ndf//2)
        self.bn2 = nn.BatchNorm2d(ndf)

    def forward(self, x):
        input = x
        out1 = self.conv1(x)
        out1 = self.relu(self.bn1(out1))
        out2 = self.conv2(out1)
        out2 = self.relu(self.bn1(out2))
        out3 = self.conv3(out2)
        out3 = self.relu(self.bn1(out3))
        res1 = torch.add(out1, out3)
        out4 = self.conv4(res1)
        out4 = self.relu(self.bn2(out4))
        out = torch.add(input, out4)       ##### wheather residual connection need to use add as connection method

        return out

class RIRBEncoder(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(RIRBEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2, True)
        )
        
        self.rirb = RIRBBlock(outchannels)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.rirb(out1)
        return out2

class RIRBDecoder(nn.Module):
    def __init__(self, inchannels, outchannels, attName = ''):
        super(RIRBDecoder, self).__init__()
        self.conv = Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2, True)
        )
        self.rirb = RIRBBlock(outchannels)
        if attName == 'PSA':
            print('psa')
            self.att = PSAModule(outchannels)
        elif attName == 'SA':
            print("sa")
            self.att = ShuffleAttention(outchannels) 
        else:
            print("here")
            self.att = None
    
    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.rirb(out1)
        if self.att is not None:
            out = self.att(out2)
        else:
            out = out2    
        return out


class DownConv(nn.Module):
    def __init__(self, inchannels, outchannels, pooling = True):
        super(DownConv,self).__init__()
        if pooling:
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                RIRBEncoder(inchannels, outchannels)
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels=inchannels, out_channels = outchannels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(outchannels),
                nn.LeakyReLU(0.2,True),
                RIRBEncoder(outchannels, outchannels)
            )

    def forward(self, x):
        return self.down(x)

class UpConv(nn.Module):
    def __init__(self, inchannels,outchannels, bilinear = True, attName = ''):
        super(UpConv, self).__init__() 

        if bilinear:
            
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = RIRBDecoder(inchannels, outchannels, attName)
             
        else:
            self.up = nn.ConvTranspose2d(inchannels , inchannels // 2, kernel_size=4, stride=2, padding=1)
            self.conv = RIRBDecoder(inchannels, outchannels, attName)
            

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2], dim = 1)
        return self.conv(x)


