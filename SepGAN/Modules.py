import sys
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

#################################################  Depthwise convolution ##############################################

class DeepWiseConv(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(DeepWiseConv, self).__init__()
        self.deepwiseConv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=inchannels),
            nn.Conv2d(inchannels, outchannels, kernel_size=1)
        )

    def forward(self,x):
        return self.deepwiseConv(x)


################################################ ASPP module ############################################################

class ASPP(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ASPP, self).__init__()
        dilations = [1,2,4,8]
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels//4, kernel_size=1, dilation=dilations[0]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 =nn.Sequential(
            nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        ) 
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=dilations[3], dilation=dilations[3]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )

        self.imagePooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(inchannels, inchannels//4, kernel_size=1),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )

        self.outConv = nn.Sequential(
            nn.Conv2d(inchannels//4*5, outchannels, kernel_size=1),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )
        self.dropout = nn.Dropout(0.5)


    def forward(self, input):
        conv_1 = self.conv1(input)
        conv_2 = self.conv2(input)
        conv_3 = self.conv3(input)

        conv_4 = self.conv4(input)

        pool = self.imagePooling(input)
        pool = F.interpolate(pool, size=conv_1.size()[2:], mode='bilinear', align_corners=True)

        catOut = torch.cat((conv_1,conv_2, conv_3, conv_4, pool), dim=1)

        out = self.outConv(catOut)

        return self.dropout(out)


class Sep_ASPP(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Sep_ASPP, self).__init__()
        dilations = [1,2,4,8]

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels//4, kernel_size=1, dilation=dilations[0]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )

        self.conv2 =nn.Sequential(
            DeepWiseConv(inchannels, inchannels//4, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        ) 
        self.conv3 = nn.Sequential(
            DeepWiseConv(inchannels, inchannels//4, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            DeepWiseConv(inchannels, inchannels//4, kernel_size=3, padding=dilations[3], dilation=dilations[3]),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )

        self.imagePooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(inchannels, inchannels//4, kernel_size=1),
            nn.BatchNorm2d(inchannels//4),
            nn.LeakyReLU(0.2, True)
        )

        self.outConv = nn.Sequential(
            nn.Conv2d(inchannels//4*5, outchannels, kernel_size=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2, True)
        )
        self.dropout = nn.Dropout(0.5)


    def forward(self, input):
        conv_1 = self.conv1(input)
        conv_2 = self.conv2(input)
        conv_3 = self.conv3(input)
        conv_4 = self.conv4(input)

        pool = self.imagePooling(input)
        pool = F.interpolate(pool, size=conv_4.size()[2:], mode='bilinear', align_corners=True)
        catOut = torch.cat((conv_1,conv_2, conv_3, conv_4, pool), dim=1)
        out = self.outConv(catOut)

        return self.dropout(out)





################################################   RDB NET PART ################################################
"""
    It's a network structure which can extract more spatial information like aspp Module. multi-scale fusion
    Referance paper: DuDoRNet: Learning a Dual-Domain Recurrent Network for Fast MRI Reconstruction with Deep T1 Prior
"""
#### why not use relu and batchnormal
class DRDB_Conv(nn.Module):
    def __init__(self,inchannels, outchannels, k_size = 3, dilate=1, stride = 1, isSep=False):
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SDRDB(nn.Module):
    def __init__(self,channels, ratio=8):
        super(SDRDB,self).__init__()

        self.drdb = DRDB(channels)
        self.se = SELayer(channels, ratio)

    def forward(self, input):
        drdbout = self.drdb(input)
        seout = self.se(drdbout)
        out =torch.add(seout, input)

        return out



############################################# dense block #####################################################
class DenseLayer(nn.Module):
    def __init__(self, inchannels, outchannels, kernerSize, padding, isSep = True):
        super(DenseLayer, self).__init__()
        self.bn =nn.BatchNorm2d(outchannels)
        self.relu = nn.LeakyReLU(0.2,True)
        if isSep:
            self.conv = DeepWiseConv(inchannels, outchannels, kernel_size=kernerSize, padding=padding)
        else: 
            self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=kernerSize, padding=padding)
        
    def forward(self, input):
        out = (self.relu(self.bn(self.conv(input))))
        return torch.cat((input,out), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, channels = 32, GR = 4):
        super(DenseBlock, self).__init__()
        sequence = []
        for i in range(GR):
            scale = i + 1
            sequence +=  [
                DenseLayer(scale * channels, channels, kernerSize=3, padding=1, isSep=True)
            ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

