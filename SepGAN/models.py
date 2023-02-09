
import os
import torch
import torch.nn as nn
import numpy as np
from AttentionModule import *
from unet_parts import *

import yaml as yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt
#################################### A Discriminator ##################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # set parameter
        self.df_dim = 64
        self.fin = 8192

        # network
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.df_dim, kernel_size=5, stride=2, padding=2, ),
            nn.LeakyReLU(negative_slope=0.2)

        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim, out_channels=self.df_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 2, out_channels=self.df_dim * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 4, out_channels=self.df_dim * 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 8, out_channels=self.df_dim * 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 16, out_channels=self.df_dim * 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 32),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 32, out_channels=self.df_dim * 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 16, out_channels=self.df_dim * 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.res8 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 8, out_channels=self.df_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.df_dim * 2, out_channels=self.df_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.df_dim * 2, out_channels=self.df_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
        )

        self.LRelu = nn.LeakyReLU(negative_slope=0.2)

        self.out = nn.Sequential(
            nn.Linear(self.fin, 1),
            nn.Sigmoid()
        )

    # forward propagation
    def forward(self, input_image):
        net_in = input_image
        net_h0 = self.conv0(net_in)
        net_h1 = self.conv1(net_h0)
        net_h2 = self.conv2(net_h1)
        net_h3 = self.conv3(net_h2)
        net_h4 = self.conv4(net_h3)
        net_h5 = self.conv5(net_h4)
        net_h6 = self.conv6(net_h5)
        net_h7 = self.conv7(net_h6)
        res_h7 = self.res8(net_h7)
        net_h8 = self.LRelu(res_h7 + net_h7)
        net_ho = net_h8.contiguous().view(net_h8.size(0), -1)
        logits = self.out(net_ho)

        return logits


###############################################  SEP NET ###################################################

from Modules import DRDB, DeepWiseConv , Sep_ASPP
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

class SepUp(nn.Module):
    def __init__(self,inchannels, outchannels):
        super(SepUp, self).__init__()

        self.sepConv = nn.Sequential(
            DeepWiseConv(inchannels,outchannels,kernel_size=3,padding=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2,True),
            DeepWiseConv(outchannels,outchannels,kernel_size=3,padding=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2,True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input1, input2):

        upOut = self.up(input1)
        residual = torch.cat((upOut, input2), dim=1)
        out = self.sepConv(residual)

        return out

class SepUnet(nn.Module):
    def __init__(self, inchannels, outchannels, ndf, bottleneckName = 'ASPP'):
        super(SepUnet, self).__init__()

        self.inflow = nn.Sequential(
            nn.Conv2d(inchannels, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf//2, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,True)
        )

        self.down1 = SepDown(ndf, ndf*2)
        self.down2 = SepDown(ndf*2, ndf*4)
        self.down3 = SepDown(ndf*4, ndf*8)
        self.down4 = SepDown(ndf*8, ndf*16)

        self.bottleConv1 = DeepWiseConv(ndf*16,ndf*16,kernel_size=3, padding=1)
        self.bottleConv2 = DeepWiseConv(ndf*16, ndf*16, kernel_size=3, padding=1)

        if bottleneckName == 'ASPP':
            self.bottleOper = Sep_ASPP(ndf*16, ndf*8)
        elif bottleneckName == 'SESOAM': 
            self.bottleOper = nn.Sequential(
                SESOAM(ndf*16),
                DeepWiseConv(ndf*16, ndf*8, kernel_size=3, padding=1)   ##
            )
        elif bottleneckName == "DRDB":
            self.bottleOper = nn.Sequential(
                DRDB(ndf*16),
                DeepWiseConv(ndf*16, ndf*8, kernel_size=3, padding=1)   ##
            )
        else:
            self.bottleOper = DeepWiseConv(ndf*16, ndf*8, kernel_size=3, padding=1)   ##

        self.up4 = SepUp(ndf*16, ndf*4)
        self.up3 = SepUp(ndf*8, ndf*2)
        self.up2 = SepUp(ndf*4, ndf*1)
        self.up1 = SepUp(ndf*2, ndf)

        self.outflow = nn.Sequential(
            nn.Conv2d(ndf, ndf//2, kernel_size=1),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf//2, outchannels, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(outchannels,outchannels,kernel_size=1),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, input):
        in_flow  = self.inflow(input)

        en_1 = self.down1(in_flow)
        en_2  = self.down2(en_1)
        en_3 = self.down3(en_2)
        en_4 = self.down4(en_3)

        bt1 = self.bottleConv1(en_4)
        bt2 = self.bottleConv2(bt1)

        aspp_out = self.bottleOper(bt2)

        up4 = self.up4(aspp_out, en_3)
        up3 = self.up3(up4,en_2)
        up2 = self.up2(up3, en_1)
        up1 = self.up1(up2, in_flow)

        out = self.outflow(up1)

        return out

class SepDiscriminator(nn.Module):
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)

    def __init__(self, inchannels, ndf, isFC = False):
        super(SepDiscriminator, self).__init__()

        sequence = []
        sequence+=[
            nn.Conv2d(inchannels, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf//2, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,True),
            SepDown(ndf, ndf*2),
            SepDown(ndf*2, ndf*4),
            SepDown(ndf*4, ndf*8),
            SepDown(ndf*8, ndf*16)
        ]

        if isFC:
            sequence+=[
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(ndf*16, 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128,1)
            ]
        else:
            sequence+=[
                nn.Conv2d(ndf*16, 1, kernel_size=3,stride=1, padding=1)
            ]

        self.model = nn.Sequential(*sequence).apply(self.weight_init)

    def forward(self, input):
        return self.model(input)

############################################  SEP NET V2 ######################################################
"""
    Sep Net V2 using residual connection in the encoder and decoder part. 
    And using the aspp or sesoam module in the bottleNeck block.
    The whole model using the deepwise separable convolution as the basic convlution part.
"""
class SepResidualConv(nn.Module):
    def __init__(self, channels):
        super(SepResidualConv, self).__init__()
        self.conv = nn.Sequential(
            DeepWiseConv(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels), 
            nn.LeakyReLU(0.2, True),
            DeepWiseConv(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2,True)
        )

    def forward(self, x):
        input = x
        conv_o = self.conv(x)
        residual = conv_o+input
        return residual

class sepRirBlock(nn.Module):
    def __init__(self, channels):
        super(sepRirBlock, self).__init__()

        self.conv1 =DeepWiseConv(channels, channels //2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels//2)
        self.relu1 = nn.LeakyReLU(0.2, True)
        
        self.conv2 = DeepWiseConv(channels//2, channels//2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels//2)
        self.relu2 = nn.LeakyReLU(0.2, True)

        self.conv3 = DeepWiseConv(channels//2, channels//2, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels//2)
        self.relu3 = nn.LeakyReLU(0.2, True)

        self.conv4 = DeepWiseConv(channels//2, channels, kernel_size=3, padding=1)
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

class SepDownV2(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SepDownV2, self).__init__()
        self.conva = DeepWiseConv(inchannels, outchannels, kernel_size=3, padding=1)
        self.bna = nn.BatchNorm2d(outchannels)
        self.relua = nn.LeakyReLU(0.2, True)

        self.en_sepRes = SepResidualConv(outchannels)

        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        input = x
        conv_a = self.relua(self.bna(self.conva(input)))
        rirbout = self.en_sepRes(conv_a)
        out = self.down(rirbout)
        return out

class SepUpV2(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SepUpV2, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.de_sepRes = SepResidualConv(inchannels)
        self.deconvb = DeepWiseConv(inchannels, outchannels, kernel_size=3,  padding=1)   ## how to select the kernel sizes
        self.bnb = nn.BatchNorm2d(outchannels)
        self.relub = nn.LeakyReLU(0.2, True)
    def forward(self,input1, input2):
        upOut = self.up(input1)
        residual = torch.cat((upOut, input2), dim=1)
        derirbout = self.de_sepRes(residual)
        deconvb = self.relub(self.bnb(self.deconvb(derirbout)))
        return deconvb

class SepUnetV2(nn.Module):
    def __init__(self, inchannels, outchannels, ndf, bottleNeckName = 'ASPP'):
        super(SepUnetV2, self).__init__()

        self.inflow = nn.Sequential(
            nn.Conv2d(inchannels, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf//2, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,True)
        )
        self.down1 = SepDownV2(ndf, ndf*2)
        self.down2 = SepDownV2(ndf*2, ndf*4)
        self.down3 = SepDownV2(ndf*4, ndf*8)
        self.down4 = SepDownV2(ndf*8, ndf*16)

        self.bottleConv1 = DeepWiseConv(ndf*16,ndf*16,kernel_size=3, padding=1)
        self.bottleConv2 = DeepWiseConv(ndf*16, ndf*16, kernel_size=3, padding=1)


        if bottleNeckName == 'ASPP':
            self.bottleOper = Sep_ASPP(ndf*16, ndf*8)
        elif bottleNeckName == 'SESOAM': 
            self.bottleOper = nn.Sequential(
                SESOAM(ndf*16),
                DeepWiseConv(ndf*16, ndf*8, kernel_size=3, padding=1)   ##
            )
        elif bottleNeckName == "DRDB":
            self.bottleOper = nn.Sequential(
                DRDB(ndf*16),
                DeepWiseConv(ndf*16, ndf*8, kernel_size=3, padding=1)   ##
            )
        else:
            self.bottleOper = DeepWiseConv(ndf*16, ndf*8, kernel_size=3, padding=1)   ##
            
        self.up4 = SepUpV2(ndf*16, ndf*4)
        self.up3 = SepUpV2(ndf*8, ndf*2)
        self.up2 = SepUpV2(ndf*4, ndf*1)
        self.up1 = SepUpV2(ndf*2, ndf)

        self.outflow = nn.Sequential(
            nn.Conv2d(ndf, ndf//2, kernel_size=1),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf//2, outchannels, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(outchannels,outchannels,kernel_size=1),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, input):
        in_flow  = self.inflow(input)
        en_1 = self.down1(in_flow)
        en_2  = self.down2(en_1)
        en_3 = self.down3(en_2)
        en_4 = self.down4(en_3)

        bt1 = self.bottleConv1(en_4)
        bt2 = self.bottleConv2(bt1)

        aspp_out = self.bottleOper(bt2)

        up4 = self.up4(aspp_out, en_3)
        up3 = self.up3(up4,en_2)
        up2 = self.up2(up3, en_1)
        up1 = self.up1(up2, in_flow)

        out = self.outflow(up1)
        return out

class SepDiscriminatorV2(nn.Module):
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
        super(SepDiscriminatorV2, self).__init__()
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


########################################### SEP NET V3 #########################################################
class SepResidualConvV3(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SepResidualConvV3, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(0.2,True),
            DeepWiseConv(inchannels, outchannels,kernel_size=3,padding=1),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2,True),
            DeepWiseConv(outchannels, outchannels,kernel_size=3,padding=1)
        )
        self.skip_conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0)
        self.skip_bn = nn.BatchNorm2d(outchannels)

    def forward(self, input):
        out1 = self.conv(input)
        out2 = self.skip_bn(self.skip_conv(input))
        out = torch.add(out1, out2)
        return out

class SepDownV3(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SepDownV3, self).__init__()
        self.resConv = SepResidualConvV3(inchannels, outchannels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, input):
        down = self.pool(input)
        out = self.resConv(down)
        return out

class SepUpV3(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SepUpV3, self).__init__()
        self.resConv = SepResidualConvV3(inchannels, outchannels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input1, input2):
        upOut = self.up(input1)
        residual = torch.cat((upOut, input2), dim=1)

        out = self.resConv(residual)

        return out

class SepUnetV3(nn.Module):
    def __init__(self, inchannels, outchannels, ndf, bnottleNeckName = 'ASPP',DRDB_num = 0 ,useSESOAM = True):
        super(SepUnetV3, self).__init__()
        self.inflow = nn.Sequential(
            nn.Conv2d(inchannels, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf//2, ndf, kernel_size=3, stride=1, padding=1)
        )
        self.inflow_skip = nn.Sequential(
            nn.Conv2d(inchannels, ndf, kernel_size=1, padding=0),
            nn.BatchNorm2d(ndf)
        )
        self.down1 = SepDownV3(ndf, ndf*2)
        self.down2 = SepDownV3(ndf*2, ndf*4)
        self.down3 = SepDownV3(ndf*4, ndf*8)
        self.down4 = SepDownV3(ndf*8, ndf*16)


        sequence = []
        sequence +=[
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, True),
            DeepWiseConv(ndf*16,ndf*8,kernel_size=3, padding=1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True)
        ]
        if DRDB_num !=0:
            for _ in range(DRDB_num):
                sequence+=[
                    DRDB(ndf*8,isSeparable=True)
                ]
        sequence +=[
            DeepWiseConv(ndf*8,ndf*8,kernel_size=3, padding=1)
        ]
        self.bottleOper = nn.Sequential(*sequence)
        self.up4 = SepUpV3(ndf*16, ndf*4)
        self.up3 = SepUpV3(ndf*8, ndf*2)
        self.up2 = SepUpV3(ndf*4, ndf*1)
        self.up1 = SepUpV3(ndf*2, ndf)

        if useSESOAM:
            self.sesoam =nn.Sequential(           
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2,True),
                SESOAM(ndf)
                )
            # self.sesoam =nn.Sequential(           
            #     nn.BatchNorm2d(ndf),
            #     nn.LeakyReLU(0.2,True),
            #     SOAM(ndf)
            #     )
        else:
            self.sesoam = nn.Sequential(
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2,True),
                nn.Identity()
            )

        self.outflow = nn.Sequential(
            nn.Conv2d(ndf, ndf//2, kernel_size=1),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf//2, outchannels, kernel_size=1),
        )

    def forward(self, input):
        in_flow  = self.inflow(input)
        en_1 = self.down1(in_flow)
        en_2  = self.down2(en_1)
        en_3 = self.down3(en_2)
        en_4 = self.down4(en_3)

        aspp_out = self.bottleOper(en_4)

        up4 = self.up4(aspp_out, en_3)
        up3 = self.up3(up4,en_2)
        up2 = self.up2(up3, en_1)
        up1 = self.up1(up2, in_flow)
        att = self.sesoam(up1)
        # att, _, _ = self.sesoam(up1)

        out = self.outflow(att)
        return out, att

class sepDiscriminatorV3(nn.Module):

    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)

    def __init__(self, inchannels, ndf = 64, isFc = False, num_layers=3):
        super(sepDiscriminatorV3, self).__init__()
        ks = 3
        padw = 1
        sequence = [
            nn.Conv2d(inchannels, ndf, kernel_size=ks, padding=padw),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf, ndf, kernel_size=ks, padding=padw),
        ]
        ndf_num_pre = 1
        ndf_num = 1

        for n in range(1, num_layers):
            ndf_num_pre = ndf_num
            ndf_num = min(2**n, 8)

            sequence += [
                SepDownV3(ndf*ndf_num_pre, ndf*ndf_num)
            ]

        ndf_num_pre = ndf_num
        ndf_num = min(2**num_layers, 8)

        sequence+=[
            nn.MaxPool2d(2),
            nn.BatchNorm2d(ndf*ndf_num_pre),
            nn.LeakyReLU(0.2,True),
            DeepWiseConv(ndf*ndf_num_pre, ndf*ndf_num, kernel_size=ks, padding=padw),
            nn.BatchNorm2d(ndf*ndf_num),
            nn.LeakyReLU(0.2,True)
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
                nn.Conv2d(ndf*ndf_num, 1, kernel_size=ks, padding=padw)
            ]
        self.model = nn.Sequential(*sequence).apply(self.weight_init)

    def forward(self, input):
        return self.model(input)
################################################################################################################

def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args



import pickle
####  using the args to control the generator version
class SepWnet(nn.Module):
    def __init__(self, args):
        super(SepWnet, self).__init__()

        inchannels = args.num_input_slices*2
        outchannels = args.outchannels
        nb_filters = args.nb_filters
        ## control the mid dc and out dc operation
        self.isKDc = args.isKDC  
        self.outDc = args.outDC
        self.device = args.device

        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(np.fft.fftshift(masks['mask1']) == 1, device=args.device)
        # self.mask = torch.tensor(np.fft.fftshift(masks['mask1']) == 1)

        self.maskNot = self.mask == 0

        if args.netV =="V2":
            if args.bottleNeckV == "SESOAM":
                print("The Generator using SepNet version 2 with SESOAM as bottleneck!!")
                self.kUnet = SepUnetV2(inchannels, 2, nb_filters, 'SESOAM') 
                self.iUnet =SepUnetV2(1, outchannels, nb_filters, 'SESOAM')
            else:
                print("The Generator using SepNet version 2 with ASPP as bottleneck!!")
                self.kUnet = SepUnetV2(inchannels, 2, nb_filters, 'ASPP') 
                self.iUnet =SepUnetV2(1, outchannels, nb_filters, 'ASPP')
        elif args.netV =="V3":
            print("The Generator using SepNet version 3 with DRDB as bottleneck and SESOAM as the output layer!!")
            self.kUnet = SepUnetV3(inchannels, 2, nb_filters, 'ASPP', DRDB_num=args.DRDBnum, useSESOAM=True) 
            self.iUnet =SepUnetV3(1, outchannels, nb_filters, 'ASPP', DRDB_num=args.DRDBnum, useSESOAM=True)
        else:
            if args.bottleNeckV == "SESOAM":
                print("The Generator using SepNet version 1 with SESOAM as bottleneck!!")
                self.kUnet = SepUnet(inchannels, 2, nb_filters, 'SESOAM') 
                self.iUnet =SepUnet(1, outchannels, nb_filters, 'SESOAM')
            else:
                print("The Generator using SepNet version 1 with ASPP as bottleneck!!")
                self.kUnet = SepUnet(inchannels, 2, nb_filters, 'ASPP') 
                self.iUnet =SepUnet(1, outchannels, nb_filters, 'ASPP')
        
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
        img = torch.absolute(img_cmplx[:,:,:,0]+ 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img
    
    def KDC_layer(self, rec_K, und_K, mask, maskNot):
        rec_Kspace =self.mask*und_K[:, int(und_K.shape[1]/2)-1:int(und_K.shape[1]/2)+1, :, :]+self.maskNot*rec_K
        final_k = torch.complex(rec_Kspace[:,0:1,:,:], rec_Kspace[:,1:2,:,:])
        final_rec = torch.abs(torch.fft.ifft2(final_k, dim=(2,3)))
        return final_rec, rec_Kspace


    def IDC_layer(self, rec_img, und_K, mask, maskNot):
        und_k_cplx = torch.complex(und_K[:,int(und_K.shape[1]/2)-1,:,:], und_K[:,int(und_K.shape[1]/2),:,:])[:,None,:,:] 
        rec_k = self.FT(rec_img)
        final_k = torch.mul(mask, und_k_cplx) + torch.mul(maskNot, rec_k)
        final_rec  = torch.abs(torch.fft.ifft2(final_k, dim=(2,3)))
        return final_rec
    
    def FT(self, image):
        kspace_cplx = torch.fft.fft2(image, dim=(2,3))
        return kspace_cplx

    def forward(self, kspace):
        # recon_all_kspace = self.kUnet(kspace)
        recon_all_kspace, map = self.kUnet(kspace)

        if self.isKDc:
            rec_mid_img , rec_Kspace= self.KDC_layer(recon_all_kspace, kspace, self.mask, self.maskNot)
        else:
            rec_Kspace = recon_all_kspace
            rec_mid_img = self.inverseFT(rec_Kspace)

        # refine_img = self.iUnet(rec_mid_img)
        refine_img, att = self.iUnet(rec_mid_img)
        rec_img = torch.clamp(torch.tanh(refine_img+ rec_mid_img), 0,1)

        ### operate the DC operation after the I net
        if self.outDc:
            final_rec = self.IDC_layer(rec_img, kspace, self.mask, self.maskNot)
        else:
            final_rec = rec_img
        # return final_rec, rec_Kspace,rec_mid_img

        return final_rec, rec_Kspace,rec_mid_img, att



class SepPatchGan(nn.Module):
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)
    def __init__(self, inchannels, ndf, n_layers, isFC):
        super(SepPatchGan,self).__init__()
        ks = 3
        padw=1
        sequence = [
            nn.Conv2d(inchannels, ndf, kernel_size=ks, stride=1, padding=padw), 
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf, ndf, kernel_size=ks, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace= True)
        ]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence+=[
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks,stride=1, padding=padw),  ###BN之前的卷积层中的偏置b不会起到作用
                nn.BatchNorm2d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf*nf_mult, ndf*nf_mult, kernel_size=ks, stride=2, padding=padw),  ###BN之前的卷积层中的偏置b不会起到作用
                nn.BatchNorm2d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True),
                # nn.MaxPool2d(2)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence +=[
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks, stride = 1, padding=padw),
            nn.BatchNorm2d(ndf*nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]


        if isFC:
            sequence += [
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(ndf*nf_mult, 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128,1) 
            ]
        else:
            sequence+=[
                nn.Conv2d(ndf*nf_mult, 1,kernel_size=ks, stride=1, padding=padw)
            ]
        
        self.model = nn.Sequential(*sequence).apply(self.weight_init)  ##apply函数将函数递归传递到网络模型的每一个子模型中，主要用于参数的初始化过程中

    def forward(self,x):
        return self.model(x)


############################################# the transformer Unet #############################################

from Transformer import *
from einops.layers.torch import Rearrange
class TransUNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """
        super(TransUNet, self).__init__()
        patch_size = 16
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)

        ####### transformer part
        patch_dim = 32 * patch_size * patch_size  ##patch size=16
        num_patches = (256//patch_size) **2
        self.trans = VisionTransformer(embed_dim=patch_dim, depth=4, num_heads=4,mlp_ratio=4)
        self.patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size,
                         p2=patch_size),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, patch_dim))
        self.p1 = patch_size
        self.p2 = patch_size

        self.outc = OutConv(32, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        ################## transformer part

        b,_,h,w = x.shape
        ## embed
        x = self.patch_embbeding(x)
        x += self.pos_embedding
        ##
        x = self.trans(x)

        H = int(h/self.p1)
        W = int(w/self.p2)
        c = int(x.shape[2]/(self.p1*self.p2))

        x = x.reshape(b, H, W, self.p1, self.p2, c)  # b H W p1 p2 c
        x = x.permute(0, 5, 1, 3, 2, 4)  # b c H p1 W p2
        x = x.reshape(b, -1, h, w,)

        out = self.outc(x)
        return out


if __name__ == "__main__":
    args = get_args()

    netG = SepWnet(args)
    netD = SepDiscriminatorV2(1, 64, True, 4)

    print('    Total params of G: %.4fM' % (sum(p.numel() for p in netG.parameters()) / 1000000)) 
    print('    Total params of D: %.4fM' % (sum(p.numel() for p in netD.parameters()) / 1000000))
    print('    Total params : %.4fM' %((sum(p.numel() for p in netG.parameters()) / 1000000)+(sum(p.numel() for p in netD.parameters()) / 1000000)))

    from torchstat import stat
    from KINet import WNet
    # from KINet import WNet
    args = get_args()
    conv1 = SepDiscriminatorV2(inchannels=1, ndf=64, isFc=True, num_layers=4)
    # conv1 = SepWnet(args)
    stat(conv1,(1,256,256))

    # conv1 = SepWnet(args)
    # D = sepDiscriminatorV3(1)
    # summary(D, (1,256,256), device='cuda')

    # tensor = torch.rand((4,1,256,256))
    # model = SepUnetV3(1,1,32)
    # out = model(tensor)

    # model = SepWnet(args).to(device='cuda')
    # from torchsummary import summary

    # summary(model, (6,256,256), device='cuda')