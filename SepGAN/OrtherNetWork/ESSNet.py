import os
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np
from AttentionModule import *
from unet_parts import *

import yaml as yaml
from types import SimpleNamespace

################################# Module part ##########################################################
class RIRB(nn.Module):
    def __init__(self, channel_num , stride = 1):
        super(RIRB, self).__init__()

        self.channel_num = channel_num
        self.stride = stride


        self.conv_1 = nn.Conv2d(self.channel_num, self.channel_num // 2, kernel_size=3, stride =1, padding=1)
        self.conv_2 = nn.Conv2d(self.channel_num //2, self.channel_num //2, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(self.channel_num //2, self.channel_num //2, kernel_size=1, stride=1)
        self.conv_4 = nn.Conv2d(self.channel_num //2, self.channel_num, kernel_size=3, stride=1, padding=1)


        self.bn = nn.BatchNorm2d(self.channel_num)
        self.bn_half = nn.BatchNorm2d(self.channel_num // 2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.activation(self.bn_half(self.conv_1(x)))
        out2 = self.activation(self.bn_half(self.conv_2(out1)))
        out3 = self.activation(self.bn_half(self.conv_3(out2)))
        res1 = torch.add(out1, out3)
        out4 = self.activation(self.bn(self.conv_4(res1)))  
        out = torch.add(x, out4)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.doubleConv(x)

class EncoderBlock(nn.Module):
    '''
        Encoder block first using the stride conv to achieve the down sample.
        Then the double conv is used.
    '''
    def __init__(self, in_channels, out_channels, is_residual = False):
        super(EncoderBlock, self).__init__()
        self.is_residual = is_residual
        self.down_conv = nn.Conv2d(in_channels, out_channels,kernel_size=3,stride=2,padding=1)
        self.relu = nn.LeakyReLU(0.2, True)
        self.bn = nn.BatchNorm2d(out_channels)

        self.doubleConv = DoubleConv(out_channels,out_channels)

    def forward(self, x):
        down_x = self.down_conv(x)
        down_x = self.bn(down_x)
        down_x = self.relu(down_x)

        out = self.doubleConv(down_x)
        if self.is_residual:
            output = out + down_x
        else:
            output = out

        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attName = '', is_residual = False):
        super(DecoderBlock, self).__init__()
        self.attName = attName

        self.doubleConv = DoubleConv(in_channels, out_channels)
        self.upConv = nn.ConvTranspose2d(out_channels,out_channels,kernel_size=4, stride=2, padding=1)
        self.is_residual = is_residual

        if attName == 'CBAM':
            self.att = CBAM(out_channels)
        elif attName == 'SA':
            self.att = ShuffleAttention(out_channels, 16)
        elif attName == 'PSA':
            self.att = PSAModule(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)

        ## for ressidual match in the Channel 
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        x = self.doubleConv(input)
        ## attention step
        if self.attName != '':
            att_x = self.att(x)
            att_x = self.relu(self.bn(att_x))
        else: 
            att_x = x

        ##  residual step 
        if self.is_residual:
            residualOut = self.conv1x1(input) + att_x
        else: 
            residualOut = att_x
        out = self.upConv(residualOut)
        return out

################################ Generator for image domain inputs #####################################
class PsychoGenerator(nn.Module):
    '''
        if you use psa, the nb_filter must large than 64
    '''
    def __init__(self, in_channels, out_channels, nb_filters=32, attName=''):
        super(PsychoGenerator, self).__init__()
        ###### first unet part
        self.input_layer_1 = DoubleConv(in_channels, nb_filters)  # 1*256*256 -> 32*256*256
        self.encoder1_0 = EncoderBlock(nb_filters, nb_filters*2)  # 32*256*256 -> 64*128*128
        self.encoder1_1 = EncoderBlock(nb_filters*2, nb_filters*4) # 64*128*128 -> 128*64*64
        self.encoder1_2 = EncoderBlock(nb_filters*4, nb_filters*8) # 128*64*64 -> 256*32*32
        self.encoder1_3 = EncoderBlock(nb_filters*8, nb_filters*16)  # 256*32*32 -> 512*16*16

        self.rirb_en_1_0 = RIRB(nb_filters)
        self.rirb_en_1_1 = RIRB(nb_filters*2)
        self.rirb_en_1_2 = RIRB(nb_filters*4)
        self.rirb_en_1_3 = RIRB(nb_filters*8)
        self.rirb_en_1_wave = RIRB(nb_filters*16)

        self.decoder1_3 = DecoderBlock(nb_filters*16, nb_filters*8,attName)
        self.decoder1_2 = DecoderBlock(nb_filters*8, nb_filters*4,attName)  
        self.decoder1_1 = DecoderBlock(nb_filters*4, nb_filters*2,attName)
        self.decoder1_0 = DecoderBlock(nb_filters*2, nb_filters, attName)

        self.output_layer_1 = DoubleConv(nb_filters, nb_filters)      

        ###### wave connection part
        self.rirb_wave_0 = RIRB(nb_filters)
        self.rirb_wave_1 = RIRB(nb_filters*2)
        self.rirb_wave_2 = RIRB(nb_filters*4)
        self.rirb_wave_3 = RIRB(nb_filters*8)

        ###### second unet part
        self.input_layer_2 = DoubleConv(1, nb_filters)  # 1*256*256 -> 32*256*256
        self.encoder2_0 = EncoderBlock(nb_filters, nb_filters*2)  # 32*256*256 -> 64*128*128
        self.encoder2_1 = EncoderBlock(nb_filters*2, nb_filters*4) # 64*128*128 -> 128*64*64
        self.encoder2_2 = EncoderBlock(nb_filters*4, nb_filters*8) # 128*64*64 -> 256*32*32
        self.encoder2_3 = EncoderBlock(nb_filters*8, nb_filters*16)  # 256*32*32 -> 512*16*16

        self.rirb_en_2_0 = RIRB(nb_filters)
        self.rirb_en_2_1 = RIRB(nb_filters*2)
        self.rirb_en_2_2 = RIRB(nb_filters*4)
        self.rirb_en_2_3 = RIRB(nb_filters*8)

        self.decoder2_3 = DecoderBlock(nb_filters*16, nb_filters*8, attName)
        self.decoder2_2 = DecoderBlock(nb_filters*8, nb_filters*4, attName)
        self.decoder2_1 = DecoderBlock(nb_filters*4, nb_filters*2, attName)
        self.decoder2_0 = DecoderBlock(nb_filters*2, nb_filters, attName)

        self.output_layer_2 = DoubleConv(nb_filters, nb_filters)

        self.tanh = nn.Tanh()
        self.conv1x1 = nn.Conv2d(nb_filters, 1,kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(nb_filters, out_channels,kernel_size=1)

    def forward(self, input):

        ## first unet part
        conv1_0 = self.input_layer_1(input) # 4*32*256*256
        conv1_1 = self.encoder1_0(conv1_0) #4*64*128*128
        conv1_2 = self.encoder1_1(conv1_1) #4*128*64*64
        conv1_3 = self.encoder1_2(conv1_2) #4*256*32*32
        conv1_4 = self.encoder1_3(conv1_3) #4*512*16*16

        RIR_en_1_0 = self.rirb_en_1_0(conv1_0)# 4*32*256*256
        RIR_en_1_1 = self.rirb_en_1_1(conv1_1)#4*64*128*128
        RIR_en_1_2 = self.rirb_en_1_2(conv1_2)#4*128*64*64
        RIR_en_1_3 = self.rirb_en_1_3(conv1_3)#4*256*32*32
        RIR_wave = self.rirb_en_1_wave(conv1_4)#4*512*16*16

        deconv1_4 = self.decoder1_3(conv1_4) #4*256*32*32
        residual1_4 = torch.add(RIR_en_1_3,deconv1_4) #4*256*32*32
        # residual1_4 = torch.cat((RIR_en_1_3,deconv1_4),1) #4*256*32*32

        deconv1_3 = self.decoder1_2(residual1_4) # 4*128*64*64
        residual1_3 = torch.add(RIR_en_1_2,deconv1_3) #4*128*64*64
        # residual1_3 = torch.cat((RIR_en_1_2,deconv1_3),1) #4*128*64*64

        deconv1_2 = self.decoder1_1(residual1_3) #4*64*128*128
        residual1_2 = torch.add(RIR_en_1_1,deconv1_2) #4*64*128*128
        # residual1_2 = torch.cat((RIR_en_1_1,deconv1_2),1) #4*64*128*128

        deconv1_1 = self.decoder1_0(residual1_2) #4*32*256*256
        residual1_1 = torch.add(RIR_en_1_0,deconv1_1) # 4*32*256*256
        # residual1_1 = torch.cat((RIR_en_1_0,deconv1_1),1) # 4*32*256*256

        out_1 = self.output_layer_1(residual1_1)  # 4*1*256*256
        out_1 = self.conv1x1(out_1)
        # out1 = torch.tanh(out_1)

        # out_1_refine = torch.add(input, out_1)

        # out_1_refine = self.tanh(out_1)
        out_1_refine = torch.clamp(out_1, 0,1)
        ## wave connection between two net
        RIR_wave_0 = self.rirb_wave_0(residual1_1)  #4*32*256*256
        RIR_wave_1 = self.rirb_wave_1(residual1_2) #4*64*128*128
        RIR_wave_2 = self.rirb_wave_2(residual1_3) #4*128*64*64
        RIR_wave_3 = self.rirb_wave_3(residual1_4) #4*256*32*32


        # # ## second unet part

        conv2_0 = self.input_layer_2(out_1_refine)
        wave_conn0 = torch.add(RIR_wave_0,conv2_0) #4*32*256*256

        conv2_1 = self.encoder2_0(wave_conn0)
        wave_conn1 = torch.add(RIR_wave_1, conv2_1)#4*64*128*128

        conv2_2 = self.encoder2_1(wave_conn1)
        wave_conn2 = torch.add(RIR_wave_2, conv2_2) #4*128*64*64

        conv2_3 = self.encoder2_2(wave_conn2)
        wave_conn3 = torch.add(RIR_wave_3, conv2_3)#4*256*32*32

        conv2_4 = self.encoder2_3(wave_conn3)
        wave_conn4 = torch.add(RIR_wave, conv2_4)#4*512*16*16

        RIR_en_2_0 = self.rirb_en_2_0(wave_conn0) #4*32*256*256
        RIR_en_2_1 = self.rirb_en_2_1(wave_conn1) #4*64*128*128
        RIR_en_2_2 = self.rirb_en_2_2(wave_conn2) #4*128*64*64
        RIR_en_2_3 = self.rirb_en_2_3(wave_conn3) #4*256*32*32

        deconv2_4 = self.decoder2_3(wave_conn4)
        residual2_4 = torch.add(deconv2_4,RIR_en_2_3) #4*256*32*32

        deconv2_3 = self.decoder2_2(residual2_4)
        residual2_3 = torch.add(deconv2_3,RIR_en_2_2)  

        deconv2_2 = self.decoder2_1(residual2_3)
        residual2_2 = torch.add(deconv2_2,RIR_en_2_1)    

        deconv2_1 = self.decoder2_0(residual2_2)
        residual2_1 = torch.add(deconv2_1,RIR_en_2_0)

        out_2 = self.output_layer_2(residual2_1)
        out_2 = self.conv1x1_2(out_2)

        out = torch.clamp(out_2, 0,1)
        return out, out_1_refine

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nb_filters):
        super(Generator, self).__init__()

        self.inLayer1 = ConvLayer(in_channels, nb_filters)

        self.encoder1_0 = EncoderBlock(nb_filters, nb_filters, is_residual=True)
        self.encoder1_1 = EncoderBlock(nb_filters, nb_filters*2, is_residual=True)
        self.encoder1_2 = EncoderBlock(nb_filters*2, nb_filters*4, is_residual=True)
        self.encoder1_3 = EncoderBlock(nb_filters*4, nb_filters*8, is_residual=True)


        self.rirb_en_1_0 = RIRB(nb_filters)
        self.rirb_en_1_1 = RIRB(nb_filters )
        self.rirb_en_1_2 = RIRB(nb_filters * 2)
        self.rirb_en_1_3 = RIRB(nb_filters * 4)
        self.rirb_en_1_4 = RIRB(nb_filters * 8)

        
        self.decoder1_3 = DecoderBlock(nb_filters*8, nb_filters*4, is_residual=True)
        self.decoder1_2 = DecoderBlock(nb_filters*4, nb_filters*2, is_residual=True)
        self.decoder1_1 = DecoderBlock(nb_filters*2, nb_filters*1, is_residual=True)
        self.decoder1_0 = DecoderBlock(nb_filters*1, nb_filters*1, is_residual=True)

        self.rirb_wave_0 = RIRB(nb_filters*1)
        self.rirb_wave_1 = RIRB(nb_filters*1)
        self.rirb_wave_2 = RIRB(nb_filters*2)
        self.rirb_wave_3 = RIRB(nb_filters*4)

        self.out_layer1 = nn.Conv2d(nb_filters, 1, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(1,1,1)
        ########################################## second part ###########################################
        
        self.inLayer2 =ConvLayer(1, nb_filters)

        self.encoder2_0 = EncoderBlock(nb_filters, nb_filters, is_residual=True)
        self.encoder2_1 = EncoderBlock(nb_filters, nb_filters*2, is_residual=True)
        self.encoder2_2 = EncoderBlock(nb_filters*2, nb_filters*4, is_residual=True)
        self.encoder2_3 = EncoderBlock(nb_filters*4, nb_filters*8, is_residual=True)


        self.rirb_en_2_0 = RIRB(nb_filters)
        self.rirb_en_2_1 = RIRB(nb_filters )
        self.rirb_en_2_2 = RIRB(nb_filters * 2)
        self.rirb_en_2_3 = RIRB(nb_filters * 4)

        
        self.decoder2_3 = DecoderBlock(nb_filters*8, nb_filters*4, is_residual=True)
        self.decoder2_2 = DecoderBlock(nb_filters*4, nb_filters*2, is_residual=True)
        self.decoder2_1 = DecoderBlock(nb_filters*2, nb_filters*1, is_residual=True)
        self.decoder2_0 = DecoderBlock(nb_filters*1, nb_filters*1, is_residual=True)

        self.out_layer2 = nn.Conv2d(nb_filters, 1, kernel_size=3, stride=1, padding=1)
        self.bn = BatchNorm2d(1)
        self.tanh = nn.Tanh()

    def forward(self, x):


        input1 = self.inLayer1(x)         #64*256*256
        en_1_0 = self.encoder1_0(input1)   #64*128*128
        en_1_1 = self.encoder1_1(en_1_0)    #128*64*64
        en_1_2 = self.encoder1_2(en_1_1)     #256*32*32
        en_1_3 = self.encoder1_3(en_1_2)      #512*16*16

        rirb_en_1_0 = self.rirb_en_1_0(input1)  #64*256*256
        rirb_en_1_1 = self.rirb_en_1_1(en_1_0)  #64*128*128
        rirb_en_1_2 = self.rirb_en_1_2(en_1_1)   #128*64*64
        rirb_en_1_3 = self.rirb_en_1_3(en_1_2)    #256*32*32
        rirb_wave_4 = self.rirb_en_1_4(en_1_3)     #512*16*16

        de_1_3 = self.decoder1_3(en_1_3)     
        residual1_3 = torch.add(de_1_3, rirb_en_1_3)    #256*32*32

        de_1_2 = self.decoder1_2(residual1_3)    
        residual1_2 = torch.add(de_1_2, rirb_en_1_2)   #128*64*64

        de_1_1 = self.decoder1_1(residual1_2)   #64*128*128
        residual1_1 = torch.add(de_1_1, rirb_en_1_1)

        de_1_0 = self.decoder1_0(residual1_1)   #64*256*256
        residual1_0 = torch.add(de_1_0, rirb_en_1_0)

        out1 = self.out_layer1(residual1_0)   #1*256*256
        out1 = self.conv1x1(out1)
        # out1 = torch.tanh(out1)

        # out1 = torch.clamp(out1, 0 ,1)

        rirb_wave_0 = self.rirb_wave_0(de_1_0)  #64*256*256
        rirb_wave_1 = self.rirb_wave_1(de_1_1)  #64*128*128
        rirb_wave_2 = self.rirb_wave_2(de_1_2)  #128*64*64
        rirb_wave_3 = self.rirb_wave_3(de_1_3)  #256*32*32

        ####################################################################################

        input2 = self.inLayer2(out1)  #64*256*256
        residual2_0 = torch.add(input2, rirb_wave_0)

        en_2_0 = self.encoder2_0(residual2_0) #64*128*128
        residual2_1 = torch.add(en_2_0, rirb_wave_1)

        en_2_1 = self.encoder2_1(residual2_1) #128*64*64
        residual2_2 = torch.add(en_2_1, rirb_wave_2)

        en_2_2 = self.encoder2_2(residual2_2) #256*32*32
        residual2_3 = torch.add(en_2_2, rirb_wave_3)

        en_2_3 = self.encoder2_3(residual2_3)  #512*16*16
        residual_bottle = torch.add(en_2_3, rirb_wave_4) 

        rirb_en_2_0 = self.rirb_en_2_0(input2)  #64*256*256
        rirb_en_2_1 = self.rirb_en_2_1(en_2_0)  #64*128*128
        rirb_en_2_2 = self.rirb_en_2_2(en_2_1)  #128*64*64
        rirb_en_2_3 = self.rirb_en_2_3(en_2_2)  #256*32*32

        de_2_3 = self.decoder2_3(residual_bottle)  #256*32*32
        residual2_3 = torch.add(de_2_3, rirb_en_2_3)
        de_2_2 = self.decoder2_2(residual2_3)  #128*64*64
        residual2_2 = torch.add(de_2_2, rirb_en_2_2)
        de_2_1 = self.decoder2_1(residual2_2)  #64*128*128
        residual2_1 = torch.add(de_2_1, rirb_en_2_1)
        de_2_0 = self.decoder2_0(residual2_1)  #64*256*256
        residual2_0 = torch.add(de_2_0, rirb_en_2_0)
        out2 = self.out_layer2(residual2_0)

        out = self.conv1x1(out2)
 
        out = torch.tanh(out)

        out = torch.clamp(out, 0, 1)
        return out, out1
