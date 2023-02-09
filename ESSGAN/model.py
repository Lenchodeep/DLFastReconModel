import os
from numpy import pad
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules.activation import LeakyReLU

class RIRB(nn.Module):
    def __init__(self, channel_num, stride = 1):
        super(RIRB, self).__init__()

        self.channel_num = channel_num
        self.stride = stride

        self.conv_1 = nn.Conv2d(self.channel_num, self.channel_num // 2, kernel_size=3, stride =1, padding=1)
        self.conv_2 = nn.Conv2d(self.channel_num //2, self.channel_num //2, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(self.channel_num //2, self.channel_num //2, kernel_size=1, stride=1)
        self.conv_4 = nn.Conv2d(self.channel_num //2, self.channel_num, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(self.channel_num)
        self.bn_half = nn.BatchNorm2d(self.channel_num //2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    

    def forward(self, x):
        out1 = self.activation(self.bn_half(self.conv_1(x)))
        out2 = self.activation(self.bn_half(self.conv_2(out1)))
        out3 = self.activation(self.bn_half(self.conv_3(out2)))
        res1 = torch.add(out1, out3)
        out4 = self.activation(self.bn(self.conv_4(res1)))
        out = torch.add(x, out4)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_i = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.rirb = RIRB(self.out_channels, stride = 1)
        self.conv_o = nn.Conv2d(self.out_channels, self.out_channels , kernel_size=3, stride=1, padding=1) ##the output channel is double


        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = self.activation(self.bn(self.conv_i(x)))
        out2 = self.rirb(out1)
        out3 = self.activation(self.bn(self.conv_o(out2)))

        return out3      
 


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_i = nn.ConvTranspose2d(self.in_channels, self.out_channels , kernel_size=3 ,stride=1 ,padding=1)
        self.rirb = RIRB(self.out_channels, stride=1)
        self.conv_o = nn.ConvTranspose2d(self.out_channels, self.out_channels,kernel_size=4, stride=2, padding=1)  #$ different with original code

        self.activation = nn.LeakyReLU(0.2 , inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = self.activation(self.bn(self.conv_i(x)))
        out2 = self.rirb(out1)
        out3 = self.activation(self.bn(self.conv_o(out2)))
        return out3

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels , nb_filters ):
        super(Discriminator , self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_filters = nb_filters

        self.input_layer = nn.Conv2d(self.in_channels, self.nb_filters, kernel_size=3, stride=2, padding=1)
        self.conv1_1 = EncoderBlock(self.nb_filters,self.nb_filters )
        self.conv1_2 = EncoderBlock(self.nb_filters, self.nb_filters*2)
        self.conv1_3 = EncoderBlock(self.nb_filters*2, self.nb_filters*4)
        self.conv1_4 = EncoderBlock(self.nb_filters*4, self.nb_filters*8)

        self.conv1_5 = nn.Conv2d(self.nb_filters*8, 1, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(64,1)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
         
    def forward(self, x):
        out1 = self.activation(self.input_layer(x))
        out2 = self.conv1_1(out1)
        out3 = self.conv1_2(out2)
        out4 = self.conv1_3(out3)
        out5 = self.conv1_4(out4)
        out6 = self.conv1_5(out5)
        out7 = self.flatten(out6)
        out = self.linear(out7)

        return out


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nb_filters):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_layer = nn.Conv2d(self.in_channels, nb_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(nb_filters)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.encoder1_0 = EncoderBlock(nb_filters , nb_filters)
        self.encoder1_1 = EncoderBlock(nb_filters , nb_filters*2)
        self.encoder1_2 = EncoderBlock(nb_filters*2 , nb_filters*4)
        self.encoder1_3 = EncoderBlock(nb_filters*4 , nb_filters*8)

        self.rirb_en_1_0 = RIRB(nb_filters)
        self.rirb_en_1_1 = RIRB(nb_filters )
        self.rirb_en_1_2 = RIRB(nb_filters * 2)
        self.rirb_en_1_3 = RIRB(nb_filters * 4)
        self.rirb_en_1_4 = RIRB(nb_filters * 8)

        self.decoder1_3 = DecoderBlock(nb_filters*8, nb_filters*4)
        self.decoder1_2 = DecoderBlock(nb_filters*4, nb_filters*2)
        self.decoder1_1 = DecoderBlock(nb_filters*2, nb_filters*1)
        self.decoder1_0 = DecoderBlock(nb_filters*1, nb_filters*1)

        self.rirb_wave_0 = RIRB(nb_filters*1)
        self.rirb_wave_1 = RIRB(nb_filters*1)
        self.rirb_wave_2 = RIRB(nb_filters*2)
        self.rirb_wave_3 = RIRB(nb_filters*4)

        self.out_layer = nn.Conv2d(nb_filters, 1, kernel_size=3, stride=1, padding=1)


        ########################################## second part ###########################################
        self.input_layer_2 = nn.Conv2d(1, nb_filters, kernel_size=3, stride=1, padding=1)

        self.encoder_2_0 = EncoderBlock(nb_filters, nb_filters)
        self.encoder_2_1 = EncoderBlock(nb_filters, nb_filters*2)
        self.encoder_2_2 = EncoderBlock(nb_filters*2, nb_filters*4)
        self.encoder_2_3 = EncoderBlock(nb_filters*4, nb_filters*8)

        self.rirb_en_2_0 = RIRB(nb_filters)
        self.rirb_en_2_1 = RIRB(nb_filters*1)
        self.rirb_en_2_2 = RIRB(nb_filters*2)
        self.rirb_en_2_3 = RIRB(nb_filters*4)

        self.decoder_2_3 = DecoderBlock(nb_filters*8, nb_filters*4)
        self.decoder_2_2 = DecoderBlock(nb_filters*4, nb_filters*2)
        self.decoder_2_1 = DecoderBlock(nb_filters*2, nb_filters)
        self.decoder_2_0 = DecoderBlock(nb_filters, nb_filters)

        self.out_layer_2 = nn.Conv2d(nb_filters, self.out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        input = self.activation(self.bn(self.input_layer(x)))   #4*64*256*256

        en_1_0 = self.encoder1_0(input)  #4*64*128*128
        en_1_1 = self.encoder1_1(en_1_0)  #4*128*64*64
        en_1_2 = self.encoder1_2(en_1_1)  #2*256*32*32
        en_1_3 = self.encoder1_3(en_1_2)  #48512*16*16

        rirb_en_1_0 = self.rirb_en_1_0(input)
        rirb_en_1_1 = self.rirb_en_1_1(en_1_0)
        rirb_en_1_2 = self.rirb_en_1_2(en_1_1)
        rirb_en_1_3 = self.rirb_en_1_3(en_1_2)
        rirb_en_1_4 = self.rirb_en_1_4(en_1_3)

        de_1_3 = self.decoder1_3(en_1_3)   #4*256*32*32
        de_1_3 = torch.add(rirb_en_1_3, de_1_3)
        de_1_2 = self.decoder1_2(de_1_3)   #4*128*64*64
        de_1_2 = torch.add(rirb_en_1_2, de_1_2)
        de_1_1 = self.decoder1_1(de_1_2)   #4*64*128*128
        de_1_1 = torch.add(rirb_en_1_1, de_1_1)
        de_1_0 = self.decoder1_0(de_1_1)   #4*64*256*256 
        de_1_0 = torch.add(rirb_en_1_0 , de_1_0)

        out1 = self.out_layer(de_1_0)
        out1 = torch.add(x, out1)
        # out1 = torch.tanh(out1)


        rirb_wave_00 = self.rirb_wave_0(de_1_0)
        rirb_wave_01 = self.rirb_wave_1(de_1_1)
        rirb_wave_02 = self.rirb_wave_2(de_1_2)
        rirb_wave_03 = self.rirb_wave_3(de_1_3)


        ################################################################################################################

        input2 = self.activation(self.bn(self.input_layer_2(out1)))
        input2 = torch.add(rirb_wave_00, input2)

        en_2_0 = self.encoder_2_0(input2)
        en_2_0 = torch.add(rirb_wave_01, en_2_0)

        en_2_1 = self.encoder_2_1(en_2_0)
        en_2_1 = torch.add(rirb_wave_02, en_2_1)


        en_2_2 = self.encoder_2_2(en_2_1)
        en_2_2 = torch.add(rirb_wave_03, en_2_2)

        en_2_3 = self.encoder_2_3(en_2_2)
        en_2_3 = torch.add(rirb_en_1_4, en_2_3)

        
        rirb_en_2_0 = self.rirb_en_2_0(input2)
        rirb_en_2_1 = self.rirb_en_2_1(en_2_0)
        rirb_en_2_2 = self.rirb_en_2_2(en_2_1)
        rirb_en_2_3 = self.rirb_en_2_3(en_2_2)

        de_2_3 = self.decoder_2_3(en_2_3)
        de_2_3 = torch.add(rirb_en_2_3,de_2_3)

        de_2_2 = self.decoder_2_2(de_2_3)
        de_2_2 = torch.add(rirb_en_2_2, de_2_2)

        de_2_1 = self.decoder_2_1(de_2_2)
        de_2_1 = torch.add(rirb_en_2_1 ,de_2_1)

        de_2_0 = self.decoder_2_0(de_2_1)
        de_2_0 = torch.add(rirb_en_2_0, de_2_0)

        out2 = self.out_layer_2(de_2_0)
        out = torch.add(out1, out2)

        # out = torch.tanh(out)
        out = torch.clamp(out, 0,1)
        return out

class Generator2(nn.Module):
    def __init__(self, in_channels, out_channels, nb_filters = 64):
        super(Generator2, self).__init__()
        
        self.inputLayer = nn.Conv2d(in_channels, nb_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(nb_filters)
        self.relu = nn.LeakyReLU(0.2, True)

        self.encoder_1_0 = EncoderBlock(nb_filters, nb_filters)
        self.encoder_1_1 = EncoderBlock(nb_filters*1, nb_filters*2)
        self.encoder_1_2 = EncoderBlock(nb_filters*2, nb_filters*4)
        self.encoder_1_3 = EncoderBlock(nb_filters*4, nb_filters*8)

        self.rirb_en_1_0 = RIRB(nb_filters)
        self.rirb_en_1_1 = RIRB(nb_filters)
        self.rirb_en_1_2 = RIRB(nb_filters*2)
        self.rirb_en_1_3 = RIRB(nb_filters*4)
        self.rirb_en_1_4 = RIRB(nb_filters*8)

        self.decoder_1_3 = DecoderBlock(nb_filters*8, nb_filters*4)
        self.decoder_1_2 = DecoderBlock(nb_filters*4, nb_filters*2)
        self.decoder_1_1 = DecoderBlock(nb_filters*2, nb_filters*1)
        self.decoder_1_0 = DecoderBlock(nb_filters*1, nb_filters*1)
        
        # self.decoder_1_3 = DecoderBlock(nb_filters*8, nb_filters*4)
        # self.decoder_1_2 = DecoderBlock(nb_filters*4*2, nb_filters*2)
        # self.decoder_1_1 = DecoderBlock(nb_filters*2*2, nb_filters*1)
        # self.decoder_1_0 = DecoderBlock(nb_filters*1*2, nb_filters*1)

        self.rirb_wave_0 = RIRB(nb_filters*1)
        self.rirb_wave_1 = RIRB(nb_filters*1)
        self.rirb_wave_2 = RIRB(nb_filters*2)
        self.rirb_wave_3 = RIRB(nb_filters*4)

        self.outputLayer = nn.Conv2d(nb_filters*1, 1,kernel_size=3,padding=1) 
        # self.outputLayer = nn.Conv2d(nb_filters*2, 1,kernel_size=3,padding=1) 

        ############################# second part ####################################################
        self.inputLayer2 = nn.Conv2d(1,nb_filters, kernel_size=3,padding=1)
        
        self.encoder_2_0 = EncoderBlock(nb_filters, nb_filters)
        self.encoder_2_1 = EncoderBlock(nb_filters*1, nb_filters*2)
        self.encoder_2_2 = EncoderBlock(nb_filters*2, nb_filters*4)
        self.encoder_2_3 = EncoderBlock(nb_filters*4, nb_filters*8)

        self.rirb_en_2_0 = RIRB(nb_filters*1)
        self.rirb_en_2_1 = RIRB(nb_filters*1)
        self.rirb_en_2_2 = RIRB(nb_filters*2)
        self.rirb_en_2_3 = RIRB(nb_filters*4)

        self.decoder_2_3 = DecoderBlock(nb_filters*8, nb_filters*4)
        self.decoder_2_2 = DecoderBlock(nb_filters*4, nb_filters*2)
        self.decoder_2_1 = DecoderBlock(nb_filters*2, nb_filters*1)
        self.decoder_2_0 = DecoderBlock(nb_filters*1, nb_filters*1)

        self.oputputLayer2 = nn.Conv2d(nb_filters*1, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        conv1_0 = self.inputLayer(input)
        conv1_0 = self.relu(self.bn(conv1_0))

        conv1_1 = self.encoder_1_0(conv1_0)
        conv1_2 = self.encoder_1_1(conv1_1)
        conv1_3 = self.encoder_1_2(conv1_2)
        conv1_4 = self.encoder_1_3(conv1_3)

        RIR_en_1_0 = self.rirb_en_1_0(conv1_0)
        RIR_en_1_1 = self.rirb_en_1_1(conv1_1)
        RIR_en_1_2 = self.rirb_en_1_2(conv1_2)
        RIR_en_1_3 = self.rirb_en_1_3(conv1_3)
        RIR_en_1_4 = self.rirb_en_1_4(conv1_4)

        deconv1_4 = self.decoder_1_3(conv1_4)
        residual1_1 = torch.add(RIR_en_1_3, deconv1_4)
        # residual1_1 = torch.add(conv1_3, deconv1_4)
        # residual1_1 = torch.cat((RIR_en_1_3, deconv1_4), dim=1)


        deconv1_3 = self.decoder_1_2(residual1_1)
        residual1_2 = torch.add(RIR_en_1_2, deconv1_3)
        # residual1_2 = torch.add(conv1_2, deconv1_3)
        # residual1_2 = torch.cat((RIR_en_1_2, deconv1_3), dim=1)


        deconv1_2 = self.decoder_1_1(residual1_2)
        residual1_3 = torch.add(RIR_en_1_1, deconv1_2)
        # residual1_3 = torch.add(conv1_1, deconv1_2)
        # residual1_3 = torch.cat((RIR_en_1_1, deconv1_2), dim=1)



        deconv1_1 = self.decoder_1_0(residual1_3)
        residual1_4 = torch.add(RIR_en_1_0, deconv1_1)
        # residual1_4 = torch.add(conv1_0, deconv1_1)
        # residual1_4 = torch.cat((RIR_en_1_0, deconv1_1), dim=1)



        output1_1 = self.outputLayer(residual1_4)
        output1_1 = self.tanh(output1_1)
        output1_2 = torch.add(input, output1_1)
        # output1_2 = torch.clamp(output1_2, 0,1)

        RIR_de_1_0 = self.rirb_wave_0(residual1_4)
        RIR_de_1_1 = self.rirb_wave_1(residual1_3)
        RIR_de_1_2 = self.rirb_wave_2(residual1_2)
        RIR_de_1_3 = self.rirb_wave_3(residual1_1)

        conv2_0 = self.inputLayer2(output1_2)
        conv2_0 = self.relu(self.bn(conv2_0))

        Wave_conn1_0 = torch.add(RIR_de_1_0, conv2_0)
        conv2_1 = self.encoder_2_0(Wave_conn1_0)

        Wave_conn1_1 = torch.add(RIR_de_1_1, conv2_1)
        conv2_2 = self.encoder_2_1(Wave_conn1_1)

        Wave_conn1_2 = torch.add(RIR_de_1_2, conv2_2)
        conv2_3 = self.encoder_2_2(Wave_conn1_2)

        Wave_conn1_3 = torch.add(RIR_de_1_3, conv2_3)
        conv2_4 = self.encoder_2_3(Wave_conn1_3)

        Wave_conn1_4 = torch.add(RIR_en_1_4, conv2_4)

        RIR_en_2_0 = self.rirb_en_2_0(Wave_conn1_0)
        RIR_en_2_1 = self.rirb_en_2_1(Wave_conn1_1)
        RIR_en_2_2 = self.rirb_en_2_2(Wave_conn1_2)
        RIR_en_2_3 = self.rirb_en_2_3(Wave_conn1_3)

        deconv2_4 = self.decoder_2_3(Wave_conn1_4)
        residual2_1 = torch.add(RIR_en_2_3, deconv2_4)

        deconv2_3 = self.decoder_2_2(residual2_1)
        residual2_2 = torch.add(RIR_en_2_2, deconv2_3)      

        deconv2_2 = self.decoder_2_1(residual2_2)  
        residual2_3 = torch.add(RIR_en_2_1, deconv2_2)

        deconv2_1 = self.decoder_2_0(residual2_3)
        residual2_4 = torch.add(RIR_en_2_0, deconv2_1)

        out = self.oputputLayer2(residual2_4)
        out = self.tanh(out)

        out = torch.add(out, output1_2)
    
        out = torch.clamp(out, 0,1)

        return out



def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
        nn.init.zeros_(m.bias)



# tensor = torch.rand((4,1,256,256))
# net = Unet(1,1)
# net(tensor)
