import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

############################ component ############################
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

## depthwise seperable convolution
class DSC(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(DSC, self).__init__()
        self.deepwiseConv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=inchannels),
            nn.Conv2d(inchannels, outchannels, kernel_size=1)
        )  

    def forward(self,x):
        return self.deepwiseConv(x)

class ResicualBlock(nn.Module):
    def __init__(self, chan):
        super(ResicualBlock, self).__init__()
        self.block = nn.Sequential(
            DSC(chan, chan, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2),
            DSC(chan, chan // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan // 2),
            nn.LeakyReLU(0.2),
            DSC(chan // 2, chan, kernel_size=3, stride=1, padding=1),  # nl=tf.identity
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return x + self.block(x)


class Residual_enc(nn.Module):
    def __init__(self, in_chan, chan):
        super(Residual_enc, self).__init__()
        self.block = nn.Sequential(
            DSC(in_chan, chan, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2),
            ResicualBlock(chan),
            DSC(chan, chan, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


class Residual_dec(nn.Module):
    def __init__(self, in_chan, chan):
        super(Residual_dec, self).__init__()
        self.block = nn.Sequential(
            DSC(in_chan, chan, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2),
            ResicualBlock(chan),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.block(x)

class DRDB(nn.Module):
    def __init__(self, channels,dilateSet = None):
        super(DRDB, self).__init__()
        if dilateSet is None:
            dilateSet = [1,2,4,4]

        self.conv = nn.Sequential(
            DRDB_Conv(channels*1, channels, k_size=3, dilate=dilateSet[0]),
            DRDB_Conv(channels*2, channels, k_size=3, dilate=dilateSet[1]),
            DRDB_Conv(channels*3, channels, k_size=3, dilate=dilateSet[2]),
            DRDB_Conv(channels*4, channels, k_size=3, dilate=dilateSet[3])
        )

        self.outConv = nn.Conv2d(channels*5, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        
        fusion_out = self.conv(input)
        out = self.relu(self.bn(self.outConv(fusion_out)))

        return out


class DRDB_Conv(nn.Module):
    def __init__(self,inchannels, outchannels, k_size = 3, dilate=1, stride = 1):
        super(DRDB_Conv, self).__init__()
        self.conv = DSC(inchannels, outchannels, kernel_size=k_size, padding=dilate*(k_size-1)//2
                        , dilation=dilate, stride=stride)

        self.relu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat((input, out), dim=1)        
###########################  single Model #################################
NB_FILTERS = 32

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.e0 = Residual_enc(2, NB_FILTERS * 1)
        self.e1 = Residual_enc(NB_FILTERS * 1, NB_FILTERS * 2)
        self.e2 = Residual_enc(NB_FILTERS * 2, NB_FILTERS * 4)
        self.e3 = Residual_enc(NB_FILTERS * 4, NB_FILTERS * 8)

        self.drdb= DRDB(NB_FILTERS * 8)

        self.d3 = Residual_dec(NB_FILTERS * 8, NB_FILTERS * 4)
        self.d2 = Residual_dec(NB_FILTERS * 4, NB_FILTERS * 2)
        self.d1 = Residual_dec(NB_FILTERS * 2, NB_FILTERS * 1)
        self.d0 = Residual_dec(NB_FILTERS * 1, NB_FILTERS * 1)

        self.dd = nn.Sequential(
            DSC(NB_FILTERS, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_e0 = self.e0(x)
        x_e1 = self.e1(x_e0)
        x_e2 = self.e2(x_e1)
        x_e3 = self.e3(x_e2)

        bottom = self.drdb(x_e3)

        x_d3 = self.d3(bottom)
        x_d2 = self.d2(x_d3 + x_e2)
        x_d1 = self.d1(x_d2 + x_e1)
        x_d0 = self.d0(x_d1 + x_e0)

        out = self.dd(x_d0)

        return out + x


class Refine_G_light(nn.Module):

    def __init__(self, args):
        super(Refine_G_light, self).__init__()
        self.g0 = Generator()
        self.g1 = Generator()
        self.dc = Dataconsistency()
        self.args = args
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_un, k_un, mask):
 
        x_rec0 = self.g0(x_un)
        x_dc0 = self.dc(x_rec0, mask, k_un)
        x_rec1 = self.g1(x_dc0)
        x_dc1 = self.dc(x_rec1, mask, k_un)
        x_dc1 =  torch.clamp(x_dc1, 0, 1)

        return x_rec0, x_dc0, x_rec1, x_dc1

########################### direct Connect Model ##########################

class GeneratorDirectConnect(nn.Module):

    def __init__(self):
        super(GeneratorDirectConnect, self).__init__()

        self.e0_T1 = Residual_enc(2, NB_FILTERS * 1)
        self.e0_T2 = Residual_enc(2, NB_FILTERS * 1)

        self.e1_T1 = Residual_enc(NB_FILTERS * 1, NB_FILTERS * 2)
        self.e1_T2 = Residual_enc(NB_FILTERS * 1 , NB_FILTERS * 2)

        self.e2_T1 = Residual_enc(NB_FILTERS * 2, NB_FILTERS * 4)
        self.e2_T2 = Residual_enc(NB_FILTERS * 2 , NB_FILTERS * 4)

        self.e3_T1 = Residual_enc(NB_FILTERS * 4, NB_FILTERS * 8)
        self.e3_T2 = Residual_enc(NB_FILTERS * 4 , NB_FILTERS * 8)

        self.drdb_T1= DRDB(NB_FILTERS * 8)
        self.drdb_T2= DRDB(NB_FILTERS * 8)

        self.d3_T1 = Residual_dec(NB_FILTERS * 8, NB_FILTERS * 4)
        self.d3_T2 = Residual_dec(NB_FILTERS * 8, NB_FILTERS * 4)

        self.d2_T1 = Residual_dec(NB_FILTERS * 4, NB_FILTERS * 2)
        self.d2_T2 = Residual_dec(NB_FILTERS * 4, NB_FILTERS * 2)

        self.d1_T1 = Residual_dec(NB_FILTERS * 2, NB_FILTERS * 1)
        self.d1_T2 = Residual_dec(NB_FILTERS * 2, NB_FILTERS * 1)

        self.d0_T1 = Residual_dec(NB_FILTERS * 1, NB_FILTERS * 1)
        self.d0_T2 = Residual_dec(NB_FILTERS * 1, NB_FILTERS * 1)

        self.dd_T1 = nn.Sequential(
            DSC(NB_FILTERS, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        self.dd_T2 = nn.Sequential(
            DSC(NB_FILTERS, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, T1, T2):

        T1_e0 = self.e0_T1(T1)
        # T2 = torch.cat([T1,T2], dim = 1)
        T2_e0 = self.e0_T2(T2)

        T1_e1 = self.e1_T1(T1_e0)
        T2_e1 = self.e1_T2(T1_e0+T2_e0)

        T1_e2 = self.e2_T1(T1_e1)
        T2_e2 = self.e2_T2(T1_e1+T2_e1)

        T1_e3 = self.e3_T1(T1_e2)
        T2_e3 = self.e3_T2(T1_e2+T2_e2)

        T1_bottom = self.drdb_T1(T1_e3)
        T2_bottom = self.drdb_T2(T1_e3+T2_e3)

        T1_d3 = self.d3_T1(T1_bottom)
        T2_d3 = self.d3_T2(T1_bottom+T2_bottom)

        T1_d2 = self.d2_T1(T1_d3)
        T2_d2 = self.d2_T2(T1_d3+(T2_d3+T2_e2))

        T1_d1 = self.d1_T1(T1_d2)
        T2_d1 = self.d1_T2(T1_d2+(T2_d2+T2_e1))

        T1_d0 = self.d0_T1(T1_d1)
        T2_d0 = self.d0_T2(T1_d1+(T2_d1+T2_e0))

        T1_out = self.dd_T1(T1_d0)
        T2_out = self.dd_T2(T2_d0)

        return T1_out, T2_out

class Refine_G_DirectConnet(nn.Module):
    def __init__(self, args):
        super(Refine_G_DirectConnet, self).__init__()
        self.g0 = GeneratorDirectConnect()
        self.g1 = GeneratorDirectConnect()
        self.dc = Dataconsistency()
        self.args = args
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_un_T1, k_un_T1, x_un_T2, k_un_T2, mask):
 
        x_rec0_T1, x_rec0_T2 = self.g0(x_un_T1, x_un_T2)
        x_dc0_T1 = self.dc(x_rec0_T1, mask, k_un_T1)
        x_dc0_T2 = self.dc(x_rec0_T2, mask, k_un_T2)

        x_rec1_T1, x_rec1_T2 = self.g1(x_dc0_T1, x_dc0_T2)
        x_dc1_T1 = self.dc(x_rec1_T1, mask, k_un_T1)
        x_dc1_T2 = self.dc(x_rec1_T2, mask, k_un_T2)

        x_dc1_T1 =  torch.clamp(x_dc1_T1, 0, 1)
        x_dc1_T2 =  torch.clamp(x_dc1_T2, 0, 1)

        return x_rec0_T1, x_rec0_T2, x_dc0_T1, x_dc0_T2, x_rec1_T1, x_rec1_T2, x_dc1_T1, x_dc1_T2

########################### discriminator #################################

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.e0 = Residual_enc(2, NB_FILTERS * 1)
        self.e1 = Residual_enc(NB_FILTERS * 1, NB_FILTERS * 2)
        self.e2 = Residual_enc(NB_FILTERS * 2, NB_FILTERS * 4)
        self.e3 = Residual_enc(NB_FILTERS * 4, NB_FILTERS * 8)

        self.ret = DSC(NB_FILTERS * 8, 1, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_e0 = self.e0(x)
        x_e1 = self.e1(x_e0)
        x_e2 = self.e2(x_e1)
        x_e3 = self.e3(x_e2)

        ret = self.ret(x_e3)

        return ret
import yaml
from types import SimpleNamespace

def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args
from torchstat import stat
if __name__ == '__main__':
    args = get_args()
    # netG = Generator()
    # netD = Discriminator()
    # # stat(netG, ((2,256,256),(2,256,256),(2,256,256)))
    # # stat(netD, (2,256,256))

    # print('    Total params: %.4fMB' % (sum(p.numel() for p in netG.parameters()) / 1000000))
    # print('    Total params: %.4fMB' % (sum(p.numel() for p in netD.parameters()) / 1000000))


    # stat(netD, (2,256,256))

    imput1 = torch.rand((1,2,256,256))    
    imput2 = torch.rand((1,2,256,256))
    model = Refine_G_DirectConnet(args)

    x_rec0_T1, x_rec0_T2, x_dc0_T1, x_dc0_T2, x_rec1_T1, x_rec1_T2, x_dc1_T1, x_dc1_T2= model(imput1,imput1,imput1,imput1,imput1)