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
        out1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)

        ##second conv
        out2 = F.leaky_relu(self.bn2(self.conv2(out1)), 0.1)

        ##third conv

        out3 = F.leaky_relu(self.bn3(self.conv3(out2)), 0.1)
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
        out1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        ##second conv
        out2 = F.leaky_relu(self.bn2(self.conv2(out1)), 0.1)
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
## Unet architectures
class BasicUnet(nn.Module):
    def __init__(self, in_chan, out_chan, filters):
        super(BasicUnet, self).__init__()
        self.conv1T2 = nn.Conv2d(in_chan*2,int(filters/2),1)

        self.dT2_1 = BasicBlockDown(int(filters/2), filters)
        self.downT2_1 = DownSample()
        self.dT2_2 = BasicBlockDown(filters, filters*2)
        self.downT2_2 = DownSample()
        self.dT2_3 = BasicBlockDown(filters*2, filters*4)
        self.downT2_3 = DownSample()
        self.dT2_4 = BasicBlockDown(filters*4, filters*8)
        self.downT2_4 = DownSample()

        self.bottomT2 = DRDB(filters*8)

        self.up4_T2 = UpSample_T(filters*8)
        self.u4_T2 = BasicBlockUp(filters*4)
        self.up3_T2 = UpSample_T(filters*4)
        self.u3_T2 = BasicBlockUp(filters*2)
        self.up2_T2 = UpSample_T(filters*2)
        self.u2_T2 = BasicBlockUp(filters)
        self.up1_T2 = UpSample_T(filters)
        self.u1_T2 = BasicBlockUp(int(filters/2))

        self.outconvT2 = nn.Conv2d(int(filters/2),out_chan, 1)

    def forward(self, T1, T2):
        T2 = torch.cat([T1,T2], dim = 1)

        T2_x1 = self.conv1T2(T2)
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        T2_x3,T2_y2 = self.downT2_2(self.dT2_2(T2_x2))
        T2_x4,T2_y3 = self.downT2_3(self.dT2_3(T2_x3))
        T2_x5,T2_y4 = self.downT2_4(self.dT2_4(T2_x4))

        T2_x = self.bottomT2(T2_x5)

        T2_1x = self.u4_T2(self.up4_T2(T2_x,T2_y4))
        T2_2x = self.u3_T2(self.up3_T2(T2_1x,T2_y3))
        T2_3x = self.u2_T2(self.up2_T2(T2_2x,T2_y2))
        T2_4x = self.u1_T2(self.up1_T2(T2_3x,T2_y1))

        T2 = self.outconvT2(T2_4x)

        return T2
    

class UnetK(nn.Module):
    def __init__(self, in_chan,out_chan, filters):
        super(UnetK, self).__init__()

        self.conv1T1 = nn.Conv2d(in_chan,int(filters/2),1)
        # self.conv1T2 = nn.Conv2d(in_chan*2,int(filters/2),1)
        self.conv1T2 = nn.Conv2d(in_chan,int(filters/2),1)


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


        self.bottomT1 = DRDB(filters*8)
        self.bottomT2 = DRDB(filters*8)

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


    def forward(self, T1, T2):
        ## input feature extract
        T1_x1 = self.conv1T1(T1)
        # T2 = torch.cat([T1,T2], dim = 1)
        T2_x1 = self.conv1T2(T2)

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
        T1 = self.outconvT1(T1_4x)
        T2 = self.outconvT2(T12_x)

        return T1,T2

class UnetI(nn.Module):
    def __init__(self, in_chan,out_chan, filters):
        super(UnetI, self).__init__()

        self.conv1T1 = nn.Conv2d(in_chan,int(filters/2),1)
        # self.conv1T2 = nn.Conv2d(in_chan*2,int(filters/2),1)
        self.conv1T2 = nn.Conv2d(in_chan,int(filters/2),1)


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


        self.bottomT1 = DRDB(filters*8)
        self.bottomT2 = DRDB(filters*8)

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


    def forward(self, T1, T2):
        ## input feature extract
        T1_x1 = self.conv1T1(T1)
        # T2 = torch.cat([T1,T2], dim = 1)
        T2_x1 = self.conv1T2(T2)

        ##encoder part
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

        #bootleneck part
        T1_x = self.bottomT1(T1_x5)
        T2_x = self.bottomT2(T12_x5)
        #decoder part
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
        #output part
        T1 = self.outconvT1(T1_4x)
        T2 = self.outconvT2(T12_x)
        
        return T1,T2


############################## Generator ##########################

class ImgGenerator(nn.Module):
    def __init__(self, args):
        super(ImgGenerator,self).__init__()
        self.filters = args.filters
        self.device = args.device

        ## init the mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1'] == 1, device=args.device)
        self.maskNot = self.mask == 0

        self.IUnet_1 = UnetI(1,1,self.filters)
        self.IUnet_2 = UnetI(1,1,self.filters)

    def fftshift(self, img):
        '''
            4d tensor FFT operation
        '''
        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2
    
    def FT(self, image):
        '''
            Fourier operation 
        '''

        kspace_cplx = self.fftshift(torch.fft.fft2(image, dim=(2,3)))
        return kspace_cplx

    def inverseFT(self, Kspace):
        """The input Kspace has two channels(real and img)"""
        Kspace = Kspace.permute(0, 2, 3, 1)
        img_cmplx = torch.fft.ifft2(Kspace, dim=(1,2))
        img = torch.absolute(img_cmplx[:,:,:,0]+ 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img

    
    def IDC_layer(self, rec_Img, und_K):
        und_k_cplx = torch.complex(und_K[:,0,:,:], und_K[:,1,:,:])[:,None,:,:] 
        rec_k = self.FT(rec_Img)
        final_k = (torch.mul(self.mask, und_k_cplx) + torch.mul(self.maskNot, rec_k))
        final_rec  =  torch.absolute(torch.fft.ifft2(self.fftshift(final_k), dim=(2,3)))

        return final_rec


    def forward(self, T1masked, T2masked, T1k, T2k):
        ## first stage kspace restoration
        recon_img_T1, recon_img_T2 = self.IUnet_1(T1masked, T2masked)
        ## DC operation

        rec_mid_T1 = self.IDC_layer(recon_img_T1, T1k)
        rec_mid_T2 = self.IDC_layer(recon_img_T2, T2k)

        ## second stage image dealiasing
        """ not use tye refine technique"""
        rec_T1, rec_T2 = self.IUnet_2(rec_mid_T1,rec_mid_T2)
        ##IDC
        rec_T1 = self.IDC_layer(rec_T1, T1k)
        rec_T2 = self.IDC_layer(rec_T2, T2k)

        # rec_T1 = torch.clamp(rec_T1, 0,1)
        # rec_T2 = torch.clamp(rec_T2, 0,1)

        return rec_T1, rec_mid_T1, rec_T2, rec_mid_T2

class BasicGenerator(nn.Module):
    def __init__(self, args):
        super(BasicGenerator,self).__init__()
        self.filters = args.filters
        self.device = args.device

        ## init the mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1'] == 1, device=args.device)
        self.maskNot = self.mask == 0

        self.KUnet = BasicUnet(2,2,self.filters)
        self.IUnet = BasicUnet(1,1,self.filters)

    def fftshift(self, img):
        '''
            4d tensor FFT operation
        '''
        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2

    def FT(self, image):
        '''
            Fourier operation 
        '''
        
        kspace_cplx = self.fftshift(torch.fft.fft2(image, dim=(2,3)))
        return kspace_cplx

    def inverseFT(self, Kspace):
        """The input Kspace has two channels(real and img)"""
        Kspace = Kspace.permute(0, 2, 3, 1)
        img_cmplx = torch.fft.ifft2(Kspace, dim=(1,2))
        img = torch.absolute(img_cmplx[:,:,:,0]+ 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img

    def KDC_layer(self, rec_K, und_K):
        '''
            K data consistency layer
        '''
        rec_Kspace = (self.mask*torch.complex(und_K[:, 0, :, :], und_K[:, 1, :, :]) + self.maskNot*torch.complex(rec_K[:, 0, :, :], rec_K[:, 1, :, :]))[:, None, :, :]
        final_rec =torch.absolute(torch.fft.ifft2((rec_Kspace),dim=(2,3)))

        return final_rec, rec_Kspace
    
    def IDC_layer(self, rec_Img, und_K):
        und_k_cplx = torch.complex(und_K[:,0,:,:], und_K[:,1,:,:])[:,None,:,:] 
        rec_k = self.FT(rec_Img)
        final_k = (torch.mul(self.mask, und_k_cplx) + torch.mul(self.maskNot, rec_k))
        final_rec  =  torch.absolute(torch.fft.ifft2((final_k), dim=(2,3)))

        return final_rec


    def forward(self, T1kspace, T2kspace, T1tar):
        ## first stage kspace restoration
        recon_K_T2 = self.KUnet(T1kspace, T2kspace)
        ## DC operation
        rec_mid_T2 , rec_Kspace_T2= self.KDC_layer(recon_K_T2, T2kspace)

        ## second stage image dealiasing
        """ not use tye refine technique"""
        rec_T2 = self.IUnet(T1tar,rec_mid_T2)
        ##IDC
        rec_T2 = self.IDC_layer(rec_T2, T2kspace)

        return rec_T2, recon_K_T2, rec_T2, recon_K_T2

################################ discriminator ####################################
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

class Discriminator(nn.Module):
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
        super(Discriminator, self).__init__()
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

def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args


if __name__ == "__main__":
    # args = get_args()
    # t = torch.rand((2,2,256,256)).to("cuda", dtype=torch.float32)
    # netG = Generator(args).to("cuda")
    # _,_,_,_ = netG(t,t)
    # netD = Discriminator(1, 64, True, 4)
    # print('    Total params of G: %.4fM' % (sum(p.numel() for p in netG.parameters()) / 1000000)) 
    # print('    Total params of D: %.4fM' % (sum(p.numel() for p in netD.parameters()) / 1000000))
    # print('    Total params : %.4fM' %((sum(p.numel() for p in netG.parameters()) / 1000000)+(sum(p.numel() for p in netD.parameters()) / 1000000)))
    pass