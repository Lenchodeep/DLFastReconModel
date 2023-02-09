import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

from CommonPart import *

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
        self.conv1T2 = nn.Conv2d(in_chan*2,int(filters/2),1)
        # self.conv1T2 = nn.Conv2d(in_chan,int(filters/2),1)


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
        T2 = torch.cat([T1,T2], dim = 1)
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
        self.conv1T2 = nn.Conv2d(in_chan*2,int(filters/2),1)
        # self.conv1T2 = nn.Conv2d(in_chan,int(filters/2),1)


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
        T2 = torch.cat([T1,T2], dim = 1)
        T2_x1 = self.conv1T2(T2)

        ##encoder part
        T1_x2,T1_y1 = self.downT1_1(self.dT1_1(T1_x1))
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        # print("x max", torch.max(T1_x2), "x min", torch.min(T1_x2))

        temp = T1_x2.mul(self.msca1(T1_x2))
        # print("map max", torch.max(temp), "map min", torch.min(temp))

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

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator,self).__init__()
        self.filters = args.filters
        self.device = args.device

        ## init the mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1'] == 1, device=args.device)
        self.maskNot = self.mask == 0

        self.KUnet = UnetK(2,2,self.filters)
        self.IUnet = UnetI(1,1,self.filters)

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
        final_rec =torch.absolute(torch.fft.ifft2(self.fftshift(rec_Kspace),dim=(2,3)))

        return final_rec, rec_Kspace
    
    def IDC_layer(self, rec_Img, und_K):
        und_k_cplx = torch.complex(und_K[:,0,:,:], und_K[:,1,:,:])[:,None,:,:] 
        rec_k = self.FT(rec_Img)
        final_k = (torch.mul(self.mask, und_k_cplx) + torch.mul(self.maskNot, rec_k))
        final_rec  =  torch.absolute(torch.fft.ifft2(self.fftshift(final_k), dim=(2,3)))

        return final_rec


    def forward(self, T1kspace, T2kspace):
        ## first stage kspace restoration
        recon_K_T1, recon_K_T2 = self.KUnet(T1kspace, T2kspace)
        ## DC operation
        rec_mid_T1 , rec_Kspace_T1= self.KDC_layer(recon_K_T1, T1kspace)
        rec_mid_T2 , rec_Kspace_T2= self.KDC_layer(recon_K_T2, T2kspace)

        ## second stage image dealiasing
        """ not use tye refine technique"""
        rec_T1, rec_T2 = self.IUnet(rec_mid_T1,rec_mid_T2)
        ##IDC
        rec_T1 = self.IDC_layer(rec_T1, T1kspace)
        rec_T2 = self.IDC_layer(rec_T2, T2kspace)
        rec_T2 =  torch.tanh(rec_T2)
        rec_T1 =  torch.tanh(rec_T1)

        return rec_T1, recon_K_T1, rec_T2, recon_K_T2

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