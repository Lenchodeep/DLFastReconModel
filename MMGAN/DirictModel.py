import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

from CommonPart import *

class DirectUnet(nn.Module):
    def __init__(self, in_chan,out_chan, filters):
        super(DirectUnet,self).__init__()
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

    def forward(self, T1, T2):
        ## input feature extract
        T1_x1 = self.conv1T1(T1)
        T2 = torch.cat([T1,T2], dim = 1)
        T2_x1 = self.conv1T2(T2)

        #encoder part
        T1_x2,T1_y1 = self.downT1_1(self.dT1_1(T1_x1))
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        T12_x2 = T2_x2+T1_x2

        T1_x3,T1_y2 = self.downT1_2(self.dT1_2(T1_x2))
        T2_x3,T2_y2 = self.downT2_2(self.dT2_2(T12_x2))
        T12_x3 = T1_x3+ T2_x3

        T1_x4,T1_y3 = self.downT1_3(self.dT1_3(T1_x3))
        T2_x4,T2_y3 = self.downT2_3(self.dT2_3(T12_x3))
        T12_x4 = T1_x4 + T2_x4    
           
        T1_x5,T1_y4 = self.downT1_4(self.dT1_4(T1_x4))
        T2_x5,T2_y4 = self.downT2_4(self.dT2_4(T12_x4))
        T12_x5 = T1_x5+T2_x5   

        ##bootleneck part
        T1_x = self.bottomT1(T1_x5)
        T2_x = self.bottomT2(T12_x5)

        ##decoder part
        T1_1x = self.u4_T1(self.up4_T1(T1_x))
        T2_1x = self.u4_T2(self.up4_T2(T2_x,T2_y4))
        T12_x = T1_1x+T2_1x   

        T1_2x = self.u3_T1(self.up3_T1(T1_1x))
        T2_2x = self.u3_T2(self.up3_T2(T12_x,T2_y3))
        T12_x = T1_2x+T2_2x   

        T1_3x = self.u2_T1(self.up2_T1(T1_2x))
        T2_3x = self.u2_T2(self.up2_T2(T12_x,T2_y2))
        T12_x = T1_3x+T2_3x    

        T1_4x = self.u1_T1(self.up1_T1(T1_3x))
        T2_4x = self.u1_T2(self.up1_T2(T12_x,T2_y1))
        T12_x = T1_4x+T2_4x

        ##output part
        T1 = self.outconvT1(T1_4x)
        T2 = self.outconvT2(T12_x)

        return T1,T2

class DirectG(nn.Module):
    def __init__(self, args):
        super(DirectG,self).__init__()
        self.filters = args.filters
        self.device = args.device

         ## init the mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1'] == 1, device=args.device)
        self.maskNot = self.mask == 0
        self.KUnet = DirectUnet(2,2,self.filters)
        self.IUnet = DirectUnet(1,1,self.filters)
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


        return rec_T1, recon_K_T1, rec_T2, recon_K_T2
