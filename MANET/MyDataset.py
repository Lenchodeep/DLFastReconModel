'''
Mydataset of multi-modal for train and eval
It contains FastMRI dataset and BraTS dataset
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import h5py
import yaml
import pickle
import logging

from os import listdir, path
from types import SimpleNamespace
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########################## FastMRI dataset #############################
class FastMRIDataset(Dataset):
    def __init__(self, args, istrain = True, transform = None):
        self.args = args
        self.istrain = istrain
        self.imageSize = args.imgSize
        self.num_input_slices = args.num_input_slices  

        ## The noise level added to Kspace data
        # self.fillNoise = args.fillnoise 
        # self.minmax_noise_val = args.minmax_noise_val
        
        ## generate the data dir
        if self.istrain:
            self.dataDir = path.join(args.imageDataDir, "train/")
        else:
            self.dataDir = path.join(args.imageDataDir, "val/")

        ## data augmentation
        self.transform = transform

        ## generate the pdwi and corresponding fspdwi file
        if self.istrain:
            self.data = pd.read_csv("./data/MiniFastMRI/singlecoil_train_split_less.csv")
        else:
            self.data = pd.read_csv("./data/MiniFastMRI/singlecoil_val_split_less.csv")

        PDWIList =  self.data['PDWI']
        FSPDWIList =  self.data['FSPDWI']
        ## counting valid file num
        self.ids = list()

        for idx in range(len(PDWIList)):
            try:
                full_file_path = path.join(self.dataDir, PDWIList[idx]+'.hdf5')
                with h5py.File(full_file_path, 'r') as f:
                    num_slice = f['reconstruction_rss'].shape[0]

                if num_slice < self.args.slice_range[1]:
                    continue

                for slice in range(self.args.slice_range[0], self.args.slice_range[1]):
                    self.ids.append((PDWIList[idx], FSPDWIList[idx], slice))
            except:
                continue


        if self.istrain:
            print(f'Creating training datasets using FastMRI with {len(self.ids)} examples')
        else:
            print(f'Creating val dataset using FastMRI with {len(self.ids)} examples')
       
        ## generate the sampling mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
        self.masks = np.dstack((masks_dictionary['mask0'],masks_dictionary['mask1'], masks_dictionary['mask2']))
        self.maskNot = 1-masks_dictionary['mask1']

    def __len__(self):
        return len(self.ids)

    def fft2(self, image):
        '''
            Fourier operation
        '''
        return np.fft.fftshift(np.fft.fft2(image))

    def ifft2(self, kspace_cplx):
        '''
            Inverse Fourier operation
        '''
        ''''''
        return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace_cplx)))[None, :, :]


    def crop_toshape(self, image):
        '''
            crop to target image size
        '''
        if image.shape[0] == self.imageSize:
            return image
        if image.shape[0] % 2 == 1:
            image = image[:-1, :-1]
        crop = int((image.shape[0] - self.imageSize)/2)
        image = image[crop:-crop, crop:-crop]
        return image

    def regular(self,image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    def slice_preprocess(self,kspace_cplx, slice_num):
        """
            kspace_cplx: an one channel complex matrix
            slice_num: the current slice number
        """
        #splits to real and imag channels
        kspace = np.zeros((self.imageSize, self.imageSize, 2))
        kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)

        ## this is the target image, need use the ifft result instead of the original one because after the fft and the ifft the image has changed
        image = self.ifft2(kspace_cplx)                             ## conduct the fftshift operation
        
        kspace = kspace.transpose((2, 0, 1))

        masked_Kspace = kspace*self.masks[:, :, slice_num]

        masked_img = self.ifft2((masked_Kspace[0,:,:]+1j*masked_Kspace[1,:,:]))
        # if self.fillNoise:
        #     masked_Kspace += np.random.uniform(low=self.minmax_noise_val[0], high=self.minmax_noise_val[1],
        #                                     size=masked_Kspace.shape)*self.maskNot
        return masked_Kspace, kspace, image, masked_img

    def __getitem__(self, i):
        PDWIFileName, FSPDWIFileName, slice_num = self.ids[i]
        PDWIFilePath =  path.join(self.dataDir,PDWIFileName + '.hdf5')
        FSPDWIFilePath = path.join(self.dataDir,FSPDWIFileName + '.hdf5')

        with h5py.File(PDWIFilePath, 'r') as f, h5py.File(FSPDWIFilePath, 'r') as s:
            add = int(self.num_input_slices / 2)  ## add controls the slices range
            PDWIimgs = f['reconstruction_rss'][ slice_num-add:slice_num+add+1, :, :]
            FSPDWIimgs = s['reconstruction_rss'][ slice_num-add:slice_num+add+1, :, :]

            masked_Kspaces = np.zeros((self.num_input_slices*2, self.imageSize, self.imageSize))
            reference_masked_Kspace = np.zeros((self.num_input_slices*2, self.imageSize, self.imageSize))
            reference_Kspace = np.zeros((2, self.imageSize, self.imageSize))
            target_Kspace = np.zeros((2, self.imageSize, self.imageSize))
            target_img = np.zeros((1, self.imageSize, self.imageSize))
            reference_img = np.zeros((1, self.imageSize, self.imageSize))
            target_masked_img = np.zeros((1, self.imageSize, self.imageSize))
            reference_masked_img = np.zeros((1, self.imageSize, self.imageSize))
            for slice in range(self.num_input_slices):
                PDWIimg = self.regular(self.crop_toshape(PDWIimgs[slice,:,:]))
                FSPDWIimg = self.regular(self.crop_toshape(FSPDWIimgs[slice,:,:]))

                if self.transform:
                    PDWIimg = self.transform(PDWIimg)   
                    FSPDWIimg = self.transform(FSPDWIimg)   

                PDWIkspace = self.fft2(PDWIimg)
                FSPDWIkspace = self.fft2(FSPDWIimg)

                masked_FSPDWIkspace, full_FSPDWIkspace, full_FSPDWIimg, masked_FSPDWIimg = self.slice_preprocess(FSPDWIkspace, slice)
                masked_PDWIkspace, full_PDWIkspace, full_PDWIimg, masked_PDWIimg = self.slice_preprocess(PDWIkspace, slice)

                # print(masked_FSPDWIimg.shape, masked_FSPDWIimg.max(), masked_FSPDWIimg.min())
                # plt.figure()
                # plt.imshow(np.squeeze(masked_FSPDWIimg),plt.cm.gray)
                # plt.show()

                masked_Kspaces[slice*2:slice*2+2, :, :] = masked_FSPDWIkspace
                reference_masked_Kspace[slice*2:slice*2+2, :, :] = masked_PDWIkspace

                if slice == int(self.num_input_slices/2):
                    target_Kspace = full_FSPDWIkspace
                    target_img = full_FSPDWIimg
                    target_masked_img = masked_FSPDWIimg
                    reference_img = full_PDWIimg
                    reference_Kspace = full_PDWIkspace
                    reference_masked_img = masked_PDWIimg

            return {
                "masked_K":torch.from_numpy(masked_Kspaces),
                "refer_masked_K":torch.from_numpy(reference_masked_Kspace),
                "target_K":torch.from_numpy(target_Kspace),
                "target_img": torch.from_numpy(target_img),
                "target_masked_img": torch.from_numpy(target_masked_img),
                "refer_K":torch.from_numpy(reference_Kspace),
                "refer_img": torch.from_numpy(reference_img),
                "refer_masked_img": torch.from_numpy(reference_masked_img)
            }


############################## BraTS dataset ######################################
class BraTSDataset(Dataset):
    def __init__(self, args, istrain = True, transform = None):
        self.args = args
        self.istrain = istrain
        self.imageSize = args.imgSize
        self.num_input_slices = args.num_input_slices  

        ## generate the data dir
        if self.istrain:
            self.dataDir = path.join(args.imageDataDir, "train/")
        else:
            self.dataDir = path.join(args.imageDataDir, "val/")

        ## data augmentation
        self.transform = transform

        ## generate the pdwi and corresponding fspdwi file
        if self.istrain:
            self.data = pd.read_csv("./data/BraTS/T1andT2Train.csv")
        else:
            self.data = pd.read_csv("./data/BraTS/T1andT2Val.csv")

        T1ceList = self.data['T1CE']
        T2List = self.data['T2']
        ## counting valid file num
        self.ids = list()

        for idx in range(len(T1ceList)):
            try:
                full_file_path = path.join(self.dataDir, T1ceList[idx])
                with h5py.File(full_file_path, 'r') as f:
                    num_slice = f['data'].shape[0]

                if num_slice < self.args.slice_range[1]:
                    continue

                for slice in range(self.args.slice_range[0], self.args.slice_range[1]):
                    self.ids.append((T1ceList[idx], T2List[idx], slice))
            except:
                continue


        if self.istrain:
            print(f'Creating training datasets using BraTS with {len(self.ids)} examples')
        else:
            print(f'Creating val dataset using BraTS with {len(self.ids)} examples')

        ## generate the sampling mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
        self.masks = np.dstack((masks_dictionary['mask0'],masks_dictionary['mask1'], masks_dictionary['mask2']))
        self.maskNot = 1-masks_dictionary['mask1']

    def __len__(self):

        return len(self.ids)

    def fft2(self, image):
        '''
            Fourier operation
        '''
        return np.fft.fftshift(np.fft.fft2(image))

    def ifft2(self, kspace_cplx):
        '''
            Inverse Fourier operation
        '''
        return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace_cplx)))[None, :, :]

    def expand_toshape(self, image):
        '''
            crop to target image size
        '''
        if image.shape[0] == self.imageSize:
            return image
        expand = int((self.imageSize-image.shape[0])/2)
        retimg = np.zeros((self.imageSize,self.imageSize))

        retimg[expand:-expand, expand:-expand] = image
        return retimg

    def slice_preprocess(self,kspace_cplx, slice_num):
        """
            kspace_cplx: an one channel complex matrix
            slice_num: the current slice number
        """
        #splits to real and imag channels
        kspace = np.zeros((self.imageSize, self.imageSize, 2))
        kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)

        ## this is the target image, need use the ifft result instead of the original one because after the fft and the ifft the image has changed
        image = self.ifft2(kspace_cplx)                             ## conduct the fftshift operation
        kspace = kspace.transpose((2, 0, 1))
        masked_Kspace = kspace*self.masks[:, :, slice_num]
        return masked_Kspace, kspace, image
    
    def __getitem__(self, i):
        T1CEFileName, T2FileName, slice_num = self.ids[i]
        T1CEFilePath =  path.join(self.dataDir,T1CEFileName)
        T2FilePath = path.join(self.dataDir,T2FileName)

        with h5py.File(T1CEFilePath, 'r') as f, h5py.File(T2FilePath, 'r') as s:
            add = int(self.num_input_slices / 2)  ## add controls the slices range
            T1CEimgs = f['data'][ :, :, slice_num-add:slice_num+add+1]
            T2imgs = s['data'][ :, :, slice_num-add:slice_num+add+1]

            reference_masked_Kspace = np.zeros((self.num_input_slices*2, self.imageSize, self.imageSize))
            masked_Kspaces = np.zeros((self.num_input_slices*2, self.imageSize, self.imageSize))
            reference_Kspace = np.zeros((2, self.imageSize, self.imageSize))
            target_Kspace = np.zeros((2, self.imageSize, self.imageSize))
            target_img = np.zeros((1, self.imageSize, self.imageSize))
            reference_img = np.zeros((1, self.imageSize, self.imageSize))
            
            for slice in range(self.num_input_slices):
                T1img = self.expand_toshape(T1CEimgs[:,:,slice])
                T1img = np.rot90(T1img,-1)
                T2img = self.expand_toshape(T2imgs[:,:,slice])
                T2img = np.rot90(T2img,-1)

                T1kspace = self.fft2(T1img)
                T2kspace = self.fft2(T2img)

                masked_T1kspace, full_T1kspace, full_T1img = self.slice_preprocess(T1kspace, slice)
                masked_T2kspace, full_T2kspace, full_T2img = self.slice_preprocess(T2kspace, slice)

                masked_Kspaces[slice*2:slice*2+2, :, :] = masked_T2kspace
                reference_masked_Kspace[slice*2:slice*2+2, :, :] = masked_T1kspace

                if slice == int(self.num_input_slices/2):
                    target_Kspace = full_T2kspace
                    reference_Kspace = full_T1kspace
                    target_img = full_T2img
                    reference_img = full_T1img

            return {
                "masked_K":torch.from_numpy(masked_Kspaces),
                "refer_masked_K":torch.from_numpy(reference_masked_Kspace),
                "target_K":torch.from_numpy(target_Kspace),
                "target_img": torch.from_numpy(target_img),
                "refer_K":torch.from_numpy(reference_Kspace),
                "refer_img": torch.from_numpy(reference_img)
            }


def main():
    with open('./config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)

    dataset = FastMRIDataset(args=args, istrain = False,transform = None)
    dataloader = DataLoader(dataset , batch_size=1 , shuffle=True , num_workers=0 , drop_last=True)

    data_iter = iter(dataloader)
    data = data_iter.next()

    tar = data["target_img"]
    maskK = data["masked_K"]
    refer_k = data["refer_K"]
    refer_img = data["refer_img"]

    tar = tar.detach().numpy()
    maskK = maskK.detach().numpy()
    refer_k = refer_k.detach().numpy()
    refer_img = refer_img.detach().numpy()

    mask_kspace1 = maskK[0,0,:,:]+1j*maskK[0,1,:,:]
    # mask_kspace2 = maskK[0,2,:,:]+1j*maskK[0,3,:,:]
    # mask_kspace3 = maskK[0,4,:,:]+1j*maskK[0,5,:,:]

    mask_kimg1 =np.log(np.abs(mask_kspace1))
    # mask_kimg2 =np.log(np.abs(mask_kspace2))
    # mask_kimg3 = np.log(np.abs(mask_kspace3))

    refer_kspace1 = refer_k[0,0,:,:]+1j*refer_k[0,1,:,:]
    # refer_kspace2 = refer_k[0,2,:,:]+1j*refer_k[0,3,:,:]
    # refer_kspace3 = refer_k[0,4,:,:]+1j*refer_k[0,5,:,:]

    refer_kimg1 =np.log(np.abs(refer_kspace1))
    # refer_kimg2 =np.log(np.abs(refer_kspace2))
    # refer_kimg3 = np.log(np.abs(refer_kspace3))

    zfimg = np.abs(np.fft.ifft2((mask_kspace1)))
    tar_img = np.squeeze(tar)
    print(tar_img.max(), tar_img.min())

    refer_img = np.squeeze(refer_img)
    test = abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(refer_img))))
    plt.figure()
    plt.imshow(mask_kimg1,plt.cm.gray)
    plt.title("mask_kimg2")
    plt.figure()
    plt.imshow(refer_kimg1,plt.cm.gray)
    plt.title("refer_kimg2")    
    plt.figure()
    plt.imshow(zfimg,plt.cm.gray)
    plt.title("zfimg")
    plt.figure()
    plt.imshow(tar_img,plt.cm.gray)
    plt.title("tar_img")
    plt.figure()
    plt.imshow(refer_img,plt.cm.gray)
    plt.title("refer_img") 
    plt.show()

if __name__ == "__main__":
    # main()
    pass


