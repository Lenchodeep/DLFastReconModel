'''
    This file is a file for the dataset of IXI open source dataset
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from os.path import splitext
from os import listdir, path
import torch
import logging
import h5py
import yaml
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle

class IXIDatasetImage(Dataset):
    def __init__(self, data_dir, args, is_train = True, transform = None):
        self.args = args
        self.is_train = is_train
        if self.is_train:
            self.data_dir = path.join(data_dir, 'train/')
        else:
            self.data_dir = path.join(data_dir, 'test/')
        self.transform = transform
        self.file_names = [splitext(file)[0] for file in listdir(self.data_dir)]    ##get all the h5py file in the folder
        self.imageSize = args.imgSize
        self.ids = list()      ## to save the slices and the file name


        for file_name in self.file_names:
            try:
                fullPath = path.join(self.data_dir, file_name+'.hdf5')
                with h5py.File(fullPath,'r') as f:
                    num_slice = f['data'].shape[2]
                if num_slice < self.args.slice_range[1]:
                    continue
                for slice in range(self.args.slice_range[0], self.args.slice_range[1]):
                    self.ids.append((file_name, slice))
            except:
                continue

        if self.is_train:
            logging.info(f'Creating training datasets with {len(self.ids)} examples')
        else:
            logging.info(f'Creating test dataset with {len(self.ids)} examples')  

        self.mask = np.load(args.mask_path)
        self.maskNot = 1-self.mask

        self.minmax_noise_val = args.minmax_noise_val

    def __len__(self):
        return len(self.ids)

    def fft2(self, image):
        '''
            get K space from image  domain, and the result is after the fftshift operation
        '''
        return np.fft.fftshift(np.fft.fft2(image))
    
    def ifft2(self, kspace):
        '''
            get image from the K space, first need fftshift operation
        '''
        return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace)))

    def crop_to_shape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.imageSize:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.imageSize)/2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def preProcess(self, complexK):
        complexK = self.crop_to_shape(complexK)
        label = self.ifft2(complexK)

        masked_kspace = complexK * self.mask
        masked_kspace += np.random.uniform(low=self.minmax_noise_val[0], high=self.minmax_noise_val[1],    ## add noise to the image
                                           size=masked_kspace.shape)*self.maskNot                      
        fold_img = self.ifft2(masked_kspace) 

        return fold_img, label


    def __getitem__(self, i):
        file_name, slice_num = self.ids[i]
        full_file_path = path.join(self.data_dir,file_name + '.hdf5')

        with h5py.File(full_file_path, 'r') as f:
            image = f['data'][:,:,slice_num]        ##get the 2d image by the slice number
        
        # target_image = np.zeros((1, self.imageSize, self.imageSize))
        # fold_image = np.zeros((1,self.imageSize, self.imageSize))

        image = image.astype(np.float32)
        image = np.rot90(image, 1)  ## need to rotate 90 degree

        if self.transform:
            image = self.transform(image)       ## the input of the transform is a 2d slice
        
        kspace = self.fft2(image)    ## the kspace is after the fftshift

        fold_img_tmp, label_img_tmp = self.preProcess(kspace)

        fold_image = np.expand_dims(fold_img_tmp, 0)
        label_img = np.expand_dims(label_img_tmp, 0)
        return {
            "fold_img": torch.from_numpy(fold_image),
            "target_img": torch.from_numpy(label_img)
        }

class IXIDataSetComplex(Dataset):
    def __init__(self, data_dir, args, is_train = True, transform = None):
        self.args = args
        
        self.is_train = is_train
        if self.is_train:
            self.data_dir = path.join(data_dir, "train/")
        else:
            self.data_dir = path.join(data_dir, "test/")
        self.transform = transform
        self.num_input_slice = args.num_input_slices
        self.file_names = [splitext(file)[0] for file in listdir(self.data_dir)]
        self.imageSize = args.imgSize
        self.ids = list()
        self.fillNoise = args.fillnoise
        for file_name in self.file_names:
            try:
                full_file_path = path.join(self.data_dir, file_name+'.hdf5')
                with h5py.File(full_file_path, 'r') as f:
                    num_slice = f['data'].shape[2]
                if num_slice < self.args.slice_range[1]:
                    continue
                for slice in range(self.args.slice_range[0], self.args.slice_range[1]):
                    self.ids.append((file_name, slice))
            except:
                continue

        if self.is_train:
            logging.info(f'Creating training datasets with {len(self.ids)} examples')
        else:
            logging.info(f'Creating test dataset with {len(self.ids)} examples')
       
        self.mask = np.load(args.mask_path)
        self.maskNot = 1-self.mask

        #random noise:
        self.minmax_noise_val = args.minmax_noise_val
    def __len__(self):
        return len(self.ids)

    def fft2(self, image):
        return np.fft.fftshift(np.fft.fft2(image))
    
    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace_cplx)))[None, :, :]
   
    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.imageSize:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.imageSize)/2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def preprocess(self, complexK):
        complexK = self.crop_toshape(complexK)

        kspace = np.zeros((self.imageSize, self.imageSize, 2))
        kspace[:,:,0] = np.real(complexK).astype(np.float32)
        kspace[:,:,1] = np.imag(complexK).astype(np.float32)
        image = self.ifft2(complexK)   #正常图像

        kspace = kspace.transpose((2,0,1))
        masked_kspace = kspace* self.mask   #kspace和masked_kspace 均为shift后
        if self.fillNoise:
            masked_kspace += np.random.uniform(low=self.minmax_noise_val[0], high=self.minmax_noise_val[1],
                                           size=masked_kspace.shape)*self.maskNot

        return masked_kspace, kspace, image
   
    def __getitem__(self, i):

        file_name , slice_num = self.ids[i]
        full_file_path = path.join(self.data_dir,file_name + '.hdf5')

        with h5py.File(full_file_path, 'r') as f:
            image = f['data'][:, :, slice_num]

        masked_Kspace = np.zeros((2, self.imageSize, self.imageSize))
        target_Kspace = np.zeros((2, self.imageSize, self.imageSize))
        target_image = np.zeros((1, self.imageSize, self.imageSize))

        image = image.astype(np.float32)
        image = np.rot90(image, 1)
        if self.transform:
            image = self.transform(image)

        kspace = self.fft2(image)
        masked_Kspace, target_Kspace, target_image = self.preprocess(kspace)
        return {
            "masked_K":torch.from_numpy(masked_Kspace),
            "target_K":torch.from_numpy(target_Kspace),
            "target_img": torch.from_numpy(target_image)
        }  

class IXIDatasetMultiSlices(Dataset):

    def __init__(self, data_dir, args, isTrainData = True, transform = None):

        self.args = args
        self.is_train = isTrainData
        self.imageSize = args.imgSize       
        self.num_input_slices = args.num_input_slices    ## control the input slice number
        self.fillNoise = args.fillnoise                 ## if fill noise to the sampled kspace
        self.minmax_noise_val = args.minmax_noise_val

        if self.is_train:
            self.data_dir = path.join(data_dir, "train/")
        else:
            self.data_dir = path.join(data_dir, "val/")

        self.transform = transform

        #make an image id's list
        self.file_names = [splitext(file)[0] for file in listdir(self.data_dir)]

        self.ids = list()
        
        for file_name in self.file_names:
            try:
                full_file_path = path.join(self.data_dir, file_name+'.hdf5')
                with h5py.File(full_file_path, 'r') as f:
                    num_slice = f['data'].shape[2]

                if num_slice < self.args.slice_range[1]:
                    continue

                for slice in range(self.args.slice_range[0], self.args.slice_range[1]):
                    self.ids.append((file_name, slice))
            except:
                continue

        if self.is_train:
            logging.info(f'Creating training datasets with {len(self.ids)} examples')
        else:
            logging.info(f'Creating val dataset with {len(self.ids)} examples')
       

        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
        mask0 = np.fft.fftshift(masks_dictionary['mask0'])
        mask1 = np.fft.fftshift(masks_dictionary['mask1'])
        mask2 = np.fft.fftshift(masks_dictionary['mask2'])
        self.masks = np.dstack((mask0,mask1,mask2))

        self.maskNot = 1-mask1
        
    def __len__(self):
        return len(self.ids)

    def fft2(self, image):
        return np.fft.fft2(image)
    
    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]
   
    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.imageSize:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.imageSize)/2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def slice_preprocess(self,kspace_cplx, slice_num):
        """
            kspace_cplx: an one channel complex matrix
            slice_num: the current slice number
        """
        #crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx) 
        #splits to real and imag channels
        kspace = np.zeros((self.imageSize, self.imageSize, 2))
        kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)
        ## this is the target image, need use the ifft result instead of the original one because after the fft and the ifft the image has changed
        image = self.ifft2(kspace_cplx)                             ## conduct the fftshift operation
        
        kspace = kspace.transpose((2, 0, 1))

        masked_Kspace = kspace*self.masks[:, :, slice_num]
        if self.fillNoise:
            masked_Kspace += np.random.uniform(low=self.minmax_noise_val[0], high=self.minmax_noise_val[1],
                                            size=masked_Kspace.shape)*self.maskNot
        return masked_Kspace, kspace, image

    def __getitem__(self, i):
        file_name, slice_num = self.ids[i]

        full_file_path = path.join(self.data_dir,file_name + '.hdf5')

        with h5py.File(full_file_path, 'r') as f:
            add = int(self.num_input_slices / 2)  ## add controls the slices range
            imgs = f['data'][:, :, slice_num-add:slice_num+add+1]
            
            masked_Kspaces = np.zeros((self.num_input_slices*2, self.imageSize, self.imageSize))
            target_Kspace = np.zeros((2, self.imageSize, self.imageSize))
            target_img = np.zeros((1, self.imageSize, self.imageSize))

            for sliceNum in range(self.num_input_slices):
                img = imgs[:,:,sliceNum]
                img = np.rot90(img, 1)
                kspace = self.fft2(img)
                slice_masked_Kspace, slice_full_Kspace, slice_full_image = self.slice_preprocess(kspace, sliceNum)

                masked_Kspaces[sliceNum*2:sliceNum*2+2, :, :] = slice_masked_Kspace
                ## if it's the middle slice append the image
                if sliceNum == int(self.num_input_slices/2):
                    target_Kspace = slice_full_Kspace
                    target_img = slice_full_image

            return {
                "masked_K":torch.from_numpy(masked_Kspaces),
                "target_K":torch.from_numpy(target_Kspace),
                "target_img": torch.from_numpy(target_img)
            }


def main():
    with open('./config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    args = SimpleNamespace(**data)
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10)
    ])
    dataset = IXIDatasetMultiSlices('D:/datas/IXI-T1-save/', args=args,isTrainData= True ,transform=transform)    
    dataloader = DataLoader(dataset , batch_size=1 , shuffle=True , num_workers=0 , drop_last=True)
    data_iter = iter(dataloader)
    data = data_iter.next()

    tar = data["target_img"]
    label = data["masked_K"]
    tar = tar.detach().numpy()
    label = label.detach().numpy()

    print("size", tar.shape)
    kspace1 = label[0,0,:,:]+1j*label[0,1,:,:]
    kspace2 = label[0,2,:,:]+1j*label[0,3,:,:]
    kspace3 = label[0,4,:,:]+1j*label[0,5,:,:]

    kimg1 =np.log(np.abs(kspace1))
    kimg2 =np.log(np.abs(kspace2))
    kimg3 = np.log(np.abs(kspace3))

    zfimg = np.abs(np.fft.ifft2(kimg2))


    tar_img = np.squeeze(tar)

    plt.figure()
    plt.imshow(kimg1,plt.cm.gray)
    plt.title("kimg1")
    plt.figure()
    plt.imshow(np.fft.fftshift(kimg2),plt.cm.gray)
    plt.title("kimg1")    
    plt.figure()
    plt.imshow(kimg3,plt.cm.gray)
    plt.title("kimg1")
    plt.figure()
    plt.imshow(tar_img,plt.cm.gray)
    plt.title("tar_img")
    plt.figure()
    plt.imshow(zfimg,plt.cm.gray)
    plt.title("zfimg")

    plt.show()


from scipy import fftpack
        
if __name__ =='__main__':
    main()
