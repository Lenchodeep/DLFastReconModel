import os 
from os.path import splitext
from os import listdir, path
import torch 
import logging
import h5py
import yaml
import pickle
import matplotlib.pyplot as plt
import numpy as np
from  torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace

class IXIDataSetImage(Dataset):
    def __init__(self, data_dir, args, isTrainData = True, transform = None):
        if isTrainData:
            self.dataDir = path.join(data_dir, 'train/')
        else:
            self.dataDir = path.join(data_dir, 'Val')
        self.transform = transform
        self.file_names = [splitext(file)[0] for file in listdir(self.dataDir)]    ##get all the h5py file in the folder

        self.imageSize = args.image_size
        self.ids = list()      ## to save the slices and the file name

        for file_name in self.file_names:
            try:
                fullPath = path.join(self.dataDir, file_name+'.hdf5')
                with h5py.File(fullPath,'r') as f:
                    num_slice = f['data'].shape[2]
                if num_slice < args.slice_range[1]:
                    continue
                for slice in range(args.slice_range[0], args.slice_range[1]):
                    self.ids.append((file_name, slice))
            except:
                continue

        if isTrainData:
            logging.info(f'Creating training datasets with {len(self.ids)} examples')
        else:
            logging.info(f'Creating val dataset with {len(self.ids)} examples') 

        ## get the sample mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        ## for DAGan, we just use the sigle mask to under sample
        self.mask = masks['mask1'] == 1
        self.maskNot = self.mask == 0


    def __len__(self):
        return len(self.ids)

    def crop_toshape(self, kspace_cplx):
            if kspace_cplx.shape[0] == self.imageSize:
                return kspace_cplx
            if kspace_cplx.shape[0] % 2 == 1:
                kspace_cplx = kspace_cplx[:-1, :-1]
            crop = int((kspace_cplx.shape[0] - self.imageSize)/2)
            kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
            return kspace_cplx

    def fft2(self, image):
        return np.fft.fftshift(np.fft.fft2(image))

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace_cplx)))

    def preProcess(self, complexK):
        complexK = self.crop_toshape(complexK)
        label = self.ifft2(complexK)

        masked_kspace = complexK * self.mask
                    
        fold_img = self.ifft2(masked_kspace) 
        return fold_img, label

    def __getitem__(self, i):
        file_name, slice_num = self.ids[i]
        full_file_path = path.join(self.dataDir,file_name + '.hdf5')

        with h5py.File(full_file_path, 'r') as f:
            image = f['data'][:,:,slice_num]        ##get the 2d image by the slice number
        
        image = np.rot90(image, 1)  ## need to rotate 90 degree
        image = image.astype(np.float32)

        if self.transform:
            image = self.transform(image)       ## the input of the transform is a 2d slice
        
        kspace = self.fft2(image)    ## the kspace is after the fftshift

        fold_img_tmp, label_img_tmp = self.preProcess(kspace)
        ## from [0,1] to [-1,1]
        fold_img_tmp = (fold_img_tmp*2)-1
        label_img_tmp = (label_img_tmp*2)-1

        fold_img_tmp = np.expand_dims(fold_img_tmp, 0)
        label_img_tmp = np.expand_dims(label_img_tmp, 0)
        
        return {
            "fold_img": torch.from_numpy(fold_img_tmp),
            "target_img": torch.from_numpy(label_img_tmp)
        }

def main():

    with open('./config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        
        args = SimpleNamespace(**data)
        print(args)
        dataset = IXIDataSetImage('D:/datas/IXI-T1-save/', args = args, isTrainData=True)    
        dataloader = DataLoader(dataset , batch_size=4 , shuffle=True , num_workers=0 )
        data_iter = iter(dataloader)
        data = data_iter.next()

        zf_image = data['fold_img']
        label = data['target_img']
        print(zf_image.max(), zf_image.min())

        zf_image = zf_image.detach().numpy() 
        label = label.detach().numpy() 

        ZF_image = np.squeeze(zf_image)
        label = np.squeeze(label)

        plt.figure()
        plt.imshow(ZF_image[0,:,:],plt.cm.gray)
        plt.title("zfImage")
        plt.figure()
        plt.imshow(label[0,:,:],plt.cm.gray)
        plt.title("label")    


    plt.show()
# main()