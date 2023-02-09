"""
    This file contains a dataset foe IXI database
"""

from os.path import splitext
from os import listdir, path
from h5py._hl.files import File
import numpy as np
from glob import glob
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import logging
from PIL import Image
import h5py
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt
from torchvision import transforms

class IXIDataSet(Dataset):
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
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]
   
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


def main():
    with open('./config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    args = SimpleNamespace(**data)
    print(args)
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
    ])
    dataset = IXIDataSet(args.imageDataDir, args=args,is_train=True ,transform=transform)    
    dataloader = DataLoader(dataset , batch_size=4 , shuffle=True , num_workers=0 , drop_last=True)
    data_iter = iter(dataloader)
    data = data_iter.next()
    label = data["target_img"]
    targetK = data["target_K"]
    masked_k = data["masked_K"]
    label = label.detach().numpy()
    targetK = targetK.detach().numpy()
    masked_k = masked_k.detach().numpy()
    for i in range(masked_k.shape[0]):
        image = abs(np.log(masked_k[i,:,:,:][0,:,:]+1j*masked_k[i,:,:,:][1,:,:]))
        image = np.squeeze(image)
        plt.figure()
        plt.imshow(image, plt.cm.gray)
    plt.show()
    print(masked_k.shape)

# if __name__ == '__main__':
#     main()