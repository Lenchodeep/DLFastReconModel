"""
    This file contains a dataset foe IXI database
"""

from os.path import splitext
from os import listdir, path
import numpy as np
from glob import glob
import torch
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
        # self.num_input_slice = args.num_input_slices
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
        complexK = self.crop_toshape(complexK)   #correct

        ##get the original image
        image = np.absolute(np.fft.ifft2(np.fft.fftshift(complexK)))
        # image = image / np.amax(image)
        image = np.expand_dims(image, -1)
        image = np.transpose(image, (2,0,1))

        m_complexK = complexK * self.mask


        masked_img = np.absolute(np.fft.ifft2(np.fft.fftshift(m_complexK)))
        # masked_img = masked_img / np.amax(masked_img)
        masked_img = np.expand_dims(masked_img, -1)
        masked_img = np.transpose(masked_img, (2,0,1))


        return masked_img ,image
   
    def __getitem__(self, i):

        file_name , slice_num = self.ids[i]
        full_file_path = path.join(self.data_dir,file_name + '.hdf5')

        with h5py.File(full_file_path, 'r') as f:
            image = f['data'][:, :, slice_num]

        target_image = np.zeros((1, self.imageSize, self.imageSize))
        masked_img = np.zeros((1, self.imageSize, self.imageSize))

        image = image.astype(np.float32)
        image = np.rot90(image, 1)
        if self.transform:
            image = self.transform(image)

        kspace = self.fft2(image)  #shift to center

        masked_img ,target_image = self.preprocess(kspace)
        image = np.expand_dims(image, -1)
        image = np.transpose(image, (2,0,1))

        return {
            "target_img": torch.from_numpy(target_image),
            "masked_img" : torch.from_numpy(masked_img)
        }


def main():
    with open(r'E:\DLCode\DLFastReconstruction\ESSGAN\config.yaml') as f:
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
    masked_img = data["masked_img"]
    label = label.detach().numpy()
    masked_img = masked_img.detach().numpy()
    print("label", label.max(), label.min(), "fold", masked_img.max(), masked_img.min())
    for i in range(masked_img.shape[0]):
        image = masked_img[i,:,:]
        image = np.squeeze(image)
        print(image.max(), image.min())
        plt.figure()
        plt.imshow(image, plt.cm.gray)
        plt.title("masked")

        plt.figure()
        plt.imshow(np.squeeze(label[i,:,:]), plt.cm.gray)
        plt.title("label")
    plt.show()

# if __name__ == '__main__':
#     main()