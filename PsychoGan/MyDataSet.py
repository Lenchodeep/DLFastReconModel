import os
from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.io import savemat
from scipy.io import loadmat
from torchvision.transforms.transforms import ToTensor
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

class MyDataSet(Dataset):

    def __init__(self, is_train, args, transform = None):
        super(MyDataSet, self).__init__()
        self.imageDataDir = args.imageDataDir
        self.dataFileName = args.dataFileName
        self.maskDir = args.mask_path
        self.is_train = is_train

        if self.is_train:
            # self.dataPath = path.join(self.imageDataDir, ('/train/' +self.dataFileName))
            self.dataPath =(self.imageDataDir+'/train/' +self.dataFileName)
        else:
            # self.dataPath = path.join(self.imageDataDir, ('/test/'+ self.dataFileName))
            self.dataPath =(self.imageDataDir+'/test/'+ self.dataFileName)
        print(self.dataPath)
        self.imageLabel = loadmat(self.dataPath)["data"]
        self.mask = np.load(self.maskDir)
        self.imageSize = args.imgSize
        self.transform = transform

    def __len__(self):
        return self.imageLabel.shape[0]  ##dara is orgnized in CHW

    def fft2(self, image):
        return np.fft.fftshift(np.fft.fft2(image))

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def preprocess(self, complexK):
        kspace = np.zeros((self.imageSize, self.imageSize, 2))
        kspace[:,:,0] = np.real(complexK).astype(np.float32)
        kspace[:,:,1] = np.imag(complexK).astype(np.float32)
        image = self.ifft2(complexK)   #正常图像

        kspace = kspace.transpose((2,0,1))
        masked_kspace = kspace* self.mask   #kspace和masked_kspace 均为shift后

        return masked_kspace, kspace, image

    def __getitem__(self, i):

        masked_Kspace = np.zeros((2, self.imageSize, self.imageSize))
        target_Kspace = np.zeros((2, self.imageSize, self.imageSize))
        target_image = np.zeros((1, self.imageSize, self.imageSize))

        image = self.imageLabel[i,:,:]
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
    with open('E:/DLCode/GanCode/config.yml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    args = SimpleNamespace(**data)
    print(args)
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15)
    ])
    dataset = MyDataSet(is_train=True, args=args, transform=transform)    
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