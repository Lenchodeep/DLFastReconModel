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
import random
from utils import undersample


class FastMRIDatasetRefinrGan(Dataset):
    '''
        The data range is 0-1
    '''
    def __init__(self,data_dir, args, isTrainData = True, transform = None):
        super().__init__()

        self.args = args

        self.is_train = isTrainData        
        self.imageSize = args.imgSize       
       
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
                full_file_path = path.join(self.data_dir, file_name+'.h5')
                with h5py.File(full_file_path, 'r') as f:
                    num_slice = f['reconstruction_rss'].shape[2]

                if num_slice < self.args.slice_range[1]:
                    continue

                for slice in range(self.args.slice_range[0], self.args.slice_range[1]):
                    self.ids.append((file_name, slice))
            except:
                continue
        # get the data lenth
        self.len = len(self.ids)
        if self.is_train:
            logging.info(f'Creating training datasets with {len(self.ids)} examples')
        else:
            logging.info(f'Creating val dataset with {len(self.ids)} examples')

        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
        ## shift the mask instead of shift the kspace image
        self.mask = torch.fft.fftshift(torch.from_numpy(masks_dictionary['mask1']))
        self.maskNot = 1-self.mask
        
    def __len__(self):
        return self.len

    def __getitem__(self, i):
        ## get the imageA
        if self.is_train:
            file_name, slice_num = self.ids[i]
            full_file_path = path.join(self.data_dir,file_name + '.h5')

            with h5py.File(full_file_path, 'r') as f:
                imageA = f['reconstruction_rss'][slice_num,:, :]
                # imageA = np.rot90(imageA, 1)
                imageA = self.image_regular(self.cropCenter(imageA))
                ## if transform using the transform
                if self.transform:
                    imageA = self.transform(imageA)

            ## get the imageB
            index_B = random.randint(0, self.len - 1)   ## get another image index
            file_name_B, slice_num_B = self.ids[index_B]
            full_file_path_B = path.join(self.data_dir,file_name_B + '.h5')

            with h5py.File(full_file_path_B, 'r') as f:
                imageB = f['reconstruction_rss'][slice_num_B,:, :]
                # imageB = np.rot90(imageB, 1)
                imageB = self.image_regular(self.cropCenter(imageB))
                ## if transform using the transform
                if self.transform:
                    imageB = self.transform(imageB)


            imageA = imageA[..., np.newaxis]
            imageB = imageB[..., np.newaxis]

            imageA = np.transpose(imageA, (2,0,1))
            imageB = np.transpose(imageB, (2,0,1))
            if self.args.isZeroToOne:
                imageA = (imageA[0] - (0 + 0j)) 
                imageB = (imageB[0] - (0 + 0j)) 
            else:
                imageA = (imageA[0] - (0.5 + 0.5j)) * 2.0
                imageB = (imageB[0] - (0.5 + 0.5j)) * 2.0


            imageA = torch.from_numpy(imageA)
            imageB = torch.from_numpy(imageB)
            # generate zero-filled image x_und, k_und, k
            image_A_und, k_A_und, k_A = undersample(imageA, self.mask)
            image_B_und, k_B_und, k_B = undersample(imageB, self.mask)
            ########################## complex to 2 channel ##########################
            im_A = torch.view_as_real(imageA).permute(2, 0, 1).contiguous()
            im_A_und = torch.view_as_real(image_A_und).permute(2, 0, 1).contiguous()
            k_A_und = torch.view_as_real(k_A_und).permute(2, 0, 1).contiguous()
            im_B = torch.view_as_real(imageB).permute(2, 0, 1).contiguous()
            im_B_und = torch.view_as_real(image_B_und).permute(2, 0, 1).contiguous()
            k_B_und = torch.view_as_real(k_B_und).permute(2, 0, 1).contiguous()
            mask = torch.view_as_real(self.mask * (1. + 1.j)).permute(2, 0, 1).contiguous()

            return {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und,
                    'im_B': im_B, 'im_B_und': im_B_und, 'k_B_und': k_B_und, 'mask': mask}
        else:
            val_file_name, val_slice_num = self.ids[i]
            full_file_path = path.join(self.data_dir,val_file_name + '.h5')

            with h5py.File(full_file_path, 'r') as f:
                val_imageA = f['reconstruction_rss'][val_slice_num,:, :]
                # val_imageA = np.rot90(val_imageA, 1)
                val_imageA = self.image_regular(self.cropCenter(val_imageA))
                ## if transform using the transform
                if self.transform:
                    val_imageA = self.transform(val_imageA)

            val_imageA = val_imageA[..., np.newaxis]
            val_imageA = np.transpose(val_imageA, (2,0,1))
            if self.args.isZeroToOne:
                val_imageA = (val_imageA[0] - (0 + 0j)) 
            else:
                val_imageA = (val_imageA[0] - (0.5 + 0.5j)) * 2.0


            val_imageA = torch.from_numpy(val_imageA.copy())
            val_image_A_und, val_k_A_und, k_A = undersample(val_imageA, self.mask)
            ########################## complex to 2 channel ##########################
            val_im_A = torch.view_as_real(val_imageA).permute(2, 0, 1).contiguous()
            val_im_A_und = torch.view_as_real(val_image_A_und).permute(2, 0, 1).contiguous()
            val_k_A_und = torch.view_as_real(val_k_A_und).permute(2, 0, 1).contiguous()
            val_mask = torch.view_as_real(self.mask * (1. + 1.j)).permute(2, 0, 1).contiguous()

            return {'im_A': val_im_A, 'im_A_und': val_im_A_und, 'k_A_und': val_k_A_und, 'mask': val_mask}
    def cropCenter(self, image):
        if image.shape[0] == self.imageSize:
            return image
        if image.shape[0] % 2 == 1:
            image = image[:-1, :-1]
        crop = int((image.shape[0] - self.imageSize)/2)
        image = image[crop:-crop, crop:-crop]
        return image
    def image_regular(self, image):
        image = (image - np.min(image))/(np.max(image)-np.min(image))
        return image


from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def main():

    with open('./config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    args = SimpleNamespace(**data)
    print(args)
    transform  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10)
    ])
    dataset = FastMRIDatasetRefinrGan('D:/datas/knee/T1', args=args,isTrainData=True ,transform=None)    
    dataloader = DataLoader(dataset , batch_size=1, shuffle=True , num_workers=0 , drop_last=True)
    data_iter = iter(dataloader)
    data = data_iter.next()


    imagea = data['im_A']
    image_a_und = data['im_A_und']
    imageb = data['im_B']
    image_b_und = data['im_B_und']
    mask = data['mask']

    imagea = torch.abs(torch.complex(imagea[:,0,:,:], imagea[:,1,:,:]))
    imageb = torch.abs(torch.complex(imageb[:,0,:,:], imageb[:,1,:,:]))
    image_a_und = torch.abs(torch.complex(image_a_und[:,0,:,:], image_a_und[:,1,:,:]))
    image_b_und = torch.abs(torch.complex(image_b_und[:,0,:,:], image_b_und[:,1,:,:]))

    imagea = imagea.detach().numpy()
    imageb = imageb.detach().numpy()
    image_a_und = image_a_und.detach().numpy()
    image_b_und = image_b_und.detach().numpy()


    imagea = np.squeeze(imagea)
    imageb = np.squeeze(imageb)
    image_a_und = np.squeeze(image_a_und)
    image_b_und = np.squeeze(image_b_und)

    print(image_a_und.max(), image_a_und.min())

    plt.figure()
    plt.imshow(imagea,plt.cm.gray)
    plt.title("imagea")    
    plt.figure()
    plt.imshow(imageb,plt.cm.gray)
    plt.title("imageb")  
    plt.figure()
    plt.imshow(image_a_und,plt.cm.gray)
    plt.title("image_a_und")  
    plt.figure()
    plt.imshow(image_b_und,plt.cm.gray)
    plt.title("image_a_und")  
    plt.show()

# main()