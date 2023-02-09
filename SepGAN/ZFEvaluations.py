'''
This file is used for evaluating the zero-filled images' evaluation value(PSNR,SSIM,NMSE)
'''
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from os.path import splitext
from os import listdir, path

import numpy as np
import h5py
from types import SimpleNamespace
import pickle
from utils import *
import  matplotlib.pyplot as plt


def fft2(image):
    return np.fft.fftshift(np.fft.fft2(image))
def ifft2( kspace_cplx):
    return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace_cplx)))[None, :, :]
def ZF_evaluation(args):
    data_dir=args.predictDir
    file_names = [splitext(file)[0] for file in listdir(data_dir)]
    with open(args.mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    mask1 = masks_dictionary['mask1']
    # mask1 = np.fft.fftshift(np.load(args.mask_path))
    maskNot = 1-mask1

    zfpsnr = list()
    zfssim = list()
    zfnmse = list()
    count = 0
    totalNum = len(file_names)
    for filename in file_names:
        full_file_path = path.join(data_dir,filename + '.hdf5')

        with h5py.File(full_file_path, 'r') as f:
            data =f['data'] 
            num_slice = data.shape[2]
            if num_slice < args.slice_range[1]:
                continue
            count = count+1
            for slice in range(args.slice_range[0], args.slice_range[1]):
                image = data[:,:,slice]
                image = np.rot90(image,1)
                kspace = fft2(image)       
                masked_kspace = kspace*mask1
                # masked_kspace += np.random.uniform(low=args.minmax_noise_val[0], high=args.minmax_noise_val[1],
                #                             size=masked_kspace.shape)*maskNot
                label = ifft2(kspace)
                zf_img = ifft2(masked_kspace)



                # plt.figure()
                # plt.imshow(zf_img[0,:,:], plt.cm.gray)
                # plt.show()
                psnr = calculatePSNR(label,zf_img, True)
                zfpsnr.append(psnr)
                ssim = calculateSSIM(label, zf_img, True)
                zfssim.append(ssim)
                nmse = calcuNMSE(label, zf_img, True)
                zfnmse.append(nmse)
            percent = count / totalNum
            print("calculate percentage" ,percent, count)
    psnrarray = np.array(zfpsnr)
    ssimarray = np.array(zfssim)
    nmsearray = np.array(zfnmse)
    avg_psnr = np.mean(psnrarray)
    avg_ssim = np.mean(ssimarray)
    avg_nmse = np.mean(nmsearray)



    stdpsnr = np.std(psnrarray)
    stdssim = np.std(ssimarray)
    stdnmse = np.std(nmsearray)

    print("avg_psnr",avg_psnr,"avg_ssim",avg_ssim,"avg_nmse",avg_nmse,"stdpsnr",stdpsnr,"stdssim",stdssim,"stdnmse",stdnmse)
    print()


if __name__ == '__main__':
    configPath = './config.yaml'
    args = getArgs(configPath)
    ZF_evaluation(args)


