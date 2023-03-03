import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import h5py
import logging
import pickle
import yaml

import numpy as  np
import pandas as pd
import  matplotlib.pyplot as plt
from types import SimpleNamespace
'''our module'''
from Model import *
from utils import calculatePSNR, calculateSSIM, calcuNMSE, \
                imgRegular, crop_toshape, fft2, ifft2, getArgs, pltImages

def undersample(image, mask):
    k = torch.fft.fft2(image)
    k_und = mask*k
    x_und = torch.absolute(torch.fft.ifft2(k_und))
    return x_und, k_und, k

def test(args, model):
    ## load model
    model = model.to(args.device)
    checkPoint = torch.load(args.predModelPath, map_location = args.device)
    model.load_state_dict(checkPoint['moel_state_dict'])
    logging.info("Model loaded !")
    model.eval()   #no backforward

    mask_path = args.mask_path
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    mask = torch.fft.fftshift(torch.from_numpy(masks_dictionary['mask1']))

    
    psnr_list = []
    ssim_list = [] 
    nmse_list = []

    f = pd.read_csv("./data/FastMRI/singlecoil_test_split_less.csv")
    PDWIList =  f['PDWI']
    FSPDWIList =  f['FSPDWI']

    sliceNum = args.slice_range[1] - args.slice_range[0]

    for i, T2file in enumerate(FSPDWIList):
        T2_image_path = os.path.join(args.predictDir, T2file+'.hdf5')
        T1_image_path = os.path.join(args.predictDir, PDWIList[i]+'.hdf5')
        logging.info("\nTesting image {} ...".format(T2_image_path))
        with h5py.File(T1_image_path, 'r') as f, h5py.File(T2_image_path, 'r') as s:
            PDWIimgs = f['reconstruction_rss']
            FSPDWIimgs = s['reconstruction_rss']

            
            rec_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
            ZF_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
            label_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))

            for slice in range(args.slice_range[0], args.slice_range[1],1):
                PDWIimg = imgRegular(crop_toshape(PDWIimgs[slice,:,:], args.imgSize))
                FSPDWIimg = imgRegular(crop_toshape(FSPDWIimgs[slice,:,:], args.imgSize))
                
                PDWIimg = PDWIimg[..., np.newaxis]
                FSPDWIimg = FSPDWIimg[..., np.newaxis]

                PDWIimg = np.transpose(PDWIimg, (2,0,1))
                FSPDWIimg = np.transpose(FSPDWIimg, (2,0,1))
                
                ## convert to torch data
                PDWIimg = torch.from_numpy(PDWIimg)
                FSPDWIimg = torch.from_numpy(FSPDWIimg)

                T1_masked_img, T1_masked_k, _ = undersample(PDWIimg, mask)  
                T2_masked_img, T2_masked_k, _ = undersample(FSPDWIimg, mask)  

                T1_tar_img = torch.unsqueeze(PDWIimg, dim=1).to(args.device, dtype=torch.float32)
                T1_masked_img = torch.unsqueeze(T1_masked_img, dim=1).to(args.device, dtype=torch.float32)
                T2_tar_img = torch.unsqueeze(FSPDWIimg, dim=1).to(args.device, dtype=torch.float32)
                T2_masked_img = torch.unsqueeze(T2_masked_img, dim=1).to(args.device, dtype=torch.float32)
                ## forward network
                rec_img_T2 =  model(T1_masked_img, T2_masked_img)
                    
                ## get outputs
                rec_T2 = torch.squeeze(rec_img_T2)
                tar_T2 = torch.squeeze(T2_tar_img)
                ZF_img = torch.squeeze(T2_masked_img)

                rec_T2 = rec_T2.detach().cpu().numpy()
                ZF_img = ZF_img.detach().cpu().numpy()
                tar_T2 = tar_T2.detach().cpu().numpy()

                rec_ssim = calculateSSIM(tar_T2, rec_T2, False)
                rec_psnr = calculatePSNR(tar_T2, rec_T2, False)
                rec_nmse = calcuNMSE(tar_T2, rec_T2, False)
 

                psnr_list.append(rec_psnr)
                ssim_list.append(rec_ssim)
                nmse_list.append(rec_nmse)


                rec_imgs[:,:,slice-args.slice_range[0]] = rec_T2
                ZF_imgs[:,:,slice-args.slice_range[0]] = ZF_img
                label_imgs[:,:,slice-args.slice_range[0]] = tar_T2
                           
            if args.visualize_images:
                logging.info("Visualizing results for image , close to continue ...")
                pltImages(ZF_imgs, rec_imgs, label_imgs,series_name =os.path.split(T2file)[1], args = args, sliceList=[10,13,15])
        ## final 
    psnrarray = np.array(psnr_list)
    ssimarray = np.array(ssim_list)
    nmsearray = np.array(nmse_list)
        
    avg_psnr = psnrarray.mean()
    std_psnr = psnrarray.std()

    avg_ssim = ssimarray.mean()
    std_ssim = ssimarray.std()

    avg_nmse = nmsearray.mean()
    std_nmse = nmsearray.std()
   
    print(  "avg_psnr:",avg_psnr, "std_psnr:", std_psnr,
        "avg_ssim", avg_ssim, "std_ssim:", std_ssim,
        "avg_nmse", avg_nmse, "std_nmse:", std_nmse)
 

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    configPath = './config.yaml'
    args = getArgs(configPath)

    logging.info(f"Using device {args.device}")
    logging.info("Loading model {}".format(args.predModelPath))
    model = Xnet()
    test(args, model)