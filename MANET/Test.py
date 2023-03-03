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

from MANet import *
from utils import calculatePSNR, calculateSSIM, calcuNMSE, \
                imgRegular, crop_toshape, fft2, ifft2, getArgs, pltImages

def slice_preprocess(kspace_cplx, mask, args):
        kspace = np.zeros((args.imgSize, args.imgSize, 2))
        kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)

        ## this is the target image, need use the ifft result instead of the original one because after the fft and the ifft the image has changed
        image = ifft2(kspace_cplx)                             ## conduct the fftshift operation
        kspace = kspace.transpose((2, 0, 1))
        masked_Kspace = kspace*mask
        masked_img = ifft2((masked_Kspace[0,:,:]+1j*masked_Kspace[1,:,:]))

        return masked_Kspace, kspace, image, masked_img

def test(args, model):
    ## load model
    model = model.to(args.device)
    checkPoint = torch.load(args.predModelPath, map_location = args.device)
    model.load_state_dict(checkPoint['G_model_state_dict'])
    logging.info("Model loaded !")

    model.eval()   #no backforward
    ## init mask
    mask_path = args.mask_path
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    mask = masks_dictionary['mask1']
    maskNot = 1-mask

    psnr_list = []
    ssim_list = [] 
    nmse_list = []
    psnr_list_zf = []

    f = pd.read_csv("./data/FastMRI/singlecoil_test_split_less.csv")
    PDWIList =  f['PDWI']
    FSPDWIList =  f['FSPDWI']

    fileLen = len(FSPDWIList)
    sliceNum = args.slice_range[1] - args.slice_range[0]

    T2_rec_imgs_full = np.zeros((args.imgSize,args.imgSize,fileLen*sliceNum))
    T2_ZF_img_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))
    T2_tar_imgs_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))

    for i, T2file in enumerate(FSPDWIList):
        T2_image_path = os.path.join(args.predictDir, T2file+'.hdf5')
        T1_image_path = os.path.join(args.predictDir, PDWIList[i]+'.hdf5')
        logging.info("\nTesting image {} ...".format(T2_image_path))
        with h5py.File(T1_image_path, 'r') as f, h5py.File(T2_image_path, 'r') as s:
            PDWIimgs = f['reconstruction_rss']
            FSPDWIimgs = s['reconstruction_rss']

            T2_rec_img = np.zeros((args.imgSize,args.imgSize,sliceNum))
            T2_tar_img = np.zeros((args.imgSize,args.imgSize,sliceNum))
            T2_ZF_img = np.zeros((args.imgSize, args.imgSize,sliceNum))

            for slice in range(args.slice_range[0], args.slice_range[1],1):
                PDWIimg = imgRegular(crop_toshape(PDWIimgs[slice,:,:], args.imgSize))
                FSPDWIimg = imgRegular(crop_toshape(FSPDWIimgs[slice,:,:], args.imgSize))

                PDWIkspace = fft2(PDWIimg)
                FSPDWIkspace = fft2(FSPDWIimg)

                masked_PDWIkspace_np, full_PDWIkspace, full_PDWIimg, masked_PDWIimg_np = slice_preprocess(PDWIkspace, mask, args)
                masked_FSPDWIkspace_np, full_FSPDWIkspace, full_FSPDWIimg, masked_FSPDWIimg_np = slice_preprocess(FSPDWIkspace, mask, args)
                
                masked_PDWIkspace = np.expand_dims(masked_PDWIkspace_np, axis=0)
                masked_FSPDWIkspace = np.expand_dims(masked_FSPDWIkspace_np, axis=0)

                masked_PDWIkspace = torch.from_numpy(masked_PDWIkspace).to(device=args.device, dtype=torch.float32)
                masked_FSPDWIkspace = torch.from_numpy(masked_FSPDWIkspace).to(device=args.device, dtype=torch.float32)

                masked_PDWIimg = np.expand_dims(masked_PDWIimg_np, axis=0)
                masked_FSPDWIimg = np.expand_dims(masked_FSPDWIimg_np, axis=0)

                masked_PDWIimg = torch.from_numpy(masked_PDWIimg).to(device=args.device, dtype=torch.float32)
                masked_FSPDWIimg = torch.from_numpy(masked_FSPDWIimg).to(device=args.device, dtype=torch.float32)

                rec_T1, rec_mid_T1, rec_T2, rec_mid_T2 = model(masked_PDWIimg, masked_FSPDWIimg, masked_PDWIkspace, masked_FSPDWIkspace)

                rec_T1 = np.squeeze(rec_T1.data.cpu().numpy())
                rec_T2 = np.squeeze(rec_T2.data.cpu().numpy())

                zf_T2 = np.squeeze(masked_FSPDWIimg_np)

                rec_T2_psnr = calculatePSNR(FSPDWIimg,rec_T2)
                rec_T2_ssim = calculateSSIM(FSPDWIimg,rec_T2)
                rec_nmse = calcuNMSE(FSPDWIimg, rec_T2)
                zf_psnr = calculatePSNR(FSPDWIimg,zf_T2)

                psnr_list.append(rec_T2_psnr)
                ssim_list.append(rec_T2_ssim)
                nmse_list.append(rec_nmse)
                psnr_list_zf.append(zf_psnr)

                T2_rec_img[:, :, slice-args.slice_range[0]] = rec_T2
                T2_tar_img[:, :, slice-args.slice_range[0]] = FSPDWIimg
                T2_ZF_img[:, :, slice-args.slice_range[0]] = zf_T2
        T2_tar_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = T2_tar_img
        T2_rec_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = T2_rec_img
        T2_ZF_img_full[:,:,i*sliceNum:(i+1)*sliceNum] = T2_ZF_img
        
        if args.visualize_images:
            logging.info("Visualizing results for image {}, close to continue ...".format(T2_image_path))
            pltImages(T2_ZF_img, T2_rec_img, T2_tar_img, series_name =os.path.split(T2_image_path)[1],args = args, sliceList=[10,13,15])

    psnrarray = np.array(psnr_list)
    ssimarray = np.array(ssim_list)
    nmsearray = np.array(nmse_list)
    psnrzfarray = np.array(psnr_list_zf)

    avg_psnr = psnrarray.mean()
    avg_ssim = ssimarray.mean()
    avg_nmse = nmsearray.mean()
    avg_psnr_zf = psnrzfarray.mean()

    stdnmse = np.std(nmsearray)
    stdpsnr = np.std(psnrarray)
    stdssim = np.std(ssimarray)

    print("avg_psnr:",avg_psnr ,"stdpsnr:", stdpsnr,"avg_ssim", avg_ssim, 
        "stdssim:",stdssim,  "avg_nmse", avg_nmse,"stdnmse:", stdnmse, "zfpsnr: ",avg_psnr_zf)



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    configPath = './config.yaml'
    args = getArgs(configPath)

    logging.info(f"Using device {args.device}")
    logging.info("Loading model {}".format(args.predModelPath))
    model = Generator(args)
    test(args, model)