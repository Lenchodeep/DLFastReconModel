import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import yaml
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
from Models import *
from SingleModel import SingleGenerator
from DirictModel import *
from utils import calculatePSNR, calculateSSIM, calcuNMSE, \
                imgRegular, crop_toshape, fft2, ifft2, getArgs, pltImages, output2complex


def slice_preprocess(kspace_cplx, mask, args):
        kspace = np.zeros((args.imgSize, args.imgSize, 2))
        kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)

        ## this is the target image, need use the ifft result instead of the original one because after the fft and the ifft the image has changed
        image = ifft2(kspace_cplx)                             ## conduct the fftshift operation
        kspace = kspace.transpose((2, 0, 1))
        masked_Kspace = kspace*mask

        return masked_Kspace, kspace, image


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
    T2_rec_Kspaces_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum), dtype=np.csingle) #complex
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

                masked_PDWIkspace_np, full_PDWIkspace, full_PDWIimg = slice_preprocess(PDWIkspace, mask, args)
                masked_FSPDWIkspace_np, full_FSPDWIkspace, full_FSPDWIimg = slice_preprocess(FSPDWIkspace, mask, args)

                
                masked_PDWIkspace = np.expand_dims(masked_PDWIkspace_np, axis=0)
                masked_FSPDWIkspace = np.expand_dims(masked_FSPDWIkspace_np, axis=0)

                masked_PDWIkspace = torch.from_numpy(masked_PDWIkspace).to(device=args.device, dtype=torch.float32)
                masked_FSPDWIkspace = torch.from_numpy(masked_FSPDWIkspace).to(device=args.device, dtype=torch.float32)

                rec_T1, rec_k_T1, rec_T2, rec_k_T2 = model(masked_PDWIkspace, masked_FSPDWIkspace)

                rec_T1 = np.squeeze(rec_T1.data.cpu().numpy())
                rec_T2 = np.squeeze(rec_T2.data.cpu().numpy())

                if args.testDC:
                    reck = fft2(rec_T2)
                    tark = fft2(FSPDWIimg)

                    finalk = mask*tark + maskNot*reck

                    rec_T2 = np.squeeze(ifft2(finalk))

                zf_T2 = np.squeeze(ifft2((masked_FSPDWIkspace_np[0, :, :] + 1j*masked_FSPDWIkspace_np[1, :, :])))

                rec_T2_psnr = calculatePSNR(FSPDWIimg,rec_T2)
                rec_T2_ssim = calculateSSIM(FSPDWIimg,rec_T2)
                rec_nmse = calcuNMSE(FSPDWIimg, rec_T2)
                zf_psnr = calculatePSNR(FSPDWIimg,zf_T2)

                psnr_list.append(rec_T2_psnr)
                ssim_list.append(rec_T2_ssim)
                nmse_list.append(rec_nmse)
                psnr_list_zf.append(zf_psnr)
                # print(rec_T2.max())
                # plt.figure()
                # plt.imshow(rec_T2, plt.cm.gray)
                # plt.title("Recon")
                # plt.figure()
                # plt.imshow(FSPDWIimg, plt.cm.gray)
                # plt.title("Tar")
                # plt.figure()
                # plt.imshow(zf_T2, plt.cm.gray)
                # plt.title("ZF")
                # plt.figure()
                # plt.imshow(10*(FSPDWIimg-rec_T2), vmin=0, vmax=1, cmap=plt.get_cmap('jet'))
                # plt.title("Diff")
                # plt.show()
                T2_rec_img[:, :, slice-args.slice_range[0]] = rec_T2
                T2_tar_img[:, :, slice-args.slice_range[0]] = FSPDWIimg
                T2_ZF_img[:, :, slice-args.slice_range[0]] = zf_T2

        T2_tar_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = T2_tar_img
        T2_rec_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = T2_rec_img
        T2_ZF_img_full[:,:,i*sliceNum:(i+1)*sliceNum] = T2_ZF_img

        # if args.visualize_images:
        #     logging.info("Visualizing results for image {}, close to continue ...".format(T2_image_path))
        #     pltImages(zf_T2, rec_T2, FSPDWIimg, series_name =os.path.split(T2_image_path)[1],args = args, mask=mask, sliceList=[10,15,20])



    ## calculate average evaluation values
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

##################################### multi modal test part ##################################
def undersample(image, mask):
    k = torch.fft.fft2(image)
    k_und = mask*k
    x_und = torch.fft.ifft2(k_und)
    return x_und, k_und, k

def multimodalTest(args, model, singleTrain = False):
    model = model.to(args.device)
    checkPoint = torch.load(args.predModelPath, map_location = args.device)
    model.load_state_dict(checkPoint['G_model_state_dict'])
    logging.info("Model loaded !")
    model.eval()   #no backforward

    # init mask
    mask_path = args.mask_path
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    mask = torch.fft.fftshift(torch.from_numpy(masks_dictionary['mask1']))

    psnr_list = []
    ssim_list = [] 
    nmse_list = []
    psnr_list_zf = []
    ssim_list_zf = []
    nmse_list_zf = []

    f = pd.read_csv("./data/FastMRI/singlecoil_test_split_less.csv")
    PDWIList =  f['PDWI']
    FSPDWIList =  f['FSPDWI']
    
    fileLen = len(FSPDWIList)
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
                ## convert to complex data
                PDWIimg = (PDWIimg[0] - (0 + 0j)) 
                FSPDWIimg = (FSPDWIimg[0] - (0 + 0j)) 
                
                ## convert to torch data
                PDWIimg = torch.from_numpy(PDWIimg)
                FSPDWIimg = torch.from_numpy(FSPDWIimg)

                T1_masked_img, T1_masked_k, _ = undersample(PDWIimg, mask)  
                T2_masked_img, T2_masked_k, _ = undersample(FSPDWIimg, mask)  

                T1_tar_img = torch.unsqueeze(torch.view_as_real(PDWIimg).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                T1_masked_img = torch.unsqueeze(torch.view_as_real(T1_masked_img).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                T1_masked_k = torch.unsqueeze(torch.view_as_real(T1_masked_k).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                T2_tar_img = torch.unsqueeze(torch.view_as_real(FSPDWIimg).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                T2_masked_img = torch.unsqueeze(torch.view_as_real(T2_masked_img).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                T2_masked_k = torch.unsqueeze(torch.view_as_real(T2_masked_k).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                in_mask = torch.unsqueeze(torch.view_as_real(mask * (1. + 1.j)).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                ## forward network
                if singleTrain:
                    rec_mid_T2_pre, rec_mid_T2, rec_img_T2_pre, rec_img_T2 = model(T2_masked_img, T2_masked_k, in_mask)
                else:
                     rec_mid_T1_pre, rec_mid_T2_pre, rec_mid_T1, rec_mid_T2, \
                            rec_img_T1_pre, rec_img_T2_pre, rec_img_T1, rec_img_T2\
                            = model(T1_masked_img, T1_masked_k,T2_masked_img, T2_masked_k, in_mask)
                ## get outputs
                rec_T2 = torch.squeeze(output2complex(rec_img_T2, args.isZeroToOne).abs())
                tar_T2 = torch.squeeze(output2complex(T2_tar_img, args.isZeroToOne).abs())
                ZF_img = torch.squeeze(output2complex(T2_masked_img, args.isZeroToOne).abs())

                rec_T2 = rec_T2.detach().cpu().numpy()
                ZF_img = ZF_img.detach().cpu().numpy()
                tar_T2 = tar_T2.detach().cpu().numpy()
                # plt.figure()
                # plt.imshow(rec_T2, plt.cm.gray)
                # plt.show()
                ## calculate evaluation value
                rec_ssim = calculateSSIM(tar_T2, rec_T2, False)
                rec_psnr = calculatePSNR(tar_T2, rec_T2, False)
                rec_nmse = calcuNMSE(tar_T2, rec_T2, False)
                zf_ssim = calculateSSIM(tar_T2, ZF_img, False)
                zf_psnr = calculatePSNR(tar_T2, ZF_img, False)
                zf_nmse = calcuNMSE(tar_T2, ZF_img, False)   

                psnr_list.append(rec_psnr)
                ssim_list.append(rec_ssim)
                nmse_list.append(rec_nmse)

                psnr_list_zf.append(zf_psnr)
                ssim_list_zf.append(zf_ssim)
                nmse_list_zf.append(zf_nmse)

                rec_imgs[:,:,slice-args.slice_range[0]] = rec_T2
                ZF_imgs[:,:,slice-args.slice_range[0]] = ZF_img
                label_imgs[:,:,slice-args.slice_range[0]] = tar_T2
                           
            if args.visualize_images:
                logging.info("Visualizing results for image , close to continue ...")
                pltImages(ZF_imgs, rec_imgs, label_imgs,series_name =os.path.split(T2file)[1], args = args, sliceList=[10,15,20])
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
 
    ## mid
    zfpsnrarray = np.array(psnr_list_zf)
    zfssimarray = np.array(ssim_list_zf)
    zfnmsearray = np.array(nmse_list_zf)

    mid_avg_psnr = zfpsnrarray.mean()
    mid_std_psnr = zfpsnrarray.std()
  
    mid_avg_ssim = zfssimarray.mean()
    mid_std_ssim = zfssimarray.std()

    mid_avg_nmse = zfnmsearray.mean()
    mid_std_nmse = zfnmsearray.std()

    print( "zf_avg_psnr:",mid_avg_psnr, "zf_var_psnr:", mid_std_psnr,
        "zf_avg_ssim", mid_avg_ssim, "zf_var_ssim:", mid_std_ssim,
        "zf_avg_nmse", mid_avg_nmse, "zf_var_nmse:", mid_std_nmse)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    configPath = './config.yaml'
    args = getArgs(configPath)

    logging.info(f"Using device {args.device}")
    logging.info("Loading model {}".format(args.predModelPath))
    # model = DirectG(args)
    model = SingleGenerator(args)

    multimodalTest(args, model, True)