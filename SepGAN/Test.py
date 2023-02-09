import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from models import *
import os
import sys
import numpy as  np
import yaml
import  matplotlib.pyplot as plt
import torch
import h5py
from scipy.io import loadmat
import logging
from types import SimpleNamespace
import glob
from utils import calculatePSNR, calculateSSIM, saveData, pltImages,calcuNMSE, getArgs
import pandas
import pickle
import time

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def crop_toshape(kspace_cplx, args):
    if kspace_cplx.shape[0] == args.imgSize:
        return kspace_cplx
    if kspace_cplx.shape[0] % 2 == 1:
        kspace_cplx = kspace_cplx[:-1, :-1]
    crop = int((kspace_cplx.shape[0] - args.imgSize) / 2)
    kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]

    return kspace_cplx
    
def crop_toshape1( image):
    '''
        crop to target image size
    '''
    if image.shape[0] == 256:
        return image
    if image.shape[0] % 2 == 1:
        image = image[:-1, :-1]
    crop = int((image.shape[0] - 256)/2)
    image = image[crop:-crop, crop:-crop]
    return image


def regular(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))
def fft2(img):
    return np.fft.fft2(img)

def ifft2(kspace_cplx):
    return np.absolute(np.fft.ifft2(kspace_cplx))[None,:,:]

def slice_preprocess(kspace_cplx, mask, args):
    kspace = np.zeros((args.imgSize, args.imgSize, 2))
    kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
    kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)

    image = np.squeeze(ifft2(kspace_cplx)).astype(np.float32)   ## remove the 1 channel
    
    kspace = np.transpose(kspace, (2,0,1))
    masked_kspace = kspace* mask

    if args.fillnoise:
        masked_kspace += np.random.uniform(low=args.minmax_noise_val[0], high=args.minmax_noise_val[1],
                                        size=masked_kspace.shape)*(1-mask)
    return masked_kspace, kspace, image


def multiSlice_preprocess(kspace_cplx, slice_num, masks, maskedNot, args):
    kspace = np.zeros((args.imgSize, args.imgSize, 2))
    kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
    kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)
    # target image:
    image = ifft2(kspace_cplx)
    # HWC to CHW
    kspace = kspace.transpose((2, 0, 1))
    masked_Kspace = kspace * masks[:, :, slice_num]
    masked_Kspace += np.random.uniform(low=args.minmax_noise_val[0], high=args.minmax_noise_val[1],
                                       size=masked_Kspace.shape) * maskedNot
    return masked_Kspace, kspace, image

#### the single slice test
def test_Kdomain(args, model):
    model = model.to(args.device)
    checkPoint = torch.load(args.predModelPath, map_location = args.device)
    model.load_state_dict(checkPoint['G_model_state_dict'])
    model.eval()   #no backforward
    logging.info("Model loaded !")
    mask = np.load(args.mask_path)
    test_files = glob.glob(os.path.join(args.predictDir, '*.hdf5'))

    fileLen = len(test_files)
    sliceNum = args.slice_range[1] - args.slice_range[0]

    rec_img_full = np.zeros((args.imgSize,args.imgSize,fileLen*sliceNum))  
    rec_Kspace_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum),np.complex128)
    rec_mid_img_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))
    ZF_img_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))
    label_data_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))

    psnr_list = []
    ssim_list = [] 
    nmse_list = []

    for i, infile in enumerate(test_files):
        logging.info("\nPredicting image {} ...".format(infile))
        with h5py.File(infile, 'r') as f:
            img_shape = f['data'].shape
            target_img = f['data'][:]

        #data preprocess
        rec_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
        rec_Kspace = np.zeros((args.imgSize,args.imgSize,sliceNum), np.complex128)
        rec_mid_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
        ZF_img = np.zeros((args.imgSize,args.imgSize,sliceNum))
        label_data = np.zeros((args.imgSize,args.imgSize,sliceNum))

        for slice in range(20,120,1):
            img = target_img[:,:,slice]
            img = np.rot90(img, 1)   ##rotation the image to a normal condition

            kspace = fft2(img)
            
            masked_k, full_k, full_img = slice_preprocess(kspace, mask,args)

            masked_k = np.expand_dims(masked_k, axis=0)
            masked_k = torch.from_numpy(masked_k).to(device= args.device, dtype = torch.float32)
            ## reconstruction
            rec_img , rec_k, rec_mid_img = model(masked_k)

     
            ## the reconstruction image
            rec_img = np.squeeze(rec_img.data.cpu().numpy()) # squeeze the channel
            ## the reconstruction k data
            rec_k = np.squeeze(rec_k.data.cpu().numpy())
            rec_k = rec_k[0,:,:]+1j*rec_k[1,:,:]    #create the 
            ## the mid image
            rec_mid_img = np.squeeze(rec_mid_img.data.cpu().numpy())
            ## the fold image
            masked_k = masked_k.data.cpu().numpy()
            masked_k = np.squeeze(masked_k)
            maskedData = masked_k[0,:,:]+1j*masked_k[1,:,:]
            zfimage = np.absolute(np.fft.ifft2(np.fft.fftshift(maskedData)))


            ## calculate the evaluation numbers

            rec_psnr = calculatePSNR(full_img,rec_img)
            rec_ssim = calculateSSIM(full_img,rec_img)
            rec_nmse = calcuNMSE(full_img, rec_img)
        

            psnr_list.append(rec_psnr)
            ssim_list.append(rec_ssim)
            nmse_list.append(rec_nmse)
            ####

            rec_imgs[:,:,slice-args.slice_range[0]] = rec_img
            rec_Kspace[:,:,slice-args.slice_range[0]] = rec_k
            rec_mid_imgs[:,:,slice-args.slice_range[0]] = rec_mid_img
            ZF_img[:,:,slice-args.slice_range[0]] = zfimage
            label_data[:,:,slice-args.slice_range[0]] = full_img


        if args.visualize_images:
            logging.info("Visualizing results for image , close to continue ...")
            pltImages(ZF_img, rec_imgs, label_data,series_name =os.path.split(infile)[1], args = args, mask=mask, sliceList=[30, 40, 50, 60, 70, 90])

    rec_img_full[:,:,i*sliceNum:(i+1)*sliceNum] = rec_imgs
    rec_Kspace_full[:,:,i*sliceNum:(i+1)*sliceNum] = rec_Kspace
    rec_mid_img_full[:,:,i*sliceNum:(i+1)*sliceNum] = rec_mid_imgs
    ZF_img_full[:,:,i*sliceNum:(i+1)*sliceNum] = ZF_img
    label_data_full[:,:,i*sliceNum:(i+1)*sliceNum] = label_data

    if args.savePredication:

        os.makedirs(args.predictResultPath, exist_ok=True)
        filename = os.path.join(args.predictResultPath,  args.predResultName)
        saveData(filename, zf_imgs=ZF_img_full, rec_imgs=rec_img_full, rec_kspaces= rec_Kspace_full, label_imgs= label_data_full)
        logging.info("reconstructions save to: {}".format(filename))

    ## calculate the avg value
    psnrarray = np.array(psnr_list)
    ssimarray = np.array(ssim_list)
    nmsearray = np.array(nmse_list)

    avg_psnr = psnrarray.mean()
    var_psnr = psnrarray.var()
    max_psnr = psnrarray.max()
    min_psnr = psnrarray.min()

    avg_ssim = ssimarray.mean()
    var_ssim = ssimarray.var()
    max_ssim = ssimarray.max()
    min_ssim = ssimarray.min()

    avg_nmse = nmsearray.mean()
    var_nmse = nmsearray.var()
    max_nmse = nmsearray.max()
    min_nmse = nmsearray.min()


    print("max_psnr:", max_psnr, "min_psnr:", min_psnr,  "avg_psnr:",avg_psnr, "var_psnr:", var_psnr,
        "max_ssim:", max_ssim, "min_ssim:", min_ssim, "avg_ssim", avg_ssim, "var_ssim:", var_ssim,
        "max_nmse:", max_nmse, "min_nmse:", min_nmse, "avg_nmse", avg_nmse, "var_nmse:", var_nmse)

#### the multi slices test
def test_Kdomain_mutiSlices2(args, model):
    #load the model
    model = model.to(args.device)
    checkPoint = torch.load(args.predModelPath, map_location = args.device)
    model.load_state_dict(checkPoint['G_model_state_dict'])

    ## into the evalue mode
    model.eval()   #no backforward
    logging.info("Model loaded !")
    ## prepare the three slices masks
    mask_path = args.mask_path
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    mask0 = np.fft.fftshift(masks_dictionary['mask0'])
    mask1 = np.fft.fftshift(masks_dictionary['mask1'])
    mask2 = np.fft.fftshift(masks_dictionary['mask2'])

    masks = np.dstack((mask0,mask1,mask2))
    maskNot = 1-mask1

    psnr_list = []
    ssim_list = [] 
    nmse_list = []
    psnr_list_zf = []

    ## get all the test file paths
    test_files = glob.glob(os.path.join(args.predictDir, '*.hdf5'))
    ## the test slices number
    sliceNum = args.slice_range[1] - args.slice_range[0]
    fileLen = len(test_files)
    #define the  final matrixs
    rec_imgs_full = np.zeros((args.imgSize,args.imgSize,fileLen*sliceNum))
    rec_Kspaces_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum), dtype=np.csingle) #complex
    rec_mid_imgs_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))
    ZF_img_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))
    target_imgs_full = np.zeros((args.imgSize, args.imgSize,fileLen*sliceNum))

    for i, infile in enumerate(test_files):

        logging.info("\nTesting image {} ...".format(infile))

        with h5py.File(infile, 'r') as f:
            # img_shape = f['data'].shape
            img_shape = f['reconstruction_rss'].shape

            # label_img = f['data'][:]

        rec_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
        rec_Kspaces = np.zeros((args.imgSize, args.imgSize,sliceNum), dtype=np.csingle) #complex
        rec_mid_imgs = np.zeros((args.imgSize, args.imgSize,sliceNum))
        ZF_img = np.zeros((args.imgSize, args.imgSize,sliceNum))
        target_imgs = np.zeros((args.imgSize, args.imgSize,sliceNum))

        for slice_num in range(args.slice_range[0], args.slice_range[1],1):
            add = int(args.num_input_slices / 2)
            #prepare the images, if it's the first image of the file the slices will be the [0,0,1], if the last images of the file,
            # the slices will be the [last-1, last-1, last], else will be the [slice-1, slice , slice+1](this is onlyefficient for 
            # the 3 slices as input's condition)
            # with h5py.File(infile, 'r') as f:
            #     if slice_num == 0:
            #         print("first slice")
            #         imgs = np.dstack((f['data'][:, :, 0], f['data'][:, :, 0], f['data'][:, :, 1]))
            #     elif slice_num == img_shape[2]-1:
            #         print("last slice")
            #         imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num]))
            #     else:
            #         imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num + 1]))
            with h5py.File(infile, 'r') as f:
                if slice_num == 0:
                    print("first slice")
                    imgs = np.dstack((f['reconstruction_rss'][0,:, :], f['reconstruction_rss'][0,:, :], f['reconstruction_rss'][1,:, :]))
                elif slice_num == img_shape[2]-1:
                    print("last slice")
                    imgs = np.dstack((f['reconstruction_rss'][slice_num-1, :, : ], f['reconstruction_rss'][slice_num, :, :], f['reconstruction_rss'][slice_num, :, :]))
                else:
                    imgs = np.dstack((f['reconstruction_rss'][slice_num-1,:, :], f['reconstruction_rss'][slice_num,:, :], f['reconstruction_rss'][slice_num + 1,:, :]))
            # the masked kspace matrix input of the net
            masked_Kspaces_np = np.zeros((args.num_input_slices * 2, args.imgSize, args.imgSize))
            # the target kspace matrix label of the kspace
            target_Kspace = np.zeros((2, args.imgSize, args.imgSize))
            # the target image, the label
            target_img = np.zeros((1, args.imgSize, args.imgSize))
            
            for slice_j in range(args.num_input_slices):
                # img = imgs[:, :, slice_j]
             
                img = regular(crop_toshape1(imgs[ :, :, slice_j]))

                # img = np.rot90(img, 1) ## rotate the image 
                kspace = fft2(img)
                # image preprocess
                slice_masked_Kspace, slice_full_Kspace, slice_target_img = multiSlice_preprocess(kspace, slice_j, masks, maskNot, args) 

                masked_Kspaces_np[slice_j * 2:slice_j * 2 + 2, :, :] = slice_masked_Kspace
                ## if it's the middle slice
                if slice_j == int(args.num_input_slices / 2):
                    target_Kspace = slice_full_Kspace
                    target_img = slice_target_img
                
            ## prepare the input tensor
            masked_Kspaces = np.expand_dims(masked_Kspaces_np, axis=0)

            masked_Kspaces = torch.from_numpy(masked_Kspaces).to(device=args.device, dtype=torch.float32)

            rec_img, rec_Kspace, rec_mid_img, att = model(masked_Kspaces)

            ## process the result squeeze the dimentions equal one
            rec_img = np.squeeze(rec_img.data.cpu().numpy())                      ##2D tensor
            rec_Kspace = np.squeeze(rec_Kspace.data.cpu().numpy())                ##3D tensor the first dim is the real and imag
            rec_Kspace = (rec_Kspace[0, :, :] + 1j*rec_Kspace[1, :, :])           ## 2D tensor complex k space

            rec_mid_img = np.squeeze(rec_mid_img.data.cpu().numpy())              ##2D tensor
            target_img = np.squeeze(target_img)

            # att = np.squeeze(att.data.cpu().numpy())
            # if slice_num == 30:
            #     print(att.shape, 'attention map size')

            #     np.save('E:/DLCode/DLFastReconstruction/PsychoGanCode/checkpoints/IXI/DA/ABLATION/DC/test/attentioni.npy',att)
            #     np.save('E:/DLCode/DLFastReconstruction/PsychoGanCode/checkpoints/IXI/DA/ABLATION/DC/test/image.npy', target_img)


            if args.testDC:
                rec_k = np.fft.fft2(rec_img)
                tar_k = np.fft.fft2(target_img)
                final_k = mask1 * tar_k + maskNot * rec_k
                final_rec = np.abs(np.fft.ifft2(final_k))
                rec_img = final_rec
            ##get zf image
            zf_img = np.squeeze(ifft2((masked_Kspaces_np[2, :, :] + 1j*masked_Kspaces_np[3, :, :])))
            zf_psnr = calculatePSNR(target_img,zf_img)

            rec_psnr = calculatePSNR(target_img,rec_img)
            rec_ssim = calculateSSIM(target_img,rec_img)
            rec_nmse = calcuNMSE(target_img, rec_img)
            # print(target_img.max(), rec_img.max(), target_img.min(), rec_img.min())
            # print(rec_nmse, 'rec_nmse')

            psnr_list.append(rec_psnr)
            ssim_list.append(rec_ssim)
            nmse_list.append(rec_nmse)
            psnr_list_zf.append(zf_psnr)

            rec_imgs[:, :, slice_num-args.slice_range[0]] = rec_img
            rec_Kspaces[:, :, slice_num-args.slice_range[0]] = rec_Kspace
            rec_mid_imgs[:, :, slice_num-args.slice_range[0]] = rec_mid_img
            ZF_img[:, :, slice_num-args.slice_range[0]] =zf_img  ## select the mid image
            target_imgs[:,:,slice_num-args.slice_range[0]] =target_img
       
        rec_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = rec_imgs
        rec_Kspaces_full[:,:,i*sliceNum:(i+1)*sliceNum] = rec_Kspaces
        rec_mid_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = rec_mid_imgs
        ZF_img_full[:,:,i*sliceNum:(i+1)*sliceNum] = ZF_img
        target_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = target_imgs


        if args.visualize_images:
            logging.info("Visualizing results for image {}, close to continue ...".format(infile))
            pltImages(ZF_img, rec_imgs, target_imgs, series_name =os.path.split(infile)[1],args = args, mask=masks[:,:,1], sliceList=[0])
        ## calculate the avg value

    if args.savePredication:
        os.makedirs(args.predictResultPath, exist_ok=True)
        out_file_name = args.predictResultPath + os.path.split(infile)[1]
        saveData(out_file_name, ZF_img_full, rec_imgs_full, None, target_imgs_full)

        logging.info("reconstructions save to: {}".format(out_file_name))

    psnrarray = np.array(psnr_list)
    ssimarray = np.array(ssim_list)
    nmsearray = np.array(nmse_list)
    psnrzfarray = np.array(psnr_list_zf)

    savearray = np.stack((psnrarray, ssimarray, nmsearray),axis=1)
    print(savearray.shape)
    n_list = range(1,len(psnr_list)+1)
    panda_array = pandas.DataFrame(savearray)
    panda_array.columns = ['PSNR', 'SSIM', 'NMSE']
    panda_array.index = n_list
    psnrwriter = pandas.ExcelWriter(args.predictResultPath+'evaluation.xlsx')
    panda_array.to_excel(psnrwriter,'page_1',float_format='%.7f')
    print('save result')
    psnrwriter.save()

    avg_psnr = psnrarray.mean()
    avg_ssim = ssimarray.mean()
    avg_nmse = nmsearray.mean()
    avg_psnr_zf = psnrzfarray.mean()

    stdnmse = np.std(nmsearray)
    stdpsnr = np.std(psnrarray)
    stdssim = np.std(ssimarray)


    print("avg_psnr:",avg_psnr ,"stdpsnr:", stdpsnr,"avg_ssim", avg_ssim, 
            "stdssim:",stdssim,  "avg_nmse", avg_nmse,"stdnmse:", stdnmse, "zfpsnr: ",avg_psnr_zf)

from KINet import * 
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    configPath = './config.yaml'
    args = getArgs(configPath)


    logging.info(f"Using device {args.device}")

    logging.info("Loading model {}".format(args.predModelPath))
    model = SepWnet(args)
    # model =WNet(args)
    test_Kdomain_mutiSlices2(args, model)
