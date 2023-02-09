import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import numpy as np
from torch.utils.data import DataLoader
import glob
import h5py
import pickle
import yaml
from types import SimpleNamespace
import time

from model import *
from dataset import IXIDatasetRefinrGan
from utils import *
def crop_toshape( image):
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
def Test(args,model):

    ## load model
    model = model.to(args.device)
    checkPoint = torch.load(args.predModelPath, map_location = args.device)
    model.load_state_dict(checkPoint['G_model_state_dict'])
    ## into the evalue mode
    model.eval()   #no backforwardll
    logging.info("Model loaded !")

    psnr_list = []
    ssim_list = [] 
    nmse_list = []
    mid_psnr_list = []
    mid_ssim_list = [] 
    mid_nmse_list = []

    mask_path = args.mask_path
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    mask = torch.fft.fftshift(torch.from_numpy(masks_dictionary['mask1']))
    maskNot = 1-mask

    sliceNum = args.slice_range[1] - args.slice_range[0]

    test_files = glob.glob(os.path.join(args.predictDir, '*.hdf5')) 
    with torch.no_grad():
        for i, infile in enumerate(test_files):
            logging.info("\nTesting image {} ...".format(infile))
            with h5py.File(infile, 'r') as f:
                img_shape = f['data'].shape
                target_img = f['data'][:] ## for ixi and brats
                # target_img = f['reconstruction_rss'][:]## for fast mri

                rec_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
                rec_mid_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
                ZF_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
                label_imgs = np.zeros((args.imgSize,args.imgSize,sliceNum))
                

                for slice_num in range(args.slice_range[0],args.slice_range[1],1):
                    img = target_img[:,:,slice_num] ## for IXI and  brats
                    # img =regular(crop_toshape(target_img[slice_num,:,:])) ## for fast mri

                    img = np.rot90(img, 1)   ##for ixi and brats
                    img1 = img
                    ## prepare input
                    img = img[..., np.newaxis]
                    img = np.transpose(img, (2,0,1))
                    if args.isZeroToOne:
                        img = (img[0] - (0 + 0j)) 
                    else:
                        img = (img[0]-(0.5+0.5j))*2
                    tarimg = torch.from_numpy(img)
                    ZF_img, k_und, k_A = undersample(tarimg, mask)

                    tarimg = torch.unsqueeze(torch.view_as_real(tarimg).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                    ZF_img = torch.unsqueeze(torch.view_as_real(ZF_img).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype= torch.float32)
                    k_und = torch.unsqueeze(torch.view_as_real(k_und).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype= torch.float32)
                    in_mask = torch.unsqueeze(torch.view_as_real(mask * (1. + 1.j)).permute(2, 0, 1).contiguous(), dim=0).to(args.device, dtype=torch.float32)
                    Sp1, S1, Tp1, T1 = model(ZF_img, k_und, in_mask)
                    T1 = torch.squeeze(output2complex(T1, args.isZeroToOne).abs())
                    S1 = torch.squeeze(output2complex(S1, args.isZeroToOne).abs())
                    ZF_img = torch.squeeze(output2complex(ZF_img, args.isZeroToOne).abs())
                    tarimg = torch.squeeze(output2complex(tarimg, args.isZeroToOne).abs())
                    ## convert to numpy
                    rec_img = T1.detach().cpu().numpy()
                    rec_mid_img = S1.detach().cpu().numpy()
                    zf_img = ZF_img.detach().cpu().numpy()
                    label_img = tarimg.detach().cpu().numpy()

                    ## final evaluation
                    rec_ssim = calculateSSIM(img1, rec_img, False)
                    rec_psnr = calculatePSNR(img1, rec_img, False)
                    rec_nmse = calcuNMSE(img1, rec_img, False)

                    psnr_list.append(rec_psnr)
                    ssim_list.append(rec_ssim)
                    nmse_list.append(rec_nmse)
                    ## mid evaluations
                    mid_rec_ssim = calculateSSIM(img1, rec_mid_img, False)
                    mid_rec_psnr = calculatePSNR(img1, rec_mid_img, False)
                    mid_rec_nmse = calcuNMSE(img1, rec_mid_img, False)

                    mid_psnr_list.append(mid_rec_psnr)
                    mid_ssim_list.append(mid_rec_ssim)
                    mid_nmse_list.append(mid_rec_nmse)

                    ##save rec  image:
                    # imsave(args.predictResultPath+"rec_"+str(slice_num)+".png",np.squeeze(rec_img))
                    # imsave(args.predictResultPath+"rec_"+str(slice_num)+".png",np.squeeze(rec_mid_img))

                    rec_imgs[:,:,slice_num-args.slice_range[0]] = rec_img
                    rec_mid_imgs[:,:,slice_num-args.slice_range[0]] = rec_mid_img
                    ZF_imgs[:,:,slice_num-args.slice_range[0]] = zf_img
                    label_imgs[:,:,slice_num-args.slice_range[0]] = img
                
                if args.visualize_images:
                    logging.info("Visualizing results for image , close to continue ...")
                    pltImages(ZF_imgs, rec_imgs, rec_mid_imgs, label_imgs,series_name =os.path.split(infile)[1], args = args, sliceList=[0])

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
    midpsnrarray = np.array(mid_psnr_list)
    midssimarray = np.array(mid_ssim_list)
    midnmsearray = np.array(mid_nmse_list)

    mid_avg_psnr = midpsnrarray.mean()
    mid_std_psnr = midpsnrarray.std()
  

    mid_avg_ssim = midssimarray.mean()
    mid_std_ssim = midssimarray.std()


    mid_avg_nmse = midnmsearray.mean()
    mid_std_nmse = midnmsearray.std()

    print( "mid_avg_psnr:",mid_avg_psnr, "mid_var_psnr:", mid_std_psnr,
        "mid_avg_ssim", mid_avg_ssim, "mid_var_ssim:", mid_std_ssim,
        "mid_avg_nmse", mid_avg_nmse, "mid_var_nmse:", mid_std_nmse)





def getArgs(configPath):
    with open(configPath) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)    
    args = SimpleNamespace(**data)

    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    configPath = './config.yaml'
    args = getArgs(configPath)


    logging.info(f"Using device {args.device}")

    logging.info("Loading model {}".format(args.predModelPath))
    model = Refine_G(args)
    Test(args, model)

