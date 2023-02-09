import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import glob
import h5py
import pickle
import yaml

from types import SimpleNamespace
from model import *
from utils import *



def test():
    mask_path = "D:/datas/masks/gaussian/mask_50_256.pickle" 
    predictDir = "D:/datas/bwq2/"  
    psnr_list1 = []
    psnr_list2 = []
    psnr_list3 = []
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    tmp_mask = np.fft.fftshift(masks_dictionary['mask1'])

    mask = torch.fft.fftshift(torch.from_numpy(masks_dictionary['mask1']))
    test_files = glob.glob(os.path.join(predictDir, '*.hdf5')) 
    for i, infile in enumerate(test_files):
        with h5py.File(infile, 'r') as f:
            target_img = f['data'][:]

            rec_imgs = np.zeros((256,256,100))
            rec_mid_imgs = np.zeros((256,256,100))
            ZF_imgs = np.zeros((256,256,100))
            label_imgs = np.zeros((256,256,100))

            for slice_num in range(20,120,1):
                label = target_img[:,:,slice_num]
                label = np.rot90(label,1)
                zf_tmp = myundersample(label,tmp_mask)
                label_tmp = label
                label_tmp = label_tmp[..., np.newaxis]
                label_tmp = np.transpose(label_tmp, (2,0,1))
                label_tmp = (label_tmp[0] - (0 + 0j)) 
     
                tarimg = torch.from_numpy(label_tmp)
                ZF_img, k_und, k_A = undersample(tarimg, mask)
                
                tarimg = torch.unsqueeze(torch.view_as_real(tarimg).permute(2, 0, 1).contiguous(), dim=0).to("cuda", dtype=torch.float32)
                ZF_img = torch.unsqueeze(torch.view_as_real(ZF_img).permute(2, 0, 1).contiguous(), dim=0).to("cuda", dtype= torch.float32)
                k_und = torch.unsqueeze(torch.view_as_real(k_und).permute(2, 0, 1).contiguous(), dim=0).to("cuda", dtype= torch.float32)


                ZF_img = torch.squeeze(output2complex(ZF_img, 1).abs())
                tarimg = torch.squeeze(output2complex(tarimg, 1).abs())

                zf_img = ZF_img.detach().cpu().numpy()
                label_img = tarimg.detach().cpu().numpy()
                # plt.figure()
                # plt.imshow(label_img, plt.cm.gray)
                # plt.title("after")
                # plt.figure()
                # plt.imshow(label, plt.cm.gray)
                # plt.title("ori")
                # plt.figure()
                # plt.imshow(label_img-label, plt.cm.gray)
                # plt.title("sub")
                # plt.show()    
                zf_psnr1 = calculatePSNR(label_img, zf_img, False)
                zf_psnr2 = calculatePSNR(label, zf_img, False)
                zf_psnr3 = calculatePSNR(label,zf_tmp, False)
                psnr_list1.append(zf_psnr1)
                psnr_list2.append(zf_psnr2)
                psnr_list3.append(zf_psnr3)

    avg_psnr1 = np.array(psnr_list1).mean()
    avg_psnr2 = np.array(psnr_list2).mean()
    avg_psnr3 = np.array(psnr_list3).mean()
    print("psnr1", avg_psnr1, "psnr2", avg_psnr2, "psnr3", avg_psnr3)

def myundersample(image, mask):
    k = np.fft.fft2(image)
    u_k = mask*k
    img = np.fft.ifft2(u_k)
    return abs(img)

def fft2(image):
    return np.fft.fftshift(np.fft.fft2(image))
def ifft2( kspace_cplx):
    return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace_cplx)))[None, :, :]

def ZF_evaluation():
    mask_path = "D:/datas/masks/gaussian/mask_50_256.pickle" 
    predictDir = "D:/datas/bwq2/"      
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    mask1 = masks_dictionary['mask1']
    # mask1 = np.fft.fftshift(np.load(args.mask_path))
    maskNot = 1-mask1

    zfpsnr = list()
    zfssim = list()
    zfnmse = list()
    count = 0
    test_files = glob.glob(os.path.join(predictDir, '*.hdf5')) 

    for filename in test_files:

        with h5py.File(filename, 'r') as f:
            data =f['data'] 
            num_slice = data.shape[2]

            count = count+1
            for slice in range(20, 120):
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

# ZF_evaluation()

test()