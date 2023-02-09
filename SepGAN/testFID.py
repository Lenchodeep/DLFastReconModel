import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from os.path import splitext
from os import listdir, path
import glob
from Test import *

import numpy as np
import h5py
from utils import *
import  matplotlib.pyplot as plt
from types import SimpleNamespace
import pickle
import logging
from models import *

def fft2(image):
    return (np.fft.fft2(image))
def ifft2( kspace_cplx):
    return np.absolute(np.fft.ifft2((kspace_cplx)))[None, :, :]


def generateImg():
    oriPath = "D:/mytest/IXI002-Guys-0828-T1.hdf5"
    mask_path = "D:/datas/masks/radial/radial_30.pickle"
    targetpath = "D:/mytest/"
    labelsavepath = targetpath+"label/"
    zfsavepath = targetpath+"zfradial30/"
    if not os.path.exists(labelsavepath):
        os.makedirs(labelsavepath)

    if not os.path.exists(zfsavepath):
        os.makedirs(zfsavepath)

    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)

    mask1 = masks_dictionary['mask1']
    # mask1 = np.fft.fftshift(np.load(args.mask_path))
    maskNot = 1-mask1

    with h5py.File(oriPath, 'r') as f:
        data =f['data'] 


        num_slice = data.shape[2]
        for slice in range(20,120):
            image = data[:,:,slice]
            image = np.rot90(image,1)
            kspace = np.fft.fftshift(fft2(image))     
            masked_kspace = kspace*mask1

            label = np.absolute(np.fft.ifft2(np.fft.fftshift(kspace)))[None, :, :]
            zf_img = np.absolute(np.fft.ifft2(np.fft.fftshift(masked_kspace)))[None, :, :]
            imsave(labelsavepath+"label_"+str(slice)+".png",np.squeeze(label))
            imsave(zfsavepath+"zf_"+str(slice)+".png",np.squeeze(zf_img))


def generateTestImg(recsavepath):
    '''
        测试并且保存图像
    '''
    ##加载模型
    configPath = './config.yaml'
    args = getArgs(configPath)

    logging.info(f"Using device {args.device}")

    logging.info("Loading model {}".format(args.predModelPath))
    model = SepWnet(args)

    model = model.to(args.device)
    checkPoint = torch.load(args.predModelPath, map_location = args.device)
    model.load_state_dict(checkPoint['G_model_state_dict'])

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

    test_files = glob.glob(os.path.join(args.predictDir, '*.hdf5'))
    sliceNum = args.slice_range[1] - args.slice_range[0]

    for i, infile in enumerate(test_files):

        logging.info("\nTesting image {} ...".format(infile))
        with h5py.File(infile, 'r') as f:
            img_shape = f['data'].shape

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
            with h5py.File(infile, 'r') as f:
                if slice_num == 0:
                    print("first slice")
                    imgs = np.dstack((f['data'][:, :, 0], f['data'][:, :, 0], f['data'][:, :, 1]))
                elif slice_num == img_shape[2]-1:
                    print("last slice")
                    imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num]))
                else:
                    imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num + 1]))
            # the masked kspace matrix input of the net
            masked_Kspaces_np = np.zeros((args.num_input_slices * 2, args.imgSize, args.imgSize))
            # the target kspace matrix label of the kspace
            target_Kspace = np.zeros((2, args.imgSize, args.imgSize))
            # the target image, the label
            target_img = np.zeros((1, args.imgSize, args.imgSize))
            
            for slice_j in range(args.num_input_slices):
                img = imgs[:, :, slice_j]
                img = np.rot90(img, 1) ## rotate the image 
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

            if args.testDC:
                rec_k = np.fft.fft2(rec_img)
                tar_k = np.fft.fft2(target_img)
                final_k = mask1 * tar_k + maskNot * rec_k
                final_rec = np.abs(np.fft.ifft2(final_k))
                rec_img = final_rec

            tmp = (np.squeeze(target_img) - np.squeeze(rec_img))*10
            # print(np.max(tmp))
            # tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            imsave(recsavepath+"rec_"+str(slice_num)+".png",np.squeeze(rec_img))
            # plt.figure()
            # plt.imshow(tmp, vmin=0, vmax=1, cmap=plt.get_cmap('jet'))
            # plt.colorbar()
            # plt.show()
            # print(max(abs(img - rec_img)))
            
##计算FID的值
def calculateFID(dataFolder1,dataFolder2):
    '''
    环境:安装pytorch-fid
    datafolder1:存储图像文件夹1(jpg)
    datafolder2:存储图像文件夹2(jpg)
    两文件夹顺序没有要求,两文件夹的内容需要对应
    返回值:计算的FID值
    '''
    score = os.popen("{} -m pytorch_fid {} {} --num-workers 0 --device cpu".format(
        sys.executable,
        dataFolder1,
        dataFolder2)).read()
    return score


if __name__ == "__main__":
    # generateImg()

    # recsaveFolder = "D:/mytest/ablation/SepGAN-G/"
    # if not os.path.exists(recsaveFolder):
    #     os.makedirs(recsaveFolder)
    # generateTestImg(recsaveFolder)

    labelFolder = "D:/mytest/comparison/label/"
    # zfFolder = "D:/mytest/zfrec50/"
    # myreconFolder = "D:/mytest/myRec50/"
    # daReconFolder = "D:/mytest/DARec50/"
    # refineReconFolder = "D:/mytest/RefineRec50/"
    # reconReconFolder = "D:/mytest/RefineRec50Mid/"
    ablationFolder = "D:/mytest/ablation/SepGAN-G/"

    # recFID = calculateFID(labelFolder,myreconFolder)
    # zfFID = calculateFID(labelFolder,zfFolder)
    # DArecFID = calculateFID(labelFolder,daReconFolder)
    # refinerecFID = calculateFID(labelFolder,refineReconFolder)
    # reconFID = calculateFID(labelFolder,reconReconFolder)
    ablationFID = calculateFID(labelFolder,ablationFolder)

    # print("reconstruction FID: ", recFID)
    # print("refineReconstruction FID: ", refinerecFID)
    # print("reconReconstruction FID: ", reconFID)
    # print("DAReconstruction FID: ", DArecFID)
    # print("zeroFilled FID: ", zfFID)
    print("ABLATION FID: ", ablationFID)

