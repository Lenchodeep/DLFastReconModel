import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import nibabel as nib


# sourcefolder = "E:/FastMRI/train/"
# tarfolder = "E:/FastMRI/val/"
# data = pd.read_csv("E:\DLCode\DLFastReconstruction\MMGAN\data\FastMRI/tmp.csv") #读取文件中所有数据
# # 按列分离数据
# PDWIList = data['PDWI']#读取某两列
# FSPDWIList = data['FSPDWI']#读取某一列


# for itemname in PDWIList:
#     sourcepath = sourcefolder+itemname+'.hdf5'
#     targetpath = tarfolder
#     shutil.move(sourcepath, targetpath + itemname+'.hdf5')          # 移动文件



import pickle

path1 = "D:/datas/IXI-T1-save/train/IXI012-HH-1211-T1.hdf5"
mask_path = "D:/datas/masks/gaussian/mask_30_256.pickle"
with open(mask_path, 'rb') as pickle_file:
    masks = pickle.load(pickle_file)
mask = masks['mask1']
mask = np.array(mask,np.bool8)

with h5py.File(path1, 'r') as f:
    images = f['data']
    img_shape = images.shape

    print(img_shape)
    img = images[:,:,90]
    ksapce = np.fft.fftshift(np.fft.fft2(img))
    mask_k = ksapce*mask

    plt.figure()
    plt.imshow(img,plt.cm.gray)
    plt.figure()
    plt.imshow(mask,plt.cm.gray)
    plt.figure()
    plt.imshow(np.log(np.abs(ksapce)), plt.cm.gray)
    plt.figure()
    plt.imshow(np.log(np.abs(mask_k)), plt.cm.gray)
    plt.show()
# with h5py.File(path2, 'r') as f:
#     images = f['data']
#     img_shape = images.shape

#     print(img_shape)

#     plt.figure()
#     plt.imshow(images[:,:,50],plt.cm.gray)
#     plt.show()
# from skimage.io import imsave
# mask_path = "D:/datas/masks/gaussian/mask_50_256.pickle"
# tar_path = "D:/Desktop/study/文章/图片/"
# with open(mask_path, 'rb') as pickle_file:
#     masks = pickle.load(pickle_file)
# mask = masks['mask1']
# mask = np.array(mask, np.bool8)
# plt.figure()
# plt.imshow(mask,plt.cm.gray)
# plt.show()
# imsave(tar_path+"gaussian50.png",mask)