import numpy as np
import h5py
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from os import listdir, path
from os.path import splitext
IMG_SIZE = 256
SLICE_RANGE = [20,120]
def convertH5Tomat(data_dir, targetPath):
    fileNames = [splitext(file)[0] for file in listdir(data_dir)]
    count = 0
    imglist = []
    for filename in fileNames:
        count = count+1
        fullpath = path.join(data_dir, filename+'.hdf5')
        with h5py.File(fullpath, 'r') as f:
            image = f['data']
            # plt.figure()
            # plt.imshow(image[:,:,100], plt.cm.gray)
            # plt.show()
            imglist.append( image[:,:,SLICE_RANGE[0]: SLICE_RANGE[1]])
            # data_array[:,:,count *(SLICE_RANGE[1]-SLICE_RANGE[0]) : ((count+1)*(SLICE_RANGE[1]-SLICE_RANGE[0]))] = image[:,:,SLICE_RANGE[0]:SLICE_RANGE[1]]
    data_array = np.zeros((IMG_SIZE, IMG_SIZE, len(imglist)*100))
    array1 = imglist[0]
    for i in range(len(imglist)):
        if i ==0:
            continue
        array1 = np.concatenate((array1, imglist[i]), axis=2)
        print(array1.shape)

    for slice in range(array1.shape[2]):
        array1[:,:,slice] = np.rot90(array1[:,:,slice], 1)
        maxvalue = np.max(array1[:,:,slice])
        minvalue = np.min(array1[:,:,slice])
        k = 2/(maxvalue-minvalue)
        # array1[:,:,slice] = array1[:,:,slice]*2-1
        array1[:,:,slice] = -1+k*(array1[:,:,slice]-minvalue)

    f = h5py.File(targetPath, 'a')
    f.create_dataset('data', data = array1)
    f.close()

# convertH5Tomat(r'D:\datas\IXI-T1-save\train', r'D:/datas/IXI-T1-save/train/IXItrain_minus.hdf5')

# full_file_path = "D:/datas/IXI-1-save/train/IXItrain2.hdf5"
# with h5py.File(full_file_path, 'r') as f:
#     image = f['data'][:]  

# print(image.shape)
# slices=[10,20,30,40,50,110,120,130,140,150]
# for slice in slices:
#     imagetmp = image[:,:,slice]
#     # imagetmp = (imagetmp+1)/2
#     imageK = np.fft.fft2(imagetmp)

#     # imageK = imageK+1
#     image1 = np.fft.ifft2(imageK)
#     print("value range: ", imagetmp.min(), imagetmp.max(), imageK.min(), imageK.max())
#     plt.figure()
#     plt.imshow(imagetmp, plt.cm.gray)
#     plt.figure()
#     plt.imshow(abs(image1), plt.cm.gray)

# plt.show()

