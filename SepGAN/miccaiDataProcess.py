import glob
import nibabel as nib
import os
import h5py
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
# nii_path = 'D:/datas/MICCAI18/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_280_1/'
# save_path = 'D:/datas/MICCAI18/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_280_1/'
nii_path = 'D:\datas\MICCAI18\MICCAI_BraTS_2018_Data_Training\HGG\Brats18_TCIA05_396_1/'
save_path = 'D:\datas\MICCAI18\MICCAI_BraTS_2018_Data_Training\HGG\Brats18_TCIA05_396_1/'
def convert2hdf5(file_path):
    try:
        print(file_path)
        # read:
        data = nib.load(file_path).get_data()
        # Norm data:
        data = (data - data.min()) / (data.max() - data.min()).astype(np.float32)
        # save hdf5:
        data_shape = data.shape
        patient_name = os.path.split(file_path)[1].replace('nii.gz', 'hdf5')
        output_file_path = save_path + patient_name
        with h5py.File(output_file_path, 'w') as f:
            dset = f.create_dataset('data', data_shape, data=data, compression="gzip", compression_opts=9)
    except:
        print(file_path, ' Error!')

if __name__=='__main__':
    data_list = glob.glob(nii_path + '*.nii.gz')
    os.makedirs(save_path, exist_ok=True)

    P = Pool(40)
    P.map(convert2hdf5, data_list)
    full_file_path = save_path+'Brats18_TCIA05_396_1_T2.hdf5'
    with h5py.File(full_file_path, 'r') as f:
        imgs = f['data']
        print(imgs.shape)
        # image = imgs[:,:,75]
        # image1 = np.zeros((256,256))
        # image1[7:247,7:247] = image

        for i in range(80,81):
            image1 = np.zeros((256,256))
            image1[7:247,7:247] = imgs[:,:,i]
            plt.figure()
            plt.imshow(image1 , plt.cm.gray)
            plt.show() 

