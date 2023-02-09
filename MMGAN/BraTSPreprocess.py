import os
import h5py
import glob
import random
import shutil

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

from multiprocessing import Pool


def convert2hdf5(file_path,tarPath):
    try:
        print(file_path)
        # read:
        data = nib.load(file_path).get_data()
        # Norm data:
        data = (data - data.min()) / (data.max() - data.min()).astype(np.float32)
        # save hdf5:
        data_shape = data.shape
        patient_name = os.path.split(file_path)[1].replace('nii.gz', 'hdf5')
        output_file_path = tarPath + patient_name
        with h5py.File(output_file_path, 'w') as f:
            dset = f.create_dataset('data', data_shape, data=data, compression="gzip", compression_opts=9)
    except:
        print(file_path, ' Error!')

def convertProcess(originalPath, tarpath):
    paths = os.listdir(originalPath)
    for path in paths:
        data_list = glob.glob(os.path.join(originalPath, path)+"/" + '*flair*.nii.gz')
        for data in data_list:
            convert2hdf5(data , tarpath)


def randomSplit(t1filepath, t2filepath , num, targetpath1, targetpath2):
    if not os.path.exists(targetpath1):
        os.makedirs(targetpath1)
    if not os.path.exists(targetpath2):
        os.makedirs(targetpath2)
    paths = os.listdir(t1filepath)
    selectedFiles = random.sample(paths,k=num)
    list = []
    for file in selectedFiles:
        t2file = file.replace("t1ce","t2")
        src = os.path.join(t1filepath, file)
        src2 = os.path.join(t2filepath, t2file)

        shutil.move(src,targetpath1)
        shutil.move(src2,targetpath2)
        list.append([file,t2file])
    
    df1 = pd.DataFrame(data=list, columns=['T1CE', 'T2'])
    df1.to_csv(targetpath1+'T1andT2Val.csv')

# path = "D:/datas/MICCAI18/MICCAI18/T1ce/"
# tarpath = "D:/datas/MICCAI18/MICCAI18/T1CEtrain/"
# path2 = "D:/datas/MICCAI18/MICCAI18/T2/"
# tarpath2 = "D:/datas/MICCAI18/MICCAI18/T2train/"
# randomSplit(path, path2, 147, tarpath, tarpath2)


