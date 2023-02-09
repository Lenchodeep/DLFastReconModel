import nibabel as nib
import numpy as np

import os
import h5py

def convert2hdf5(file_path, save_path):
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



# convert2hdf5("D:/datas/IXI002-Guys-0828-T2.nii.gz","D:/datas/" )