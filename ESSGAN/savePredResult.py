import os
import numpy as np 
from scipy.io import savemat
import h5py
def saveData(outFileName, zf_imgs, rec_imgs, rec_kspaces, label_imgs):
    """
        In HWC form
    """
    # adict = {}
    # adict["ZF_imgs"] = zf_imgs
    # adict["rec_imgs"] = rec_imgs
    # adict["rec_kspzces"] = rec_kspaces
    # adict["label_imgs"] =label_imgs
    
    # savemat( os.path.join(outFileName), adict)
    f = h5py.File(outFileName, 'a')
    f.create_dataset('ZF_imgs', data = zf_imgs)
    f.create_dataset('rec_imgs', data = rec_imgs)
    f.create_dataset('rec_kspzces', data = rec_kspaces)
    f.create_dataset('label_imgs', data = label_imgs)
    f.close()