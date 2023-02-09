import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def main():

    maskpath1 = 'D:/datas/masks/radial/mask_radial50-1.mat'
    maskpath0 = 'D:/datas/masks/radial/mask_radial50-0.mat'
    maskpath2 = 'D:/datas/masks/radial/mask_radial0-2.mat'

    mask0 = np.array(loadmat(maskpath0)['mask_matrix'])
    mask1 = np.array(loadmat(maskpath1)['mask_matrix'])
    mask2 = np.array(loadmat(maskpath2)['mask_matrix'])
    dict = {'mask0':mask0, 'mask1':mask1, 'mask2': mask2}


    mask_out_path = 'D:/datas/masks/radial/radial_50.pickle'

    with open(mask_out_path, 'wb') as f:
        pickle.dump(dict, f)


    # with open(mask_out_path, 'rb')as pickle_file:
    #     masks = pickle.load(pickle_file)
    # mask0 = masks['mask0']
    # mask1 = masks['mask1']
    # mask2 = masks['mask2']

    # mask0 = np.fft.fftshift(mask0)
    # mask1 = np.fft.fftshift(mask1)
    # mask2 = np.fft.fftshift(mask2)
    plt.figure()
    plt.imshow(mask0, plt.cm.gray)
    plt.title('mask0')
    plt.figure()
    plt.imshow(mask1, plt.cm.gray)
    plt.title('mask1')
    plt.figure()
    plt.imshow(mask2, plt.cm.gray)
    plt.title('mask2')
    plt.show()
if __name__ =='__main__':

    main()
