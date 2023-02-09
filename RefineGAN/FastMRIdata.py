import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import splitext
from os import listdir, path

filepath='F:/data/'
file_names = [splitext(file)[0] for file in listdir(filepath)]
print(file_names)
for filename in file_names:
    full_file_path = path.join(filepath,filename + '.h5')
    with h5py.File(full_file_path, 'r') as f:
        labels = f['reconstruction_rss']
        zf_imgs = f['reconstruction_esc']
        label = labels[15,:,:]
        zf_img = zf_imgs[15,:,:]

        
        plt.figure()
        plt.imshow(label, plt.cm.gray)
        plt.title('label')
        plt.savefig(full_file_path+'.png')

    # plt.show()
    # print(f.keys())

