import numpy as np
import matplotlib.pyplot as plt
import os

path = "D:/datas/masks/spiral/"
files = os.listdir(path)
for file in files:
        
    mask = np.load(path+file)
    mask = mask/ np.max(mask)
    np.save(path+file , mask )
    plt.figure()
    plt.imshow(mask, plt.cm.gray)
plt.show()
 
