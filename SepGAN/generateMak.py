import numpy as np
import matplotlib.pyplot as plt

SLICE_HEIGHT = 256
SLICE_WIDTH = 256
def gaussian1d(factor,direction = "column",low_freq_lines = 31):
    if direction != 'column':
        img_shape = (SLICE_HEIGHT , SLICE_WIDTH)

    center = np.array([1.0 * SLICE_HEIGHT/ 2 -0.5 ])
    cov = np.array([[(1.0 * SLICE_HEIGHT / 4) ** 2]])

    factor = int(factor * SLICE_HEIGHT)
    samples = np.array([0])

    m = 1
    while(samples.shape[0] < factor):
        samples = np.random.multivariate_normal(center , cov , m*factor)
        samples = np.rint(samples).astype(int)                        #四舍五入取整
        indexes = np.logical_and(samples >= 0 , samples<SLICE_HEIGHT) #取有效部分的值
        samples = samples[indexes]
        samples = np.unique(samples)                                  #去除重复数字，并进行排序
        if samples.shape[0] < factor:
            m *= 2
            continue
    
    
    indexes = np.arange(samples.shape[0], dtype=int)
    np.random.shuffle(indexes)
    samples = samples[indexes][:factor]
    under_pattern = np.zeros((SLICE_WIDTH , SLICE_HEIGHT), dtype=bool)
    under_pattern[:, samples] = 1

    start = int(SLICE_HEIGHT//2 - low_freq_lines//2)
    end = int(SLICE_HEIGHT // 2 + low_freq_lines //2)

    for i in range(start , end):
        under_pattern[:,i] = 1

    if direction != "column":
        under_pattern = under_pattern.T

    return under_pattern


# pattern = gaussian1d(0.35)
# sump = np.sum(pattern)
# factor = sump / (256*256)
# print(factor)
# plt.figure()
# plt.imshow(pattern, plt.cm.gray)
# plt.show()

# np.save('E:/DLCode/DLFastReconstruction/gaussian40.npy',pattern)
import pickle
mask = np.load('E:/DLCode/DLFastReconstruction/gaussian40.npy')
mask = mask.astype(int)
print(mask.dtype)

plt.figure()
plt.imshow(mask, plt.cm.gray)
plt.show()

dict = {'mask0':mask, 'mask1':mask, 'mask2': mask}


mask_out_path = 'D:/datas/masks/gaussian/mask_40_256.pickle'

with open(mask_out_path, 'wb') as f:
    pickle.dump(dict, f)

# with open(mask_out_path, 'rb')as pickle_file:
#     masks = pickle.load(pickle_file)

# mask0 = masks['mask0']
# print(mask0.dtype)
# plt.figure()
# plt.imshow(mask0, plt.cm.gray)
# plt.show()