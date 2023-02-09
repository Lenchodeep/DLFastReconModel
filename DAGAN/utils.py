import logging
import os.path
import numpy as np
import torch
import torch.fft
import torchvision.transforms as transforms
from math import ceil
from skimage.io import imsave
from skimage.io import imread
from skimage.draw import  rectangle_perimeter
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity 
from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import  mean_squared_error 
# Early stopping
class EarlyStopping:
    def __init__(self , patience , savepath, cpname, verbose = False , delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False        
        self.val_loss_min = np.Inf
        self.delta = delta
        self.cpname = cpname
        self.savepath = savepath
    def __call__(self , val_loss , model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.savepath+self.cpname+'.pth')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

# Data Augment
class DataAugment:
    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0)
        ])

    def __call__(self, x):
        x = torch.div(torch.add(x, torch.ones_like(x)), 2)
        x = self.transform_train(x)
        x = torch.sub(torch.mul(x, 2), torch.ones_like(x))

        return x
        
def to_real_image(image):
    return torch.div(torch.add(image, torch.ones_like(image)), 2)

def image_regular(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

# Fourier Transform
def fft_abs_for_map_fn(x):
    x = torch.div(torch.add(x, torch.ones_like(x)), 2)
    fft_x = torch.fft.fftn(x)
    fft_abs = torch.abs(fft_x)

    return fft_abs


def calculateSSIM(tar, rec, isBatch = False):
    """
        the input should be the numpy array and return the sum of the all batch if the isbatch is true
    """
    ssimlist = []
    x_good = np.squeeze(tar)
    x_bad = np.squeeze(rec)
    if isBatch:
        for i in range(x_good.shape[0]):
            label = image_regular(x_good[i,:,:])
            pred = image_regular(x_bad[i,:,:])
            ssimlist.append(structural_similarity(label, pred, data_range=pred.max()))
    else:
        label = image_regular(tar)
        pred = image_regular(rec)
        ssimlist.append(structural_similarity(label, pred,data_range=pred.max()))
    return sum(ssimlist)



def calculatePSNR(tar, rec, isBatch = False):
    """
        the input should be the numpy array and return the sum of the all batch if the isbatch is true
    """
    ssimlist = []
    x_good = np.squeeze(tar)
    x_bad = np.squeeze(rec)
    if isBatch:
        for i in range(x_good.shape[0]):
            label = image_regular(x_good[i,:,:])
            pred = image_regular(x_bad[i,:,:])
            ssimlist.append(peak_signal_noise_ratio(label, pred, data_range=pred.max()))
    else:
        label = image_regular(tar)
        pred = image_regular(rec)
        ssimlist.append(peak_signal_noise_ratio(label, pred, data_range=pred.max()))
    return sum(ssimlist)


def nmse(X_good, X_bad):
    X_good = image_regular(np.squeeze(X_good))
    X_bad = image_regular(np.squeeze(X_bad))
    nmsea = mean_squared_error(X_bad, X_good)
    nmseb = mean_squared_error(X_bad, np.zeros_like(X_bad))
    return nmsea/ nmseb

# Preparation for VGG
class VGG_PRE:
    def __init__(self, device):
        self.transform_vgg = transforms.Compose([transforms.Resize((244, 244))])
        self.device = device

    def __call__(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = torch.mul(torch.add(x, torch.ones_like(x)), 127.5)
        mean = torch.from_numpy(np.array([123.68, 116.779, 103.939], dtype=np.float32)
                                .reshape((1, 3, 1, 1)))
        x = torch.sub(x, mean.to(self.device))
        x = self.transform_vgg(x)

        return x


# Preparation for UNet
class PREPROCESS:
    def __init__(self):
        pass

    def __call__(self, x):
        x = x.permute(0, 3, 1, 2)

        h_padding = 256 - x.shape[2]
        w_padding = 256 - x.shape[3]
        if h_padding > 0:
            h_padding_t = ceil(h_padding / 2)  # 128 + ceil(x.shape[2]/2)
            h_padding_b = h_padding - h_padding_t  # 128 - ceil(x.shape[2]/2) - x.shape[2]
            h_cutting_t = 0
            h_cutting_b = 256
        else:
            h_padding_t = 0
            h_padding_b = 0
            h_cutting_t = ceil(x.shape[2] / 2) - 128
            h_cutting_b = ceil(x.shape[2] / 2) + 128
        if w_padding > 0:
            w_padding_l = ceil(w_padding / 2)  # 128 + ceil(x.shape[3]/2)
            w_padding_r = w_padding - w_padding_l  # 128 - ceil(x.shape[3]/2) - x.shape[3]
            w_cutting_t = 0
            w_cutting_b = 256
        else:
            w_padding_l = 0
            w_padding_r = 0
            w_cutting_t = ceil(x.shape[3] / 2) - 128
            w_cutting_b = ceil(x.shape[3] / 2) + 128

        constant_padding = torch.nn.ConstantPad2d((w_padding_l, w_padding_r, h_padding_t, h_padding_b), -1)
        x = constant_padding(x)
        x = x[:, :, h_cutting_t:h_cutting_b, w_cutting_t:w_cutting_b]

        x = x.permute(0, 2, 3, 1)

        return x

### plot image part
def pltImages(ZF_imgs, rec_imgs, label_imgs, series_name,isDc = True, sliceList = None, args = None, mask = None, savepng = True):
    '''
        To plot the reconstruction image, original image, fold image and the sub image 
    '''
   
    plt.figure()
    if sliceList is None:
        slices = [30, 40, 50, 60, 70, 90]
    else:
        slices = sliceList

    for slice in slices:
        label = label_imgs[:, :, slice]
        reconImg = rec_imgs[:,:,slice]
        ZF_img = ZF_imgs[:,:,slice]

        if isDc:
        ### Kdc operation now we just add the original K to the sampled points
            # mask = np.load('D:/datas/masks/radial/radial_30.npy')
            masknot = 1-mask

            reconK = np.fft.fftshift(np.fft.fft2(reconImg))
            labelK = np.fft.fftshift(np.fft.fft2(label))
            mask_recon = masknot * reconK
            mask_label = mask * labelK

            final_K = mask_recon + mask_label
            final_recon = np.abs(np.fft.ifft2(np.fft.fftshift(final_K))) 
        else:
            final_recon = reconImg

        diff_image = 10*(label - final_recon)

        predictSsim = round(calculateSSIM(label, final_recon), 3) #保留三位小数
        predictPsnr = round(calculatePSNR(label, final_recon), 3)
        predictMSE = round(nmse(label,final_recon), 5)
        ## plot the images

        fig, ax = plt.subplots(1, 4, figsize=(40, 10))
        plt.subplots_adjust(hspace=0, wspace=0)

        ax[0].set_title('Label Image', fontsize=30)
        ax[0].imshow(label, vmin=0,
                     vmax=1, cmap=plt.get_cmap('gray'))

        ax[1].set_title('Recon Image', fontsize=30)
        ax[1].imshow(reconImg, vmin=0,
                     vmax=1, cmap=plt.get_cmap('gray'))

        ax[2].set_title('diff', fontsize=30)
        ax[2].imshow(diff_image, vmin=0, vmax=1, cmap=plt.get_cmap('jet'))

        ax[3].set_title('recon Mid Image', fontsize=30)
        ax[3].imshow(ZF_img, vmin=0,
                     vmax=1, cmap=plt.get_cmap('gray'))

        plt.xticks([]), plt.yticks([])

        ax[1].text(0, 10, "ssim={:.3f}".format(predictSsim),
                   fontdict={'size': '20', 'color': 'w'})

        ax[1].text(0, 20, "psnr={:.3f}".format(predictPsnr),
                   fontdict={'size': '20', 'color': 'w'})

        ax[1].text(0, 30, "mse={:.6f}".format(predictMSE),
                   fontdict={'size': '20', 'color': 'w'})

        plot_path = os.path.join(args.predictResultPath ,series_name+'_recon'+"{}.png".format(slice))
        plt.savefig(plot_path, bbox_inches='tight')
        # plt.show()
        if savepng:
            siglePngPath = os.path.join(args.predictResultPath, 'SplitImage')
            save_numpy_image_to_png(label, final_recon, diff_image, siglePngPath, slice)
            plotAndSaveSingleImage(siglePngPath, slice, (150,140),(200,190),True)


        imageDict = {
            "label_img": label,
            "recon_img": final_recon
        }

        # imagePixelEvaluation(imageDict,150)

def save_numpy_image_to_png(label, reconImage, diffImage, savePath, slice):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    label = np.squeeze(label).astype("float32")
    reconImage = np.squeeze(reconImage).astype("float32")
    cmap = plt.cm.jet
    diffImage = cmap(np.squeeze(diffImage).astype("float32"))
    imsave(savePath+"/label_"+str(slice)+".png", label)
    imsave(savePath+"/rec_"+str(slice)+".png", reconImage)
    imsave(savePath+"/diff_"+str(slice)+".png", diffImage)

def plotAndSaveSingleImage(folderpath, slice, starts, ends, savezoomImage=True):
    
    label_image_path = folderpath + "/label_"+ str(slice)+".png"
    rec_image_path = folderpath+ "/rec_"+ str(slice)+".png"
    diff_image_path = folderpath+"/diff_"+str(slice)+".png"
    print("label_image_path", label_image_path)
    label = imread(label_image_path)
    recon = imread(rec_image_path)
    diff = imread(diff_image_path)

    label_rgb_image = getROI(label,starts, ends)
    label_ROI = getROIImage(label,starts, ends)
    recon_rgb_image = getROI(recon,starts, ends)
    recon_ROI = getROIImage(recon,starts, ends)

    ## the diff is the RGB image and don't need the diff image ?
    diff_ROI = getROIImage(diff, starts, ends)

    # interpolation = plt.interpolation.lanczos

    plt.figure()
    plt.imshow(label_rgb_image, plt.cm.gray)
    plt.title('Label Image')
    plt.figure()
    plt.imshow(label_ROI, plt.cm.gray, interpolation="lanczos")  ## here need the Anti-aliasing
    plt.title('Label Image Zoom')
    plt.figure()
    plt.imshow(recon_rgb_image, plt.cm.gray)
    plt.title('Recon Image')
    plt.figure()
    plt.imshow(recon_ROI, plt.cm.gray,interpolation="lanczos")
    plt.title('Recon Image Zoom')
    plt.figure()
    plt.imshow(diff, plt.cm.gray)
    plt.title('Recon Image')
    plt.figure()
    plt.imshow(diff_ROI, interpolation="lanczos")
    plt.title('Diff Image Zoom')
    plt.colorbar()
    # plt.show()



    if savezoomImage:
        rgbLabelPath = folderpath + "/label_"+str(slice)+"_RGB.png"
        rgbReconPath = folderpath + "/rec_"+str(slice)+"_RGB.png"

        ROILabelPath = folderpath + "/label_"+str(slice)+"_ZOOM.png"
        ROIReconPath = folderpath + "/rec_"+str(slice)+"_ZOOM.png"
        ROIDiffPath = folderpath + "/diff_"+str(slice)+"_ZOOM.png"
        
        imsave(rgbLabelPath, label_rgb_image)
        imsave(rgbReconPath, recon_rgb_image)
        imsave(ROILabelPath, label_ROI)
        imsave(ROIReconPath, recon_ROI)
        imsave(ROIDiffPath, diff_ROI)

def getROI(image, starts,ends):

    imageShape = image.shape

    rr1, cc1 = rectangle_perimeter(
        (starts[0], starts[1]), (ends[0], ends[1]), shape=imageShape)
    outline_image = np.zeros_like(image, dtype=np.bool)
    outline_image[cc1, rr1] = True

    rgb_image = np.zeros((imageShape[0], imageShape[1], 3))
    rgb_image[:, :, 0] = image / np.max(image)
    rgb_image[:, :, 1] = image / np.max(image)
    rgb_image[:, :, 2] = image / np.max(image)
    rgb_image[outline_image] = (1, 0, 0)

    return rgb_image

def getROIImage(image, starts, ends):

    if starts[0] <= ends[0] and starts[1] <= ends[1]:
        ROI_image = image[starts[1]:ends[1], starts[0]:ends[0]]
    elif starts[0] <= ends[0] and starts[1] > ends[1]:
        ROI_image = image[starts[1]:ends[1], ends[0]:starts[0]]
    elif starts[0] > ends[0] and starts[1] <= ends[1]:
        ROI_image = image[ends[1]:starts[1], starts[0]:ends[0]]
    else:
        ROI_image = image[ends[1]:starts[1], ends[0]:starts[0]]

    return ROI_image


if __name__ == "__main__":
    pass
