'''
    This file contains the tools that used in the project
'''
import os
import numpy as np
import h5py
from PIL import Image

from skimage.metrics import structural_similarity 
from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import  mean_squared_error 
from skimage.io import imsave
from skimage.io import imread
from skimage.draw import  rectangle_perimeter

import matplotlib.pyplot as plt 
import torch
from torch import autograd
import math

############################################ early stop part #############################################################
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

############################################# gradient penalty part ######################################################
def calcu_gradient_penalty(D, batch_size, real, fake, device = "cpu"):
    """"To calculate gradient penalty"""
    alpha = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    alpha = alpha.expand(batch_size, real.size(1), real.size(2), real.size(3))
    alpha = alpha.to(device)

    interpolates = alpha * real + ((1-alpha) * fake)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


############################################## save test result part ############################################

def saveData(outFileName, zf_imgs, rec_imgs, rec_kspaces = None, label_imgs = None):
    """
        In HWC form, save result into a h5py file
    """
    f = h5py.File(outFileName, 'a')
    f.create_dataset('ZF_imgs', data = zf_imgs)
    f.create_dataset('rec_imgs', data = rec_imgs)
    if rec_kspaces is not None:
        f.create_dataset('rec_kspzces', data = rec_kspaces)
    if label_imgs is not None:
        f.create_dataset('label_imgs', data = label_imgs)
    f.close()

############################################## save png part ############################################

def png_saver(img,savePath):
    '''
        saving the reconstruction result during the training process
    '''
    image_array = img[0,0,:,:].detach().cpu().numpy()
    image_array = np.squeeze(image_array).astype("float32")
    image_array = image_array * 255

    im = Image.fromarray(image_array.astype('uint8')).convert('L')
    im.save(savePath)

def save_numpy_image_to_png(label, ZF_img, reconImage, diffImage, zfDiffImage, savePath, slice):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    label = np.squeeze(label).astype("float32")
    reconImage = np.squeeze(reconImage).astype("float32")
    ZF_img = np.squeeze(ZF_img).astype("float32")
    cmap = plt.cm.jet
    diffImage = cmap(np.squeeze(diffImage).astype("float32"))
    zfDiffImage = cmap(np.squeeze(zfDiffImage).astype("float32"))

    imsave(savePath+"/label_"+str(slice)+".png", label)
    imsave(savePath+"/rec_"+str(slice)+".png", reconImage)
    imsave(savePath+"/zfimg_"+str(slice)+".png", ZF_img)
    imsave(savePath+"/diff_"+str(slice)+".png", diffImage)
    imsave(savePath+"/zfdiff_"+str(slice)+".png", zfDiffImage)


################################################ plot image result part ##########################################
global ROI_START_X
ROI_START_X = 90
global ROI_START_Y
ROI_START_Y = 110
global ROI_END_X
ROI_END_X = 130
global ROI_END_Y
ROI_END_Y = 150

# global ROI_START_X
# ROI_START_X = 155
# global ROI_START_Y
# ROI_START_Y = 135
# global ROI_END_X
# ROI_END_X = 185
# global ROI_END_Y
# ROI_END_Y = 165

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
        zf_diff_img = 10*(label - ZF_img)
        predictSsim = round(calculateSSIM(label, final_recon), 3) #保留三位小数
        predictPsnr = round(calculatePSNR(label, final_recon), 3)
        predictMSE = round(calcuNMSE(label,final_recon), 5)
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

        ax[1].text(0, 30, "nmse={:.6f}".format(predictMSE),
                   fontdict={'size': '20', 'color': 'w'})

        plot_path = os.path.join(args.predictResultPath ,series_name+'_recon'+"{}.png".format(slice))
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()
        if savepng:
            siglePngPath = os.path.join(args.predictResultPath, 'SplitImage')
            save_numpy_image_to_png(label, ZF_img, final_recon, diff_image,zf_diff_img, siglePngPath, slice)
            plotAndSaveSingleImage(siglePngPath, slice, (ROI_START_X,ROI_START_Y),(ROI_END_X,ROI_END_Y),True)

        imageDict = {
            "label_img": label,
            "recon_img": final_recon
        }

        # imagePixelEvaluation(imageDict,150)

def imgRegular(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) 

def plotAndSaveSingleImage(folderpath, slice, starts, ends, savezoomImage=True):
    
    label_image_path = folderpath + "/label_"+ str(slice)+".png"
    rec_image_path = folderpath+ "/rec_"+ str(slice)+".png"
    diff_image_path = folderpath+"/diff_"+str(slice)+".png"
    zf_image_path = folderpath+"/zfimg_"+str(slice)+".png"
    zf_diff_image_path =  folderpath+"/zfdiff_"+str(slice)+".png"

    print("label_image_path", label_image_path)
    label = imread(label_image_path)
    recon = imread(rec_image_path)
    diff = imread(diff_image_path)
    zf = imread(zf_image_path)
    zfdiff = imread(zf_diff_image_path)

    label_rgb_image = getROI(label,starts, ends)
    label_ROI = getROIImage(label,starts, ends)
    recon_rgb_image = getROI(recon,starts, ends)
    recon_ROI = getROIImage(recon,starts, ends)
    zf_rgb_image = getROI(zf,starts, ends)
    zf_ROI = getROIImage(zf,starts, ends)
    

    ## the diff is the RGB image and don't need the diff image ?
    diff_ROI = getROIImage(diff, starts, ends)
    zfdiff_ROI = getROIImage(zfdiff, starts, ends)
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
    plt.close()


    if savezoomImage:
        rgbLabelPath = folderpath + "/label_"+str(slice)+"_RGB.png"
        rgbReconPath = folderpath + "/rec_"+str(slice)+"_RGB.png"
        rgbzfPath = folderpath + "/zfimg_"+str(slice)+"_RGB.png"

        ROILabelPath = folderpath + "/label_"+str(slice)+"_ZOOM.png"
        ROIReconPath = folderpath + "/rec_"+str(slice)+"_ZOOM.png"
        ROIDiffPath = folderpath + "/diff_"+str(slice)+"_ZOOM.png"
        ROIzfPath = folderpath + "/zfimg_"+str(slice)+"_ZOOM.png"
        ROIzfDiffPath = folderpath+"/zfdiff_"+str(slice)+"_ZOOM.png"
        imsave(rgbLabelPath, label_rgb_image)
        imsave(rgbReconPath, recon_rgb_image)
        imsave(rgbzfPath, zf_rgb_image)

        imsave(ROILabelPath, label_ROI)
        imsave(ROIReconPath, recon_ROI)
        imsave(ROIDiffPath, diff_ROI)
        imsave(ROIzfPath, zf_ROI)
        imsave(ROIzfDiffPath, zfdiff_ROI)


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

######################################  image pixel evaluation ###########################################
def imagePixelEvaluation(imagesDict,line,isRow = True):
    '''
        This function is a image pixel values evaluation method, you can use this to plot the pixel values in the 
        same line of different images to see the difference.

        input: 
            imagesDict: the images to be evaluated, a dictionary contain image name and image matrix, the image name as the key
            line: the number of the line that selected
            isRow: if true select the row-th line in row direction else in the column direction
    '''
    imageValueList = []
    RgbImageList = []
    for key in list(imagesDict.keys()):
        image = imagesDict[key]
        rgbImage = np.zeros((image.shape[0], image.shape[1], 3))

        rgbImage[:,:,0] = (image-np.min(image))/ (np.max(image) - np.min(image))
        rgbImage[:,:,1] = (image-np.min(image))/ (np.max(image) - np.min(image))
        rgbImage[:,:,2] = (image-np.min(image))/ (np.max(image) - np.min(image))

        if isRow:
            imageValueList.append(image[line,:])
            rgbImage[line,:,:] = (1,0,0)
        else:
            imageValueList.append(image[:,line])
            rgbImage[:,line,:] = (1,0,0)
        RgbImageList.append(rgbImage)

    xLabel = np.arange(0,256)
    keys = list(imagesDict.keys())
    ## plot the images
    for m in range(len(RgbImageList)):
        plt.figure()
        plt.imshow(RgbImageList[m], plt.cm.gray)
        plt.title(keys[m])

    ## plot the curves
    plt.figure()
    for j in range(len(imageValueList)):
        plt.plot(xLabel, imageValueList[j], label=keys[j], linewidth = 2)
    plt.xlabel('points')
    plt.ylabel('value')
    plt.legend()
    plt.show()

################################################ evaluation calcu #############################################

def calculatePSNR(tar, recon, isbatch = False):
    psnrlist = []
    if isbatch:
        for i in range(recon.shape[0]):
                    
            label = imgRegular(tar[i,:,:])
            pred = imgRegular(recon[i,:,:])
            psnrlist.append((peak_signal_noise_ratio(label, pred, data_range=pred.max())))
        return sum(psnrlist)
    else:   
        label = imgRegular(tar)
        pred = imgRegular(recon)
        psnrlist.append(peak_signal_noise_ratio(label, pred, data_range=pred.max()))
        return sum(psnrlist)   ## if not batch return single value

def calculateSSIM(tar, recon, isbatch = False):
    ssimlist = []
    if isbatch:
        for i in range(recon.shape[0]):
            label = imgRegular(tar[i,:,:])
            pred = imgRegular(recon[i,:,:])
            ssimlist.append((structural_similarity(label, pred, data_range=pred.max())))
        return sum(ssimlist)
    else:   
        label = imgRegular(tar)
        pred = imgRegular(recon)
        ssimlist.append(structural_similarity(label, pred, data_range=pred.max()))
        return sum(ssimlist)

def calcuNMSE(tar, recon, isbatch=False):
    nmselist = []
    if isbatch:
        for i in range(recon.shape[0]):

            label = imgRegular(tar[i,:,:])
            pred = imgRegular(recon[i,:,:])
            msea = mean_squared_error(pred, label)
            mseb = mean_squared_error(pred,np.zeros_like(pred))
            nmse = np.divide(msea, mseb)
            nmselist.append(nmse)
        return sum(nmselist)
    else:   
        label = imgRegular(tar)
        pred = imgRegular(recon)

        msea = mean_squared_error(pred, label)
        mseb = mean_squared_error(pred,np.zeros_like(pred))
        nmse = np.divide(msea, mseb)
        nmselist.append(nmse)     
        return sum(nmselist)

def calcuBatchPSNR(gt,pred):
    x_good = np.squeeze(gt)
    x_bad = np.squeeze(pred)
    psnr_res = []
    for idx in range(gt.shape[0]):
        x_good_regular = imgRegular(x_good[idx,:,:])
        x_bad_regular = imgRegular(x_bad[idx,:,:])
        psnr_res.append(peak_signal_noise_ratio(x_good_regular, x_bad_regular))
    return psnr_res


######################################     get the config args   ##############################################

import yaml
from types import SimpleNamespace
def getArgs(configPath):
    with open(configPath) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)    
    args = SimpleNamespace(**data)

    return args