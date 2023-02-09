from torch.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from skimage.io import imsave,imread
import math
from skimage.draw import  rectangle_perimeter


def undersample(image, mask):
    k = fft2(image)
    k_und = mask*k
    x_und = ifft2(k_und)
    return x_und, k_und, k

def cal_psnr(pred, gt, maxp=1.):
    # pred = pred.abs()
    pred = torch.clamp(pred, min=0., max=maxp)  # some points in pred are larger than maxp
    # gt = gt.abs()

    mse = torch.mean((pred - gt) ** 2, dim=(1, 2))

    psnr = -10. * torch.log10(mse)  # + 1e-6
    psnr = psnr + 20. * math.log10(maxp)

    return psnr.sum()

def RF(x_rec, mask, norm='ortho'):
    '''
    RF means R*F(input), F is fft, R is applying mask;
    return the masked k-space of x_rec,
    '''
    x_rec = x_rec.permute(0, 2, 3, 1)
    mask = mask.permute(0, 2, 3, 1)
    k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=norm)
    k_rec = torch.view_as_real(k_rec)
    k_rec *= mask
    k_rec = k_rec.permute(0, 3, 1, 2)
    return k_rec

def revert_scale(im_tensor, a=2., b=-1.):
    '''
    param: im_tensor : [B, 2, W, H]
    '''
    b = b * torch.ones_like(im_tensor)
    im = (im_tensor - b) / a

    return im

def output2complex(im_tensor, isZeroToOne = False):
    '''
    param: im_tensor : [B, 2, W, H]
    return : [B,W,H] complex value
    '''
    ############## revert each channel to [0,1.] range
    if not isZeroToOne:
        im_tensor = revert_scale(im_tensor)
    # 2 channel to complex
    im_tensor = torch.view_as_complex(im_tensor.permute(0, 2, 3, 1).contiguous())
    
    return im_tensor


def output2complex2(im_tensor, isZeroToOne = False):
    '''
    param: im_tensor : [B, 2, W, H]
    return : [B,W,H] complex value
    '''
    ############## revert each channel to [0,1.] range
    if not isZeroToOne:
        im_tensor = revert_scale(im_tensor)
    # 2 channel to complex
    im_tensor = torch.abs(torch.complex(im_tensor[:,0:1,:,:], im_tensor[:,1:2,:,:]))
    
    return im_tensor


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

def imgRegular(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) 


######################### plot image result part #########################
def pltImages(ZF_imgs, rec_imgs, rec_mid_imgs, label_imgs, series_name, sliceList = None, args = None, savepng = True):
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
        reconMidImg = rec_mid_imgs[:,:,slice]
        ZF_img = ZF_imgs[:,:,slice]

        diff_image = 10*(label - reconImg)
        diff_mid_img = 10*(label - reconMidImg)

        # final_recon_regular = imgRegular(reconImg)
        # label_regular = imgRegular(label)

        predictSsim = round(calculateSSIM(label, reconImg, False), 3) #保留三位小数
        predictPsnr = round(calculatePSNR(label, reconImg, False), 3)
        predictMSE = round(calcuNMSE(label,reconImg,False), 5)

        #3 calculate the mid images evaluation value
        midSsim = round(calculateSSIM(label, reconMidImg, False), 3) #保留三位小数
        midPsnr = round(calculatePSNR(label, reconMidImg, False), 3)
        midMSE = round(calcuNMSE(label,reconMidImg,False), 5)

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
            save_numpy_image_to_png(label, reconImg, reconMidImg, diff_image, diff_mid_img, siglePngPath, slice)
            # plotAndSaveSingleImage(siglePngPath, slice, (155,135),(185,165),True)
        plotAndSaveSingleImage(siglePngPath, slice, (150,140),(200,190),True)


        imageDict = {
            "label_img": label,
            "recon_img": reconImg
        }

def save_numpy_image_to_png(label, reconImage, reconMidImg, diffImage, diff_mid_img, savePath, slice):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    label = np.squeeze(label).astype("float32")
    reconImage = np.squeeze(reconImage).astype("float32")
    reconMidImg= np.squeeze(reconMidImg).astype("float32")
    cmap = plt.cm.jet
    diffImage = cmap(np.squeeze(diffImage).astype("float32"))
    diff_mid_img = cmap(np.squeeze(diff_mid_img).astype("float32"))
    imsave(savePath+"/label_"+str(slice)+".png", label)
    imsave(savePath+"/rec_"+str(slice)+".png", reconImage)
    imsave(savePath+"/recmid_"+str(slice)+".png", reconMidImg)
    imsave(savePath+"/diff_"+str(slice)+".png", diffImage)
    imsave(savePath+"/middiff_"+str(slice)+".png", diff_mid_img)


def plotAndSaveSingleImage(folderpath, slice, starts, ends, savezoomImage=True):
    ## read the images  
    label_image_path = folderpath + "/label_"+ str(slice)+".png"
    rec_image_path = folderpath+ "/rec_"+ str(slice)+".png"
    mid_rec_image_path = folderpath+ "/recmid_"+ str(slice)+".png"
    diff_image_path = folderpath+"/diff_"+str(slice)+".png"
    mid_diff_image_path = folderpath+"/diff_"+str(slice)+".png"

    label = imread(label_image_path)
    recon = imread(rec_image_path)
    mid_recon = imread(mid_rec_image_path)
    diff = imread(diff_image_path)
    mid_diff = imread(mid_diff_image_path)

    ## get the ROI reigon
    label_rgb_image = getROI(label,starts, ends)
    label_ROI = getROIImage(label,starts, ends)
    recon_rgb_image = getROI(recon,starts, ends)
    recon_ROI = getROIImage(recon,starts, ends)
    mid_recon_rgb_image = getROI(mid_recon,starts, ends)
    mid_recon_ROI = getROIImage(mid_recon,starts, ends)
    ## the diff is the RGB image and don't need the diff image ?
    diff_ROI = getROIImage(diff, starts, ends)
    mid_diff_ROI = getROIImage(mid_diff, starts, ends)

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

    ## wite the images
    if savezoomImage:
        rgbLabelPath = folderpath + "/label_"+str(slice)+"_RGB.png"
        rgbReconPath = folderpath + "/rec_"+str(slice)+"_RGB.png"
        rgbMidReconPath = folderpath + "/midrec_"+str(slice)+"_RGB.png"

        ROILabelPath = folderpath + "/label_"+str(slice)+"_ZOOM.png"
        ROIReconPath = folderpath + "/rec_"+str(slice)+"_ZOOM.png"
        ROIMidReconPath = folderpath + "/midrec_"+str(slice)+"_ZOOM.png"
        ROIDiffPath = folderpath + "/diff_"+str(slice)+"_ZOOM.png"
        ROIMidDiffPath = folderpath + "/middiff_"+str(slice)+"_ZOOM.png"
        
        imsave(rgbLabelPath, label_rgb_image)
        imsave(rgbReconPath, recon_rgb_image)
        imsave(rgbMidReconPath, mid_recon_rgb_image)
        imsave(ROILabelPath, label_ROI)
        imsave(ROIReconPath, recon_ROI)
        imsave(ROIMidReconPath, mid_recon_ROI)
        imsave(ROIDiffPath, diff_ROI)
        imsave(ROIMidDiffPath, mid_diff_ROI)

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