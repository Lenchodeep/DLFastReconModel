import matplotlib.pyplot as plt 
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def pltImages(ZF_imgs, rec_imgs, label_imgs):
    plt.figure()

    slices = [150, 250, 350 ,450, 550, 650, 750, 850, 950, 1050]

    for slice in slices:
        label = imageRegulization(label_imgs[:, :, slice])
        reconImg = imageRegulization(rec_imgs[:,:,slice])
        ZF_img = imageRegulization(ZF_imgs[:,:,slice])
        
        ### Kdc
        mask = np.load('/home/d1/share/DLreconstruction/masks/radial/radial_30.npy')
        masknot = 1-mask

        reconK = np.fft.fftshift(np.fft.fft2(reconImg))
        labelK = np.fft.fftshift(np.fft.fft2(label))
        mask_recon = masknot * reconK
        mask_label = mask * labelK

        final_K = mask_recon + mask_label
        final_recon = np.abs(np.fft.ifft2(np.fft.fftshift(final_K)))
        final_recon = imageRegulization(final_recon)
        predictSsim = round(ssim(final_recon, label), 3) #保留三位小数
        predictPsnr = round(psnr(final_recon, label), 3)
        ###


        # predictSsim = round(ssim(reconImg, label), 3) #保留三位小数
        # predictPsnr = round(psnr(reconImg, label), 3)
        fig, ax = plt.subplots(1,4,figsize=(100,100))

        plt.subplots_adjust(hspace=0, wspace=0)
        ax[0].set_title("LabelImage", fontsize= 30)
        ax[0].imshow(label, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))

        ax[1].set_title('Recon image', fontsize=30)
        ax[1].imshow(final_recon, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
        # ax[1].imshow(reconImg, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))


        ax[2].set_title('ZF image', fontsize=30)
        ax[2].imshow(ZF_img, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))


        ax[3].set_title('sub image', fontsize=30)
        ax[3].imshow(10*(reconImg-label), vmin=0, vmax=1, cmap = plt.cm.jet)
       
        plt.annotate("ssim:{}\npsnr:{}".format(predictSsim, predictPsnr), (0,40),fontsize = 20,color = "white")
        plt.xticks([]), plt.yticks([])
        plt.show()

def imageRegulization(img):
    image = (img - np.min(img))/(np.max(img) - np.min(img))
    return image


