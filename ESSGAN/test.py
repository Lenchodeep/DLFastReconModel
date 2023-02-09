import os
import sys
from types import SimpleNamespace
import numpy as np
from numpy.lib.utils import info
import yaml
import logging
import torch
import h5py
from scipy.io import loadmat

from savePredResult import saveData
from imgPlot import pltImages
def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft2(kspace_cplx):
    return np.absolute(np.fft.ifft2(kspace_cplx))[None,:,:]
    
def slice_preprocess(kspace_cplx, mask, args):

    masked_k_complex = kspace_cplx * mask
    masked_k = np.zeros((args.imgSize, args.imgSize, 2))
    kspace = np.zeros((args.imgSize, args.imgSize, 2))

    kspace[:,:,0] = np.real(kspace_cplx).astype(np.float32)
    kspace[:,:,1] = np.imag(kspace_cplx).astype(np.float32)

    masked_k[:,:,0] = np.real(masked_k_complex).astype(np.float32)
    masked_k[:,:,1] = np.imag(masked_k_complex).astype(np.float32)

    image = ifft2(kspace_cplx)

    kspace = np.transpose(kspace, (2,0,1))
    masked_k = np.transpose(masked_k, (2,0,1))

    return masked_k, kspace, image

def getArgs(configPath):
    with open(configPath) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)    
    args = SimpleNamespace(**data)

    return args

def test():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    configPath = "./config.yaml"
    args = getArgs(configPath)

    logging.info(f"Using device {args.device}")

    logging.info("Loading model {}".format(args.predModelPath))

    net = WNet(args, masked_kspace=args.masked_kspace)
    net.to(device= args.device)

    checkPoint = torch.load(args.predModelPath, map_location = args.device)

    net.load_state_dict(checkPoint['G_model_state_dict'])
    net.eval()   #no backforward

    logging.info("Model loaded !")
    #load mask from npy
    mask = np.load(args.mask_path)

    ### mat version
    # testData = loadmat(args.predictDir)["data"]
    ### hdf5 version
    with h5py.File(args.predictDir, 'r') as f:
        testData = f['data'][:]
    testData = np.array(testData)
    testData = np.transpose(testData, (2,0,1))
    ####


    recon_img = np.zeros(testData.shape)
    recon_Kspace = np.zeros(testData.shape, dtype = np.complex128)
    recon_mid_img = np.zeros(testData.shape)
    ZF_img = np.zeros(testData.shape)

    for sliceNum in range(testData.shape[0]):  #SLICE, HEIGHT, WIDTH
        img = testData[sliceNum, :, :]
        kspace = fft2(img)  #shift to center and complex

        masked_k, full_k, full_img = slice_preprocess(kspace, mask,args)

        masked_k = np.expand_dims(masked_k, axis=0)

        masked_k = torch.from_numpy(masked_k).to(device= args.device, dtype = torch.float32)

        ##predict:
        rec_img , rec_k, rec_mid_img = net(masked_k)

        rec_img = np.squeeze(rec_img.data.cpu().numpy())

        rec_k = np.squeeze(rec_k.data.cpu().numpy())
        rec_k = (rec_k[0,:,:]+1j*rec_k[1,:,:])

        rec_mid_img = np.squeeze(rec_mid_img.data.cpu().numpy())

        recon_img[sliceNum,:,:] = rec_img
        recon_Kspace[sliceNum,:,:] = rec_k
        recon_mid_img[sliceNum,:,:] = rec_mid_img

        masked_k = masked_k.data.cpu().numpy()
        masked_k = np.squeeze(masked_k)
        maskedDat = masked_k[0,:,:]+1j*masked_k[1,:,:]
        ZF_img[sliceNum,:,:] = np.absolute(np.fft.ifft2(np.fft.fftshift(maskedDat)))

    recon_img = np.transpose(recon_img,(1,2,0))
    recon_Kspace = np.transpose(recon_Kspace, (1,2,0))
    recon_mid_img = np.transpose(recon_mid_img, (1,2,0))
    ZF_img = np.transpose(ZF_img, (1,2,0))
    testData = np.transpose(testData,(1,2,0))

    if args.savePredication:
        filename = os.path.join(args.predictResultPath, args.predResultName)
        saveData(filename, zf_imgs=ZF_img, rec_imgs=recon_img, rec_kspaces= recon_Kspace, label_imgs= testData)
        logging.info("reconstructions save to: {}".format(filename))

    if args.visualize_images:
        logging.info("Visualizing results for image , close to continue ...")
        pltImages(ZF_img, recon_img, testData)

if __name__ == '__main__':
    # test()
    with h5py.File(r'/home/d1/share/DLreconstruction/Psycho/GanCode/predict/IXI/predResult_radial30withssim.hdf5', 'r') as f:
        ZF_img = f['ZF_imgs'][:]
        recon_img = f['rec_imgs'][:]
        rec_kspzces = f['rec_kspzces'][:]
        label_imgs = f['label_imgs'][:]
    
    print(recon_img.shape, label_imgs.shape)
    pltImages(ZF_img, recon_img, label_imgs)