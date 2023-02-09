import numpy as np
import yaml
import glob
import pickle
import h5py
from types import SimpleNamespace
from os import path
from  Unet import *
from skimage.io import imsave
import logging
import matplotlib.pyplot as plt



def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft2(kspace):
    return np.abs(np.fft.ifft2(np.fft.fftshift(kspace)))

def slicePreProcess(image, mask, args):
    kspace_full = fft2(image)
    und_kspace = kspace_full * mask
    zf_img = ifft2(und_kspace)
    label = ifft2(kspace_full)

    return zf_img, label

def getFeatures(args,model):
    model  =model.to(args.device)
    chepoint = torch.load(args.predModelPath, map_location=args.device)
    model.load_state_dict(chepoint['G_model_state_dict'])
    model.eval()   
    logging.info("Model loaded !")

    mask = np.load(args.mask_path)


    test_files = glob.glob(os.path.join(args.predictDir, '*.hdf5'))

    for infile in test_files:
        for slice_num in range(args.slice_range[0], args.slice_range[1],1):
            with h5py.File(infile, 'r') as f:
                img = f['data'][:,:,slice_num]
            img = np.rot90(img,1)
            zf_image, tar_img = slicePreProcess(img, mask, args) 

        
            print(tar_img.min(), tar_img.max(), "tar_img")

            ## prepare the input tensor
            zf_image = np.expand_dims(np.expand_dims(zf_image, axis=0), axis=0)  # from 2D to H,C,H,W
            zf_image = torch.from_numpy(zf_image).to(device=args.device, dtype=torch.float32)

            rec_img, featureMap= model(zf_image)
            rec_img_np = rec_img.detach().cpu().numpy()
            featuremap_np = featureMap.detach().cpu().numpy()
            
            featuremap_np = np.squeeze(featuremap_np)
            rec_img_np = np.squeeze(rec_img_np)
            # plt.figure()
            # plt.imshow(rec_img_np, plt.cm.gray)
            # plt.show()

            for i in range(featuremap_np.shape[0]):
                current_feature = featuremap_np[i,:,:]

                current_feature = (current_feature-np.min(current_feature)) / (np.max(current_feature)-np.min(current_feature))
                current_feature = current_feature *255
                current_feature = current_feature.astype(np.uint8)
   
                # plt.figure()
                # plt.imshow(current_feature, plt.cm.gray)
                # plt.show()

                imsave(path.join(args.predictResultPath,'feature_'+str(i)+'.png'), current_feature)
def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args

def main():
    args = get_args()
    logging.info(f"Using device {args.device}")

    logging.info("Loading model {}".format(args.predModelPath))
    model = Unet(1,1)
    getFeatures(args, model)

main()