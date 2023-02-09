import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import logging
import yaml
import glob
import pickle
import h5py
from types import SimpleNamespace
from model import *
from utils import *
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
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

def crop_toshape( image):
    '''
        crop to target image size
    '''
    if image.shape[0] == 256:
        return image
    if image.shape[0] % 2 == 1:
        image = image[:-1, :-1]
    crop = int((image.shape[0] - 256)/2)
    image = image[crop:-crop, crop:-crop]
    return image


def regular(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def test(args, model):
    ## load the model
    model = model.to(args.device)
    chepoint = torch.load(args.predModelPath, map_location=args.device)
    model.load_state_dict(chepoint['G_model_state_dict'])

    ## into the eval mode
    model.eval()   
    logging.info("Model loaded !")
    ## get the mask
    mask_path = args.mask_path
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    masks = np.dstack((masks_dictionary['mask0'], masks_dictionary['mask1'], masks_dictionary['mask2']))
    mask = masks_dictionary['mask1']
    maskNot = 1-mask
    test_num = 0
    tot_psnr = 0
    tot_ssim = 0
    tot_nmse = 0
    psnr_list = []
    ssim_list = [] 
    nmse_list = []

    ## get all the test file paths
    test_files = glob.glob(os.path.join(args.predictDir, '*.hdf5'))
    print(test_files)
    ## the test slices number
    sliceNum = args.slice_range[1] - args.slice_range[0]
    fileLen = len(test_files)

    #define the  final matrixs
    rec_imgs_full = np.zeros((args.image_size,args.image_size,fileLen*sliceNum))
    ZF_img_full = np.zeros((args.image_size, args.image_size,fileLen*sliceNum))
    target_imgs_full = np.zeros((args.image_size, args.image_size,fileLen*sliceNum))

    for i, infile in enumerate(test_files):
        logging.info("\nTesting image {} ...".format(infile))

        rec_imgs = np.zeros((args.image_size,args.image_size,sliceNum))
        ZF_img = np.zeros((args.image_size, args.image_size,sliceNum))
        target_imgs = np.zeros((args.image_size, args.image_size,sliceNum))

        for slice_num in range(args.slice_range[0], args.slice_range[1],1):
            test_num = test_num+1
            with h5py.File(infile, 'r') as f:
                print(f['reconstruction_rss'].shape)
                img = regular(crop_toshape(f['reconstruction_rss'][slice_num,:, :]))

                # img = f['data'][:, :, slice_num]


            # img = np.rot90(img, 1) ## rotate the image 
            zf_image, tar_img = slicePreProcess(img, mask, args) 

            ## from [0-1] to [-1-1]
            zf_image = zf_image*2-1
            tar_img = tar_img*2-1

            ## prepare the input tensor
            zf_image = np.expand_dims(np.expand_dims(zf_image, axis=0), axis=0)  # from 2D to H,C,H,W
            zf_image = torch.from_numpy(zf_image).to(device=args.device, dtype=torch.float32)
    
            rec_img= model(zf_image, True)
        
            # print(end-start)
            ## get the recon images numpy format
            rec_img = np.squeeze(rec_img.data.cpu().numpy())                      ##2D tensor
            zf_image = np.squeeze(zf_image.data.cpu().numpy())

            ## convert back to [0,1]
            rec_img = (rec_img+1)/2
            tar_img = (tar_img+1)/2
            plt.figure()
            plt.imshow(rec_img, plt.cm.gray)
            plt.show()
            imsave(args.predictResultPath+"rec_"+str(slice_num)+".png",np.squeeze(rec_img))


            tar_img = np.squeeze(tar_img)

            rec_psnr = calculatePSNR(tar_img, rec_img, isBatch=False)
            rec_ssim = calculateSSIM(tar_img, rec_img, isBatch=False)
            rec_nmse_a =  mean_squared_error(image_regular(rec_img), image_regular(tar_img))
            rec_nmse_b = mean_squared_error(image_regular(rec_img), np.zeros_like(rec_img))
            rec_nmse = rec_nmse_a / rec_nmse_b

            psnr_list.append(rec_psnr)
            ssim_list.append(rec_ssim)
            nmse_list.append(rec_nmse)

            tot_psnr = tot_psnr+rec_psnr
            tot_nmse = tot_nmse+rec_nmse
            tot_ssim = tot_ssim+rec_ssim

            rec_imgs[:, :, slice_num-args.slice_range[0]] = rec_img
            ZF_img[:, :, slice_num-args.slice_range[0]] = zf_image
            target_imgs[:,:,slice_num-args.slice_range[0]] =tar_img



        rec_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = rec_imgs
        ZF_img_full[:,:,i*sliceNum:(i+1)*sliceNum] = ZF_img
        target_imgs_full[:,:,i*sliceNum:(i+1)*sliceNum] = target_imgs

        if args.savePredication:
            os.makedirs(args.predictResultPath, exist_ok=True)
            out_file_name = args.predictResultPath + os.path.split(infile)[1]
            # save_data(out_file_name, rec_imgs, F_rec_Kspaces, fully_sampled_img, ZF_img, rec_Kspaces)

            logging.info("reconstructions save to: {}".format(out_file_name))

        if args.visualize_images:
            logging.info("Visualizing results for image {}, close to continue ...".format(infile))
            pltImages(ZF_img, rec_imgs, target_imgs, series_name =os.path.split(infile)[1],args = args, mask=masks[:,:,1], sliceList=[0])
        ## calculate the avg value
    psnrarray = np.array(psnr_list)
    ssimarray = np.array(ssim_list)
    nmsearray = np.array(nmse_list)

    avg_psnr = psnrarray.mean()
    std_psnr = psnrarray.std()

    avg_ssim = ssimarray.mean()
    std_ssim = ssimarray.std()

    avg_nmse = nmsearray.mean()
    std_nmse = nmsearray.std()

    tot_nmse = tot_nmse / test_num
    tot_psnr = tot_psnr / test_num
    tot_ssim = tot_ssim / test_num

    print("avg_psnr:", avg_psnr, "std_psnr:", std_psnr,  "avg_ssim:",avg_ssim, "std_ssim:", std_ssim, "avg_nmse:", avg_nmse,"std_nmse:", std_nmse)

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
    model = UNet()
    test(args, model)

main()