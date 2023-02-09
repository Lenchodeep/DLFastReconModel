import torch
from tqdm import tqdm
from math import log10
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnrcal

from utils import calcuBatchPSNR, calculatePSNR, calcuNMSE

def eval_net(net, loader, loss, device):
    net.eval()
    n_val = len(loader)
    tot_FullLoss = 0
    tot_ImL2 = 0
    tot_ImL1 = 0
    tot_psnr = 0
    tot_ssim = 0
    totKspaceL2 = 0
    for batch in tqdm(loader):
        fold_img = batch['fold_img']
        target_img = batch['target_img']

        fold_img = fold_img.to(device = device, dtype = torch.float32)
        target_img = target_img.to(device = device, dtype = torch.float32)

        with torch.no_grad():
            rec_img, rec_mid_img= net(fold_img)
        
        FullLoss, ImL2, ImL1, KspaceL2,_, ssimValue= loss.calc_gen_loss(rec_img, target_img)


        ### for calculating the psnr
        rec_img_detach = rec_img.detach().cpu().numpy()
        label_detach = target_img.detach().cpu().numpy()

        rec_img_detach = (rec_img_detach - np.min(rec_img_detach))/ (np.max(rec_img_detach) - np.min(rec_img_detach))
        label_detach = (label_detach - np.min(label_detach))/ (np.max(label_detach) - np.min(label_detach))

        batchSize = rec_img_detach.shape[0]


        for i in range(batchSize):
            #calculate the psnr per image instead batch
            recon_img_slice = np.squeeze(rec_img_detach[i,:,:,:])
            label_img_slice = np.squeeze(label_detach[i,:,:,:])
            psnr = psnrcal(recon_img_slice,label_img_slice)
            tot_psnr += psnr
        #######################
        
        tot_FullLoss += FullLoss.item()
        tot_ImL2 += ImL2.item()
        tot_ssim += ssimValue.item()
        tot_ImL1 += ImL1.item()
        totKspaceL2 += KspaceL2.item()

    avg_psnr = tot_psnr/(n_val*batchSize) 

    net.train() ##n整个网络结构进入到train模式中
    return rec_img, rec_mid_img, target_img, tot_FullLoss/n_val, tot_ImL2/n_val, tot_ImL1/n_val,\
           totKspaceL2/n_val, avg_psnr, tot_ssim/n_val


def psycho_eval_net(net, loader, criterion, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()  ##网络结构进入到eval模式中
    n_val = len(loader)  # number of batch
    tot_FullLoss = 0
    tot_ImL2 = 0
    tot_ImL1 = 0
    tot_psnr = 0
    totKspaceL2 = 0
    tot_ssim = 0
    tot_gradient = 0
    tot_mid_ssim = 0
    tot_nmse = 0
    for batch in tqdm(loader):
        masked_Kspace = batch['masked_K']
        full_Kspace = batch['target_K']
        full_img = batch['target_img']
        masked_Kspace = masked_Kspace.to(device=device, dtype=torch.float32)
        full_Kspace = full_Kspace.to(device=device, dtype=torch.float32)
        full_img = full_img.to(device=device, dtype=torch.float32)


        with torch.no_grad():
            rec_img, rec_Kspace, rec_mid_img = net(masked_Kspace)

        FullLoss, ImL2, ImL1, KspaceL2, adverloss , ssim, midssim, gradientLoss = criterion.calc_gen_loss(rec_img, rec_mid_img, rec_Kspace, full_img, full_Kspace, None, calcuMid=False)
        ### for calculating the psnr
        rec_img_detach = rec_img.detach().cpu().numpy()
        label_detach = full_img.detach().cpu().numpy()

        rec_img_detach = (rec_img_detach - np.min(rec_img_detach))/ (np.max(rec_img_detach) - np.min(rec_img_detach))
        label_detach = (label_detach - np.min(label_detach))/ (np.max(label_detach) - np.min(label_detach))

        batchsize = label_detach.shape[0]
        batch_nmse = (calcuNMSE(np.squeeze(label_detach), np.squeeze(rec_img_detach), isbatch=True))/batchsize
        batch_psnr = (calculatePSNR(np.squeeze(label_detach), np.squeeze(rec_img_detach), isbatch=True)) /batchsize
    

        #######################
        tot_FullLoss += FullLoss.item()
        tot_ImL2 += ImL2.item()
        tot_ImL1 += ImL1.item()
        totKspaceL2 += KspaceL2.item()
        tot_ssim += ssim
        tot_mid_ssim += midssim
        tot_nmse += batch_nmse
        tot_psnr += batch_psnr
        tot_gradient += gradientLoss.item()
    ## version 1
    avg_psnr = tot_psnr/(n_val) 

    net.train() ##n整个网络结构进入到train模式中
    return rec_img, full_img, rec_mid_img, tot_FullLoss/n_val, tot_ImL2/n_val, tot_ImL1/n_val,\
           totKspaceL2/n_val, avg_psnr, tot_ssim/n_val, tot_mid_ssim/n_val, tot_gradient/n_val, tot_nmse/n_val


def psycho_eval_net1(net, loader, criterion, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()  ##网络结构进入到eval模式中
    n_val = len(loader)  # number of batch
    tot_FullLoss = 0
    tot_ImL2 = 0
    tot_ImL1 = 0
    tot_psnr = 0
    totKspaceL2 = 0
    tot_ssim = 0
    tot_gradient = 0
    tot_mid_ssim = 0
    for batch in tqdm(loader):
        masked_Kspace = batch['masked_K']
        full_Kspace = batch['target_K']
        full_img = batch['target_img']
        masked_Kspace = masked_Kspace.to(device=device, dtype=torch.float32)
        full_Kspace = full_Kspace.to(device=device, dtype=torch.float32)
        full_img = full_img.to(device=device, dtype=torch.float32)


        with torch.no_grad():
            rec_img, rec_Kspace, rec_mid_img = net(masked_Kspace)

        FullLoss, ImL2, ImL1, KspaceL2, adverloss , ssim, midssim, gradientLoss = criterion.calc_gen_loss(rec_img, rec_mid_img, rec_Kspace, full_img, full_Kspace, None, calcuMid=True)
        ### for calculating the psnr
        rec_img_detach = rec_img.detach().cpu().numpy()
        label_detach = full_img.detach().cpu().numpy()

        rec_img_detach = (rec_img_detach - np.min(rec_img_detach))/ (np.max(rec_img_detach) - np.min(rec_img_detach))
        label_detach = (label_detach - np.min(label_detach))/ (np.max(label_detach) - np.min(label_detach))
       
        batchSize = rec_img_detach.shape[0]

        for i in range(batchSize):
            #calculate the psnr per image instead batch
            recon_img_slice = np.squeeze(rec_img_detach[i,:,:,:])
            label_img_slice = np.squeeze(label_detach[i,:,:,:])
            psnr = psnrcal(recon_img_slice,label_img_slice)
            tot_psnr += psnr

        #######################
        tot_FullLoss += FullLoss.item()
        tot_ImL2 += ImL2.item()
        tot_ImL1 += ImL1.item()
        totKspaceL2 += KspaceL2.item()
        tot_ssim += ssim.item()
        tot_mid_ssim += midssim.item()
        tot_gradient += gradientLoss.item()
    ## version 1
    avg_psnr = tot_psnr/(n_val*batchSize) 

    net.train() ##n整个网络结构进入到train模式中
    return rec_img, full_img, rec_mid_img, tot_FullLoss/n_val, tot_ImL2/n_val, tot_ImL1/n_val,\
           totKspaceL2/n_val, avg_psnr, tot_ssim/n_val, tot_mid_ssim/n_val, tot_gradient/n_val
           
           
def psycho_eval_net_wgan(net, loader, criterion, args):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()  ##网络结构进入到eval模式中
    n_val = len(loader)  # number of batch
    tot_FullLoss = 0
    tot_ImL2 = 0
    tot_ImL1 = 0
    tot_psnr = 0
    totKspaceL2 = 0
    tot_ssim = 0
    tot_ssim_mid = 0
    tot_gradient = 0

    for batch in tqdm(loader):
        masked_Kspace = batch['masked_K']
        full_Kspace = batch['target_K']
        full_img = batch['target_img']
        masked_Kspace = masked_Kspace.to(device=args.device, dtype=torch.float32)
        full_Kspace = full_Kspace.to(device=args.device, dtype=torch.float32)
        full_img = full_img.to(device=args.device, dtype=torch.float32)


        with torch.no_grad():
            rec_img, rec_Kspace, rec_mid_img = net(masked_Kspace)

        FullLoss, ImL2, ImL1, KspaceL2, adverloss , ssim, ssimmid, gradientLoss = \
            criterion.calc_gen_loss_wgan(rec_img, rec_mid_img, rec_Kspace, full_img, full_Kspace, None, args.is_mid_loss)
        ### for calculating the psnr
        rec_img_detach = rec_img.detach().cpu().numpy()
        label_detach = full_img.detach().cpu().numpy()
       
        batch_psnr = np.mean(calcuBatchPSNR(label_detach, rec_img_detach)) 
        tot_psnr += batch_psnr

        #######################
        tot_FullLoss += FullLoss.item()
        tot_ImL2 += ImL2.item()
        tot_ImL1 += ImL1.item()
        totKspaceL2 += KspaceL2.item()
        tot_ssim +=ssim.item()
        tot_ssim_mid += ssimmid.item()
        tot_gradient += gradientLoss.item()
    ## version 1
    avg_psnr = tot_psnr/(n_val) 

    net.train() ##n整个网络结构进入到train模式中
    return rec_img, full_img, rec_mid_img, tot_FullLoss/n_val, tot_ImL2/n_val, tot_ImL1/n_val,\
           totKspaceL2/n_val, avg_psnr, tot_ssim/n_val, tot_ssim_mid/n_val, tot_gradient/n_val
