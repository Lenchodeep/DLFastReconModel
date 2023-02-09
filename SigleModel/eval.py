import torch
import  torch.nn.functional as F
from tqdm import tqdm   
from math import log10
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
def eval_net(net, loader, criterion, device):
    n_val = len(loader)
    net.eval()

    tot_ImL2 = 0
    tot_psnr = 0
    for batch in tqdm(loader):
        full_img = batch['target_img'] 
        fold_img = batch['masked_img']
        full_img = full_img.to(device = device, dtype = torch.float32)
        fold_img = fold_img.to(device = device, dtype = torch.float32)
        with torch.no_grad():
            reconImg ,_= net(fold_img)

        psnr = calcuBatchPsnr(reconImg.detach().cpu().numpy().squeeze(), full_img.detach().cpu().numpy().squeeze())
        tot_psnr += psnr
        loss = criterion(reconImg,full_img)
        tot_ImL2 +=loss.item()
    net.train() ##n整个网络结构进入到train模式中

    return reconImg, full_img, tot_ImL2/n_val, tot_psnr/n_val

def calcuBatchPsnr(pred, tar):
    psnrlist = []
    for i in range(pred.shape[0]):
        rec = img_regular(pred[i,:,:])
        label = img_regular(tar[i,:,:])
        psnr = peak_signal_noise_ratio(label, rec)
        psnrlist.append(psnr)
    batchPsnr = sum(psnrlist)/pred.shape[0]
    return batchPsnr

def img_regular(img):
    return (img-np.min(img))/ (np.max(img)-np.min(img))