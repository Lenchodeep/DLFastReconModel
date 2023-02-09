import torch
import  torch.nn.functional as F
from tqdm import tqdm   
from math import log10
import matplotlib.pyplot as plt
import torch.nn as nn
def eval_net(net, loader, criterion, device):
    # net.eval()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_runing_stats=False

    n_val = len(loader)

    tot_FullLoss = 0
    tot_ImL1 = 0
    tot_MSSSIM = 0
    tot_psnr = 0
    tot_gradientLoss = 0
    tot_mseLoss = 0
    for batch in tqdm(loader):
        full_img = batch['target_img'] 
        fold_img = batch['masked_img']
        full_img = full_img.to(device = device, dtype = torch.float32)
        fold_img = fold_img.to(device = device, dtype = torch.float32)

        with torch.no_grad():
            rec_img = net(fold_img)
        FullLoss, adv_Loss, ImL1, ms_ssimLoss, gradient_Loss, mseLoss = criterion.calc_gen_loss(rec_img, full_img)

        tot_FullLoss += FullLoss.item()
        tot_ImL1 += ImL1.item()
        tot_MSSSIM += ms_ssimLoss.item()
        psnr = 10 * log10(1 / mseLoss.item())
        tot_psnr += psnr
        tot_gradientLoss += gradient_Loss
        tot_mseLoss += mseLoss
    # for m in net.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.track_runing_stats=True
    net.train() ##n整个网络结构进入到train模式中
    return rec_img, full_img, tot_FullLoss/n_val, tot_ImL1/n_val, tot_MSSSIM/n_val\
        , tot_psnr/n_val, tot_gradientLoss/n_val, tot_mseLoss/ n_val



def eval_net1(net, loader, criterion, device):
    # net.eval()
    n_val = len(loader)
    tot_ImL2 = 0
    for batch in tqdm(loader):
        full_img = batch['target_img'] 
        fold_img = batch['masked_img']
        full_img = full_img.to(device = device, dtype = torch.float32)
        fold_img = fold_img.to(device = device, dtype = torch.float32)
        with torch.no_grad():
            rec_img = net(fold_img)
        loss = criterion(rec_img,full_img)
        tot_ImL2 +=loss.item()

    # net.train()
    print(torch.max(rec_img), torch.min(rec_img),"iiiii")
    return rec_img, full_img, tot_ImL2/n_val
