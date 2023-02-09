import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_msssim import SSIM, ssim, MS_SSIM

class Gradient_Net(nn.Module):

  def __init__(self, args):

    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(args.device)
    kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(args.device)
    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    
    return gradient

class NetLoss():
    def __init__(self,args, masked_kspace_flag = True):
        self.args = args
        mask_path = args.mask_path
        mask = np.load(mask_path)

        self.masked_kspace_flag = masked_kspace_flag
        self.mask = torch.tensor(mask==1, device = self.args.device)

        self.maskNot = self.mask==0

        self.ImL1Loss = nn.SmoothL1Loss()
        self.ImL2Loss = nn.MSELoss()
        self.AdverLoss = nn.BCEWithLogitsLoss() ### BCEWithLogitsLoss就是把Sigmoid-BCELoss

        self.ms_ssimLoss = MS_SSIM(data_range=1.0, size_average=True, channel=1)
        self.loss_weight = args.loss_weights
        self.gradient = Gradient_Net(args)

    def img_space_loss(self, pred_img, tar_img):
        return self.ImL1Loss(pred_img, tar_img), self.ImL2Loss(pred_img, tar_img)

    def gen_adver_loss(self, D_fake):
        real_ = torch.tensor(1.0).expand_as(D_fake).to(self.args.device)     ###图像全部设置为1
        return self.AdverLoss(D_fake, real_)

    def calcu_gradient(x,h_x=None,w_x=None):

        if h_x is None and w_x is None:
            h_x = x.size()[2]
            w_x = x.size()[3]
        r  = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]

        gradient = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)

        return gradient

    def calcu_gradient_loss(self, pred_Im, tar_Im):

        pred_Im_gradient = self.gradient(pred_Im)
        tar_Im_gradient = self.gradient(tar_Im)
        return torch.mean(torch.abs(pred_Im_gradient - tar_Im_gradient))


    def disc_adver_loss(self, D_real, D_fake):
        """Discriminator loss"""
        real_ = torch.tensor(1.0).expand_as(D_real).to(self.args.device)
        fake_ = torch.tensor(0.0).expand_as(D_fake).to(self.args.device)
        real_loss = self.AdverLoss(D_real,real_)
        fake_loss = self.AdverLoss(D_fake,fake_)
        return real_loss,fake_loss

    def calc_gen_loss(self, pred_Im, tar_Im, D_fake = None):
        ImL1, mseLoss = self.img_space_loss(pred_Im, tar_Im)
        ms_ssimLoss = self.ms_ssimLoss(pred_Im, tar_Im)

        gradient_Loss = self.calcu_gradient_loss(pred_Im, tar_Im)

        if D_fake is not None:
            adv_Loss = self.gen_adver_loss(D_fake) 

        else:
            adv_Loss = 0  

        FullLoss = self.loss_weight[0] * adv_Loss + self.loss_weight[1] * ImL1 + self.loss_weight[2] * (1-ms_ssimLoss) + self.loss_weight[3] * gradient_Loss
        return FullLoss, adv_Loss, ImL1, ms_ssimLoss, gradient_Loss, mseLoss
        
    
    def calc_disc_loss(self,D_real,D_fake):
        real_loss,fake_loss = self.disc_adver_loss(D_real,D_fake)
        return real_loss,fake_loss, 0.5*(real_loss + fake_loss)


def set_grad(network,requires_grad):
    for param in network.parameters():
        param.requires_grad = requires_grad
