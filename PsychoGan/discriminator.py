'''
    
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class patchGAN(nn.Module):
    """
        Parameters:
            input_channels (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            crop_center      -- None ot the size of the center patch to be cropped
            FC_bottleneck      -- If True use global average pooling and output a one-dimension prediction
    """
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)
            
    def __init__(self, input_channels, ndf, n_layers = 3, crop_center = None, FC_bottleneck = False):
        super(patchGAN, self).__init__()

        self.crop_center = crop_center
        ks = 3
        padw=1
        sequence = [
            nn.Conv2d(input_channels, ndf, kernel_size=ks, stride=1, padding=padw), 
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf, ndf, kernel_size=ks, stride=1, padding=padw),
            nn.LeakyReLU(0.2, inplace= True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence+=[
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks,stride=1, padding=padw),  ###BN之前的卷积层中的偏置b不会起到作用
                nn.BatchNorm2d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf*nf_mult, ndf*nf_mult, kernel_size=ks, stride=2, padding=padw),  ###BN之前的卷积层中的偏置b不会起到作用
                nn.BatchNorm2d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence +=[
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks, stride = 1, padding=padw),
            nn.BatchNorm2d(ndf*nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        if FC_bottleneck:
            sequence += [
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(ndf*nf_mult, 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128,1)
            ]
        else:
            sequence+=[
                nn.Conv2d(ndf*nf_mult, 1,kernel_size=ks, stride=1, padding=padw)
            ]
        
        self.model = nn.Sequential(*sequence).apply(self.weight_init)   ##apply函数将函数递归传递到网络模型的每一个子模型中，主要用于参数的初始化过程中

    def forward(self, input):
        if self.crop_center is not None:
            _,_,h,w = input.shape
            x0 = (h-self.crop_center) //2
            y0 = (w-self.crop_center) //2
            input = input[:,:,x0:x0+self.crop_center,y0:y0+self.crop_center]
        return self.model(input)

