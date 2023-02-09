from numpy import packbits
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torchvision import models



class PartialConv(nn.Module):
    def weight_init(self, m):
        classname = m.__class__.__name__    ##获取当前结构名称
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.002)  ##正态分布初始化
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)   ##正态分布初始化
            nn.init.zeros_(m.bias)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0,dilation = 1, group=1, bias = False ):
        super(PartialConv, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, group, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, group, False)                                
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        self.input_conv.apply(self.weight_init)

        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, input, mask):
        output = self.input_conv(input*mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1,-1,1,1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActive(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, activ='leaky', conv_bias = False):
        super(PCBActive, self).__init__()

        self.conv = PartialConv(in_channels, out_channels, 3, stride=stride, padding=1, bias= conv_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        
    def forward(self, input, mask):
        out, out_mask = self.conv(input, mask)
        out = self.bn(out)
        out = self.activation(out)

        return out, out_mask

class DoublePConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride = 1, active='leaky', conv_bias = False):
        super(DoublePConv, self).__init__()
        self.pconv1 = PCBActive(inchannels, outchannels, stride=stride, activ= active, conv_bias=conv_bias) 
        self.pconv2 = PCBActive(outchannels, outchannels, stride=stride, activ= active, conv_bias=conv_bias) 

    def forward(self, input, mask):
        out1, mask1 = self.pconv1(input, mask)
        out, out_mask = self.pconv2(out1, mask1)

        return out, out_mask
