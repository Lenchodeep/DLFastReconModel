from UnetParts import DoubleConv
import torch
import torch.nn as nn
import torch.functional as F

from PartialConv import *
import numpy as np

class PConvUnet(nn.Module):
    def __init__(self, in_channels, out_channels,  args):
        super(PConvUnet, self).__init__()
        # self.upmode = upmode
        # self.layers = layers
        self.inchannels = in_channels
        self.outchannels = out_channels
        self.fMapNum = args.fMapNum

        self.enc_PC1 = DoublePConv(self.inchannels, self.fMapNum)
        self.enc_PC2 = DoublePConv(self.fMapNum,self.fMapNum*2)
        self.enc_PC3 = DoublePConv(self.fMapNum*2, self.fMapNum*4)
        self.enc_PC4 = DoublePConv(self.fMapNum*4, self.fMapNum*8)

        self.bottle_neck = DoublePConv(self.fMapNum*8, self.fMapNum*16)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec_PC1 = DoublePConv(self.fMapNum*(16+8), self.fMapNum*8)
        self.dec_PC2 = DoublePConv(self.fMapNum*(8+4), self.fMapNum*4)
        self.dec_PC3 = DoublePConv(self.fMapNum*(4+2), self.fMapNum*2)
        self.dec_PC4 = DoublePConv(self.fMapNum*(2+1), self.fMapNum*1)

        self.outLayer = PartialConv(self.fMapNum, self.outchannels, kernel_size=3, padding=1)

    def forward(self, input, mask):
        x1, msk1 = self.enc_PC1(input, mask)
        x1_d = self.down(x1)
        msk1_d = self.down(msk1)
        
        x2, msk2 = self.enc_PC2(x1_d, msk1_d)
        x2_d = self.down(x2)
        msk2_d = self.down(msk2)

        x3, msk3 = self.enc_PC3(x2_d, msk2_d)
        x3_d = self.down(x3)
        msk3_d = self.down(msk3)

        x4, msk4 = self.enc_PC4(x3_d, msk3_d)
        x4_d = self.down(x4)
        msk4_d = self.down(msk4)


        bottle, msk_bottle = self.bottle_neck(x4_d, msk4_d)

        bottle_u = self.up(bottle)
        msk_bottle_u = self.up(msk_bottle)
        bottle_u = torch.cat([bottle_u, x4], dim=1)
        msk_bottle_u = torch.cat([msk_bottle_u, msk4], dim = 1)

        x5 , msk5 = self.dec_PC1(bottle_u, msk_bottle_u)

        x5_u = self.up(x5)
        msk5_u = self.up(msk5)
        x5_u = torch.cat([x5_u, x3], dim=1)
        msk5_u = torch.cat([msk5_u, msk3], dim = 1)

        x6, msk6 = self.dec_PC2(x5_u, msk5_u)

        x6_u = self.up(x6)
        msk6_u = self.up(msk6)
        x6_u = torch.cat([x6_u, x2], dim=1)
        msk6_u = torch.cat([msk6_u, msk2], dim=1)

        x7, msk7 = self.dec_PC3(x6_u, msk6_u)

        x7_u = self.up(x7)
        msk7_u = self.up(msk7)
        x7_u = torch.cat([x7_u, x1], dim=1)
        msk7_u = torch.cat([msk7_u, msk1], dim=1)

        x8, msk8 = self.dec_PC4(x7_u, msk7_u)

        out, msk_out = self.outLayer(x8, msk8)

        return out, msk_out











