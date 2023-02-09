import os
import torch.nn as nn
import torch
from torch.nn.modules import channelshuffle
from torch.nn.modules.activation import ReLU
from torch.nn.modules.normalization import GroupNorm
from torch.nn.parameter import Parameter

from torchsummary import summary

from Modules import DeepWiseConv

### CBAM block part
class ChannelAttention(nn.Module):
    '''
        The code of Squeeze-and-Excitation module
        paras:
            in_channels: the input channels number
            ratio: the hiddenlayer's channels decrease ratio
    '''
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # using a number to represent the whole feature map
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
    '''
        The code of spatial attention
        kernel_size: the size of conv operation
    '''
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2,1,kernel_size,padding = kernel_size //2) #input channel number is 2, because concat max and avg feature map together
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv(x)
        out = self.sigmoid(x)
        return out

class CBAM(nn.Module):
    '''
        Implement of the CBAM module.
    '''
    def __init__(self, in_channels, ratio = 8, kernel_size = 7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):

        caWeight = self.ca(x)
        x1 = caWeight*x
        saWeight = self.sa(x1)
        out = saWeight*x1

        return out

############################################ SE LAYER ################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


#### SK conv
class SKConv(nn.Module):
    '''
        The implement of the SKConv
        paras: 
            features: the channel number of the input feature map
            M : the number of the conv we use
            G: group number
            ratio: the ratio of features decrease

    '''
    def __init__(self, features, M, G, ratio, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / ratio), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])

        for i in range(M):
            #using different conv with different kernel size
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True)
                )
            )
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])

        for i in range(M):
            self.fcs.append(nn.Linear(d, features))

        self.softmax = nn.Softmax(dim = 1)


    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas,fea], dim=1)

        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)   # avg pooling operation
        fea_z = self.fc(fea_s)
 
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1) 
        return fea_v

### ESPA PART
## referance: EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network

class SEWeightModule(nn.Module):
    '''
        channels: the channels of input
        ratio: the hiddenlayer's channels decrease ratio
    '''
    def __init__(self, channels, ratio = 8):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Conv2d(channels//ratio, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class PSAModule(nn.Module):
    '''
        planes: the channels of the input
        conv_kernels: the different kernels for extract the features which also used to calculate the padding and groups
    '''
    def __init__(self, planes, conv_kernels = [3,5,7,9], stride=1):
        super(PSAModule, self).__init__()

        self.conv_1 = nn.Conv2d(planes, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                                stride = stride, groups=int(2**((conv_kernels[0] - 1)/2)))
        self.conv_2 = nn.Conv2d(planes, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                                stride = stride, groups=int(2**((conv_kernels[1] - 1)/2)))
        self.conv_3 = nn.Conv2d(planes, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                                stride = stride, groups=int(2**((conv_kernels[2] - 1)/2)))
        self.conv_4 = nn.Conv2d(planes, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                                stride = stride, groups=int(2**((conv_kernels[3] - 1)/2)))


        self.se = SEWeightModule(planes // 4,ratio=8)

        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        batchSize = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1,x2,x3,x4), dim = 1) 
        feats = feats.view(batchSize, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim = 1)
        attention_vectors = x_se.view(batchSize, 4,self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors

        for i in range(4):
            x_se_weight_fp = feats_weight[:,i,:,:]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), dim=1)
        
        return out

#### SA module : SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS
class ShuffleAttention(nn.Module):
    def __init__(self, channels, groups = 16):
        super(ShuffleAttention, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channels//(2*groups),1,1))
        self.cbias = Parameter(torch.zeros(1, channels//(2*groups),1,1))
        self.sweight = Parameter(torch.zeros(1, channels//(2*groups),1,1))
        self.sbias = Parameter(torch.zeros(1, channels//(2*groups),1,1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channels // (2*groups), channels // (2*groups))  

    def channel_shuffle(self,x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h,w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):   
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        #channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        #spatial attention
        xs = self.gn(x_1)
        xs =- self.sweight* xs + self.sbias
        xs = x_1*self.sigmoid(xs)

        #concat 
        out = torch.cat([xn,xs], dim = 1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out,2)

        return out

### SGE Module  
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x



####################################  SOAM PART  ##########################################################
'''
    THE SOAM ATTENTION MODULE ARISED IN SOGAN
    THE ORIGINAL GITHUB LINK: https://github.com/SunnyZhouWenZhong/SOGAN
'''


class SOAM(nn.Module):
    def __init__(self, channels):
        super(SOAM, self).__init__()

        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.lRelu1 = nn.LeakyReLU(0.2, True)

        self.convAtt = nn.Conv2d(channels, channels, kernel_size=1)
        self.bnAtt = nn.BatchNorm2d(channels) 
        self.lReluAtt = nn.LeakyReLU(0.2, True)

        self.convOutput = nn.Conv2d(channels, channels, kernel_size=1)
        self.bnOutput = nn.BatchNorm2d(channels)
        self.lReluOutput = nn.LeakyReLU(0.2, True)


    def forward(self, input):
        B,C,H,W = input.size()
        #### channel Fusion
        C1 = self.lRelu1(self.bn1(self.conv1(input)))
        C1 = torch.squeeze(C1)
        if len(C1.size())==2:
            C1=torch.unsqueeze(C1, dim = 0)

        #### seperation
        ## DMM
        C1_trans = C1.transpose(1, 2)
        D1 = torch.matmul(C1,C1_trans)
        D2 = torch.matmul(C1_trans, C1)

        S1 = torch.softmax(D1, -1)
        S2 = torch.softmax(D2, -1)

        C2 = self.lReluAtt(self.bnAtt(self.convAtt(input)))
        R1 = C2.reshape(B, H, C*W)
        R2 = C2.reshape(B, W, C*H)

        M1 = torch.matmul(S1, R1)
        M2 = torch.matmul(S2, R2)

        R3 = M1.reshape(B, C, H, W)
        R4 = M2.reshape(B, C, H, W)
        outConvIn = torch.add(R3,R4)

        C3 = self.lReluOutput(self.bnOutput(self.convOutput(outConvIn)))

        out = torch.add(C2, C3)

        return  out, S1, S2 

class SESOAM(nn.Module):
    def __init__(self, channels, group = 2):
        super(SESOAM, self).__init__()
        self.channelAtt = ChannelAttention(channels // group, ratio=8)

        self.selfAtt = SOAM(channels//group)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
    def channel_shuffle(self,x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h,w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, input):
        b, c , h, w = input.shape
        ## split the input feature into group
        x0, x1 = input.chunk(2, dim=1) 
        ## calcu the channel attention feature
        channel_att = self.channelAtt(x0) * x0
        self_att, att_hh, att_ww = self.selfAtt(x1)
        out = torch.cat([channel_att, self_att], dim=1)
        ## channel shuffle operation
        out  = self.channel_shuffle(out,2)
        ## feature merge
        out = self.conv(out)

        return out 

##################################### coordinate attention ###############################################
"""
    The coordinate attention which is proposed in cvpr
    It encodes both the channel relationship and long range dependencies.
"""
class activateLayer(nn.Module):
    """
        This activate layer is a modified relu activation.
    """
    def __init__(self, inplace=True):
        super(activateLayer, self).__init__()
        self.relu = nn.ReLU6(inplace = inplace)

    def forward(self, x):
        return x * (self.relu(x+3)/6)

class CoordAtt(nn.Module):
    def __init__(self, inchannels, outchannels, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None,1))
        self.pool_w = nn.AdaptiveAvgPool2d((1,None))

        mip = max(8,inchannels//reduction)

        self.conv1 = nn.Conv2d(inchannels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act =activateLayer(inplace=True) 

        self.conv_h = nn.Conv2d(mip, outchannels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, outchannels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        input = x

        _,_,H,W = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)  ## 调换H跟W的维度，方便后面进行cat操作

        y = torch.cat([x_h, x_w], dim = 2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h,x_w = torch.split(y, [H,W], dim=2)
        x_w = x_w.permute(0,1,3,2)

        att_h = torch.sigmoid(self.conv_h(x_h))
        att_w = torch.sigmoid(self.conv_w(x_w))

        out = input* att_h*att_w
        return out


"""
    A combination of the dw conv and coordinate attention
"""
class DWCoordAtt(nn.Module):
    def __init__(self, inchannels, outchannels, stride, expand_ratio):
        super(DWCoordAtt, self).__init__()

        hidden_dim = round(inchannels * expand_ratio)
        self.identity = stride==1 and inchannels == outchannels

        self.Conv = nn.Sequential(
            ##pw
            nn.Conv2d(inchannels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace = True),
            #dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            #coordinate attention
            CoordAtt(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, outchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannels)
        )

    def forward(self, x):
        y = self.Conv(x)
        if self.identity:
            return x+y
        else:
            return y


######################################  self attention  ##################################################
'''
https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
'''
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B  C  W  H)
            returns :
                out : self attention value + input feature 
                attention: B  N  N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X (N) X C
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        print('self attention')
        return out
    




# tensor  = torch.rand((4,64,256,256))
# model = SOAM(16, False)
# model = model.to("cuda:0")
# summary(model, (16,256,256),  device='cuda')
# model(tensor)