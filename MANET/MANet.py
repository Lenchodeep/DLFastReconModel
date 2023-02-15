import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class Single_level_densenet(nn.Module): 
    def __init__(self,filters, num_conv = 4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters,filters,3, padding = 1))
            self.bn_list.append(nn.BatchNorm2d(filters))
            
    def forward(self,x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final
    
class Down_sample(nn.Module):
    def __init__(self,kernel_size = 2, stride = 2):
        super(Down_sample, self).__init__()
        self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)
    
    def forward(self,x):
        y = self.down_sample_layer(x)
        return y,x

class Upsample_n_Concat_1(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_1, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_2(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_2, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_3(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_3, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_4(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_4, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_T1(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_T1, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(filters,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x):
        x = self.upsample_layer(x)
        x = F.relu(self.bn(self.conv(x)))
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)    

class Dense_Unet_k(nn.Module):
    def __init__(self, in_chan, out_chan, filters, num_conv = 4):  #64   256
        super(Dense_Unet_k, self).__init__()
        self.conv1T1 = nn.Conv2d(in_chan,filters,1)
        self.conv1T2 = nn.Conv2d(2,filters,1)
        self.convdemD0 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemD1 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemD2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemD3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU0 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU1 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        

        self.dT1_1 = Single_level_densenet(filters,num_conv )
        self.downT1_1 = Down_sample()
        self.dT1_2 = Single_level_densenet(filters,num_conv )
        self.downT1_2 = Down_sample()
        self.dT1_3 = Single_level_densenet(filters,num_conv )
        self.downT1_3 = Down_sample()
        self.dT1_4 = Single_level_densenet(filters,num_conv )
        self.downT1_4 = Down_sample()

        self.dT2_1 = Single_level_densenet(filters,num_conv )
        self.downT2_1 = Down_sample()
        self.dT2_2 = Single_level_densenet(filters,num_conv )
        self.downT2_2 = Down_sample()
        self.dT2_3 = Single_level_densenet(filters,num_conv )
        self.downT2_3 = Down_sample()
        self.dT2_4 = Single_level_densenet(filters,num_conv )
        self.downT2_4 = Down_sample()

        self.bottom_T1 = Single_level_densenet(filters,num_conv )
        self.bottom_T2 = Single_level_densenet(filters,num_conv )

        self.up4_T1 = Upsample_n_Concat_T1(filters)
        self.u4_T1 = Single_level_densenet(filters,num_conv )
        self.up3_T1 = Upsample_n_Concat_T1(filters)
        self.u3_T1 = Single_level_densenet(filters,num_conv )
        self.up2_T1 = Upsample_n_Concat_T1(filters)
        self.u2_T1 = Single_level_densenet(filters,num_conv )
        self.up1_T1 = Upsample_n_Concat_T1(filters)
        self.u1_T1 = Single_level_densenet(filters,num_conv )

        self.up4_T2 = Upsample_n_Concat_1(filters)
        self.u4_T2 = Single_level_densenet(filters,num_conv )
        self.up3_T2 = Upsample_n_Concat_2(filters)
        self.u3_T2 = Single_level_densenet(filters,num_conv )
        self.up2_T2 = Upsample_n_Concat_3(filters)
        self.u2_T2 = Single_level_densenet(filters,num_conv )
        self.up1_T2 = Upsample_n_Concat_4(filters)
        self.u1_T2 = Single_level_densenet(filters,num_conv )

        self.outconvT1 = nn.Conv2d(filters,out_chan, 1)
        self.outconvT2 = nn.Conv2d(64,out_chan, 1)

        self.atten_depth_channel_0=ChannelAttention(64)
        self.atten_depth_channel_1=ChannelAttention(64)
        self.atten_depth_channel_2=ChannelAttention(64)
        self.atten_depth_channel_3=ChannelAttention(64)

        self.atten_depth_channel_U_0=ChannelAttention(64)
        self.atten_depth_channel_U_1=ChannelAttention(64)
        self.atten_depth_channel_U_2=ChannelAttention(64)
        self.atten_depth_channel_U_3=ChannelAttention(64)

        self.atten_depth_spatial_0=SpatialAttention()
        self.atten_depth_spatial_1=SpatialAttention()
        self.atten_depth_spatial_2=SpatialAttention()
        self.atten_depth_spatial_3=SpatialAttention()

        self.atten_depth_spatial_U_0=SpatialAttention()
        self.atten_depth_spatial_U_1=SpatialAttention()
        self.atten_depth_spatial_U_2=SpatialAttention()
        self.atten_depth_spatial_U_3=SpatialAttention()



        
        
    def forward(self,T1, T2):

        T1_x1 = self.conv1T1(T1)
        T2 = torch.cat((T2,T1),dim=1)
        T2_x1 = self.conv1T2(T2)


        T1_x2,T1_y1 = self.downT1_1(self.dT1_1(T1_x1))
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        temp = T1_x2.mul(self.atten_depth_channel_0(T1_x2))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        T12_x2 = T2_x2.mul(temp)+T2_x2


        T1_x3,T1_y2 = self.downT1_1(self.dT1_2(T1_x2))
        T2_x3,T2_y2 = self.downT2_1(self.dT2_2(T12_x2))
        temp = T1_x3.mul(self.atten_depth_channel_1(T1_x3))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        T12_x3 = T2_x3.mul(temp)+T2_x3



        T1_x4,T1_y3 = self.downT1_1(self.dT1_3(T1_x3))
        T2_x4,T2_y3 = self.downT2_1(self.dT2_3(T12_x3))
        temp = T1_x4.mul(self.atten_depth_channel_2(T1_x4))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        T12_x4 = T2_x4.mul(temp)+T2_x4        




        T1_x5,T1_y4 = self.downT1_1(self.dT1_4(T1_x4))
        T2_x5,T2_y4 = self.downT2_1(self.dT2_4(T12_x4))
        temp = T1_x5.mul(self.atten_depth_channel_3(T1_x5))
        temp = temp.mul(self.atten_depth_spatial_3(temp))
        T12_x5 = T2_x5.mul(temp)+T2_x5   


        T1_x = self.bottom_T1(T1_x5)
        T2_x = self.bottom_T2(T12_x5)


        T1_1x = self.u4_T1(self.up4_T1(T1_x))
        T2_1x = self.u4_T2(self.up4_T2(T2_x,T2_y4))
        temp = T1_1x.mul(self.atten_depth_channel_U_0(T1_1x))
        temp = temp.mul(self.atten_depth_spatial_U_0(temp))
        T12_x = T2_1x.mul(temp)+T2_1x   



        T1_2x = self.u3_T1(self.up3_T1(T1_1x))
        T2_2x = self.u3_T2(self.up3_T2(T12_x,T2_y3))
        temp = T1_2x.mul(self.atten_depth_channel_U_1(T1_2x))
        temp = temp.mul(self.atten_depth_spatial_U_1(temp))
        T12_x = T2_2x.mul(temp)+T2_2x   

        T1_3x = self.u2_T1(self.up2_T1(T1_2x))
        T2_3x = self.u2_T2(self.up2_T2(T12_x,T2_y2))
        temp = T1_3x.mul(self.atten_depth_channel_U_2(T1_3x))
        temp = temp.mul(self.atten_depth_spatial_U_2(temp))
        T12_x = T2_3x.mul(temp)+T2_3x    

        T1_4x = self.u1_T1(self.up1_T1(T1_3x))
        T2_4x = self.u1_T2(self.up1_T2(T12_x,T2_y1))
        temp = T1_4x.mul(self.atten_depth_channel_U_3(T1_4x))
        temp = temp.mul(self.atten_depth_spatial_U_3(temp))
        T12_x = T2_4x.mul(temp)+T2_4x   

        T1 = self.outconvT1(T1_4x)
        T2 = self.outconvT2(T12_x)
        
        return T1,T2


class Dense_Unet_img(nn.Module):
    def __init__(self, in_chan, out_chan, filters, num_conv = 4):  
        super(Dense_Unet_img, self).__init__()
        self.conv1T1 = nn.Conv2d(in_chan,filters,1)
        self.conv1T2 = nn.Conv2d(2,filters,1)

        self.dT1_1 = Single_level_densenet_img(filters,num_conv )
        self.downT1_1 = Down_sample_img()
        self.dT1_2 = Single_level_densenet_img(filters,num_conv )
        self.downT1_2 = Down_sample_img()
        self.dT1_3 = Single_level_densenet_img(filters,num_conv )
        self.downT1_3 = Down_sample_img()
        self.dT1_4 = Single_level_densenet_img(filters,num_conv )
        self.downT1_4 = Down_sample_img()

        self.dT2_1 = Single_level_densenet_img(filters,num_conv )
        self.downT2_1 = Down_sample_img()
        self.dT2_2 = Single_level_densenet_img(filters,num_conv )
        self.downT2_2 = Down_sample_img()
        self.dT2_3 = Single_level_densenet_img(filters,num_conv )
        self.downT2_3 = Down_sample_img()
        self.dT2_4 = Single_level_densenet_img(filters,num_conv )
        self.downT2_4 = Down_sample_img()

        self.bottom_T1 = Single_level_densenet_img(filters,num_conv )
        self.bottom_T2 = Single_level_densenet_img(filters,num_conv )

        self.up4_T1 = Upsample_n_Concat_T1_img(filters)
        self.u4_T1 = Single_level_densenet_img(filters,num_conv )
        self.up3_T1 = Upsample_n_Concat_T1_img(filters)
        self.u3_T1 = Single_level_densenet_img(filters,num_conv )
        self.up2_T1 = Upsample_n_Concat_T1_img(filters)
        self.u2_T1 = Single_level_densenet_img(filters,num_conv )
        self.up1_T1 = Upsample_n_Concat_T1_img(filters)
        self.u1_T1 = Single_level_densenet_img(filters,num_conv )

        self.up4_T2 = Upsample_n_Concat_1_img(filters)
        self.u4_T2 = Single_level_densenet_img(filters,num_conv )
        self.up3_T2 = Upsample_n_Concat_2_img(filters)
        self.u3_T2 = Single_level_densenet_img(filters,num_conv )
        self.up2_T2 = Upsample_n_Concat_3_img(filters)
        self.u2_T2 = Single_level_densenet_img(filters,num_conv )
        self.up1_T2 = Upsample_n_Concat_4_img(filters)
        self.u1_T2 = Single_level_densenet_img(filters,num_conv )

        self.outconvT1 = nn.Conv2d(filters,out_chan, 1)
        self.outconvT2 = nn.Conv2d(64,out_chan, 1)
        #Components of DEM module
        self.atten_depth_channel_0=ChannelAttention_img(64)
        self.atten_depth_channel_1=ChannelAttention_img(64)
        self.atten_depth_channel_2=ChannelAttention_img(64)
        self.atten_depth_channel_3=ChannelAttention_img(64)

        self.atten_depth_channel_U_0=ChannelAttention_img(64)
        self.atten_depth_channel_U_1=ChannelAttention_img(64)
        self.atten_depth_channel_U_2=ChannelAttention_img(64)
        self.atten_depth_channel_U_3=ChannelAttention_img(64)

        self.atten_depth_spatial_0=SpatialAttention_img()
        self.atten_depth_spatial_1=SpatialAttention_img()
        self.atten_depth_spatial_2=SpatialAttention_img()
        self.atten_depth_spatial_3=SpatialAttention_img()

        self.atten_depth_spatial_U_0=SpatialAttention_img()
        self.atten_depth_spatial_U_1=SpatialAttention_img()
        self.atten_depth_spatial_U_2=SpatialAttention_img()
        self.atten_depth_spatial_U_3=SpatialAttention_img()



        
        
    def forward(self,T1, T2):

        T1_x1 = self.conv1T1(T1)
        T2 = torch.cat((T2,T1),dim=1)
        T2_x1 = self.conv1T2(T2)

        T1_x2,T1_y1 = self.downT1_1(self.dT1_1(T1_x1))
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        temp = T1_x2.mul(self.atten_depth_channel_0(T1_x2))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        T12_x2 = T2_x2.mul(temp)+T2_x2


        T1_x3,T1_y2 = self.downT1_1(self.dT1_2(T1_x2))
        T2_x3,T2_y2 = self.downT2_1(self.dT2_2(T12_x2))
        temp = T1_x3.mul(self.atten_depth_channel_1(T1_x3))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        T12_x3 = T2_x3.mul(temp)+T2_x3

        T1_x4,T1_y3 = self.downT1_1(self.dT1_3(T1_x3))
        T2_x4,T2_y3 = self.downT2_1(self.dT2_3(T12_x3))
        temp = T1_x4.mul(self.atten_depth_channel_2(T1_x4))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        T12_x4 = T2_x4.mul(temp)+T2_x4        


        T1_x5,T1_y4 = self.downT1_1(self.dT1_4(T1_x4))
        T2_x5,T2_y4 = self.downT2_1(self.dT2_4(T12_x4))
        temp = T1_x5.mul(self.atten_depth_channel_3(T1_x5))
        temp = temp.mul(self.atten_depth_spatial_3(temp))
        T12_x5 = T2_x5.mul(temp)+T2_x5 


        T1_x = self.bottom_T1(T1_x5)
        T2_x = self.bottom_T2(T12_x5)


        T1_1x = self.u4_T1(self.up4_T1(T1_x))
        T2_1x = self.u4_T2(self.up4_T2(T2_x,T2_y4))
        temp = T1_1x.mul(self.atten_depth_channel_U_0(T1_1x))
        temp = temp.mul(self.atten_depth_spatial_U_0(temp))
        T12_x = T2_1x.mul(temp)+T2_1x

        T1_2x = self.u3_T1(self.up3_T1(T1_1x))
        T2_2x = self.u3_T2(self.up3_T2(T12_x,T2_y3))
        temp = T1_2x.mul(self.atten_depth_channel_U_1(T1_2x))
        temp = temp.mul(self.atten_depth_spatial_U_1(temp))
        T12_x = T2_2x.mul(temp)+T2_2x  


        T1_3x = self.u2_T1(self.up2_T1(T1_2x))
        T2_3x = self.u2_T2(self.up2_T2(T12_x,T2_y2))
        temp = T1_3x.mul(self.atten_depth_channel_U_2(T1_3x))
        temp = temp.mul(self.atten_depth_spatial_U_2(temp))
        T12_x = T2_3x.mul(temp)+T2_3x  

        T1_4x = self.u1_T1(self.up1_T1(T1_3x))
        T2_4x = self.u1_T2(self.up1_T2(T12_x,T2_y1))
        temp = T1_4x.mul(self.atten_depth_channel_U_3(T1_4x))
        temp = temp.mul(self.atten_depth_spatial_U_3(temp))
        T12_x = T2_4x.mul(temp)+T2_4x  

        T1 = self.outconvT1(T1_4x)
        T2 = self.outconvT2(T12_x)
        
        return T1,T2

class Single_level_densenet_img(nn.Module):
    def __init__(self,filters, num_conv = 4):
        super(Single_level_densenet_img, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters,filters,3, padding = 1))
            self.bn_list.append(nn.BatchNorm2d(filters))
            
    def forward(self,x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final
    
class Down_sample_img(nn.Module):
    def __init__(self,kernel_size = 2, stride = 2):
        super(Down_sample_img, self).__init__()
        self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)
    
    def forward(self,x):
        y = self.down_sample_layer(x)
        return y,x

class Upsample_n_Concat_1_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_1_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(filters*2,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_2_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_2_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_3_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_3_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_4_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_4_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_T1_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_T1_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(filters,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x):
        x = self.upsample_layer(x)
        x = F.relu(self.bn(self.conv(x)))
        return x
class ChannelAttention_img(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_img, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention_img(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_img, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class Generator(nn.Module):
    def __init__(self,args):
        super(Generator,self).__init__()
        self.KUnet = Dense_Unet_img(1,1,64,4)
        self.IUnet = Dense_Unet_img(1,1,64,4)

        self.filters = args.filters
        self.device = args.device

        ## init the mask
        mask_path = args.mask_path
        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
        self.mask = torch.tensor(masks['mask1'] == 1, device=args.device)
        self.maskNot = self.mask == 0

    # def inverseFT(self, Kspace):
    #     """The input Kspace has two channels(real and img)"""
    #     Kspace = Kspace.permute(0, 2, 3, 1)
    #     img_cmplx = torch.fft.ifft2(Kspace, dim=(1,2))
    #     img = torch.sqrt(img_cmplx[:, :, :, 0]**2 + img_cmplx[:, :, :, 1]**2)
    #     img = img[:, None, :, :]
    #     return img
    def fftshift(self, img):
        '''
            4d tensor FFT operation
        '''
        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2
    
    def FT(self, image):
        '''
            Fourier operation 
        '''

        kspace_cplx = self.fftshift(torch.fft.fft2(image, dim=(2,3)))
        return kspace_cplx

    def inverseFT(self, Kspace):
        """The input Kspace has two channels(real and img)"""
        Kspace = Kspace.permute(0, 2, 3, 1)
        img_cmplx = torch.fft.ifft2(Kspace, dim=(1,2))
        img = torch.absolute(img_cmplx[:,:,:,0]+ 1j*img_cmplx[:,:,:,1])
        img = img[:, None, :, :]
        return img

    # def KDC_layer(self, rec_K, und_K):
    #     '''
    #         K data consistency layer
    #     '''
    #     rec_Kspace = (self.mask*torch.complex(und_K[:, 0, :, :], und_K[:, 1, :, :]) + self.maskNot*torch.complex(rec_K[:, 0, :, :], rec_K[:, 1, :, :]))[:, None, :, :]
    #     final_rec =torch.absolute(torch.fft.ifft2(self.fftshift(rec_Kspace),dim=(2,3)))

    #     return final_rec, rec_Kspace
    
    def IDC_layer(self, rec_Img, und_K):
        und_k_cplx = torch.complex(und_K[:,0,:,:], und_K[:,1,:,:])[:,None,:,:] 
        rec_k = self.FT(rec_Img)
        final_k = (torch.mul(self.mask, und_k_cplx) + torch.mul(self.maskNot, rec_k))
        final_rec  =  torch.absolute(torch.fft.ifft2(self.fftshift(final_k), dim=(2,3)))

        return final_rec

    def forward(self, T1img, T2img, T1K, T2K):
        recon_T1, recon_T2 = self.KUnet(T1img, T2img)

        recon_mid_T1 = self.IDC_layer(recon_T1, T1K)
        recon_mid_T2 = self.IDC_layer(recon_T2, T2K)

        rec_T1, rec_T2 = self.IUnet(recon_mid_T1,recon_mid_T2)
        rec_T1 = self.IDC_layer(rec_T1, T1K)
        rec_T2 = self.IDC_layer(rec_T2, T2K)
        # rec_T1 = torch.clamp(F.tanh(rec_final_T1+recon_mid_T1), 0, 1)
        # rec_T2 = torch.clamp(F.tanh(re_final_T2+recon_mid_T2), 0, 1)

        return rec_T1, recon_mid_T1, rec_T2, recon_mid_T2
