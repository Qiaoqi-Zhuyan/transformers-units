import torch
from torch import nn

import torch.nn.functional as F
import einops

# modify from https://github.com/milesial/Pytorch-UNet/tree/master

class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim=None, kernel_size=3, padding=1):
        super(Conv_BN, self).__init__()
        if not hidden_dim:
            hidden_dim = out_channel
        self.conv1 = nn.Conv2d(in_channel, hidden_dim, kernel_size, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act_layer = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_channel, kernel_size, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act_layer(x)


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = Conv_BN(in_channel, out_channel)

    def forward(self, x):
        x = self.max_pool(x)
        return self.conv(x)


class UpSample(nn.Module):
    '''
        双线性插值, 根据四个最近的像素点估算新的像素点
        使用conv来降低通道
    '''
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(UpSample, self).__init__()
        if bilinear:
            self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = Conv_BN(in_channel, out_channel)
        else:
            self.up_sample = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = Conv_BN(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffY // 2,
                        diffY // 2, diffY - diffY // 2
                        ])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=False):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bilinear = bilinear

        self.in_conv = Conv_BN(in_channel, 64)
        self.downsample1 = DownSample(64, 128)
        self.downsample2 = DownSample(128, 256)
        self.downsample3 = DownSample(256, 512)
        factor = 2 if bilinear else 1
        self.downsample4 = DownSample(512, 1024 // factor)
        self.upsample1 = UpSample(1024, 512 // factor, bilinear)
        self.upsample2 = UpSample(512, 256 // factor, bilinear)
        self.upsample3 = UpSample(256, 128 // factor, bilinear)
        self.upsample4 = UpSample(128, 64, bilinear)
        self.out_conv = OutConv(64, out_channel)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)

        x = self.upsample1(x5, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)

        logits = self.out_conv(x)
        return logits



if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    unet = UNet(3, 3, bilinear=False)
    y = unet(x)
    print(y.shape)



