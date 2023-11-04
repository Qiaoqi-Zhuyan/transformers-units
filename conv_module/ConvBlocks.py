# Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mobile_vit.py
import torch
from torch import nn

class conv_nxn_bn(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, act=nn.ReLU6):
        super(conv_nxn_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            act()
        )

    def forward(self, x):
        return self.conv(x)


class conv_3x3_bn(nn.Module):
    def __init__(self, in_channel, out_channel, act_layer=nn.ReLU6, downsample=False):
        super(conv_3x3_bn, self).__init__()
        self.stride = 1 if downsample == False else 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=self.stride, bias=False),
            nn.BatchNorm2d(out_channel),
            act_layer()
        )
    def forward(self, x):
        return self.conv(x)


class SEModule(nn.Module):
    '''
     SEmodule, 用于模型中的特征重标定，首先对特征图进行全局平均池化以获得每个
     通道的全局信息，然后通过两个全连接层来获得每个通道的权重，最后对原始特侦图
     进行通道注意力调制
    会对特征图的每个通道进行加权，其权重是通过全局平均池化、全连接层（或称为密集层）和sigmoid激活函数学习得来的，使得网络可以重点关注更有用的特征。
     SE模组插入某些块中，增强模型对通道之间关系的捕获能力
     '''
    def __init__(self, in_channel, out_channel, reduction=4):
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(out_channel, in_channel // reduction, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.se(x)
        y = self.sigmoid(y).view(b, c, 1, 1)

        return x * y

class InvertedResiduals(nn.Module):
    '''
    参考: MobileNetv2 中 MV2Blcok 和 efficientnet中的MBConv 实现
    1. 扩张层,通过一个1x1的卷积层进行扩张,增加特征的表达能力
    2. DWConv操作,降低计算复杂度
    3. 1x1的卷积层将特征压缩回较小的维度, 作为线性瓶颈
    4. 对残差进行采样操作
    '''
    def __init__(self, in_channel, out_channel, downsample=False, expan_ratio=4):
        super(InvertedResiduals, self).__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(in_channel * expan_ratio)

        if downsample:
            self.max_pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        if expan_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channel, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),

                # pw
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv = nn.Sequential(
                # pw down_sample
                nn.Conv2d(in_channel, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SEModule(in_channel, hidden_dim),

                # linear
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        self.norm = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.proj(x)) + self.conv(self.norm(x))
        else:
            return x + self.conv(self.norm(x))







