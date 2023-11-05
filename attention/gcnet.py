# reference: https://blog.csdn.net/yumaomi/article/details/125588724
# reference: https://github.com/xvjiarui/GCNet/blob/a9fcc88c4bd3a0b89de3678b4629c9dfd190575f/mmdet/ops/gcb/context_block.py#L13

import torch
from torch import nn
import torch.nn.functional as F
import einops

'''
GCNet, 类似SEModule
1. 结合non-local操作,考虑所有位置的特征的机制
2. 全局上下文信息通过对所有位置的特征进行平均池化来获得
3. 使用1x1 卷积和激活函数得到特征, 与原始图像相乘 

Non-local
'''
# no longer use ...
class GCblock(nn.Module):
    def __init__(self, in_channel, expan_ration=16, fusion_type='add'):
        super(GCblock, self).__init__()
        assert  fusion_type == "add" or fusion_type == "mul"
        self.fusion_type = fusion_type
        self.in_channel = in_channel
        self.reduction = expan_ration
        self.hidden_dim = int(in_channel * expan_ration)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_mask = nn.Conv2d(in_channel, 1, kernel_size=1)
        self.softmax = nn.Softmax()

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hidden_dim, 1, stride=1, bias=False),
            nn.LazyBatchNorm2d([self.hidden_dim, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.in_channel, 1, stride=1, bias=False)
        )

    def spatial_pool(self, x):
        x_ = x

        x = einops.rearrange(x, 'b c h w -> b 1 c (h * w)')
        context_mask = self.softmax(self.conv_mask(x))
        context_mask = einops.rearrange(context_mask, "b 1 (h * w) -> b 1 (h * w) 1")
        context = torch.matmul(x_, context_mask)
        context = einops.rearrange(context, "b 1 c 1 -> b c 1 1")

        return context

    def forward(self, x):
        context = self.spatial_pool(x)

        if self.fusion_type == "mul":
            out = self.attn(self.conv(context))
            out = x * out
        if self.fusion_type == "add":
            out = self.attn(self.conv(context))
            out = out + x


        return out



# reference: https://paperswithcode.com/method/global-context-block
class GlobalContextBlock(nn.Module):
    def __init__(self, in_channel, expan_ration, fusion_type='add'):
        super(GlobalContextBlock, self).__init__()
        assert fusion_type == 'add' or fusion_type == 'mul'
        self.fusion_type = fusion_type
        self.in_channel = in_channel
        self.hidden_dim = int(in_channel * expan_ration)
        self.conv_v = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hidden_dim, 1, stride=1, bias=False),
            nn.LazyBatchNorm2d([self.hidden_dim, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.in_channel, 1, stride=1, bias=False)
        )

        self.conv_k = nn.Conv2d(self.in_channel, self.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        key = self.softmax(self.conv_k(x))
        key = einops.rearrange(key, "b 1 h w -> b (h w) 1")
        query = einops.rearrange(x, "b c h w -> b c (h w)")
        value = einops.rearrange(torch.matmul(key, query), "b (w h) 1 -> b c 1 1").contiguous()
        value = self.conv_v(value)

        if self.fusion_type == 'add':
            return value + x

        if self.fusion_type == 'mul':
            return value * x
