import torch
from torch import nn



# reference: https://github.com/Jongchan/attention-module

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.in_channel = in_channel
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(),
            nn.Flatten(in_channel // reduction, in_channel)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.attn = nn.Softmax()

    def forward(self, x):
        max_y = self.mlp(self.max_pool(x))
        avg_y = self.mlp(self.avg_pool(x))
        out = self.attn(max_y + avg_y)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.attn = nn.Sigmoid()

    def forward(self, x):
        max_y, _ = torch.max(x, dim=1, keepdim=True)
        avg_y, _ = torch.mean(x, dim=1, keepdim=True)

        y = torch.cat([max_y, avg_y], 1)
        out = self.attn(self.conv(y))

        return out


class CBAMBlock(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=49):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_channel=in_channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x * self.ca(x)
        y = y * self.sa(y)
        return x + y


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    cbam = CBAMBlock(3)
    y = cbam(x)
    print(y)
    print(y.shape)