import torch
from torch import nn

class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(DWConv, self).__init__()
        #dwconv
        self.depthwise = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False)
        # pwconv
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1, 0, 1,bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    dwconv_3x3 = DWConv(3, 64, 3)
    y = dwconv_3x3(x)
    print(y.shape)
