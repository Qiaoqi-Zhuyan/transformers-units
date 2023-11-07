import torch
from torch import nn

'''
yolov5的一种下采样方法, 可以很大程度的保留信息
这种方法会比pooling保留更多特征
'''
# modify from https://github.com/ultralytics/yolov5/blob/master/models/common.py#L248
class Conv_SiLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1 ):
        super(Conv_SiLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Focus(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1):
        super(Focus, self).__init__()
        self.conv = Conv_SiLU(in_channel * 4, out_channel, kernel_size, stride)
        self.fc = nn.Linear(in_channel * 4, out_channel)

    def forward(self, x):
        '''
        :param x: [b, c, w, h]
        :return: x' [b, 4c, w//2, h//2]
        '''

        x_0 = x[:, :, 0::2, 0::2]
        x_1 = x[:, :, 1::2, 0::2]
        x_2 = x[:, :, 0::2, 1::2]
        x_3 = x[:, :, 1::2, 1::2]

        x = torch.cat((x_0, x_1, x_2, x_3), 1)

        return self.conv(x)

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    f = Focus(3, 64)
    y = f(x)
    print(f'f shape: {y.shape}')
