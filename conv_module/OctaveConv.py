import torch
from torch import nn


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        print("after h2g_pool")
        print(f'X_h: {X_h.shape}')
        print(f'X_l: {X_l.shape}')

        X_h2l = self.h2g_pool(X_h)
        print(f'X_h->X_h2l: {X_h2l.shape}')

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)
        print(f'X_h->X_h2h: {X_h2h.shape}')
        print(f'X_l->X_l2h: {X_l2h.shape}')

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)
        print(f'X_l->X_l2l: {X_l2l.shape}')
        print(f'X_h2l->X_h2l: {X_h2l.shape}')

        X_l2h = self.upsample(X_l2h)
        print(f'X_l2h->X_l2h: {X_l2h.shape}')
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

if __name__ == '__main__':
    high = torch.Tensor(1, 64, 32, 32).cuda()
    low = torch.Tensor(1, 192, 16, 16).cuda()

    print(f'input high feature: {high.shape}')
    print(f'input low feature: {low.shape}')


    OCconv = OctaveConv(kernel_size=(3,3),in_channels=256,out_channels=512,bias=False,stride=2,alpha=0.75).cuda()
    i = high,low
    x_out,y_out = OCconv(i)
    print(f'out put: {x_out.size()}')
    print(f'out put: {y_out.size()}')


