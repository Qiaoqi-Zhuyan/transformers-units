import torch
from torch import nn

if __name__ == '__main__':
    x = torch.randn(1, 3, 18, 18)
    print(f'before fft : {x}')
    fft_reasult = torch.fft.rfftn(x)
    print(f'after fft: {fft_reasult}')
    inv_fft = torch.fft.irfftn(fft_reasult)
    print(f'inv fft : {inv_fft}')