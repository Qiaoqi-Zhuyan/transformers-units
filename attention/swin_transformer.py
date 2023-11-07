import MySQLdb
import torch
from torch import nn
import einops
import numpy as np
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

'''
swin transformer
1. 设计W-MSA, W-MSA 每个窗口进行独立运算
2. 设计SW-MSA, 对图形进行移动,分割出不同窗口, 加强窗口交互
3. SwinTransformerBlock, 成对使用W-MSA和SW-MSA
4. Patch Merging进行下采样
5. 充分考虑到位移不变性, 尺寸不变性, 层次变深感受野越大等特性

思考: 考虑到cnn学习表征,较小感受野, transformer学习更深层次的特征,全局感受野
在浅层网络使用cnn,深层使用transformer
4 stages, 多个stage融合? res add blocks focus fpn partial conv
'''

# reference: https://arxiv.org/pdf/2103.14030v2.pdf
# Modify from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
'''
 1. 对swintransformerBlock 进行修改, 增加残差和采样优化计算量, 
 2. 对blocks进行特征对齐和融合
'''


# tool function
def window_partition(x, window_size):
    '''
    将输入图像切成窗口大小
    :param x: [b, h, w, c]
    :param window_size: window size int
    :return: [(b*h*w) window_size window_size c]
    '''
    B, H, W, C = x.shape
    x = einops.rearrange(x, 'b (h s1) (w s2) c -> b h s1 w s2 c', s1=window_size, s2=window_size)
    windows = einops.rearrange(x, 'b h s1 w s2 c -> b h w s1 s2 c')
    windows = einops.rearrange(windows, 'b h w s1 s2 c -> (b h w) s1 s2 c')
    return windows

def window_partition_official(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, h, w):
    '''
    从窗口恢复成图形格式
    :param windows: (num_windows*b, window_size, window_size, c)
    :param window_size: int
    :param h: height of img
    :param w: width of img
    :return:  [b h w c]
    '''
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = einops.rearrange(windows, '(b h w) s1 s2 c -> b h w s1 s2 c', b=int(b), s1=window_size, s2=window_size, w=w//window_size, h=h//window_size)
    x = einops.rearrange(x, 'b h w s1 s2 c -> b (h s1) (w s2) c')
    return x

def window_reverse_offcial(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class FeedForward(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchMerge(nn.Module):
    '''
    作用类似于采样, 缩小分辨率
    相比于max_pool + conv 组合,保留更多信息
    --> yolov5 中 focus层也是类似的操作
    '''
    def __init__(self, in_channel, img_size):
        super(PatchMerge, self).__init__()
        self.img_h, self.img_w = img_size
        self.in_channel = in_channel

        self.reduction = nn.Linear(4 * in_channel, 2 * in_channel, bias=False)
        self.norm = nn.LayerNorm(2 * in_channel)

    def forward(self, x):
        '''
        :param x: [b,h * w, c]
        :return: [b, l, -]
        '''
        B, L, C = x.shape
        assert L == self.img_h * self.img_h, f'x shape{x.shape} not match'
        assert self.img_h % 2 == 0 and self.img_w % 2 == 0, f'img size should be even'

        x = einops.rearrange(x, 'b (h w) c -> b h w c', h=self.img_h, w=self.img_w)
        x_0 = x[:, 0::2, 0::2, :]
        x_1 = x[:, 1::2, 0::2, :]
        x_2 = x[:, 0::2, 1::2, :]
        x_3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x_0, x_1, x_2, x_3], -1)
        x = einops.rearrange(x, 'b h w c-> b (h w) c', c=C * 4)
        x = self.reduction(x)
        x = self.norm(x)

        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_channel, embed_dim=96, img_size=224, patch_size=4):
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_resolution = patches_resolution
        self.num_pathch = patches_resolution[0] * patches_resolution[1]

        self.in_channel = in_channel
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channel, embed_dim,kernel_size=patch_size, stride=patch_size)
        self.norm_layer = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2) # [b, ph*pw, c]
        x = self.norm_layer(x)

        return x


class WindowAttentionV2(nn.Module):
    '''
    W-MSA, 使用相对位置编码, 基于的就是MHA
    '''
    def __init__(self, in_channel, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, pretrain_window_size=[0, 0]):
        super(WindowAttentionV2, self).__init__()
        self.in_channel = in_channel
        self.img_h, self.img_w = window_size # (window_h, window_w)
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = in_channel // num_heads
        self.scale = self.head_dim ** -0.5

        # relative position part
        # copy from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py#L113
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # relative position bias
        self.position_bias = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(
            torch.meshgrid(
                [relative_coords_h, relative_coords_w]
            )
        ).permute(1, 2, 0).contiguous().unsqueeze(0) #-> [1, 2*win_h-1, 2*win_w-1, 2]

        if pretrain_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrain_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrain_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # pair-wise relative postion index for each inside the windows

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([
            coords_h, coords_w
        ]))
        coords_flatten = torch.flatten(coords, 1) # [2, win_h*win_w]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] #[2, win_h*win_w, win_h* win_w]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.to_qkv = nn.Linear(in_channel, in_channel*3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(in_channel))
            self.v_bias = nn.Parameter(torch.zeros(in_channel))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.attn = nn.Softmax(dim=-1)
        self.proj = nn.Linear(in_channel, in_channel)

    def forward(self, x, mask=None):
        '''
        mask: (0/-inf) mask with shape of [num_windows, wh*ww, wh*ww]
        :param x: input feature [num_win*b, n, c]
        :return:
        '''
        B, N, C = x.shape
        if self.q_bias is not None:
            qkv_bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias])

        qkv = F.linear(input=x, weight=self.to_qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul((F.normalize(q, dim=-1)), F.normalize(k, dim=-1).transpose(-2, -1))




        print(f'q shape: {q.shape}')
        print(f'k shape: {k.shape}')
        print(f'v shape: {v.shape}')

        return qkv


class WindowAttention(nn.Module):
    '''
    原本的实现
    copy from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    '''
    def __init__(self, in_channel, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super(WindowAttention, self).__init__()
        self.in_channel = in_channel
        self.window_size = window_size
        self.img_h, self.img_w = window_size
        self.num_heads = num_heads
        head_dim = in_channel // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias parameter table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] -1), num_heads)
        ) # [2*Wh - 1, 2*Ww - 1, num_heads]

        # pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # [1, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1) # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] #[2, Wh*Ww]
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1) # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", self.relative_position_index)

        self.to_qvk = nn.Linear(in_channel, in_channel * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channel, in_channel)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.attn = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        '''

        :param x: [num_windows*B, N, C]
        :param mask: (num_windows, Wh*Ww, Wh*Ww) or None
        :return:
        '''

        B, N, C = x.shape
        qkv = self.to_qvk(x).chunk(3, dim=-1)
        qkv = einops.rearrange(qkv, 'b n (c2 c1 nh) -> b n c1 nh c2', c1=3, c2=C//self.num_heads, nh=self.num_heads)
        #q, k, v = map(lambda t: einops.rearrange(t, 'b n (c nh) -> b n c nh', nh=self.num_heads), qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dot = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = einops.rearrange(relative_position_bias, 'hw1 hw2 nh -> nh hw1 hw2').contiguous()
        dot = dot + relative_position_bias.unsqueeze(0)
        '''
         attn = softmax(q·k.T/sqrt(d) + b)·v
        '''

        if mask is not None:
            # [num_windows, Wh * Ww, Wh * Ww]
            nW = mask.shape[0]
            # dot = einops.rearrange(dot, '(b nw) nw nh')
            dot = dot.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            dot = dot.view(-1, self.num_heads, N, N)
            attn = self.attn(dot)
        else:
            attn = self.attn(dot)

        attn = self.attn_drop(attn)
        attn = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn

class SwinTransformerBlock(nn.Module):
    def __init__(self, in_channel, img_size, num_heads, window_size=7,
                 shift_size=0, expan_ration=4, qkv_bias=True, qk_scale=None,
                 dropout=0.0, attn_drop=0.0, drop_path=0.0):
        super(SwinTransformerBlock, self).__init__()
        self.in_channel = in_channel
        self.img_size = img_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.expan_ration = expan_ration
        self.window_size=window_size
        if min(self.img_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.img_size)

        self.norm = nn.LayerNorm(in_channel)
        '''
            def __init__(self, in_channel, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        '''
        self.attn = WindowAttention(
            in_channel=in_channel,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=dropout
        )

        self.drop_path = DropPath(drop_path)
        hidden_dim = int(in_channel * expan_ration)
        self.ffn = FeedForward(in_channel=in_channel, out_channel=in_channel,hidden_dim=hidden_dim)

        if self.shift_size > 0:
            img_h, img_w = self.img_size
            img_mask = torch.zeros((1, img_h, img_w, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)
                        )
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size) # [nW, window_size, window_size, 1]
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            self.attn_mask = attn_mask.masked_fill(attn_mask !=0, float(-100.0)).mased_fill(attn_mask == 0, float(0.0))

        else:
            self.attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.img_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res = x
        x = self.norm(x)
        x = einops.rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)

        # cyclic shift -> 个人认为是swin transformer 最精华的部分
        # 很好的利用了平移不变性,并且考虑到不同窗口的交流
        # 但是这个方法的泛化能力还是稍差, 需要大量的数据才能实现
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size) # [nW*B, window_size, window_size, C]
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)
            # return [nW*B, window_size, window_size, C]

        x_windows = einops.rearrange(x_windows, "wb ws1 ws2 c -> wb (ws1 ws2) c")

        # attn
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dim=(1, 2))

        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
            # [B H' W' C]

        x = einops.rearrange(x, 'b h w c -> b (h w) c')
        x = res + self.drop_path(x)

        #ffn
        x = x + self.drop_path(self.ffn(self.norm(x)))

        return x



if __name__ == '__main__':

    x = torch.randn(1, 3, 54, 54)
    x = einops.rearrange(x, 'b c w h -> b (w h) c')
    patch_merging = PatchMerge(3, (54, 54))
    patch_merging_test = patch_merging(x)
    print(f'patch_merging_test: {patch_merging_test.shape}')

    ffn = FeedForward(3, 64, 32)
    ffn_test = ffn(x)
    print(f'ffn_test: {ffn_test.shape}')

    x_1 = torch.randn(1, 3, 54, 54)
    x_1 = einops.rearrange(x_1, "b c w h -> b w h c")
    window_partition_test = window_partition(x_1, 9)
    print(f'window_partition_test: {window_partition_test.shape}')

    window_reverse_test = window_reverse(window_partition_test, 9, 54, 54)
    print(f'after window_reverse: {window_reverse_test.shape}')
    y_2 = window_partition_test
    #     def __init__(self, in_channel, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, mask=None, pretrain_window_size=[0, 0]):

    to_three_dim = nn.Linear(4, 12, bias=False)
    x = torch.randn(1, 64, 4)
    to_three_dim_test = to_three_dim(x)
    print(f'x shape {x.shape}')
    print(f'after to_qvk: {to_three_dim_test.shape}')