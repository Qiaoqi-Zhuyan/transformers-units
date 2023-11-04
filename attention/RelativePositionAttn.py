# reference:  https://github.com/chinhsuanwu/coatnet-pytorch/blob/master/coatnet.py

import torch
from torch import nn
import einops

class Attention(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, heads=8, head_channel=32, dropout=0.0):
        super(Attention, self).__init__()
        inner_channel = head_channel * heads
        self.img_h, self.img_w = img_size

        project_out = not (heads == 1 and head_channel == inner_channel)

        self.heads = heads
        self.scale = head_channel ** -0.5

        # Relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.img_h - 1) * (2 * self.img_w - 1), heads)
        )

        coords = torch.meshgrid((torch.arange(self.img_h), torch.arange(self.img_w)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.img_h - 1
        relative_coords[1] += self.img_w - 1
        relative_coords[0] *= 2 * self.img_w - 1
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attn = nn.Softmax(dim=-1)
        self.to_qvk = nn.Linear(in_channel, inner_channel * 3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(inner_channel, out_channel),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        qkv = self.to_qvk(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads) , qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # gather
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = einops.rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.img_h * self.img_w, w=self.img_h * self.img_w)

        dots = dots + relative_bias

        attn = self.attn(dots)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.out(out)

        return out
