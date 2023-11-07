import numpy as np
import torch
from torch import nn
import einops
from einops.layers.torch import Rearrange

# reference: Attention Is All You Need https://arxiv.org/abs/1706.03762
'''
Attention Is All You Need 中的实现
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, d_k, d_v, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.w_q = nn.Linear(d_model, heads * d_k)
        self.w_k = nn.Linear(d_model, heads * d_k)
        self.w_v = nn.Linear(d_model, heads * d_v)
        self.fc = nn.Linear(heads * d_v, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn = nn.Softmax()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        q_ = q
        q = einops.rearrange(self.w_q(q), "b l (h k) -> h b l k", h=self.heads)
        k = einops.rearrange(self.w_k(k), "b t (h k) -> h b l k", h=self.heads)
        v = einops.rearrange(self.w_v(v), "b t (h v) -> h b t v", h=self.heads)

        attn = torch.einsum("hblk,hbtk->hblt", [q, k]) / np.sort(q.shape[-1])
        attn = self.dropout(attn.softmax(dim=-1))
        out = torch.einsum('hblt, hbtv->hblv', [attn, v])
        out = einops.rearrange(out, 'h b l v -> b l (h v)')
        out = self.layer_norm(self.dropout(self.fc(out)) + q_)

        return out, attn


'''
reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
'''
class MHA(nn.Module):
    def __init__(self, in_channel,out_channel ,heads=8, head_dim=32, dropout=0.0):
        super(MHA, self).__init__()
        hidden_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == in_channel)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attn = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(in_channel, hidden_dim * 3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, out_channel),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        q_ = q
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attn(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d->b n (h d)')

        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, heads=8, head_dim=32, downsample=False, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.img_w, self.img_h = img_size
        self.hidden_dim = int(in_channel * 4)
        self.downsample = downsample

        if self.downsample:
            self.max_pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        self.attn = MHA(in_channel=in_channel,out_channel=out_channel, heads=heads, head_dim=head_dim, dropout=dropout)

        self.attn = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(in_channel),
            self.attn,
            Rearrange('b (h w) c -> b c h w', h=self.img_h, w=self.img_w)
        )


        self.feed_forward = nn.Sequential(
            nn.Linear(out_channel, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, out_channel),
            nn.Dropout(dropout)
        )

        self.FFN = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(out_channel),
            self.feed_forward,
            Rearrange("b (h w) c -> b c h w", h=self.img_h, w=self.img_w)
        )

    def forward(self, x):
        if self.downsample:
            y = self.max_pool(x)
            #print(f'max_pool y: {y.shape}')

            y = self.proj(y)
            #print(f'proj y: {y.shape}')

            y2 = self.max_pool(x)
            #print(f'max_pool y2: {y2.shape}')

            y2 = self.attn(y2)
            #print(f'attn y2: {y2.shape}')

            x = self.proj(self.max_pool(x)) + self.attn(self.max_pool(x))
        else:

            x = x + self.attn(x)

        x = x + self.FFN(x)

        return x



class Transformer(nn.Module):
    def __init__(self, in_channel, depth, hidden_dim ,heads=8, head_dim=32,  dropout=0.0):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(in_channel)
        self.layers = nn.ModuleList([])
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(in_channel),
            nn.Linear(in_channel, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channel),
            nn.Dropout(dropout)
        )
        for layer in range(depth):
            self.layers.append(nn.ModuleList([
                MHA(in_channel=in_channel, out_channel=in_channel, heads=heads, head_dim=head_dim, dropout=dropout),
                self.feed_forward
            ]))

    def forward(self, x):
        for attn, ff  in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


if __name__ == '__main__':
    x = torch.randn(1, 3, 18, 18)
    transformer = TransformerBlock(3, 64, (9, 9), downsample=True, dropout=0.5)
    y = transformer(x)


    x = torch.randn(1, 3, 18, 18)
    x = einops.rearrange(x, 'b c h w -> b (h w) c')
    transformer2 = Transformer(3, 64, 8, 64)
    y = transformer2(x)

    print(y)
    print(y.shape)
