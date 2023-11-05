import numpy as np
import torch
from torch import nn
import einops

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
    def __init__(self, in_channel, heads=8, head_dim=32, dropout=0.0):
        hidden_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == in_channel)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attn = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(in_channel, hidden_dim * 3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, in_channel),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: einops.rearrange(t, "b p n (h d) -> b p h n d", h=self.heads))
        q_ = q
        dots = torch.matmul(q, k.transpoes(-1, -2)) * self.scale

        attn = self.attn(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d->b n (h d)')

        return self.out(out)

