
import torch
from torch import nn
import torch.nn.functional as F

from .helpers import to_2tuple, make_divisible
from .weight_init import trunc_normal_


def rel_pos_indices(size):
    size = to_2tuple(size)
    pos = torch.stack(torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))).flatten(1)
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos  


class LambdaLayer(nn.Module):
    
    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, dim_head=16, r=9,
            qk_ratio=1.0, qkv_bias=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, ' should be divided by num_heads'
        self.dim_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.num_heads = num_heads
        self.dim_v = dim_out // num_heads

        self.qkv = nn.Conv2d(
            dim,
            num_heads * self.dim_qk + self.dim_qk + self.dim_v,
            kernel_size=1, bias=qkv_bias)
        self.norm_q = nn.BatchNorm2d(num_heads * self.dim_qk)
        self.norm_v = nn.BatchNorm2d(self.dim_v)

        if r is not None:
            
            self.conv_lambda = nn.Conv3d(1, self.dim_qk, (r, r, 1), padding=(r // 2, r // 2, 0))
            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            
            assert feat_size is not None
            feat_size = to_2tuple(feat_size)
            rel_size = [2 * s - 1 for s in feat_size]
            self.conv_lambda = None
            self.pos_emb = nn.Parameter(torch.zeros(rel_size[0], rel_size[1], self.dim_qk))
            self.register_buffer('rel_pos_indices', rel_pos_indices(feat_size), persistent=False)

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)  
        if self.conv_lambda is not None:
            trunc_normal_(self.conv_lambda.weight, std=self.dim_qk ** -0.5)
        if self.pos_emb is not None:
            trunc_normal_(self.pos_emb, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        M = H * W
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [
            self.num_heads * self.dim_qk, self.dim_qk, self.dim_v], dim=1)
        q = self.norm_q(q).reshape(B, self.num_heads, self.dim_qk, M).transpose(-1, -2)  
        v = self.norm_v(v).reshape(B, self.dim_v, M).transpose(-1, -2)  
        k = F.softmax(k.reshape(B, self.dim_qk, M), dim=-1)  

        content_lam = k @ v  
        content_out = q @ content_lam.unsqueeze(1)  

        if self.pos_emb is None:
            position_lam = self.conv_lambda(v.reshape(B, 1, H, W, self.dim_v))  
            position_lam = position_lam.reshape(B, 1, self.dim_qk, H * W, self.dim_v).transpose(2, 3)  
        else:
            
            pos_emb = self.pos_emb[self.rel_pos_indices[0], self.rel_pos_indices[1]].expand(B, -1, -1, -1)
            position_lam = (pos_emb.transpose(-1, -2) @ v.unsqueeze(1)).unsqueeze(1)  
        position_out = (q.unsqueeze(-2) @ position_lam).squeeze(-2)  

        out = (content_out + position_out).transpose(-1, -2).reshape(B, C, H, W)  
        out = self.pool(out)
        return out
