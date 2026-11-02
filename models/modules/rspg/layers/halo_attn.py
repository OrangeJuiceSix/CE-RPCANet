
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .helpers import make_divisible
from .weight_init import trunc_normal_
from .trace_utils import _assert


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    
    B, H, W, dim = q.shape
    rel_size = rel_k.shape[0]
    win_size = (rel_size + 1) // 2

    x = (q @ rel_k.transpose(-1, -2))
    x = x.reshape(-1, W, rel_size)

    
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, rel_size - W])

    
    x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x = x_pad[:, :W, win_size - 1:]

    
    x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    return x.permute(permute_mask)


class PosEmbedRel(nn.Module):
    
    def __init__(self, block_size, win_size, dim_head, scale):
        
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        self.height_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * scale)
        self.width_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * scale)

    def forward(self, q):
        B, BB, HW, _ = q.shape

        
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, BB, HW, -1)
        return rel_logits


class HaloAttn(nn.Module):
    
    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, num_heads=8, dim_head=None, block_size=8, halo_size=3,
            qk_ratio=1.0, qkv_bias=False, avg_down=False, scale_pos_embed=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  
        self.block_stride = 1
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_avg_pool else stride
            self.block_size_ds = self.block_size // self.block_stride

        
        
        
        self.q = nn.Conv2d(dim, self.dim_out_qk, 1, stride=self.block_stride, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, self.dim_out_qk + self.dim_out_v, 1, bias=qkv_bias)

        self.pos_embed = PosEmbedRel(
            block_size=self.block_size_ds, win_size=self.win_size, dim_head=self.dim_head_qk, scale=self.scale)

        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5  
        trunc_normal_(self.q.weight, std=std)
        trunc_normal_(self.kv.weight, std=std)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H % self.block_size == 0, '')
        _assert(W % self.block_size == 0, '')
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.q(x)
        
        q = q.reshape(
            -1, self.dim_head_qk,
            num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        

        kv = self.kv(x)
        
        
        
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
            B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        

        if self.scale_pos_embed:
            attn = (q @ k.transpose(-1, -2) + self.pos_embed(q)) * self.scale
        else:
            attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 3)  
        
        out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        out = out.permute(0, 3, 1, 4, 2).contiguous().view(
            B, self.dim_out_v, H // self.block_stride, W // self.block_stride)
        
        out = self.pool(out)
        return out



