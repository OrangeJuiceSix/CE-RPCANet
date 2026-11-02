







from functools import partial
from typing import Optional, Tuple, List

import math
import torch
from omegaconf import DictConfig
from torch import nn

from .layers import DropPath
from .layers import get_act_layer, get_norm_layer
from .layers import to_2tuple
from .ops import window_partition, window_reverse, grid_partition, \
                 grid_reverse, LayerScale, MLP


class SparseSupportRefinement(nn.Module):
    

    def __init__(
            self,
            dim: int,
            attention_cfg: DictConfig,
            first_block: bool=False,
    ):
        super().__init__()
        norm_eps = attention_cfg.get('norm_eps', 1e-5)
        partition_size = attention_cfg.partition_size
        dim_head = attention_cfg.get('dim_head', 32)
        attention_bias = attention_cfg.get('attention_bias', True)
        mlp_act_string = attention_cfg.mlp_activation
        mlp_bias = attention_cfg.get('mlp_bias', True)
        mlp_expand_ratio = attention_cfg.get('mlp_ratio', 4)

        drop_path = attention_cfg.get('drop_path', 0.0)
        drop_mlp = attention_cfg.get('drop_mlp', 0.0)
        ls_init_value = attention_cfg.get('ls_init_value', 1e-5)
        
        if isinstance(partition_size, int):
            partition_size = to_2tuple(partition_size)
        else:
            partition_size = tuple(partition_size)
            assert len(partition_size) == 2
        self.partition_size = partition_size

        norm_layer = partial(get_norm_layer('layernorm'), eps=norm_eps)

        mlp_act_layer = get_act_layer(mlp_act_string)

        sub_layer_params = (ls_init_value, drop_path, mlp_expand_ratio, mlp_act_layer, mlp_bias, drop_mlp)
        
        self_attn_module = KernelizedSparseAttention
        self.enable_CB = attention_cfg.get('enable_CB', False)

        self.local_sparse_attention = self_attn_module(dim,
                                         dim_head=dim_head,
                                         bias=attention_bias,
                                         sub_layer_params=sub_layer_params,
                                         norms=[norm_layer(dim), norm_layer(dim)])

        self.cross_window_sparse_attention = self_attn_module(dim,
                                          dim_head=dim_head,
                                          bias=attention_bias,
                                          sub_layer_params=sub_layer_params,
                                          norms=[norm_layer(dim), norm_layer(dim)])
        if first_block:
            self.to_scores = nn.Linear(dim, dim)
            self.to_controls = PositiveLinear(20, dim, bias=False)
            torch.nn.init.constant_(self.to_controls.weight, 1)
            self.act = nn.ReLU()

        self.amp_value = attention_cfg.get('AMP', 2e-4)
        self.bounce_value = attention_cfg.get('BOUNCE', 1e-3)
        self.first_block = first_block
        self.B, self.N, self.dim = None, None, dim

    def local_sparse_support_selection(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = h * w
        norm_window = (torch.norm(scores, dim=[2, 3], p=1) / temp).softmax(-1) 
        index_window = get_score_index_2d21d(norm_window.view(B, N), 1 / N, self.bounce_value) 
        return index_window
    
    def cross_window_sparse_support_refinement(self, scores: torch.Tensor, index_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = 1
        norm_token = (torch.norm(scores, dim=[3], p=1) / temp).view(B * N, -1)[index_window].softmax(-1)
        index_token, asy_index_partition, K = get_score_index_with_padding(norm_token, 1 / (h * w), self.bounce_value) 
        return index_token, asy_index_partition, K
        
    def _partition_attn(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor, index_list: List) -> Tuple[torch.Tensor, ...]:
        index_count = 0
        self.B = x.shape[0]
        img_size = x.shape[1:3]
        self.N = img_size[0] * img_size[1] // (self.partition_size[0] * self.partition_size[1])

        
        x = x + pos_emb(x)
        x = window_partition(x, self.partition_size).view(self.B, self.N, -1, self.dim) 
        if self.first_block:
            
            scale = self.to_controls(r + 1e-6)[:, None, None, :]  
            scores = self.act(self.to_scores(x)) 

            
            weight = scale.sigmoid() * scores.sigmoid() 
            x = (weight * x).view(self.B * self.N, -1, self.dim) 

            
            scale = self.amp_value / scale
            scale[scale==torch.inf] = 0
            scores = scale * scores
            index_window = self.local_sparse_support_selection(scores)
            index_token, asy_index, K = self.cross_window_sparse_support_refinement(scores, index_window)
            padding_index = index_token[torch.isin(index_token, asy_index, assume_unique=True, invert=True)] 
            index_list1 = [index_window, index_token, padding_index, asy_index, K] 
        else:
            
            x = x.view(self.B * self.N, -1, self.dim)
            index_list1, index_list2 = index_list
            index_window, index_token, padding_index, asy_index, K = index_list1
        M = len(index_window)
        
        if len(index_token):
            
            x = self.local_sparse_attention(x, index_window, index_token, padding_index, asy_index, M, self.B, self.enable_CB)
        x = window_reverse(x, self.partition_size, (img_size[0], img_size[1]))

        
        index_count += len(asy_index) // self.B

        
        if self.first_block:
            
            scores = window_reverse(scores.view_as(x), self.partition_size, (img_size[0], img_size[1]))
            scores = grid_partition(scores, self.partition_size).view(self.B, self.N, -1, self.dim)

            
            index_window = self.local_sparse_support_selection(scores) 
            index_token, asy_index, K = self.cross_window_sparse_support_refinement(scores, index_window)
            padding_index = index_token[torch.isin(index_token, asy_index, assume_unique=True, invert=True)]
            index_list2 = [index_window, index_token, padding_index, asy_index, K]
        else:
            index_window, index_token, padding_index, asy_index, K = index_list2
        x = x.view(self.B, img_size[0], img_size[1], self.dim)
        x = grid_partition(x, self.partition_size).view(self.B * self.N, -1, self.dim)
        
        M = len(index_window)
        if len(index_token): 
            
            x = self.cross_window_sparse_attention(x, index_window, index_token, padding_index, asy_index, M, self.B, self.enable_CB)
        x = grid_reverse(x, self.partition_size, (img_size[0], img_size[1]))
        index_count += len(asy_index) // self.B
        return x, index_count, [index_list1, index_list2]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor, index_list: List) -> Tuple[torch.Tensor, ...]:
        x, index_count, index_list = self._partition_attn(x, pos_emb, r, index_list)
        return x, index_count, index_list
    


class KernelizedSparseAttention(nn.Module):
    
    

    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True,
            sub_layer_params: Optional[List[nn.Module]] = None,
            norms: nn.Module = None,):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 =norms[0]

        ls_init_value, drop_path, mlp_expand_ratio, mlp_act_layer, mlp_bias, drop_mlp = sub_layer_params
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norms[1]
        self.mlp = MLP(dim=dim, channel_last=True, expansion_ratio=mlp_expand_ratio,
                       act_layer=mlp_act_layer, bias=mlp_bias, drop_prob=drop_mlp)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.sub_layers = nn.ModuleList([self.ls1, self.drop1, self.norm2, self.mlp, self.ls2, self.drop2])
            
        self.eps = 1e-6

        
        self.ess_out = nn.Linear(dim, dim, bias=True)
        

    def forward(self, x: torch.Tensor, index_window: torch.Tensor, 
                index_token: torch.Tensor, padding_index: torch.Tensor, 
                asy_index: torch.Tensor, M: int, B: torch.Tensor, enable_CB: bool) -> torch.Tensor:
        
        N, C = x.shape[0], x.shape[-1]
        restore_shape = x.shape
        x = x.view(N, -1, C)
        x = self.norm1(x)  
        if len(index_token) == 0: 
            return x.view(*restore_shape)
        
        
        X = x.clone() 
        x = x[index_window].view(-1, C) 
        XX = x.clone() 
        x[asy_index] = self.norm2(x[asy_index])  
        shortcut = x[asy_index]  
        x = x[index_token].view(M, -1, C)  

        
        Lmax = x.shape[1]                                       
        is_pad_flat = torch.isin(index_token, padding_index)     
        valid_mask = (~is_pad_flat).view(M, Lmax)                
        mask = valid_mask[:, None, :, None].to(x.dtype)          

        
        qkv = self.qkv(x).view(M, Lmax, self.num_heads, 3 * self.dim_head).transpose(1, 2)  
        q, k, v = qkv.chunk(3, dim=-1)                              

        
        orig_dtype = q.dtype
        if q.dtype in (torch.float16, torch.bfloat16):
            q = q.float(); k = k.float(); v = v.float()

        
        q = q - q.mean(dim=-1, keepdim=True)
        k = k - k.mean(dim=-1, keepdim=True)

        
        q2 = q * q                    
        k2 = k * k

        
        q2s = q2.sum(dim=-1, keepdim=True).clamp_min_(1e-7)   
        q2n = q2 / q2s

        
        k2s = k2.sum(dim=-1, keepdim=True).clamp_min_(1e-7)   
        k2n = k2 / k2s                                        
        k2n = k2n * mask                                      
        
        den_k = torch.sqrt((k2n * k2n).sum(dim=2, keepdim=True) + 1e-7)   
        k2n = k2n / (den_k + 1e-7)

        
        v_masked = v * mask

        
        KV = k2n.transpose(-2, -1) @ v_masked                              
        t2 = q2n @ KV                                                      
        n_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(t2.dtype)  
        t2 = t2 / n_valid.sqrt().view(M, 1, 1, 1)

        
        attn_out = v + t2

        
        x = attn_out.transpose(1, 2).reshape(M, Lmax, -1)                  
        x = self.proj(x).to(XX.dtype)                                      
        if x.dtype != orig_dtype:
            x = x.to(orig_dtype)



        XX[index_token] = x.view(-1, C)
        x = XX[asy_index] 

        x = shortcut + self.drop1(self.ls1(x))
        shortcut = x
        x = self.mlp(x).to(X.dtype)

        
        if enable_CB: 
            temp_X, temp_XX = torch.zeros_like(X), torch.zeros_like(XX)
            temp_XX[asy_index] = x
            temp_X[index_window] = temp_XX.view(M, -1, C)
            temp_X = temp_X.view(B, -1, C)
            temp_X = (0.5 * temp_X + (1 - 0.5) * temp_X.mean(dim=1, keepdim=True)).view(*restore_shape)
            x = temp_X[index_window].view(-1, C)[asy_index]

        x = shortcut + self.drop2(self.ls2(x))

        
        XX[asy_index] = x.view(-1, C)
        XX[padding_index] = X[index_window].view(-1, C)[padding_index]
        X[index_window] = XX.view(M, -1, C) 
        x = X.view(*restore_shape) 
        return x






























            

        




        






        











        









        

































































































            

        




        






        











        




































def get_score_index_2d21d(x: torch.Tensor, d: float, b: float) -> torch.Tensor:
    
    if x.shape[0] == 1:
        
        return torch.nonzero(x >= d / (1 + b))[:, 1]
    
    gt = x >= d / (1 + b)
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return index_1d


def get_score_index_with_padding(x: torch.Tensor, d: float, b: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    gt = x >= d / (1 + b)
    K = torch.sum(gt, dim=1)
    
    top_indices = torch.topk(x, k=K.max(), dim=1, largest=True, sorted=False)[1]
    
    arange = torch.arange(0, x.shape[0] * x.shape[1], x.shape[1], device=x.device).view(-1, 1)
    
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return (top_indices + arange).view(-1), index_1d, K
    

def get_non_zero_ratio(x: torch.Tensor) -> torch.Tensor:
    
    
    
    x_down_4 = torch.nn.functional.max_pool2d(x.float(), kernel_size=4, stride=4)
    x_down_8 = torch.nn.functional.max_pool2d(x_down_4, kernel_size=2, stride=2)
    x_down_16 = torch.nn.functional.max_pool2d(x_down_8, kernel_size=2, stride=2)
    x_down_32 = torch.nn.functional.max_pool2d(x_down_16, kernel_size=2, stride=2)
    
    num_nonzero_1 = torch.sum(torch.sum(x_down_4 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_2 = torch.sum(torch.sum(x_down_8 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_3 = torch.sum(torch.sum(x_down_16 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_4 = torch.sum(torch.sum(x_down_32 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    result1 = x.shape[0] / x_down_4.numel() * num_nonzero_1.float()
    result2 = x.shape[0] / x_down_8.numel() * num_nonzero_2.float()
    result3 = x.shape[0] / x_down_16.numel() * num_nonzero_3.float()
    result4 = x.shape[0] / x_down_32.numel() * num_nonzero_4.float()
    
    return [result1, result2, result3, result4]


class PositiveLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        
        positive_weights = torch.exp(self.weight)
        return nn.functional.linear(input, positive_weights, self.bias)