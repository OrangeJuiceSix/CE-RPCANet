
from typing import Union, List, Optional, Any

import torch
from torch import nn as nn
from torch.nn import functional as F

from .create_act import get_act_layer
from .fast_norm import is_fast_norm, fast_group_norm, fast_layer_norm
from .trace_utils import _assert


class BatchNormAct2d(nn.BatchNorm2d):
    
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            apply_act=True,
            act_layer=nn.ReLU,
            inplace=True,
            drop_layer=None,
            device=None,
            dtype=None
    ):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAct2d, self).__init__(
                num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats,
                **factory_kwargs
            )
        except TypeError:
            
            super(BatchNormAct2d, self).__init__(
                num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        
        _assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')

        
        
        
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            
            if self.num_batches_tracked is not None:  
                self.num_batches_tracked = self.num_batches_tracked + 1  
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  
                    exponential_average_factor = self.momentum

        
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        
        x = F.batch_norm(
            x,
            
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x


class SyncBatchNormAct(nn.SyncBatchNorm):
    
    
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)  
        if hasattr(self, "drop"):
            x = self.drop(x)
        if hasattr(self, "act"):
            x = self.act(x)
        return x


def convert_sync_batchnorm(module, process_group=None):
    
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if isinstance(module, BatchNormAct2d):
            
            module_output = SyncBatchNormAct(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group=process_group,
            )
            
            module_output.act = module.act
            module_output.drop = module.drop
        else:
            
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, process_group))
    del module
    return module_output


def _num_groups(num_channels, num_groups, group_size):
    if group_size:
        assert num_channels % group_size == 0
        return num_channels // group_size
    return num_groups


class GroupNormAct(nn.GroupNorm):
    
    def __init__(
            self, num_channels, num_groups=32, eps=1e-5, affine=True, group_size=None,
            apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(GroupNormAct, self).__init__(
            _num_groups(num_channels, num_groups, group_size), num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct(nn.LayerNorm):
    def __init__(
            self, normalization_shape: Union[int, List[int], torch.Size], eps=1e-5, affine=True,
            apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(LayerNormAct, self).__init__(normalization_shape, eps=eps, elementwise_affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct2d(nn.LayerNorm):
    def __init__(
            self, num_channels, eps=1e-5, affine=True,
            apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(LayerNormAct2d, self).__init__(num_channels, eps=eps, elementwise_affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        x = self.drop(x)
        x = self.act(x)
        return x
