

from .mixed_conv2d import MixedConv2d
from .cond_conv2d import CondConv2d
from .conv2d_same import create_conv2d_pad


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  
        if 'groups' in kwargs:
            groups = kwargs.pop('groups')
            if groups == in_channels:
                kwargs['depthwise'] = True
            else:
                assert groups == 1
        
        
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m
