import torch
import math
import warnings

from torch.nn.init import _calculate_fan_in_and_fan_out


def _trunc_normal_(tensor, mean, std, a, b):
    
    
    def norm_cdf(x):
        
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    
    
    
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    
    
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    
    
    tensor.erfinv_()

    
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    
    
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def trunc_normal_tf_(tensor, mean=0., std=1., a=-2., b=2.):
    
    
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
