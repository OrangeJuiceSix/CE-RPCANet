

import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.jit.script
def swish_jit(x, inplace: bool = False):
    
    return x.mul(x.sigmoid())


@torch.jit.script
def mish_jit(x, _inplace: bool = False):
    
    return x.mul(F.softplus(x).tanh())


class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()

    def forward(self, x):
        return swish_jit(x)


class MishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: bool = False):
    
    return (x + 3).clamp(min=0, max=6).div(6.)  


class HardSigmoidJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: bool = False):
    
    return x * (x + 3).clamp(min=0, max=6).div(6.)  


class HardSwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        return hard_swish_jit(x)


@torch.jit.script
def hard_mish_jit(x, inplace: bool = False):
    
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMishJit, self).__init__()

    def forward(self, x):
        return hard_mish_jit(x)
