import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.lce.models.nn.ops import LightweightContextEncoder

__all__ = ["CERPCANet"]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):

        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x

class CERPCANet(nn.Module):
    def __init__(self, stage_num=6, slayers=6, mlayers=3, channel=32, mode='train'):
        super(CERPCANet, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for i in range(stage_num):
            self.decos.append(CERPCANetStage(slayers=slayers, mlayers=mlayers, channel=channel))
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros(D.shape).to(D.device)
        [h, c] = [None, None]
        
        for i in range(self.stage_num):
            D, T, h, c = self.decos[i](D, T, h, c)
        if self.mode == 'train':
            return D,T
        else:
            return T

class CERPCANetStage(nn.Module):
    def __init__(self, slayers=6, mlayers=3, channel=32):
        super(CERPCANetStage, self).__init__()
        self.lowrank = GlobalLocalBackgroundUpdate(channel=channel)
        self.sparse = PriorGuidedSparseObjectUpdate(channel=channel, layers=slayers)
        self.merge = ObservationReconstruction(channel=channel, layers=mlayers)

    def forward(self, D, T, h, c):
        B, h, c = self.lowrank(D, T, h, c)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T, h, c
        

class GlobalContextBranch(nn.Module):
    def __init__(self, cin: int, h: int, w: int, mid: int = None, scale: float = 0.1):
        super().__init__()
        self.cin, self.h, self.w = cin, h, w
        self.out_dim = cin + h + w
        self.scale = scale                   

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        mid = mid if mid is not None else max(16, cin // 4)
        self.fc = nn.Sequential(
            nn.Linear(cin, mid),
            nn.GELU(),                       
            nn.Linear(mid, self.out_dim)
        )
        
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.cin and H == self.h and W == self.w, \
            f"尺寸不匹配: got {(C,H,W)} expected {(self.cin,self.h,self.w)}"

        s = self.global_avgpool(x).view(B, C)        
        z = self.fc(s)                                

        dz_c, dz_h, dz_w = torch.split(z, [C, H, W], dim=1)  

        
        g_c = 1 + self.scale * torch.tanh(dz_c)      
        g_h = 1 + self.scale * torch.tanh(dz_h)      
        g_w = 1 + self.scale * torch.tanh(dz_w)      

        g_c = g_c.view(B, C, 1, 1)
        g_h = g_h.view(B, 1, H, 1)
        g_w = g_w.view(B, 1, 1, W)

        gate = g_c * g_h * g_w                        
        return x * gate                               



class GlobalLocalBackgroundUpdate(nn.Module):
    def __init__(self, channel=32,**lowformer_kwargs):
        super(GlobalLocalBackgroundUpdate, self).__init__()
        self.conv1_C = nn.Sequential(nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                         nn.BatchNorm2d(channel),
                         nn.ReLU(True))
        self.RB_1 = ResidualBlock(channel, channel, 3, bias=True, res_scale=1)
        self.RB_2 = ResidualBlock(channel, channel, 3, bias=True, res_scale=1)
        self.convC_1 = nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1)
        self.global_context_branch = GlobalContextBranch(channel, 256, 256)
        self.lce = LightweightContextEncoder(in_channels=channel, **lowformer_kwargs)
        self.conv_back = nn.Conv2d(2*channel, channel, kernel_size=1)

    def forward(self, D, T, h, c):
        x = D - T
        x_c = self.conv1_C(x)
        x_c1 = self.RB_1(x_c)
        x_ct1 = self.global_context_branch(x_c1)
        x_ct2 = self.lce(x_c1)
        x_ct = torch.cat((x_ct1, x_ct2), dim=1)
        x_ct = self.conv_back(x_ct)
        x_c2 = self.RB_2(x_ct)
        x_1 = self.convC_1(x_c2)
        B = x + x_1
        return B, h, c



from .modules.rspg.sparse_support_refinement import SparseSupportRefinement
from typing import Optional
from types import SimpleNamespace
from typing import Optional, Union
from omegaconf import OmegaConf, DictConfig


class ConvolutionalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)

    def forward(self, x_bhwc: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x_bhwc.shape
        x = x_bhwc.permute(0, 3, 1, 2).contiguous()
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class ResidualResponseSparsePriorGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_cfg: Optional[Union[DictConfig, dict]] = None,
        partition_size: Union[int, tuple] = 8,
        enable_CB: bool = False,
        embed_dim: int = 32,     
    ):
        super().__init__()

        default_cfg = {
            "partition_size": partition_size,
            "dim_head": 32,
            "attention_bias": True,
            "mlp_activation": "gelu",
            "mlp_bias": True,
            "mlp_ratio": 4,
            "drop_path": 0.0,
            "drop_mlp": 0.0,
            "ls_init_value": 1e-5,
            "enable_CB": enable_CB,
            "AMP": 2e-4,
            "BOUNCE": 1e-3,
            "norm_eps": 1e-5,
        }
        base_cfg = OmegaConf.create(default_cfg)
        if attention_cfg is None:
            self.attn_cfg: DictConfig = base_cfg
        elif isinstance(attention_cfg, DictConfig):
            self.attn_cfg = OmegaConf.merge(base_cfg, attention_cfg)
        else:
            self.attn_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(attention_cfg))

        dh = int(self.attn_cfg.dim_head)
        if dh <= 0 or (embed_dim % dh != 0):
            self.attn_cfg.dim_head = 1

        
        self.in_proj  = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=True) \
                        if in_channels != embed_dim else nn.Identity()
        self.out_proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=True)

        self.pos_emb = ConvolutionalPositionEmbedding(embed_dim)
        self.sparse_support_refinement = SparseSupportRefinement(dim=embed_dim, attention_cfg=self.attn_cfg, first_block=True)

    @torch.no_grad()
    def _make_control(self, B: int, device, dtype):
        return torch.ones(B, 20, device=device, dtype=dtype)

    def _get_partition_hw(self):
        ps = self.attn_cfg.partition_size
        return (ps, ps) if isinstance(ps, int) else (int(ps[0]), int(ps[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.in_proj(x)  

        B, C, H, W = x.shape
        ph, pw = self._get_partition_hw()
        pad_h, pad_w = (-H) % ph, (-W) % pw
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else x

        x_nhwc = x_pad.permute(0, 2, 3, 1).contiguous()
        r = self._make_control(B, x.device, x.dtype)

        y_nhwc, _, _ = self.sparse_support_refinement(x_nhwc, self.pos_emb, r, index_list=[])
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        if pad_h or pad_w:
            y = y[:, :, :H, :W]

        w = self.out_proj(y)  
        return w


class PriorGuidedSparseObjectUpdate(nn.Module):
    def __init__(self, in_channels=1, channel=32, layers=6,
                 attention_cfg: Optional[dict] = None):
        super(PriorGuidedSparseObjectUpdate, self).__init__()

        
        convs = [nn.Conv2d(in_channels, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for _ in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, in_channels, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)

        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

        self.rspg = ResidualResponseSparsePriorGenerator(in_channels=in_channels,
                                     out_channels=in_channels,
                                     attention_cfg=attention_cfg)

    def forward(self, D, B, T):
        
        x = T + D - B                 
        w = self.rspg(x)          
        T = x - self.epsilon * self.convs(x + w)
        return T
    

class ObservationReconstruction(nn.Module):
    def __init__(self,  channel=32, layers=3):
        super(ObservationReconstruction, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        D = self.mapping(x)
        return D


