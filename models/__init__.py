from .ce_rpcanet import CERPCANet
from .ce_rpcanet_without_aocm import CERPCANet as CERPCANetWithoutAOCM

__all__ = ["CERPCANet", "CERPCANetWithoutAOCM", "get_model"]


def get_model(name, net=None):
    if name == "ce_rpcanet":
        net = CERPCANet(stage_num=6, slayers=6, mlayers=3, channel=32)
    elif name == "ce_rpcanet_without_aocm":
        net = CERPCANetWithoutAOCM(
            stage_num=6, slayers=6, mlayers=3, channel=32
        )
    else:
        raise NotImplementedError(f"Unknown model: {name}")
    return net
