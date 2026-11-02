try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def _float_to_int(x: float) -> int:
    
    return int(x)
