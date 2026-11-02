
from typing import Any, Optional

__all__ = [
    'is_exportable', 'is_scriptable', 'is_no_jit',
    'set_exportable', 'set_scriptable', 'set_no_jit', 'set_layer_config'
]


_NO_JIT = False




_NO_ACTIVATION_JIT = False


_EXPORTABLE = False


_SCRIPTABLE = False


def is_no_jit():
    return _NO_JIT


class set_no_jit:
    def __init__(self, mode: bool) -> None:
        global _NO_JIT
        self.prev = _NO_JIT
        _NO_JIT = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _NO_JIT
        _NO_JIT = self.prev
        return False


def is_exportable():
    return _EXPORTABLE


class set_exportable:
    def __init__(self, mode: bool) -> None:
        global _EXPORTABLE
        self.prev = _EXPORTABLE
        _EXPORTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _EXPORTABLE
        _EXPORTABLE = self.prev
        return False


def is_scriptable():
    return _SCRIPTABLE


class set_scriptable:
    def __init__(self, mode: bool) -> None:
        global _SCRIPTABLE
        self.prev = _SCRIPTABLE
        _SCRIPTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _SCRIPTABLE
        _SCRIPTABLE = self.prev
        return False


class set_layer_config:
    
    def __init__(
            self,
            scriptable: Optional[bool] = None,
            exportable: Optional[bool] = None,
            no_jit: Optional[bool] = None,
            no_activation_jit: Optional[bool] = None):
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        self.prev = _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        if scriptable is not None:
            _SCRIPTABLE = scriptable
        if exportable is not None:
            _EXPORTABLE = exportable
        if no_jit is not None:
            _NO_JIT = no_jit
        if no_activation_jit is not None:
            _NO_ACTIVATION_JIT = no_activation_jit

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT = self.prev
        return False
