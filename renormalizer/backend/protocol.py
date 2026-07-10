# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""The minimal structural interface shared by numerical backends."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BackendProtocol(Protocol):
    name: str
    array_namespace: Any
    memory_errors: tuple[type[BaseException], ...]
    ndarray: type[Any] | tuple[type[Any], ...]
    opt_einsum_name: str
    supports_gpu: bool

    @property
    def real_dtype(self) -> Any: ...

    @property
    def complex_dtype(self) -> Any: ...

    @property
    def is_32bits(self) -> bool: ...

    def use_32bits(self) -> None: ...

    def use_64bits(self) -> None: ...

    def to_numpy(self, value: Any) -> Any: ...

    def from_numpy(self, value: Any) -> Any: ...
