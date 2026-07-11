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
    supports_execution_ir: bool
    supports_batched_matmul: bool
    supports_grouped_gemm: bool

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

    def array(self, *args: Any, **kwargs: Any) -> Any: ...

    def asarray(self, *args: Any, **kwargs: Any) -> Any: ...

    def to_host(self, value: Any) -> Any: ...

    def to_backend(self, value: Any, *, dtype: Any = None) -> Any: ...

    def activate(self) -> None: ...

    def deactivate(self) -> None: ...

    def current_device(self) -> str: ...

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any: ...

    def transpose(self, value: Any, axes: Any = None) -> Any: ...

    def reshape(self, value: Any, shape: Any) -> Any: ...

    def matmul(
        self, a: Any, b: Any, *, stream: Any = None, workspace: Any = None
    ) -> Any: ...

    def batched_matmul(
        self, a: Any, b: Any, *, stream: Any = None, workspace: Any = None
    ) -> Any: ...

    def grouped_gemm(
        self,
        descriptors: Any,
        tensors: Any,
        *,
        stream: Any = None,
        workspace: Any = None,
        policy: str = "direct",
    ) -> Any: ...

    def execute_plan(
        self, plan: Any, bindings: Any, *, stream: Any = None, workspace: Any = None
    ) -> Any: ...

    def sync(self) -> None: ...

    def free_all_blocks(self) -> None: ...
