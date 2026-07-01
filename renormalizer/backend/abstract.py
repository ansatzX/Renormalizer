# -*- coding: utf-8 -*-

import os
from typing import Any

import numpy as _np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.execution import (
    BackendCapabilities,
    CopyPolicy,
    DeviceSpec,
    array_info_for_backend,
    layout_from_array,
    legacy_device_kind,
    lower_pair_contraction_to_matmul,
)
from renormalizer.backend.mpi import SingleProcessDistributedMixin
from renormalizer.backend.transforms import UnavailableTransforms


class AbstractBackend(SingleProcessDistributedMixin):
    name = "abstract"
    supported_device_kinds = ("cpu",)
    available_device_kinds = ("cpu",)
    supports_cpu = True
    array_namespace = None
    ndarray = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"
    supports_gpu = False
    supports_autodiff = False
    supports_jit = False
    supports_sparse = False
    supports_functional_update = True
    host_array_types = (_np.ndarray,)
    device_array_types = ()

    def __init__(self, config=None):
        self.config = BackendConfig.from_config(config)
        self.device = None
        self.device_spec = self.config.device_spec
        self.first_mp = False
        self._real_dtype = None
        self._complex_dtype = None
        self.transforms = UnavailableTransforms(self.name)
        self.use_64bits()
        self._set_configured_device(self.supported_device_kinds, default="cpu")
        self._apply_precision_config()

    def _apply_precision_config(self):
        if self.config.precision == 32:
            self.use_32bits()
        elif self.config.precision == 64:
            self.use_64bits()
        elif os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

    def _set_configured_device(self, supported, default=None, available=None):
        supported = tuple(supported)
        available = tuple(available or supported)
        self.supported_device_kinds = supported
        self.available_device_kinds = available
        requested = self.config.device
        if requested is None:
            self.device = default or (available[0] if available else None)
            self.device_spec = self.config.device_spec or self._device_spec_from_kind(self.device)
            return
        if requested not in supported:
            raise ValueError(
                "{0} backend does not support device '{1}'. Supported devices: {2}."
                .format(self.name, requested, ", ".join(supported) or "none")
            )
        if requested not in available:
            raise ValueError(
                "{0} backend device '{1}' was requested but is not available. "
                "Available devices: {2}."
                .format(self.name, requested, ", ".join(available) or "none")
            )
        self.device = requested
        self.device_spec = self.config.device_spec or self._device_spec_from_kind(self.device)

    def _device_spec_from_kind(self, kind):
        if kind == "cpu" or kind is None:
            return DeviceSpec(kind="cpu")
        if kind == "gpu":
            return DeviceSpec(kind="cuda")
        return DeviceSpec(kind=kind)

    @property
    def capabilities(self):
        return BackendCapabilities(
            cpu=self.supports_cpu,
            gpu=self.supports_gpu,
            autodiff=self.supports_autodiff,
            jit=self.supports_jit,
            sparse=self.supports_sparse,
            functional_update=self.supports_functional_update,
            device_index=self.supports_gpu,
            streams=False,
            events=False,
            memory_pool=self.supports_gpu,
            matmul=True,
            batched_matmul=False,
            grouped_gemm=False,
            strided_batched_gemm=False,
            einsum=True,
            contract_expression=True,
            contraction_path=True,
            custom_contraction_plan=True,
            distributed=self.is_distributed,
            distributed_array=False,
            allreduce=self.size > 1,
            allgather=self.size > 1,
        )

    @property
    def fallback_policy(self):
        return self.config.fallback_policy

    def current_device(self):
        return self.device_spec or self._device_spec_from_kind(self.device)

    def set_device(self, device):
        spec = self.config.device_spec if device is None else BackendConfig(device=device).device_spec
        kind = legacy_device_kind(spec)
        if kind is None:
            kind = self.device or (self.available_device_kinds[0] if self.available_device_kinds else None)
        if kind not in self.supported_device_kinds:
            raise ValueError(
                "{0} backend does not support device '{1}'. Supported devices: {2}."
                .format(self.name, kind, ", ".join(self.supported_device_kinds) or "none")
            )
        if kind not in self.available_device_kinds:
            raise ValueError(
                "{0} backend device '{1}' was requested but is not available. Available devices: {2}."
                .format(self.name, kind, ", ".join(self.available_device_kinds) or "none")
            )
        self.device = kind
        self.device_spec = spec or self._device_spec_from_kind(kind)

    def device_count(self):
        return 1 if self.device in self.available_device_kinds else 0

    def use_32bits(self):
        self.dtypes = (_np.float32, _np.complex64)

    def use_64bits(self):
        self.dtypes = (_np.float64, _np.complex128)

    @property
    def is_32bits(self) -> bool:
        return self._real_dtype == _np.float32

    @property
    def real_dtype(self):
        return self._real_dtype

    @real_dtype.setter
    def real_dtype(self, tp):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._real_dtype = tp

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, tp):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._complex_dtype = tp

    @property
    def dtypes(self):
        return self.real_dtype, self.complex_dtype

    @dtypes.setter
    def dtypes(self, target):
        self.real_dtype, self.complex_dtype = target

    @property
    def canonical_atol(self):
        return getattr(self, "_canonical_atol", 1e-4 if self.is_32bits else 1e-8)

    @canonical_atol.setter
    def canonical_atol(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f'canonical_atol must be a non-negative number, got {value!r}')
        self._canonical_atol = value

    @property
    def canonical_rtol(self):
        return getattr(self, "_canonical_rtol", 1e-4 if self.is_32bits else 1e-5)

    @canonical_rtol.setter
    def canonical_rtol(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f'canonical_rtol must be a non-negative number, got {value!r}')
        self._canonical_rtol = value

    def numpy(self, x: Any):
        raise NotImplementedError

    def from_numpy(self, x: _np.ndarray):
        raise NotImplementedError

    def to_numpy(self, x: Any):
        return self.numpy(x)

    def to_host(self, x: Any):
        return self.to_numpy(x)

    def to_backend(self, x: Any):
        if self.is_host_array(x):
            return self.from_numpy(x)
        return self.asarray(x)

    def astype(self, x: Any, dtype, *, copy=CopyPolicy.IF_NEEDED):
        copy = CopyPolicy.from_value(copy)
        if copy is CopyPolicy.NEVER and getattr(x, "dtype", None) != dtype:
            from renormalizer.backend.execution import BackendCopyError

            raise BackendCopyError("astype would require a copy")
        return self.asarray(x, dtype=dtype)

    def ascontiguousarray(self, x: Any, *, copy=CopyPolicy.IF_NEEDED):
        return self.asarray(x)

    def is_array(self, x: Any) -> bool:
        return isinstance(x, self.ndarray)

    def is_host_array(self, x: Any) -> bool:
        return isinstance(x, self.host_array_types)

    def is_device_array(self, x: Any) -> bool:
        return isinstance(x, self.device_array_types)

    def is_distributed_array(self, x: Any) -> bool:
        return False

    def array_info(self, x: Any):
        return array_info_for_backend(self, x, self.current_device())

    def layout(self, x: Any):
        return layout_from_array(x)

    def lower_pair_contraction_to_matmul(self, spec):
        plan = lower_pair_contraction_to_matmul(spec, self.capabilities)
        try:
            from renormalizer.utils import profiling

            if profiling.should_record_op():
                profiling.record(
                    "contraction_plan",
                    backend=self.name,
                    lowering=plan.kind,
                    left_modes=[str(mode) for mode in spec.left.modes],
                    right_modes=[str(mode) for mode in spec.right.modes],
                    output_modes=[str(mode) for mode in spec.output_modes],
                    batch_modes=[str(mode) for mode in spec.left_batch_modes],
                    contracted_modes=[str(mode) for mode in spec.contracted_modes],
                    left_only_modes=[str(mode) for mode in spec.left_only_modes],
                    right_only_modes=[str(mode) for mode in spec.right_only_modes],
                    input_shapes=[tuple(spec.left.array.shape), tuple(spec.right.array.shape)],
                    output_shape=plan.output_shape,
                    input_dtypes=[
                        str(getattr(spec.left.array, "dtype", None)),
                        str(getattr(spec.right.array, "dtype", None)),
                    ],
                    device=str(self.current_device()),
                    flops=plan.estimated_flops,
                    read_bytes=sum(desc.estimated_read_bytes for desc in plan.descs),
                    write_bytes=sum(desc.estimated_write_bytes for desc in plan.descs),
                    copy_bytes=plan.copy_bytes,
                    workspace_bytes=plan.workspace_bytes,
                    num_gemm=1 if plan.kind == "gemm" else 0,
                    num_batched_gemm=1 if plan.kind in ("batched_gemm", "strided_batched_gemm") else 0,
                    num_grouped_tasks=0,
                    num_blocks=0,
                    num_shape_buckets=0,
                    fallback_reason=plan.fallback_reason,
                )
        except Exception:
            pass
        return plan

    def sync(self):
        return None

    def free_all_blocks(self):
        return None

    def log_memory_usage(self, header=""):
        return None

    def at_set(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] = value
        return y

    def at_add(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] += value
        return y

    def at_sub(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] -= value
        return y

    def at_mul(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] *= value
        return y
