# -*- coding: utf-8 -*-

import os
from typing import Any

import numpy as _np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.execution import (
    BackendCapabilities,
    BackendCopyError,
    BackendFeatureError,
    CopyPolicy,
    DeviceSpec,
    FallbackPolicy,
    array_info_for_backend,
    layout_from_array,
    legacy_device_kind,
    lower_pair_contraction_to_matmul,
    parse_einsum,
)
from renormalizer.backend.gemm import GemmTask, array_nbytes, grouped_gemm_fallback, grouped_gemm_stats, run_gemm_task
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
    supports_batched_matmul = True
    supports_grouped_gemm = False
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
            batched_matmul=self.supports_batched_matmul,
            grouped_gemm=self.supports_grouped_gemm,
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
            raise BackendCopyError("astype would require a copy")
        return self.asarray(x, dtype=dtype)

    def ascontiguousarray(self, x: Any, *, copy=CopyPolicy.IF_NEEDED):
        return self.make_contiguous(x, copy_policy=copy)

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

    def permute(self, x: Any, perm, *, copy_policy=CopyPolicy.IF_NEEDED):
        copy_policy = CopyPolicy.from_value(copy_policy)
        perm = tuple(int(index) for index in perm)
        xp = self.array_namespace or _np
        if hasattr(x, "permute"):
            result = x.permute(*perm)
        elif hasattr(xp, "permute"):
            result = xp.permute(x, perm)
        else:
            result = xp.transpose(x, perm)
        if copy_policy is CopyPolicy.ALWAYS:
            return self.make_contiguous(result, copy_policy=CopyPolicy.ALWAYS)
        return result

    def can_reshape_view(self, x: Any, shape) -> bool:
        shape = tuple(int(dim) for dim in shape)
        if self._prod_shape(shape) != self._prod_shape(getattr(x, "shape", ())):
            return False
        if hasattr(x, "view") and self.name == "torch":
            try:
                x.view(*shape)
                return True
            except Exception:
                return False
        xp = self.array_namespace or _np
        try:
            reshaped = xp.reshape(x, shape)
        except Exception:
            return False
        if self.name == "jax":
            return True
        return self._shares_memory(reshaped, x)

    def reshape_view(self, x: Any, shape):
        shape = tuple(int(dim) for dim in shape)
        if not self.can_reshape_view(x, shape):
            raise BackendCopyError("reshape would require a copy")
        if hasattr(x, "view") and self.name == "torch":
            return x.view(*shape)
        xp = self.array_namespace or _np
        return xp.reshape(x, shape)

    def make_contiguous(self, x: Any, *, copy_policy=CopyPolicy.IF_NEEDED):
        copy_policy = CopyPolicy.from_value(copy_policy)
        if self._is_c_contiguous(x):
            if copy_policy is CopyPolicy.ALWAYS:
                return self._copy_array(x)
            return x
        if copy_policy is CopyPolicy.NEVER:
            raise BackendCopyError("make_contiguous would require a copy")
        if hasattr(x, "contiguous"):
            return x.contiguous()
        xp = self.array_namespace or _np
        ascontiguousarray = getattr(xp, "ascontiguousarray", None)
        if ascontiguousarray is not None:
            return ascontiguousarray(x)
        return self.asarray(x)

    def parse_einsum(self, equation, *operands, constants=(), optimize=None):
        return parse_einsum(equation, *operands, constants=constants, optimize=optimize)

    def synchronize(self, device=None, stream=None):
        return self.sync()

    def unpack_masked_vectors(self, x: Any, spec):
        mask = self._validated_packed_mask(spec)
        shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if shape == (spec.packed_dim,):
            struct = self._zeros_backend(spec.center_shape, getattr(x, "dtype", None))
            return self._masked_set(struct, mask, x)
        if shape != (spec.packed_dim, spec.nrhs):
            raise ValueError(
                "packed vector shape must be ({0},) or ({0}, {1}); got {2}"
                .format(spec.packed_dim, spec.nrhs, shape)
            )
        struct = self._zeros_backend(spec.center_shape + (spec.nrhs,), getattr(x, "dtype", None))
        struct = self._masked_set(struct, mask, x)
        batch_axis = self._normalize_batch_axis(spec.batch_axis, len(spec.center_shape) + 1)
        if batch_axis == len(spec.center_shape):
            return struct
        return self._move_axis(struct, len(spec.center_shape), batch_axis)

    def pack_masked_vectors(self, x_struct: Any, spec):
        mask = self._validated_packed_mask(spec)
        shape = tuple(int(dim) for dim in getattr(x_struct, "shape", ()))
        if shape == spec.center_shape:
            return x_struct[mask]

        ndim = len(spec.center_shape) + 1
        batch_axis = self._normalize_batch_axis(spec.batch_axis, ndim)
        expected_shape = list(spec.center_shape)
        expected_shape.insert(batch_axis, spec.nrhs)
        expected_shape = tuple(expected_shape)
        if shape != expected_shape:
            raise ValueError(
                "center tensor shape must be {0} or {1}; got {2}"
                .format(spec.center_shape, expected_shape, shape)
            )
        if batch_axis == len(spec.center_shape):
            struct_last_batch = x_struct
        else:
            struct_last_batch = self._move_axis(x_struct, batch_axis, len(spec.center_shape))
        packed = struct_last_batch[mask]
        packed_shape = tuple(int(dim) for dim in getattr(packed, "shape", ()))
        if packed_shape != (spec.packed_dim, spec.nrhs):
            packed = self.reshape_view(packed, (spec.packed_dim, spec.nrhs))
        return packed

    def lower_pair_contraction_to_matmul(self, spec):
        plan = lower_pair_contraction_to_matmul(spec, self.capabilities)
        try:
            from renormalizer.utils import profiling

            if profiling.should_record_op():
                def operand_payload(operand):
                    info = self.array_info(operand.array)
                    return {
                        "name": operand.name,
                        "modes": [str(mode) for mode in operand.modes],
                        "shape": info.shape,
                        "dtype": str(info.dtype),
                        "nbytes": info.nbytes,
                        "ndim": info.ndim,
                        "strides": info.strides,
                        "order": info.order,
                        "contiguous": info.contiguous,
                        "backend": info.backend_name,
                        "device": str(info.device),
                        "device_kind": info.device.kind,
                        "device_index": info.device.index,
                        "is_host": info.is_host,
                        "is_device": info.is_device,
                    }

                profiling.record(
                    "contraction_plan",
                    backend=self.name,
                    lowering=plan.kind,
                    operands=[operand_payload(spec.left), operand_payload(spec.right)],
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

    def _desc_mode_groups(self, desc):
        if desc.layout_a is None or desc.layout_b is None or desc.layout_c is None:
            raise ValueError("MatmulDesc execution requires input and output layouts")
        left_modes = tuple(desc.layout_a.logical_modes)
        right_modes = tuple(desc.layout_b.logical_modes)
        output_modes = tuple(desc.layout_c.logical_modes)
        left_set = set(left_modes)
        right_set = set(right_modes)
        output_set = set(output_modes)
        batch_modes = tuple(mode for mode in output_modes if mode in left_set and mode in right_set)
        contracted_modes = tuple(mode for mode in left_modes if mode in right_set and mode not in output_set)
        left_only_modes = tuple(mode for mode in output_modes if mode in left_set and mode not in right_set)
        right_only_modes = tuple(mode for mode in output_modes if mode in right_set and mode not in left_set)
        return batch_modes, left_only_modes, contracted_modes, right_only_modes, output_modes

    @staticmethod
    def _mode_size_map(array, modes):
        return {mode: int(dim) for mode, dim in zip(modes, getattr(array, "shape", ()))}

    @staticmethod
    def _shape_for_modes(modes, size_map):
        return tuple(size_map[mode] for mode in modes)

    @staticmethod
    def _prod_shape(shape):
        result = 1
        for dim in shape:
            result *= int(dim)
        return result

    def _shares_memory(self, left, right):
        if left is right:
            return True
        xp = self.array_namespace or _np
        shares_memory = getattr(xp, "shares_memory", None)
        if shares_memory is not None:
            try:
                return bool(shares_memory(left, right))
            except Exception:
                pass
        try:
            return bool(_np.shares_memory(left, right))
        except Exception:
            return False

    def _is_c_contiguous(self, x):
        is_contiguous = getattr(x, "is_contiguous", None)
        if callable(is_contiguous):
            try:
                return bool(is_contiguous())
            except Exception:
                pass
        try:
            return self.layout(x).order == "C"
        except Exception:
            return False

    def _copy_array(self, x):
        copy = getattr(x, "copy", None)
        if callable(copy):
            return copy()
        clone = getattr(x, "clone", None)
        if callable(clone):
            return clone()
        xp = self.array_namespace or _np
        xp_copy = getattr(xp, "copy", None)
        if xp_copy is not None:
            return xp_copy(x)
        return self.asarray(x)

    def _validated_packed_mask(self, spec):
        mask_host = _np.asarray(self.to_numpy(spec.qn_mask) if self.is_array(spec.qn_mask) else spec.qn_mask, dtype=bool)
        if mask_host.shape != tuple(spec.center_shape):
            raise ValueError(
                "qn_mask shape must match center_shape {0}; got {1}"
                .format(spec.center_shape, mask_host.shape)
            )
        if int(mask_host.sum()) != int(spec.packed_dim):
            raise ValueError(
                "packed_dim must match qn_mask true count {0}; got {1}"
                .format(int(mask_host.sum()), spec.packed_dim)
            )
        return self.to_backend(mask_host)

    def _zeros_backend(self, shape, dtype):
        xp = self.array_namespace or _np
        if self.name == "torch":
            kwargs = {"dtype": dtype}
            if hasattr(self, "_kwargs_with_configured_device"):
                kwargs = self._kwargs_with_configured_device(kwargs)
            return xp.zeros(tuple(shape), **kwargs)
        return xp.zeros(tuple(shape), dtype=dtype)

    def _masked_set(self, array, mask, values):
        at = getattr(array, "at", None)
        if at is not None:
            return at[mask].set(values)
        array[mask] = values
        return array

    @staticmethod
    def _normalize_batch_axis(batch_axis, ndim):
        axis = int(batch_axis)
        if axis < 0:
            axis += int(ndim)
        if axis < 0 or axis >= int(ndim):
            raise ValueError("batch_axis {0} is out of bounds for ndim {1}".format(batch_axis, ndim))
        return axis

    def _move_axis(self, array, source, destination):
        xp = self.array_namespace or _np
        moveaxis = getattr(xp, "moveaxis", None)
        if moveaxis is not None:
            return moveaxis(array, source, destination)
        movedim = getattr(xp, "movedim", None)
        if movedim is not None:
            return movedim(array, source, destination)
        return _np.moveaxis(array, source, destination)

    def _transpose_modes(self, array, current_modes, target_modes):
        current_modes = tuple(current_modes)
        target_modes = tuple(target_modes)
        if current_modes == target_modes:
            return array
        perm = tuple(current_modes.index(mode) for mode in target_modes)
        xp = self.array_namespace or _np
        return xp.transpose(array, perm)

    def _prepare_matmul_desc(self, desc):
        xp = self.array_namespace or _np
        left_modes = tuple(desc.layout_a.logical_modes)
        right_modes = tuple(desc.layout_b.logical_modes)
        groups = self._desc_mode_groups(desc)
        batch_modes, left_only_modes, contracted_modes, right_only_modes, output_modes = groups
        left_sizes = self._mode_size_map(desc.A, left_modes)
        right_sizes = self._mode_size_map(desc.B, right_modes)
        batch_shape = self._shape_for_modes(batch_modes, left_sizes)
        left_shape = self._shape_for_modes(left_only_modes, left_sizes)
        contracted_shape = self._shape_for_modes(contracted_modes, left_sizes)
        right_shape = self._shape_for_modes(right_only_modes, right_sizes)

        left_target_modes = batch_modes + left_only_modes + contracted_modes
        right_target_modes = batch_modes + contracted_modes + right_only_modes
        a = self._transpose_modes(desc.A, left_modes, left_target_modes)
        b = self._transpose_modes(desc.B, right_modes, right_target_modes)
        a = xp.reshape(a, batch_shape + (self._prod_shape(left_shape), self._prod_shape(contracted_shape)))
        b = xp.reshape(b, batch_shape + (self._prod_shape(contracted_shape), self._prod_shape(right_shape)))
        return a, b, {
            "batch_modes": batch_modes,
            "left_only_modes": left_only_modes,
            "right_only_modes": right_only_modes,
            "output_modes": output_modes,
            "batch_shape": batch_shape,
            "left_shape": left_shape,
            "right_shape": right_shape,
        }

    def _finalize_matmul_result(self, result, groups):
        xp = self.array_namespace or _np
        result = xp.reshape(result, groups["batch_shape"] + groups["left_shape"] + groups["right_shape"])
        current_modes = groups["batch_modes"] + groups["left_only_modes"] + groups["right_only_modes"]
        return self._transpose_modes(result, current_modes, groups["output_modes"])

    @staticmethod
    def _is_matmul_desc(value):
        return all(hasattr(value, attr) for attr in ("A", "B", "m", "n", "k"))

    @staticmethod
    def _desc_to_task(desc):
        return GemmTask(
            desc.A,
            desc.B,
            C=desc.C,
            trans_a=desc.trans_a,
            trans_b=desc.trans_b,
            conj_a=desc.conj_a,
            conj_b=desc.conj_b,
            alpha=desc.alpha,
            beta=desc.beta,
        )

    def _execute_matmul_desc(self, desc, *, stream=None, workspace=None):
        if desc.batch_shape:
            raise BackendFeatureError("matmul received a batched descriptor; use batched_matmul")
        a, b, groups = self._prepare_matmul_desc(desc)
        result = self.matmul(
            a,
            b,
            C=desc.C,
            trans_a=desc.trans_a,
            trans_b=desc.trans_b,
            conj_a=desc.conj_a,
            conj_b=desc.conj_b,
            alpha=desc.alpha,
            beta=desc.beta,
        )
        return self._finalize_matmul_result(result, groups)

    def matmul(
        self,
        A,
        B=None,
        *,
        C=None,
        trans_a=False,
        trans_b=False,
        conj_a=False,
        conj_b=False,
        alpha=1.0,
        beta=0.0,
        stream=None,
        workspace=None,
    ):
        if B is None and self._is_matmul_desc(A):
            return self._execute_matmul_desc(A, stream=stream, workspace=workspace)
        xp = self.array_namespace or _np
        return run_gemm_task(
            GemmTask(
                A,
                B,
                C=C,
                trans_a=trans_a,
                trans_b=trans_b,
                conj_a=conj_a,
                conj_b=conj_b,
                alpha=alpha,
                beta=beta,
            ),
            xp=xp,
        )

    def _execute_batched_matmul_desc(self, desc, *, stream=None, workspace=None):
        a, b, groups = self._prepare_matmul_desc(desc)
        result = self.batched_matmul(
            a,
            b,
            C=desc.C,
            trans_a=desc.trans_a,
            trans_b=desc.trans_b,
            conj_a=desc.conj_a,
            conj_b=desc.conj_b,
            alpha=desc.alpha,
            beta=desc.beta,
        )
        return self._finalize_matmul_result(result, groups)

    def batched_matmul(
        self,
        A,
        B=None,
        *,
        C=None,
        trans_a=False,
        trans_b=False,
        conj_a=False,
        conj_b=False,
        alpha=1.0,
        beta=0.0,
        stream=None,
        workspace=None,
    ):
        if B is None and self._is_matmul_desc(A):
            return self._execute_batched_matmul_desc(A, stream=stream, workspace=workspace)
        xp = self.array_namespace or _np
        if conj_a:
            A = xp.conj(A)
        if conj_b:
            B = xp.conj(B)
        if trans_a:
            A = xp.swapaxes(A, -1, -2)
        if trans_b:
            B = xp.swapaxes(B, -1, -2)
        result = xp.matmul(A, B)
        if alpha != 1.0:
            result = alpha * result
        if C is not None:
            if beta != 0.0:
                result = result + beta * C
            C[...] = result
            return C
        return result

    def _loop_matmul(self, desc, *, stream=None, workspace=None):
        xp = self.array_namespace or _np
        a, b, groups = self._prepare_matmul_desc(desc)
        if desc.conj_a:
            a = xp.conj(a)
        if desc.conj_b:
            b = xp.conj(b)
        if desc.trans_a:
            a = xp.swapaxes(a, -1, -2)
        if desc.trans_b:
            b = xp.swapaxes(b, -1, -2)
        result = xp.matmul(a, b)
        if desc.alpha != 1.0:
            result = desc.alpha * result
        return self._finalize_matmul_result(result, groups)

    def grouped_gemm(self, tasks, *, pack_threshold=4, stream=None, workspace=None):
        xp = self.array_namespace or _np
        converted = [
            self._desc_to_task(task) if self._is_matmul_desc(task) else task
            for task in tasks
        ]
        fallback_reason = self._handle_grouped_gemm_fallback()
        stats = grouped_gemm_stats(converted, xp=xp, pack_threshold=pack_threshold)
        try:
            from renormalizer.utils import profiling

            should_profile = profiling.should_record_op()
        except Exception:
            should_profile = False
        if should_profile:
            import time

            started = time.perf_counter()
            result = grouped_gemm_fallback(converted, xp=xp, pack_threshold=pack_threshold)
            wall_s = time.perf_counter() - started
            try:
                from renormalizer.utils import profiling

                profiling.record(
                    "contraction_execute",
                    backend=self.name,
                    equation=None,
                    lowering="grouped_gemm",
                    input_shapes=[
                        [tuple(getattr(task.A, "shape", ())), tuple(getattr(task.B, "shape", ()))]
                        for task in converted
                    ],
                    output_shape=[tuple(getattr(item, "shape", ())) for item in result],
                    dtype=str(getattr(result[0], "dtype", None)) if result else None,
                    device=str(self.current_device()),
                    flops=stats.flops,
                    read_bytes=stats.read_bytes,
                    write_bytes=stats.write_bytes,
                    copy_bytes=stats.copy_bytes,
                    workspace_bytes=stats.workspace_bytes,
                    largest_intermediate=max((array_nbytes(item) for item in result), default=0),
                    num_gemm=stats.loop_task_count,
                    num_batched_gemm=stats.batched_bucket_count,
                    num_grouped_tasks=stats.task_count,
                    num_blocks=stats.task_count,
                    num_shape_buckets=stats.shape_bucket_count,
                    fallback_reason=fallback_reason,
                    bucket_task_counts=stats.bucket_task_counts,
                    batched_bucket_count=stats.batched_bucket_count,
                    loop_bucket_count=stats.loop_bucket_count,
                    batched_task_count=stats.batched_task_count,
                    loop_task_count=stats.loop_task_count,
                    pack_threshold=pack_threshold,
                    wall_s=wall_s,
                )
            except Exception:
                pass
            return result
        return grouped_gemm_fallback(converted, xp=xp, pack_threshold=pack_threshold)

    def _handle_grouped_gemm_fallback(self):
        if self.supports_grouped_gemm:
            return None
        reason = "native grouped_gemm unavailable; used bucketed fallback"
        if self.fallback_policy is FallbackPolicy.FORBID:
            raise BackendFeatureError(reason)
        if self.fallback_policy is FallbackPolicy.WARN:
            import warnings

            warnings.warn(reason, RuntimeWarning, stacklevel=3)
        return reason

    def _handle_plan_fallback(self, plan):
        if plan.fallback_reason is None:
            return
        if self.fallback_policy is FallbackPolicy.FORBID:
            raise BackendFeatureError(plan.fallback_reason)
        if self.fallback_policy is FallbackPolicy.WARN:
            import warnings

            warnings.warn(plan.fallback_reason, RuntimeWarning, stacklevel=3)

    def _execute_plan_impl(self, plan, *, stream=None, workspace=None):
        if not plan.descs:
            raise ValueError("MatmulPlan has no descriptors to execute")
        if plan.kind == "gemm":
            return self.matmul(plan.descs[0], stream=stream, workspace=workspace)
        if plan.kind in ("batched_gemm", "strided_batched_gemm"):
            return self.batched_matmul(plan.descs[0], stream=stream, workspace=workspace)
        if plan.kind == "grouped_gemm":
            return self.grouped_gemm(plan.descs, stream=stream, workspace=workspace)
        if plan.kind in ("fallback_tensordot", "fallback_einsum"):
            self._handle_plan_fallback(plan)
            if len(plan.descs) != 1:
                raise BackendFeatureError("fallback execution expects exactly one descriptor")
            return self._loop_matmul(plan.descs[0], stream=stream, workspace=workspace)
        raise BackendFeatureError("Unknown MatmulPlan kind {0!r}".format(plan.kind))

    def _record_contraction_execute(self, plan, result, wall_s):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            desc = plan.descs[0] if plan.descs else None
            profiling.record(
                "contraction_execute",
                backend=self.name,
                equation=None,
                lowering=plan.kind,
                input_shapes=[
                    tuple(getattr(desc.A, "shape", ())),
                    tuple(getattr(desc.B, "shape", ())),
                ] if desc is not None else [],
                output_shape=tuple(getattr(result, "shape", plan.output_shape)),
                dtype=str(getattr(result, "dtype", None)),
                device=str(self.current_device()),
                flops=plan.estimated_flops,
                read_bytes=sum(desc.estimated_read_bytes for desc in plan.descs),
                write_bytes=sum(desc.estimated_write_bytes for desc in plan.descs),
                copy_bytes=plan.copy_bytes,
                workspace_bytes=plan.workspace_bytes,
                largest_intermediate=getattr(result, "nbytes", None),
                num_gemm=1 if plan.kind == "gemm" else 0,
                num_batched_gemm=1 if plan.kind in ("batched_gemm", "strided_batched_gemm") else 0,
                num_grouped_tasks=len(plan.descs) if plan.kind == "grouped_gemm" else 0,
                num_blocks=0,
                num_shape_buckets=0,
                fallback_reason=plan.fallback_reason,
                wall_s=wall_s,
            )
        except Exception:
            pass

    def execute_matmul_plan(self, plan, *, stream=None, workspace=None):
        try:
            from renormalizer.utils import profiling

            should_profile = profiling.should_record_op()
        except Exception:
            should_profile = False
        if should_profile:
            import time

            start = time.perf_counter()
            result = self._execute_plan_impl(plan, stream=stream, workspace=workspace)
            wall_s = time.perf_counter() - start
            self._record_contraction_execute(plan, result, wall_s)
            return result
        return self._execute_plan_impl(plan, stream=stream, workspace=workspace)

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
