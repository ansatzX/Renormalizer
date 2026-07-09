# -*- coding: utf-8 -*-

import contextlib
import os
from dataclasses import replace
from typing import Any

import numpy as _np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.execution import (
    ArrayInfo,
    BackendCapabilities,
    BackendCopyError,
    BackendFeatureError,
    BlockTensor,
    CommunicationPlan,
    ContractionPlan,
    ContractionStep,
    CopyPolicy,
    CostEstimate,
    DeviceMesh,
    DeviceSpec,
    DistributedContractionPlan,
    DistributedContractionSpec,
    DistributedStepPlan,
    DistributedTensor,
    DistributionState,
    EinsumSpec,
    FallbackPolicy,
    DenseBlock,
    GroupedGemmPlan,
    HardwareModel,
    LayoutSpec,
    MatmulDesc,
    MatmulPlan,
    PairContractionSpec,
    ShardingSpec,
    SlicedContractionPlan,
    StreamEvent,
    TensorOperand,
    Workspace,
    array_info_for_backend,
    layout_from_array,
    layout_from_modes_shape,
    legacy_device_kind,
    lower_pair_contraction_to_matmul,
    parse_einsum,
    parse_device_spec,
    parse_einsum_equation,
)
from renormalizer.backend.gemm import (
    BufferRef,
    GemmBatch,
    GemmTask,
    GemvBatch,
    GemvDesc,
    MatmulDesc as BufferMatmulDesc,
    array_nbytes,
    execute_prepacked_grouped_gemm as execute_prepacked_grouped_gemm_plan,
    gemm_task_key,
    grouped_gemm_bucketed_profiled,
    grouped_gemm_stats,
    prepack_grouped_gemm as prepack_grouped_gemm_plan,
    run_gemm_task,
)
from renormalizer.backend.mpi import SingleProcessDistributedMixin
from renormalizer.backend.transforms import UnavailableTransforms


class _ShapeOnlyArray:
    def __init__(self, shape, dtype, itemsize):
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.ndim = len(self.shape)
        self.size = 1
        for dim in self.shape:
            self.size *= int(dim)
        self.nbytes = int(self.size) * int(itemsize or 0)


class _BackendContractExpression:
    def __init__(
        self,
        expression,
        backend_name,
        backend=None,
        equation=None,
        expression_operands=(),
        constants=(),
        contract_kwargs=None,
    ):
        self.expression = expression
        self.backend_name = backend_name
        self._backend = backend
        self._planned_equation = equation
        self._expression_operands = tuple(expression_operands)
        self._constants = frozenset(int(index) for index in constants)
        self._contract_kwargs = {} if contract_kwargs is None else dict(contract_kwargs)

    def _planned_operands(self, arrays):
        variable_count = len(self._expression_operands) - len(self._constants)
        if len(arrays) != variable_count:
            return None
        array_iter = iter(arrays)
        operands = []
        for index, operand in enumerate(self._expression_operands):
            if index in self._constants:
                operands.append(operand)
            else:
                operands.append(next(array_iter))
        return tuple(operands)

    def __call__(
        self,
        *arrays,
        out=None,
        backend=None,
        evaluate_constants=False,
        stream=None,
        workspace=None,
    ):
        if (
            self._backend is not None
            and self._planned_equation is not None
            and out is None
            and backend is None
            and not evaluate_constants
        ):
            operands = self._planned_operands(arrays)
            if operands is None:
                return self.expression(
                    *arrays,
                    out=out,
                    backend=self.backend_name,
                    evaluate_constants=evaluate_constants,
                )
            kwargs = dict(self._contract_kwargs)
            if stream is not None:
                kwargs["stream"] = stream
            if workspace is not None:
                kwargs["workspace"] = workspace
            return self._backend.contract(self._planned_equation, *operands, **kwargs)
        return self.expression(
            *arrays,
            out=out,
            backend=self.backend_name if backend is None else backend,
            evaluate_constants=evaluate_constants,
        )

    def __getattr__(self, name):
        return getattr(self.expression, name)


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
    supports_complex64 = True
    supports_complex128 = True
    supports_fp32 = True
    supports_fp64 = True
    supports_mixed_precision = False
    supports_device_index = False
    supports_matmul = True
    supports_batched_matmul = True
    supports_grouped_gemm = False
    supports_strided_batched_gemm = True
    supports_einsum = True
    supports_contract_expression = True
    supports_contraction_path = True
    supports_custom_contraction_plan = True
    supports_streams = False
    supports_events = False
    supports_memory_pool = False
    supports_block_sparse = True
    supports_packed_blocks = False
    supports_scatter_add = True
    supports_distributed_array = True
    supports_allreduce = True
    supports_broadcast = True
    supports_allgather = True
    supports_reduce_scatter = True
    supports_alltoall = True
    supports_point_to_point = False
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
        self._last_execution_profile = None
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
            complex64=self.supports_complex64,
            complex128=self.supports_complex128,
            fp32=self.supports_fp32,
            fp64=self.supports_fp64,
            mixed_precision=self.supports_mixed_precision,
            device_index=self.supports_device_index,
            streams=bool(self.supports_streams and self.device == "gpu"),
            events=bool(self.supports_events and self.device == "gpu"),
            memory_pool=self.supports_memory_pool,
            matmul=self.supports_matmul,
            batched_matmul=self.supports_batched_matmul,
            grouped_gemm=self.supports_grouped_gemm,
            strided_batched_gemm=bool(self.supports_strided_batched_gemm and self.supports_batched_matmul),
            einsum=self.supports_einsum,
            contract_expression=self.supports_contract_expression,
            contraction_path=self.supports_contraction_path,
            custom_contraction_plan=self.supports_custom_contraction_plan,
            block_sparse=self.supports_block_sparse,
            packed_blocks=bool(self.supports_packed_blocks and self.supports_block_sparse),
            scatter_add=self.supports_scatter_add,
            distributed=self.is_distributed,
            distributed_array=bool(self.supports_distributed_array and self.is_distributed),
            allreduce=bool(self.supports_allreduce and self.size > 1),
            broadcast=bool(self.supports_broadcast and self.size > 1),
            allgather=bool(self.supports_allgather and self.size > 1),
            reduce_scatter=bool(self.supports_reduce_scatter and self.size > 1),
            alltoall=bool(self.supports_alltoall and self.size > 1),
            point_to_point=bool(self.supports_point_to_point and self.size > 1),
        )

    @property
    def fallback_policy(self):
        return self.config.fallback_policy

    @property
    def supports_distributed(self):
        return bool(self.is_distributed)

    def last_execution_profile(self):
        if self._last_execution_profile is None:
            return None
        return dict(self._last_execution_profile)

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

    def _default_dtype_for(self, x):
        if hasattr(x, "dtype"):
            return None
        try:
            dtype = _np.asarray(x).dtype
        except (TypeError, ValueError):
            return None
        if _np.issubdtype(dtype, _np.floating):
            return self.real_dtype
        if _np.issubdtype(dtype, _np.complexfloating):
            return self.complex_dtype
        return None

    def _kwargs_with_default_dtype(self, args, kwargs):
        kwargs = dict(kwargs)
        if "dtype" not in kwargs and args:
            dtype = self._default_dtype_for(args[0])
            if dtype is not None:
                kwargs["dtype"] = dtype
        return kwargs

    def to_numpy(self, x: Any):
        return self.numpy(x)

    def to_host(self, x: Any, *, copy=CopyPolicy.IF_NEEDED):
        copy = CopyPolicy.from_value(copy)
        if self.is_host_array(x):
            if copy is CopyPolicy.ALWAYS:
                return self._copy_array(x)
            return x
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_host would require creating a host array")
        result = self.to_numpy(x)
        if copy is CopyPolicy.ALWAYS:
            return _np.array(result, copy=True)
        return result

    def to_backend(self, x: Any, *, device=None, dtype=None, copy=CopyPolicy.IF_NEEDED):
        del device
        copy = CopyPolicy.from_value(copy)
        dtype_requires_copy = dtype is not None and getattr(x, "dtype", None) != dtype
        if self.is_array(x):
            if copy is CopyPolicy.NEVER and dtype_requires_copy:
                raise BackendCopyError("to_backend would require a dtype conversion copy")
            if copy is CopyPolicy.ALWAYS:
                return self.array(x, dtype=dtype, copy=True)
            if dtype_requires_copy:
                return self.asarray(x, dtype=dtype)
            return x
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_backend would require creating a backend array")
        if self.is_host_array(x):
            result = self.from_numpy(x)
            if dtype is not None and getattr(result, "dtype", None) != dtype:
                result = self.asarray(result, dtype=dtype)
            return result
        return self.asarray(x, dtype=dtype)

    def _promote_tensordot_operands(self, a, b):
        if not (hasattr(a, "dtype") and hasattr(b, "dtype")):
            return a, b
        if a.dtype == b.dtype:
            return a, b
        xp = self.array_namespace or _np
        promote_types = getattr(xp, "promote_types", None)
        if promote_types is not None:
            target = promote_types(a.dtype, b.dtype)
        else:
            target = _np.result_type(a.dtype, b.dtype)
        return xp.asarray(a, dtype=target), xp.asarray(b, dtype=target)

    @staticmethod
    def _normalize_tensordot_axes(axes, a_ndim, b_ndim):
        if isinstance(axes, (int, _np.integer)):
            axes = int(axes)
            if axes < 0:
                raise ValueError("tensordot axes must be non-negative when given as an integer")
            if axes > a_ndim or axes > b_ndim:
                raise ValueError("tensordot axes exceeds input rank")
            return tuple(range(a_ndim - axes, a_ndim)), tuple(range(axes))
        if not (isinstance(axes, (tuple, list)) and len(axes) == 2):
            raise ValueError("tensordot axes must be an integer or a pair of axis lists")
        left_axes, right_axes = axes
        if isinstance(left_axes, (int, _np.integer)):
            left_axes = (int(left_axes),)
        else:
            left_axes = tuple(int(axis) for axis in left_axes)
        if isinstance(right_axes, (int, _np.integer)):
            right_axes = (int(right_axes),)
        else:
            right_axes = tuple(int(axis) for axis in right_axes)

        def normalize(axis_values, ndim, side):
            normalized = []
            for axis in axis_values:
                if axis < 0:
                    axis += ndim
                if axis < 0 or axis >= ndim:
                    raise ValueError("{0} tensordot axis {1} is out of range for rank {2}".format(side, axis, ndim))
                normalized.append(axis)
            if len(set(normalized)) != len(normalized):
                raise ValueError("{0} tensordot axes must be unique".format(side))
            return tuple(normalized)

        left_axes = normalize(left_axes, a_ndim, "left")
        right_axes = normalize(right_axes, b_ndim, "right")
        if len(left_axes) != len(right_axes):
            raise ValueError("tensordot axis lists must have the same length")
        return left_axes, right_axes

    @classmethod
    def _tensordot_contract_modes(cls, a_ndim, b_ndim, axes):
        left_axes, right_axes = cls._normalize_tensordot_axes(axes, a_ndim, b_ndim)
        left_modes = list(range(a_ndim))
        right_modes = [None] * b_ndim
        for left_axis, right_axis in zip(left_axes, right_axes):
            right_modes[right_axis] = left_modes[left_axis]
        next_mode = a_ndim
        for axis in range(b_ndim):
            if right_modes[axis] is None:
                right_modes[axis] = next_mode
                next_mode += 1
        output_modes = [left_modes[axis] for axis in range(a_ndim) if axis not in left_axes]
        output_modes.extend(right_modes[axis] for axis in range(b_ndim) if axis not in right_axes)
        return left_modes, right_modes, output_modes

    def tensordot(self, a, b, axes=2, *, stream=None, workspace=None):
        a, b = self._promote_tensordot_operands(a, b)
        left_modes, right_modes, output_modes = self._tensordot_contract_modes(a.ndim, b.ndim, axes)
        spec = EinsumSpec(
            operands=(
                TensorOperand(a, left_modes, name="left"),
                TensorOperand(b, right_modes, name="right"),
            ),
            output_modes=tuple(output_modes),
        )
        plan = self.plan_contraction(spec)
        return self.execute(plan, stream=stream, workspace=workspace)

    def einsum(self, subscripts, *operands, **kwargs):
        return self.contract(subscripts, *operands, **kwargs)

    @staticmethod
    def _dtype_matches(actual, requested):
        if requested is None:
            return True
        if actual == requested:
            return True
        try:
            return _np.dtype(actual) == _np.dtype(requested)
        except Exception:
            return False

    def astype(self, x: Any, dtype, *, copy=CopyPolicy.IF_NEEDED):
        copy = CopyPolicy.from_value(copy)
        dtype_matches = self._dtype_matches(getattr(x, "dtype", None), dtype)
        if copy is CopyPolicy.NEVER and not dtype_matches:
            raise BackendCopyError("astype would require a copy")
        if dtype_matches and copy is not CopyPolicy.ALWAYS:
            return x
        if copy is CopyPolicy.ALWAYS:
            return self.array(x, dtype=dtype, copy=True)
        return self.asarray(x, dtype=dtype)

    def ascontiguousarray(self, x: Any, *, copy=CopyPolicy.IF_NEEDED):
        return self.make_contiguous(x, copy_policy=copy)

    def is_array(self, x: Any) -> bool:
        return isinstance(x, self.ndarray) or self.is_distributed_array(x)

    def is_host_array(self, x: Any) -> bool:
        return isinstance(x, self.host_array_types)

    def is_device_array(self, x: Any) -> bool:
        return isinstance(x, self.device_array_types)

    def is_distributed_array(self, x: Any) -> bool:
        return isinstance(x, DistributedTensor)

    def _device_spec_for_array(self, x: Any):
        return self.current_device()

    def array_info(self, x: Any):
        if self.is_distributed_array(x):
            local_info = array_info_for_backend(self, x.local_array, self._device_spec_for_array(x.local_array))
            return ArrayInfo(
                shape=tuple(x.global_shape),
                dtype=x.dtype,
                itemsize=local_info.itemsize,
                ndim=len(x.global_shape),
                size=self._prod_shape(x.global_shape),
                nbytes=int(x.local_nbytes),
                device=DeviceSpec(
                    kind="distributed",
                    local_rank=x.mesh.local_rank,
                    global_rank=x.mesh.global_rank,
                ),
                is_host=False,
                is_device=local_info.is_device,
                is_distributed=True,
                strides=None,
                order="distributed",
                contiguous=False,
                writeable=local_info.writeable,
                owns_data=local_info.owns_data,
                backend_name=self.name,
            )
        return array_info_for_backend(self, x, self._device_spec_for_array(x))

    def layout(self, x: Any):
        if self.is_distributed_array(x):
            local_layout = layout_from_array(x.local_array, x.modes)
            return LayoutSpec(
                logical_shape=tuple(x.global_shape),
                physical_shape=tuple(x.local_shape),
                logical_modes=tuple(x.modes),
                strides=local_layout.strides,
                order="distributed",
                contiguous_groups=local_layout.contiguous_groups,
                requires_transpose=local_layout.requires_transpose,
                transpose_perm=local_layout.transpose_perm,
                estimated_copy_bytes=local_layout.estimated_copy_bytes,
            )
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
        if copy_policy is CopyPolicy.NEVER and result is not x and not self._shares_memory(result, x):
            raise BackendCopyError("permute would require a copy")
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

    def make_contiguous(self, x: Any, *, mode_groups=None, copy_policy=CopyPolicy.IF_NEEDED):
        self._validate_mode_groups(x, mode_groups)
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

    @staticmethod
    def _validate_mode_groups(x, mode_groups):
        if mode_groups is None:
            return
        ndim = len(getattr(x, "shape", ()))
        seen = set()
        for group in mode_groups:
            axes = tuple(group)
            if not axes:
                raise ValueError("mode_groups entries must not be empty")
            normalized = []
            for axis in axes:
                axis = int(axis)
                if axis < 0:
                    axis += ndim
                if axis < 0 or axis >= ndim:
                    raise ValueError("mode_groups axes must be valid for array rank {0}".format(ndim))
                if axis in seen:
                    raise ValueError("mode_groups must not repeat axes")
                normalized.append(axis)
            if tuple(normalized) != tuple(range(normalized[0], normalized[0] + len(normalized))):
                raise ValueError("mode_groups axes must be adjacent in logical axis order")
            seen.update(normalized)

    def parse_einsum(self, equation, *operands, constants=(), optimize=None):
        return parse_einsum(equation, *operands, constants=constants, optimize=optimize)

    @staticmethod
    def _explicit_contract_is_plannable(args, kwargs):
        if len(args) < 3 or not isinstance(args[0], str):
            return False
        equation = "".join(args[0].split())
        if "->" not in equation or "..." in equation:
            return False
        supported_kwargs = {
            "optimize",
            "memory_limit",
            "prefer",
            "allow_slicing",
            "allow_distribution",
            "target_devices",
            "stream",
            "workspace",
        }
        for key, value in kwargs.items():
            if key == "backend" and value is None:
                continue
            if key == "out" and value is None:
                continue
            if key not in supported_kwargs:
                return False
        return True

    @staticmethod
    def _plan_policy_kwargs(kwargs):
        return {
            key: kwargs[key]
            for key in (
                "memory_limit",
                "prefer",
                "allow_slicing",
                "allow_distribution",
                "target_devices",
            )
            if key in kwargs
        }

    @staticmethod
    def _contract_expression_is_plannable(args, kwargs):
        if len(args) < 3 or not isinstance(args[0], str):
            return False
        equation = "".join(args[0].split())
        if "->" not in equation or "..." in equation:
            return False
        try:
            constants = tuple(int(index) for index in (kwargs.get("constants") or ()))
        except TypeError:
            return False
        operand_count = len(args) - 1
        if any(index < 0 or index >= operand_count for index in constants):
            return False
        supported_kwargs = {
            "optimize",
            "constants",
            "memory_limit",
            "prefer",
            "allow_slicing",
            "allow_distribution",
            "target_devices",
        }
        return all(key in supported_kwargs for key in kwargs)

    def _execute_explicit_planned_contract(self, args, kwargs):
        equation = args[0]
        operands = args[1:]
        optimize = kwargs.get("optimize")
        spec = self.parse_einsum(equation, *operands, optimize=optimize)
        plan_kwargs = self._plan_policy_kwargs(kwargs)
        plan = self.plan_contraction(spec, **plan_kwargs)
        execute_kwargs = {
            key: kwargs[key]
            for key in ("stream", "workspace")
            if key in kwargs
        }
        return self.execute(plan, **execute_kwargs)

    def contract(self, *args, **kwargs):
        import opt_einsum as oe

        if self._explicit_contract_is_plannable(args, kwargs):
            return self._execute_explicit_planned_contract(args, kwargs)
        if kwargs.get("backend") is None:
            kwargs = dict(kwargs)
            kwargs["backend"] = self.opt_einsum_name
        return oe.contract(*args, **kwargs)

    def contract_expression(self, *args, **kwargs):
        import opt_einsum as oe

        planned = self._contract_expression_is_plannable(args, kwargs)
        if planned:
            planned_constants = kwargs.get("constants") or ()
            oe_kwargs = {
                key: kwargs[key]
                for key in ("optimize", "constants")
                if key in kwargs
            }
            contract_kwargs = {
                key: kwargs[key]
                for key in ("optimize",)
                if key in kwargs
            }
            contract_kwargs.update(self._plan_policy_kwargs(kwargs))
        else:
            planned_constants = ()
            oe_kwargs = kwargs
            contract_kwargs = None
        return _BackendContractExpression(
            oe.contract_expression(*args, **oe_kwargs),
            self.opt_einsum_name,
            backend=self if planned else None,
            equation="".join(args[0].split()) if planned else None,
            expression_operands=args[1:] if planned else (),
            constants=planned_constants,
            contract_kwargs=contract_kwargs,
        )

    def contract_path(self, *args, **kwargs):
        import opt_einsum as oe

        return oe.contract_path(*args, **kwargs)

    @staticmethod
    def _contraction_step_from_matmul_plan(plan, input_modes, output_modes, *, inputs=(0, 1), output=2):
        write_bytes = sum(desc.estimated_write_bytes for desc in plan.descs)
        peak_bytes = write_bytes + int(plan.copy_bytes) + int(plan.workspace_bytes)
        return ContractionStep(
            kind=plan.kind,
            inputs=tuple(inputs),
            output=int(output),
            input_modes=tuple(tuple(modes) for modes in input_modes),
            output_modes=tuple(output_modes),
            plan=plan,
            estimated_flops=plan.estimated_flops,
            estimated_read_bytes=sum(desc.estimated_read_bytes for desc in plan.descs),
            estimated_write_bytes=write_bytes,
            estimated_copy_bytes=plan.copy_bytes,
            estimated_peak_bytes=peak_bytes,
            estimated_comm_bytes=0,
            required_workspace_bytes=plan.workspace_bytes,
            reason=plan.reason,
            fallback_reason=plan.fallback_reason,
        )

    @staticmethod
    def _contraction_plan_from_step(step, input_specs, output_modes, *, sliced_modes=(), distributed_modes=()):
        return ContractionPlan(
            steps=(step,),
            input_specs=input_specs,
            output_modes=output_modes,
            estimated_flops=step.estimated_flops,
            estimated_peak_bytes=step.estimated_peak_bytes,
            estimated_read_bytes=step.estimated_read_bytes,
            estimated_write_bytes=step.estimated_write_bytes,
            estimated_copy_bytes=step.estimated_copy_bytes,
            estimated_comm_bytes=step.estimated_comm_bytes,
            required_workspace_bytes=step.required_workspace_bytes,
            sliced_modes=tuple(sliced_modes),
            distributed_modes=tuple(distributed_modes),
        )

    def _shape_only_operand(self, shape, modes, left_operand, right_operand, *, name):
        itemsize = max(
            self._operand_itemsize(left_operand.array),
            self._operand_itemsize(right_operand.array),
            1,
        )
        dtype = getattr(left_operand.array, "dtype", None) or getattr(right_operand.array, "dtype", None)
        array = _ShapeOnlyArray(shape, dtype, itemsize)
        return TensorOperand(
            array,
            tuple(modes),
            layout=layout_from_modes_shape(modes, shape),
            name=name,
        )

    @staticmethod
    def _estimate_multi_step_peak_bytes(steps, *, input_count):
        live_temporary_bytes = [0] * int(input_count)
        peak = 0
        for step_index, step in enumerate(steps):
            current_live = sum(live_temporary_bytes)
            peak = max(peak, current_live + int(step.estimated_peak_bytes or 0))
            for index in sorted((int(index) for index in step.inputs), reverse=True):
                del live_temporary_bytes[index]
            output_bytes = int(step.estimated_write_bytes or step.estimated_peak_bytes or 0)
            live_temporary_bytes.append(output_bytes if step_index < len(steps) - 1 else 0)
        return int(peak)

    @classmethod
    def _contraction_step_largest_intermediate_memory(cls, step):
        plan = getattr(step, "plan", None)
        if isinstance(plan, MatmulPlan):
            return cls._matmul_plan_largest_intermediate_memory(plan)
        return int(getattr(step, "estimated_write_bytes", 0) or 0), 0

    @classmethod
    def _contraction_plan_largest_intermediate_memory(cls, plan):
        candidates = [
            cls._contraction_step_largest_intermediate_memory(step)
            for step in tuple(getattr(plan, "steps", ()))
        ]
        return max(candidates, default=(0, 0), key=lambda item: item[0])

    @staticmethod
    def _dtype_label(dtype):
        if dtype is None:
            return None
        name = getattr(dtype, "name", None)
        if name is not None:
            return str(name)
        try:
            return str(_np.dtype(dtype))
        except (TypeError, ValueError):
            return str(dtype)

    @classmethod
    def _matmul_plan_output_dtype(cls, plan):
        for desc in reversed(tuple(getattr(plan, "descs", ())) or ()):
            dtype = getattr(desc, "dtype_output", None)
            if dtype is None and getattr(desc, "C", None) is not None:
                dtype = getattr(desc.C, "dtype", None)
            if dtype is None:
                input_dtypes = [
                    getattr(getattr(desc, "A", None), "dtype", None),
                    getattr(getattr(desc, "B", None), "dtype", None),
                ]
                try:
                    dtype = _np.result_type(*[item for item in input_dtypes if item is not None])
                except (TypeError, ValueError):
                    labels = [cls._dtype_label(item) for item in input_dtypes]
                    labels = [item for item in labels if item is not None]
                    if labels and len(set(labels)) == 1:
                        return labels[0]
            label = cls._dtype_label(dtype)
            if label is not None:
                return label
        return None

    @classmethod
    def _contraction_plan_output_dtype(cls, plan):
        for step in reversed(tuple(getattr(plan, "steps", ())) or ()):
            step_plan = getattr(step, "plan", None)
            if isinstance(step_plan, MatmulPlan):
                label = cls._matmul_plan_output_dtype(step_plan)
                if label is not None:
                    return label
        return None

    @staticmethod
    def _profile_contraction_steps(steps):
        items = []
        for index, step in enumerate(steps):
            plan = getattr(step, "plan", None)
            lowering = getattr(plan, "kind", None) or step.kind
            execution_items = {}
            if isinstance(plan, MatmulPlan):
                execution_items = AbstractBackend._matmul_plan_execution_items(
                    plan,
                    fallback_reason=getattr(step, "fallback_reason", None),
                )
            items.append({
                "index": int(index),
                "kind": str(step.kind),
                "lowering": str(lowering),
                **execution_items,
                "inputs": [int(item) for item in step.inputs],
                "output": int(step.output),
                "input_modes": [
                    [str(mode) for mode in modes]
                    for modes in step.input_modes
                ],
                "output_modes": [str(mode) for mode in step.output_modes],
                "plan_hash": getattr(plan, "plan_hash", None),
                "flops": int(step.estimated_flops),
                "read_bytes": int(step.estimated_read_bytes),
                "write_bytes": int(step.estimated_write_bytes),
                "copy_bytes": int(step.estimated_copy_bytes),
                "peak_bytes": int(step.estimated_peak_bytes),
                "comm_bytes": int(step.estimated_comm_bytes),
                "workspace_bytes": int(step.required_workspace_bytes),
                "fallback_reason": getattr(step, "fallback_reason", None),
                "reason": getattr(step, "reason", None),
            })
        return items

    def _multi_operand_contraction_plan(
        self,
        spec,
        *,
        sliced_modes=(),
        distributed_modes=(),
        comm_bytes=0,
        record_profile=True,
    ):
        import opt_einsum as oe

        equation = self._equation_from_modes(
            tuple(operand.modes for operand in spec.operands),
            spec.output_modes,
        )
        _, contraction_list = oe.contract_path(
            equation,
            *(operand.array for operand in spec.operands),
            optimize=spec.optimize,
            einsum_call=True,
        )
        current_operands = list(spec.operands)
        steps = []
        next_output = len(spec.operands)
        for step_index, contraction in enumerate(contraction_list):
            indices, _, einsum_str, _, _ = contraction
            _, output_text = einsum_str.split("->", 1)
            output_modes = tuple(output_text)
            selected = tuple(current_operands[int(index)] for index in indices)
            if len(selected) != 2:
                raise BackendFeatureError("multi-step contraction currently supports pairwise opt_einsum steps")
            pair_spec = PairContractionSpec.from_operands(selected[0], selected[1], output_modes)
            matmul_plan = self.lower_pair_contraction_to_matmul(pair_spec, record_profile=record_profile)
            self._handle_plan_fallback(matmul_plan)
            step = self._contraction_step_from_matmul_plan(
                matmul_plan,
                input_modes=(selected[0].modes, selected[1].modes),
                output_modes=output_modes,
                inputs=indices,
                output=next_output,
            )
            if comm_bytes and step_index == len(contraction_list) - 1:
                step = replace(
                    step,
                    kind="distributed_contract",
                    estimated_comm_bytes=comm_bytes,
                    reason="distributed contraction with local multi-step matmul plan",
                )
            steps.append(step)
            for index in sorted((int(index) for index in indices), reverse=True):
                del current_operands[index]
            current_operands.append(
                self._shape_only_operand(
                    matmul_plan.output_shape,
                    output_modes,
                    selected[0],
                    selected[1],
                    name="intermediate{0}".format(step_index),
                )
            )
            next_output += 1

        plan = ContractionPlan(
            steps=tuple(steps),
            input_specs=spec.operands,
            output_modes=spec.output_modes,
            estimated_flops=sum(step.estimated_flops for step in steps),
            estimated_peak_bytes=self._estimate_multi_step_peak_bytes(
                steps,
                input_count=len(spec.operands),
            ),
            estimated_read_bytes=sum(step.estimated_read_bytes for step in steps),
            estimated_write_bytes=sum(step.estimated_write_bytes for step in steps),
            estimated_copy_bytes=sum(step.estimated_copy_bytes for step in steps),
            estimated_comm_bytes=sum(step.estimated_comm_bytes for step in steps),
            required_workspace_bytes=max((step.required_workspace_bytes for step in steps), default=0),
            sliced_modes=tuple(sliced_modes),
            distributed_modes=tuple(distributed_modes),
        )
        self._record_multi_step_contraction_plan(plan, record_profile=record_profile)
        return plan

    def _record_multi_step_contraction_plan(self, plan, *, record_profile=True):
        try:
            from renormalizer.utils import profiling

            if not record_profile or not profiling.should_record_op():
                return
            equation, input_modes, output_modes = self._contraction_plan_profile_metadata(plan)
            step_lowerings = [
                getattr(step.plan, "kind", step.kind)
                for step in plan.steps
            ]
            largest_intermediate_bytes, largest_intermediate_elements = (
                self._contraction_plan_largest_intermediate_memory(plan)
            )

            profiling.record(
                "contraction_plan",
                backend=self.name,
                equation=equation,
                lowering="multi_step",
                dtype=self._contraction_plan_output_dtype(plan),
                step_lowerings=step_lowerings,
                step_count=len(plan.steps),
                contraction_steps=self._profile_contraction_steps(plan.steps),
                plan_hash=plan.plan_hash,
                operands=[self._profile_tensor_operand(operand) for operand in plan.input_specs],
                input_modes=[[str(mode) for mode in modes] for modes in input_modes],
                output_modes=[str(mode) for mode in output_modes],
                input_shapes=[
                    tuple(getattr(operand.array, "shape", ()))
                    for operand in plan.input_specs
                ],
                output_shape=self._output_shape_for_contraction_plan(plan),
                input_dtypes=[
                    str(getattr(operand.array, "dtype", None))
                    for operand in plan.input_specs
                ],
                **self._profile_device_execution(self.current_device()),
                flops=plan.estimated_flops,
                read_bytes=plan.estimated_read_bytes,
                write_bytes=plan.estimated_write_bytes,
                copy_bytes=plan.estimated_copy_bytes,
                workspace_bytes=plan.required_workspace_bytes,
                peak_bytes=plan.estimated_peak_bytes,
                largest_intermediate=largest_intermediate_bytes,
                largest_intermediate_elements=largest_intermediate_elements,
                largest_intermediate_bytes=largest_intermediate_bytes,
                num_gemm=sum(1 for lowering in step_lowerings if lowering == "gemm"),
                num_batched_gemm=sum(
                    1
                    for lowering in step_lowerings
                    if lowering in ("batched_gemm", "strided_batched_gemm")
                ),
                num_grouped_tasks=sum(
                    len(getattr(step.plan, "descs", ()) or ())
                    for step, lowering in zip(plan.steps, step_lowerings)
                    if lowering == "grouped_gemm"
                ),
                num_blocks=0,
                num_shape_buckets=0,
                fallback_reason=next((step.fallback_reason for step in plan.steps if step.fallback_reason), None),
            )
        except Exception:
            pass

    def _plan_einsum_contraction(
        self,
        spec,
        *,
        sliced_modes=(),
        distributed_modes=(),
        comm_bytes=0,
        record_profile=True,
    ):
        if len(spec.operands) != 2:
            return self._multi_operand_contraction_plan(
                spec,
                sliced_modes=sliced_modes,
                distributed_modes=distributed_modes,
                comm_bytes=comm_bytes,
                record_profile=record_profile,
            )
        pair_spec = PairContractionSpec.from_operands(spec.operands[0], spec.operands[1], spec.output_modes)
        matmul_plan = self.lower_pair_contraction_to_matmul(pair_spec, record_profile=record_profile)
        self._handle_plan_fallback(matmul_plan)
        step = self._contraction_step_from_matmul_plan(
            matmul_plan,
            input_modes=(spec.operands[0].modes, spec.operands[1].modes),
            output_modes=spec.output_modes,
        )
        if comm_bytes:
            step = ContractionStep(
                kind="distributed_contract",
                inputs=step.inputs,
                output=step.output,
                input_modes=step.input_modes,
                output_modes=step.output_modes,
                plan=step.plan,
                estimated_flops=step.estimated_flops,
                estimated_read_bytes=step.estimated_read_bytes,
                estimated_write_bytes=step.estimated_write_bytes,
                estimated_copy_bytes=step.estimated_copy_bytes,
                estimated_peak_bytes=step.estimated_peak_bytes,
                estimated_comm_bytes=comm_bytes,
                required_workspace_bytes=step.required_workspace_bytes,
                reason="distributed contraction with local matmul plan",
                fallback_reason=step.fallback_reason,
            )
        return self._contraction_plan_from_step(
            step,
            input_specs=spec.operands,
            output_modes=spec.output_modes,
            sliced_modes=tuple(sliced_modes),
            distributed_modes=tuple(distributed_modes),
        )

    @staticmethod
    def _operand_global_shape(operand):
        if isinstance(operand, DistributedTensor):
            return tuple(operand.global_shape)
        return tuple(int(dim) for dim in getattr(operand, "shape", ()))

    @staticmethod
    def _operand_dtype(operand):
        if isinstance(operand, DistributedTensor):
            return getattr(operand, "dtype", None)
        return getattr(operand, "dtype", None)

    def _operand_itemsize(self, operand):
        if isinstance(operand, DistributedTensor):
            size = self._prod_shape(operand.local_shape)
            if size:
                return int(operand.local_nbytes // size)
            return int(getattr(getattr(operand, "dtype", None), "itemsize", 0) or 0)
        size = self._prod_shape(getattr(operand, "shape", ()))
        if size:
            return int(self._array_nbytes(operand) // size)
        return int(getattr(getattr(operand, "dtype", None), "itemsize", 0) or 0)

    def _distribution_state_for_operand(self, index, operand, modes):
        modes = tuple(modes)
        if isinstance(operand, DistributedTensor):
            distributed_modes = tuple(mode for mode in modes if mode in operand.sharding.sharded_modes)
            replicated_modes = tuple(mode for mode in modes if mode not in operand.sharding.sharded_modes)
            return DistributionState(
                operand_index=index,
                tensor_id=index,
                modes=modes,
                sharding=operand.sharding,
                shape=tuple(operand.global_shape),
                distributed_modes=distributed_modes,
                replicated_modes=replicated_modes,
                local_shape=tuple(operand.local_shape),
                local_nbytes=int(operand.local_nbytes),
            )
        shape = self._operand_global_shape(operand)
        return DistributionState(
            operand_index=index,
            tensor_id=index,
            modes=modes,
            sharding=None,
            shape=shape,
            distributed_modes=(),
            replicated_modes=modes,
            local_shape=shape,
            local_nbytes=self._array_nbytes(operand),
        )

    def _distribution_state_for_output(self, tensor_id, modes, shape, sharding, itemsize):
        modes = tuple(modes)
        shape = tuple(int(dim) for dim in shape)
        if sharding is None:
            distributed_modes = ()
            replicated_modes = modes
            local_shape = shape
        else:
            distributed_modes = tuple(mode for mode in modes if mode in sharding.sharded_modes)
            replicated_modes = tuple(mode for mode in modes if mode not in sharding.sharded_modes)
            local_shape = self._local_shape_for_sharding(sharding)
        return DistributionState(
            operand_index=tensor_id,
            tensor_id=tensor_id,
            modes=modes,
            sharding=sharding,
            shape=shape,
            distributed_modes=distributed_modes,
            replicated_modes=replicated_modes,
            local_shape=local_shape,
            local_nbytes=self._prod_shape(local_shape) * int(itemsize or 0),
        )

    @staticmethod
    def _mode_sizes_from_equation(input_modes, operands):
        sizes = {}
        for modes, operand in zip(input_modes, operands):
            shape = AbstractBackend._operand_global_shape(operand)
            if len(modes) != len(shape):
                raise ValueError(
                    "einsum operand rank mismatch: equation has {0} modes but array rank is {1}"
                    .format(len(modes), len(shape))
                )
            for mode, dim in zip(modes, shape):
                dim = int(dim)
                if mode in sizes and sizes[mode] != dim:
                    raise ValueError("einsum mode {0!r} has inconsistent sizes".format(mode))
                sizes[mode] = dim
        return sizes

    @staticmethod
    def _output_shape_from_sizes(output_modes, sizes):
        return tuple(int(sizes[mode]) for mode in output_modes)

    @staticmethod
    def _distributed_modes_for_operands(operands):
        distributed = []
        for operand in operands:
            if not isinstance(operand, DistributedTensor):
                continue
            for mode in operand.sharding.sharded_modes:
                if mode not in distributed:
                    distributed.append(mode)
        return tuple(distributed)

    @staticmethod
    def _reduced_distributed_modes(distributed_modes, output_modes):
        output_set = set(output_modes)
        return tuple(mode for mode in distributed_modes if mode not in output_set)

    def _derive_output_sharding(self, dist_spec, input_modes, output_modes, output_shape, *, use_requested=True):
        if use_requested and dist_spec.output_sharding is not None:
            return dist_spec.output_sharding
        sources = tuple(operand for operand in dist_spec.operands if isinstance(operand, DistributedTensor))
        if not sources:
            return None
        source = sources[0]
        mesh = source.mesh
        ranks_per_mode = {}
        mode_to_mesh_axis = {}
        selected_modes = []
        used_mesh_axes = set()
        for operand in sources:
            if operand.mesh != mesh:
                continue
            for mode in output_modes:
                if mode not in operand.sharding.sharded_modes:
                    continue
                if mode in selected_modes:
                    continue
                mesh_axis = operand.sharding.mode_to_mesh_axis.get(mode)
                if mesh_axis is not None and mesh_axis in used_mesh_axes:
                    continue
                selected_modes.append(mode)
                if mesh_axis is not None:
                    used_mesh_axes.add(mesh_axis)
                if mode in operand.sharding.ranks_per_mode:
                    ranks_per_mode[mode] = operand.sharding.ranks_per_mode[mode]
                if mesh_axis is not None:
                    mode_to_mesh_axis[mode] = mesh_axis
        sharded_modes = tuple(mode for mode in output_modes if mode in set(selected_modes))
        return ShardingSpec(
            global_shape=output_shape,
            modes=output_modes,
            mesh=mesh,
            ranks_per_mode=ranks_per_mode,
            mode_to_mesh_axis=mode_to_mesh_axis,
            sharded_modes=sharded_modes,
            replicated_modes=tuple(mode for mode in output_modes if mode not in sharded_modes),
        )

    @staticmethod
    def _sharding_mode_compatible(src, dst, mode):
        if dst is None:
            return False
        if mode not in dst.sharded_modes:
            return False
        return (
            src.mesh == dst.mesh
            and src.ranks_per_mode.get(mode) == dst.ranks_per_mode.get(mode)
            and src.mode_to_mesh_axis.get(mode) == dst.mode_to_mesh_axis.get(mode)
        )

    @staticmethod
    def _sharding_specs_equivalent(left, right):
        if left is right:
            return True
        if left is None or right is None:
            return False
        return (
            tuple(left.global_shape) == tuple(right.global_shape)
            and tuple(left.modes) == tuple(right.modes)
            and left.mesh == right.mesh
            and dict(left.ranks_per_mode) == dict(right.ranks_per_mode)
            and dict(left.mode_to_mesh_axis) == dict(right.mode_to_mesh_axis)
            and tuple(left.sharded_modes) == tuple(right.sharded_modes)
            and tuple(left.replicated_modes) == tuple(right.replicated_modes)
        )

    def _input_redistribution_items(
        self,
        operands,
        input_modes,
        output_sharding,
        reduced_distributed_modes,
    ):
        reduced = set(reduced_distributed_modes)
        items = []
        seen = set()
        for index, (operand, modes) in enumerate(zip(operands, input_modes)):
            if not isinstance(operand, DistributedTensor):
                continue
            incompatible = tuple(
                mode
                for mode in operand.sharding.sharded_modes
                if mode in modes
                and mode not in reduced
                and not self._sharding_mode_compatible(operand.sharding, output_sharding, mode)
            )
            if not incompatible:
                continue
            items.append((index, incompatible))
            for mode in incompatible:
                if mode not in seen:
                    seen.add(mode)
        return tuple(items), tuple(seen)

    def _input_redistribution_local_bytes(self, operands, items):
        total = 0
        for index, _ in items:
            operand = operands[index]
            if isinstance(operand, DistributedTensor):
                total += int(operand.local_nbytes)
            else:
                total += self._array_nbytes(operand)
        return total

    def _redistribute_incompatible_input_operands(
        self,
        operands,
        input_modes,
        output_sharding,
        reduced_distributed_modes,
    ):
        items, _ = self._input_redistribution_items(
            operands,
            input_modes,
            output_sharding,
            reduced_distributed_modes,
        )
        if not items:
            return tuple(operands)
        redistributed = list(operands)
        for index, _ in items:
            operand = redistributed[index]
            dense = self.gather_tensor(operand)
            mesh = output_sharding.mesh if output_sharding is not None else operand.mesh
            redistributed[index] = self.replicate_tensor(dense, mesh, modes=operand.modes)
        return tuple(redistributed)

    def _einsum_spec_from_distributed_spec(self, dist_spec):
        input_modes, output_modes = parse_einsum_equation(dist_spec.equation)
        operands = tuple(
            TensorOperand(operand, modes, name="operand{0}".format(index))
            for index, (operand, modes) in enumerate(zip(dist_spec.operands, input_modes))
        )
        return EinsumSpec(operands=operands, output_modes=output_modes, optimize=dist_spec.optimize)

    def plan_contraction(
        self,
        spec,
        *,
        memory_limit=None,
        prefer="balanced",
        allow_slicing=True,
        allow_distribution=False,
        target_devices=None,
        record_profile=True,
    ):
        self._validate_plan_prefer(prefer)
        target_device_specs = self._validate_plan_target_devices(
            target_devices,
            allow_distribution=allow_distribution,
        )
        if isinstance(spec, DistributedContractionSpec):
            if target_device_specs is not None:
                raise BackendFeatureError(
                    "target_devices cannot be combined with DistributedContractionSpec; "
                    "use the DistributedContractionSpec sharding instead"
                )
            input_modes, output_modes = parse_einsum_equation(spec.equation)
            sizes = self._mode_sizes_from_equation(input_modes, spec.operands)
            output_shape = self._output_shape_from_sizes(output_modes, sizes)
            itemsize = max((self._operand_itemsize(operand) for operand in spec.operands), default=0)
            comm_bytes = self._prod_shape(output_shape) * int(itemsize or 0)
            distributed_modes = self._distributed_modes_for_operands(spec.operands)
            reduced_distributed_modes = self._reduced_distributed_modes(distributed_modes, output_modes)
            einsum_spec = self._einsum_spec_from_distributed_spec(spec)
            if not allow_distribution and distributed_modes:
                raise BackendFeatureError("distributed contraction requires allow_distribution=True")
            plan = self._plan_einsum_contraction(
                einsum_spec,
                distributed_modes=distributed_modes,
                comm_bytes=comm_bytes if distributed_modes else 0,
                record_profile=record_profile,
            )
            if distributed_modes:
                output_sharding = self._derive_output_sharding(spec, input_modes, output_modes, output_shape)
                natural_output_sharding = self._derive_output_sharding(
                    spec,
                    input_modes,
                    output_modes,
                    output_shape,
                    use_requested=False,
                )
                output_redistribution_required = (
                    output_sharding is not None
                    and natural_output_sharding is not None
                    and not self._sharding_specs_equivalent(output_sharding, natural_output_sharding)
                )
                input_redistribution_items, input_redistribution_modes = self._input_redistribution_items(
                    spec.operands,
                    input_modes,
                    natural_output_sharding or output_sharding,
                    reduced_distributed_modes,
                )
                input_redistribution_bytes = sum(
                    self._prod_shape(spec.operands[index].global_shape) * self._operand_itemsize(spec.operands[index])
                    for index, _ in input_redistribution_items
                )
                input_redistribution_local_bytes = self._input_redistribution_local_bytes(
                    spec.operands,
                    input_redistribution_items,
                )
                active_distributed_modes = tuple(
                    mode for mode in distributed_modes if mode not in set(input_redistribution_modes)
                )
                reduce_scatter_required = bool(reduced_distributed_modes and output_redistribution_required)
                reduce_scatter_modes = tuple(dict.fromkeys(
                    tuple(reduced_distributed_modes) + tuple(output_sharding.sharded_modes if output_sharding is not None else ())
                ))
                output_comm_bytes = plan.estimated_comm_bytes
                output_comm_local_bytes = self._output_communication_local_bytes(
                    output_sharding,
                    itemsize,
                    output_comm_bytes,
                )
                total_comm_bytes = output_comm_bytes + input_redistribution_bytes
                states = tuple(
                    self._distribution_state_for_operand(index, operand, modes)
                    for index, (operand, modes) in enumerate(zip(spec.operands, input_modes))
                )
                local_step = self._local_step_for_activate_distribution(plan, output_sharding)
                step_plan = DistributedStepPlan(
                    local_step=local_step,
                    input_states=states,
                    output_sharding=output_sharding,
                    output_state=self._distribution_state_for_output(
                        local_step.output,
                        output_modes,
                        output_shape,
                        output_sharding,
                        itemsize,
                    ),
                    communication=tuple(
                        (
                            CommunicationPlan(
                                kind="redistribute",
                                bytes=input_redistribution_bytes,
                                local_bytes=input_redistribution_local_bytes,
                                modes=input_redistribution_modes,
                                reason="redistribute incompatible input sharding before contraction",
                            ),
                        )
                        if input_redistribution_modes
                        else ()
                    )
                    + (
                        CommunicationPlan(
                            kind=(
                                "reduce_scatter"
                                if reduce_scatter_required
                                else "allreduce"
                                if reduced_distributed_modes
                                else "alltoall"
                                if output_redistribution_required
                                else "gather"
                            ),
                            bytes=output_comm_bytes,
                            local_bytes=output_comm_local_bytes,
                            modes=reduce_scatter_modes if reduce_scatter_required else reduced_distributed_modes or active_distributed_modes,
                            reason=(
                                "sum partial outputs and scatter to requested output sharding"
                                if reduce_scatter_required
                                else "sum partial outputs across reduced sharded modes"
                                if reduced_distributed_modes
                                else "redistribute contraction output to requested sharding"
                                if output_redistribution_required
                                else "materialize distributed contraction output"
                            ),
                        ),
                    ),
                )
                distributed_plan = self._with_distributed_plan_totals(DistributedContractionPlan(
                    path=plan,
                    steps=(step_plan,),
                    output_sharding=output_sharding,
                    estimated_comm_bytes=total_comm_bytes,
                    equation=spec.equation,
                ))
                step = plan.steps[0]
                step = ContractionStep(
                    kind=step.kind,
                    inputs=step.inputs,
                    output=step.output,
                    input_modes=step.input_modes,
                    output_modes=step.output_modes,
                    plan=distributed_plan,
                    estimated_flops=step.estimated_flops,
                    estimated_read_bytes=step.estimated_read_bytes,
                    estimated_write_bytes=step.estimated_write_bytes,
                    estimated_copy_bytes=step.estimated_copy_bytes,
                    estimated_peak_bytes=step.estimated_peak_bytes,
                    estimated_comm_bytes=total_comm_bytes,
                    required_workspace_bytes=step.required_workspace_bytes,
                    reason=step.reason,
                    fallback_reason=step.fallback_reason,
                )
                wrapped_plan = ContractionPlan(
                    steps=(step,),
                    input_specs=plan.input_specs,
                    output_modes=plan.output_modes,
                    estimated_flops=plan.estimated_flops,
                    estimated_peak_bytes=plan.estimated_peak_bytes,
                    estimated_read_bytes=plan.estimated_read_bytes,
                    estimated_write_bytes=plan.estimated_write_bytes,
                    estimated_copy_bytes=plan.estimated_copy_bytes,
                    estimated_comm_bytes=total_comm_bytes,
                    required_workspace_bytes=plan.required_workspace_bytes,
                    sliced_modes=plan.sliced_modes,
                    distributed_modes=active_distributed_modes,
                )
                wrapped_plan = self._enforce_plan_memory_limit(
                    wrapped_plan,
                    memory_limit=memory_limit,
                    allow_slicing=allow_slicing,
                )
                if record_profile:
                    self._record_distributed_contraction_plan(spec, wrapped_plan)
                return wrapped_plan
            return self._enforce_plan_memory_limit(plan, memory_limit=memory_limit, allow_slicing=allow_slicing)
        plan = self._plan_einsum_contraction(spec, record_profile=record_profile)
        if target_device_specs is None and allow_distribution:
            target_device_specs = self._auto_plan_target_devices()
        if target_device_specs is not None:
            mesh = self._mesh_from_target_devices(target_device_specs)
            distributed_plan = self.plan_distributed_contraction_path(
                plan,
                mesh,
                memory_limit_per_device=memory_limit,
            )
            wrapped_plan = self._wrap_distributed_contraction_plan(plan, distributed_plan)
            if record_profile:
                self._record_distributed_contraction_plan(
                    self._distributed_spec_from_plan(distributed_plan),
                    wrapped_plan,
                )
            return wrapped_plan
        plan = self._apply_contraction_plan_preference(
            plan,
            prefer=prefer,
            memory_limit=memory_limit,
            allow_slicing=allow_slicing,
        )
        return self._enforce_plan_memory_limit(plan, memory_limit=memory_limit, allow_slicing=allow_slicing)

    @staticmethod
    def _validate_plan_prefer(prefer):
        if prefer not in ("time", "memory", "balanced"):
            raise ValueError("prefer must be one of 'time', 'memory', or 'balanced'")

    def _validate_plan_target_devices(self, target_devices, *, allow_distribution):
        if target_devices is None:
            return None
        devices = tuple(parse_device_spec(device) for device in target_devices)
        if not devices:
            raise ValueError("target_devices must not be empty")
        if not allow_distribution:
            raise BackendFeatureError("target_devices require allow_distribution=True")
        return devices

    def _auto_plan_target_devices(self):
        try:
            count = int(self.device_count())
        except Exception:
            return None
        if count <= 1:
            return None
        current = self.current_device()
        kind = current.kind
        devices = []
        for index in range(count):
            if kind in ("cuda", "gpu", "rocm", "mps", "tpu"):
                devices.append(
                    DeviceSpec(kind=kind, index=index, local_rank=index, global_rank=index)
                )
            else:
                devices.append(
                    DeviceSpec(kind=kind, local_rank=index, global_rank=index)
                )
        return tuple(devices)

    def _mesh_from_target_devices(self, devices):
        devices = tuple(devices)
        return DeviceMesh(
            devices=devices,
            shape=(len(devices),),
            axis_names=("rank",),
            backend=self.name,
            local_rank=0,
            global_rank=0,
        )

    @staticmethod
    def _wrap_distributed_contraction_plan(path, distributed_plan):
        step = path.steps[0]
        output_sharding = distributed_plan.output_sharding
        distributed_modes = tuple(output_sharding.sharded_modes) if output_sharding is not None else ()
        comm_bytes = int(distributed_plan.total_comm_bytes or distributed_plan.estimated_comm_bytes)
        wrapped_step = ContractionStep(
            kind=step.kind,
            inputs=step.inputs,
            output=step.output,
            input_modes=step.input_modes,
            output_modes=step.output_modes,
            plan=distributed_plan,
            estimated_flops=step.estimated_flops,
            estimated_read_bytes=step.estimated_read_bytes,
            estimated_write_bytes=step.estimated_write_bytes,
            estimated_copy_bytes=step.estimated_copy_bytes,
            estimated_peak_bytes=distributed_plan.peak_local_bytes or step.estimated_peak_bytes,
            estimated_comm_bytes=comm_bytes,
            required_workspace_bytes=step.required_workspace_bytes,
            reason=step.reason,
            fallback_reason=step.fallback_reason,
        )
        return ContractionPlan(
            steps=(wrapped_step,),
            input_specs=path.input_specs,
            output_modes=path.output_modes,
            estimated_flops=path.estimated_flops,
            estimated_peak_bytes=distributed_plan.peak_local_bytes or path.estimated_peak_bytes,
            estimated_read_bytes=path.estimated_read_bytes,
            estimated_write_bytes=path.estimated_write_bytes,
            estimated_copy_bytes=path.estimated_copy_bytes,
            estimated_comm_bytes=comm_bytes,
            required_workspace_bytes=path.required_workspace_bytes,
            sliced_modes=path.sliced_modes,
            distributed_modes=distributed_modes,
        )

    def _slice_shape_nbytes(self, shape, slices, itemsize):
        local_shape = self._local_shape_for_slices(tuple(int(dim) for dim in shape), tuple(slices))
        return self._prod_shape(local_shape) * int(itemsize or 0)

    def _sliced_read_bytes(self, plan, operand_slices):
        total = 0
        for slices_for_step in operand_slices:
            for operand, slices in zip(plan.input_specs, slices_for_step):
                total += self._slice_shape_nbytes(
                    self._operand_global_shape(operand.array),
                    slices,
                    self._operand_itemsize(operand.array),
                )
        return int(total)

    def _build_output_slicing_plan(self, plan, *, memory_limit, peak):
        if len(plan.steps) != 1 or len(plan.input_specs) != 2:
            raise BackendFeatureError(
                "slicing planner currently supports single-step two-operand contraction plans"
            )
        if not isinstance(plan.steps[0].plan, MatmulPlan):
            raise BackendFeatureError(
                "slicing planner currently supports dense MatmulPlan contractions"
            )
        output_shape = self._output_shape_for_contraction_plan(plan)
        output_size = self._prod_shape(output_shape)
        if output_size <= 0 or peak <= 0:
            raise BackendFeatureError("cannot slice empty contraction output")
        if memory_limit <= 0:
            raise BackendFeatureError(
                "memory_limit {0} bytes cannot hold one output slice from peak {1} bytes"
                .format(memory_limit, peak)
            )
        candidates = sorted(
            tuple(zip(plan.output_modes, output_shape)),
            key=lambda item: (int(item[1]) >= 2, int(item[1])),
            reverse=True,
        )
        for sliced_mode, mode_dim in candidates:
            mode_dim = int(mode_dim)
            if mode_dim <= 1:
                continue
            chunk = max(1, int(memory_limit) * mode_dim // int(peak))
            chunk = min(mode_dim, chunk)
            while chunk > 1 and ((int(peak) * chunk + mode_dim - 1) // mode_dim) > memory_limit:
                chunk -= 1
            if ((int(peak) * chunk + mode_dim - 1) // mode_dim) > memory_limit:
                continue
            output_axis = tuple(plan.output_modes).index(sliced_mode)
            output_slices = []
            operand_slices = []
            max_slice_peak = 0
            for start in range(0, mode_dim, chunk):
                stop = min(mode_dim, start + chunk)
                output_slice = [slice(None)] * len(output_shape)
                output_slice[output_axis] = slice(start, stop)
                output_slice = tuple(output_slice)
                output_slices.append(output_slice)
                slice_len = stop - start
                max_slice_peak = max(
                    max_slice_peak,
                    (int(peak) * slice_len + mode_dim - 1) // mode_dim,
                )
                per_operand = []
                for operand in plan.input_specs:
                    slices = [slice(None)] * len(operand.modes)
                    if sliced_mode in operand.modes:
                        slices[tuple(operand.modes).index(sliced_mode)] = slice(start, stop)
                    per_operand.append(tuple(slices))
                operand_slices.append(tuple(per_operand))
            estimated_peak = max(int(max_slice_peak), int(plan.required_workspace_bytes or 0))
            if estimated_peak > memory_limit:
                continue
            sliced_plan = SlicedContractionPlan(
                base_plan=plan,
                sliced_mode=sliced_mode,
                output_axis=output_axis,
                output_slices=tuple(output_slices),
                operand_slices=tuple(operand_slices),
            )
            base_step = plan.steps[0]
            step = ContractionStep(
                kind="slice",
                inputs=base_step.inputs,
                output=base_step.output,
                input_modes=base_step.input_modes,
                output_modes=base_step.output_modes,
                plan=sliced_plan,
                estimated_flops=plan.estimated_flops,
                estimated_read_bytes=self._sliced_read_bytes(plan, tuple(operand_slices)),
                estimated_write_bytes=plan.estimated_write_bytes,
                estimated_copy_bytes=plan.estimated_copy_bytes,
                estimated_peak_bytes=estimated_peak,
                estimated_comm_bytes=plan.estimated_comm_bytes,
                required_workspace_bytes=plan.required_workspace_bytes,
                reason="slice output mode {0!r} to satisfy memory_limit {1}".format(sliced_mode, memory_limit),
                fallback_reason=base_step.fallback_reason,
            )
            sliced_contraction_plan = ContractionPlan(
                steps=(step,),
                input_specs=plan.input_specs,
                output_modes=plan.output_modes,
                estimated_flops=plan.estimated_flops,
                estimated_peak_bytes=estimated_peak,
                estimated_read_bytes=step.estimated_read_bytes,
                estimated_write_bytes=plan.estimated_write_bytes,
                estimated_copy_bytes=plan.estimated_copy_bytes,
                estimated_comm_bytes=plan.estimated_comm_bytes,
                required_workspace_bytes=plan.required_workspace_bytes,
                sliced_modes=(sliced_mode,),
                distributed_modes=plan.distributed_modes,
            )
            self._record_sliced_contraction_plan(sliced_contraction_plan)
            return sliced_contraction_plan
        raise BackendFeatureError(
            "slicing planner could not satisfy memory_limit {0}; plan peak is {1} bytes"
            .format(memory_limit, peak)
        )

    def _record_sliced_contraction_plan(self, plan):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            step = plan.steps[0] if plan.steps else None
            sliced_plan = step.plan if step is not None else None
            if not isinstance(sliced_plan, SlicedContractionPlan):
                return
            base_plan = sliced_plan.base_plan
            base_step = base_plan.steps[0] if base_plan.steps else None
            base_lowering = getattr(getattr(base_step, "plan", None), "kind", getattr(base_step, "kind", None))
            equation, input_modes, output_modes = self._contraction_plan_profile_metadata(plan)
            output_shape = self._output_shape_for_contraction_plan(plan)
            slice_output_shapes = [
                self._local_shape_for_slices(tuple(output_shape), output_slice)
                for output_slice in sliced_plan.output_slices
            ]
            largest_intermediate_elements = max(
                (self._prod_shape(shape) for shape in slice_output_shapes),
                default=0,
            )
            itemsize = max(
                (self._operand_itemsize(operand.array) for operand in plan.input_specs),
                default=0,
            )
            largest_intermediate_bytes = int(largest_intermediate_elements) * int(itemsize)

            profiling.record(
                "contraction_plan",
                backend=self.name,
                equation=equation,
                lowering="slice",
                dtype=self._contraction_plan_output_dtype(base_plan),
                plan_hash=plan.plan_hash,
                input_modes=input_modes,
                output_modes=output_modes,
                input_shapes=[
                    tuple(getattr(operand.array, "shape", ()))
                    for operand in plan.input_specs
                ],
                operands=[self._profile_tensor_operand(operand) for operand in plan.input_specs],
                input_dtypes=[
                    str(getattr(operand.array, "dtype", None))
                    for operand in plan.input_specs
                ],
                output_shape=output_shape,
                **self._profile_device_execution(self.current_device()),
                flops=plan.estimated_flops,
                read_bytes=plan.estimated_read_bytes,
                write_bytes=plan.estimated_write_bytes,
                copy_bytes=plan.estimated_copy_bytes,
                workspace_bytes=plan.required_workspace_bytes,
                peak_bytes=plan.estimated_peak_bytes,
                largest_intermediate=largest_intermediate_bytes,
                largest_intermediate_elements=largest_intermediate_elements,
                largest_intermediate_bytes=largest_intermediate_bytes,
                num_gemm=len(sliced_plan.output_slices) if base_lowering == "gemm" else 0,
                num_batched_gemm=len(sliced_plan.output_slices) if base_lowering in ("batched_gemm", "strided_batched_gemm") else 0,
                num_grouped_tasks=0,
                num_blocks=0,
                num_shape_buckets=0,
                fallback_reason=getattr(step, "fallback_reason", None),
                sliced_modes=[str(mode) for mode in plan.sliced_modes],
                num_slices=len(sliced_plan.output_slices),
                base_lowering=base_lowering,
                slice_output_axis=sliced_plan.output_axis,
                slice_output_shapes=slice_output_shapes,
            )
        except Exception:
            pass

    def _enforce_plan_memory_limit(self, plan, *, memory_limit=None, allow_slicing=True):
        if memory_limit is None:
            return plan
        limit = int(memory_limit)
        if limit < 0:
            raise ValueError("memory_limit must be non-negative")
        peak = int(getattr(plan, "estimated_peak_bytes", 0) or 0)
        if peak <= limit:
            return plan
        if allow_slicing:
            return self._build_output_slicing_plan(plan, memory_limit=limit, peak=peak)
        raise BackendFeatureError(
            "memory_limit {0} bytes is below contraction plan peak {1} bytes"
            .format(limit, peak)
        )

    def _memory_preferred_slicing_limit(self, plan, peak):
        output_shape = self._output_shape_for_contraction_plan(plan)
        candidates = [
            int(dim)
            for dim in output_shape
            if int(dim) > 1
        ]
        if not candidates:
            return None
        largest_dim = max(candidates)
        return max(1, (int(peak) + largest_dim - 1) // largest_dim)

    def _apply_contraction_plan_preference(self, plan, *, prefer, memory_limit=None, allow_slicing=True):
        if prefer != "memory" or memory_limit is not None or not allow_slicing:
            return plan
        peak = int(getattr(plan, "estimated_peak_bytes", 0) or 0)
        if peak <= 0:
            return plan
        limit = self._memory_preferred_slicing_limit(plan, peak)
        if limit is None or limit >= peak:
            return plan
        try:
            candidate = self._build_output_slicing_plan(plan, memory_limit=limit, peak=peak)
        except BackendFeatureError:
            return plan
        if int(candidate.estimated_peak_bytes or 0) < peak:
            return candidate
        return plan

    @staticmethod
    def _rate_seconds(amount, rate):
        if not amount:
            return 0.0
        if not rate:
            return float("inf")
        return float(amount) / float(rate)

    @staticmethod
    def _matmul_desc_layout_copy_bytes(desc):
        total = 0
        for layout in (getattr(desc, "layout_a", None), getattr(desc, "layout_b", None), getattr(desc, "layout_c", None)):
            total += int(getattr(layout, "estimated_copy_bytes", 0) or 0)
        return int(total)

    @staticmethod
    def _hardware_flop_rate(hw):
        return hw.flop_per_s or hw.device_flop_s

    @staticmethod
    def _hardware_memory_bandwidth(hw):
        return hw.memory_bandwidth_Bps or hw.device_bandwidth_bytes_s or hw.host_bandwidth_bytes_s

    @staticmethod
    def _hardware_copy_bandwidth(hw):
        return (
            hw.p2p_bandwidth_Bps
            or hw.device_bandwidth_bytes_s
            or hw.h2d_bandwidth_Bps
            or hw.d2h_bandwidth_Bps
            or hw.host_bandwidth_bytes_s
        )

    @staticmethod
    def _hardware_comm_bandwidth(hw):
        return hw.network_bandwidth_Bps or hw.interconnect_bandwidth_bytes_s or hw.p2p_bandwidth_Bps

    @staticmethod
    def _hardware_point_to_point_bandwidth(hw):
        return hw.p2p_bandwidth_Bps or hw.interconnect_bandwidth_bytes_s or hw.network_bandwidth_Bps

    @classmethod
    def _hardware_bandwidth_for_communication(cls, item, hw):
        if getattr(item, "kind", None) == "point_to_point":
            return cls._hardware_point_to_point_bandwidth(hw)
        return cls._hardware_comm_bandwidth(hw)

    @staticmethod
    def _validate_cost_model_workspace_limit(hw, workspace_bytes):
        if hw is None or getattr(hw, "workspace_limit_bytes", None) is None:
            return
        limit = int(hw.workspace_limit_bytes)
        workspace_bytes = int(workspace_bytes or 0)
        if workspace_bytes > limit:
            raise BackendFeatureError(
                "workspace_limit {0} bytes is below plan workspace; requires {1} bytes"
                .format(limit, workspace_bytes)
            )

    @staticmethod
    def _validate_cost_model_peak_memory_limit(hw, peak_bytes):
        if hw is None or getattr(hw, "max_memory_bytes", None) is None:
            return
        limit = int(hw.max_memory_bytes)
        peak_bytes = int(peak_bytes or 0)
        if peak_bytes > limit:
            raise BackendFeatureError(
                "max_memory {0} bytes is below plan peak; requires {1} bytes"
                .format(limit, peak_bytes)
            )

    def _make_cost_estimate(
        self,
        hw,
        *,
        flops=0,
        read_bytes=0,
        write_bytes=0,
        copy_bytes=0,
        comm_bytes=0,
        workspace_bytes=0,
        peak_bytes=0,
    ):
        hw = HardwareModel() if hw is None else hw
        self._validate_cost_model_workspace_limit(hw, workspace_bytes)
        self._validate_cost_model_peak_memory_limit(hw, peak_bytes)
        compute_s = self._rate_seconds(flops, self._hardware_flop_rate(hw))
        memory_s = self._rate_seconds(read_bytes + write_bytes, self._hardware_memory_bandwidth(hw))
        copy_s = self._rate_seconds(copy_bytes, self._hardware_copy_bandwidth(hw))
        comm_s = self._rate_seconds(comm_bytes, self._hardware_comm_bandwidth(hw))
        if comm_bytes and hw.latency_s:
            comm_s += float(hw.latency_s)
        total_s = compute_s + memory_s + copy_s + comm_s
        return CostEstimate(
            flops=int(flops),
            read_bytes=int(read_bytes),
            write_bytes=int(write_bytes),
            copy_bytes=int(copy_bytes),
            comm_bytes=int(comm_bytes),
            workspace_bytes=int(workspace_bytes),
            peak_bytes=int(peak_bytes),
            compute_s=compute_s,
            memory_s=memory_s,
            copy_s=copy_s,
            comm_s=comm_s,
            total_s=total_s,
        )

    def estimate_matmul(self, desc, hw=None):
        if isinstance(desc, MatmulPlan):
            write_bytes = sum(item.estimated_write_bytes for item in desc.descs)
            copy_bytes = int(desc.copy_bytes)
            workspace_bytes = int(desc.workspace_bytes)
            return self._make_cost_estimate(
                hw,
                flops=desc.estimated_flops,
                read_bytes=sum(item.estimated_read_bytes for item in desc.descs),
                write_bytes=write_bytes,
                copy_bytes=copy_bytes,
                workspace_bytes=workspace_bytes,
                peak_bytes=write_bytes + copy_bytes + workspace_bytes,
            )
        batch = self._prod_shape(desc.batch_shape)
        flops = desc.estimated_flops or int(2 * batch * desc.m * desc.n * desc.k)
        read_bytes = desc.estimated_read_bytes or (self._array_nbytes(desc.A) + self._array_nbytes(desc.B))
        itemsize = max(self._operand_itemsize(desc.A), self._operand_itemsize(desc.B), 1)
        write_bytes = desc.estimated_write_bytes or int(batch * desc.m * desc.n * itemsize)
        copy_bytes = self._matmul_desc_layout_copy_bytes(desc)
        workspace_bytes = int(desc.estimated_workspace_bytes)
        return self._make_cost_estimate(
            hw,
            flops=flops,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            copy_bytes=copy_bytes,
            workspace_bytes=workspace_bytes,
            peak_bytes=write_bytes + copy_bytes + workspace_bytes,
        )

    def _estimate_distributed_contraction(self, plan, hw=None):
        hw = HardwareModel() if hw is None else hw
        local_steps = tuple(step.local_step for step in plan.steps)
        read_bytes = sum(int(step.estimated_read_bytes) for step in local_steps)
        write_bytes = sum(int(step.estimated_write_bytes) for step in local_steps)
        copy_bytes = sum(int(step.estimated_copy_bytes) for step in local_steps)
        workspace_bytes = max((int(step.required_workspace_bytes) for step in local_steps), default=0)
        self._validate_cost_model_workspace_limit(hw, workspace_bytes)
        self._validate_cost_model_peak_memory_limit(hw, plan.peak_local_bytes)
        local_flops = sum(int(step.estimated_flops) for step in local_steps)
        communication = tuple(item for step in plan.steps for item in step.communication)

        compute_s = self._rate_seconds(local_flops, self._hardware_flop_rate(hw))
        memory_s = self._rate_seconds(read_bytes + write_bytes, self._hardware_memory_bandwidth(hw))
        copy_s = self._rate_seconds(copy_bytes, self._hardware_copy_bandwidth(hw))
        comm_s = self._estimate_communication_sequence_s(communication, hw)
        total_s = compute_s + memory_s + copy_s + comm_s
        return CostEstimate(
            flops=int(plan.total_flops or plan.path.estimated_flops),
            read_bytes=int(read_bytes),
            write_bytes=int(write_bytes),
            copy_bytes=int(copy_bytes),
            comm_bytes=int(plan.total_comm_bytes or plan.estimated_comm_bytes),
            workspace_bytes=int(workspace_bytes),
            peak_bytes=int(plan.peak_local_bytes),
            compute_s=compute_s,
            memory_s=memory_s,
            copy_s=copy_s,
            comm_s=comm_s,
            total_s=total_s,
        )

    def estimate_contraction(self, plan, hw=None):
        if isinstance(plan, MatmulPlan):
            return self.estimate_matmul(plan, hw)
        if isinstance(plan, GroupedGemmPlan):
            workspace_bytes = int(plan.estimated_workspace_bytes or 0)
            copy_bytes = int(plan.estimated_copy_bytes or 0)
            return self._make_cost_estimate(
                hw,
                flops=plan.estimated_flops,
                read_bytes=plan.estimated_read_bytes,
                write_bytes=plan.estimated_write_bytes,
                copy_bytes=copy_bytes,
                workspace_bytes=workspace_bytes,
                peak_bytes=int(plan.estimated_write_bytes or 0) + copy_bytes + workspace_bytes,
            )
        if isinstance(plan, DistributedContractionPlan):
            return self._estimate_distributed_contraction(plan, hw)
        distributed_plan = self._distributed_plan_from_contraction_plan(plan)
        if distributed_plan is not None:
            return self._estimate_distributed_contraction(distributed_plan, hw)
        peak_bytes = int(plan.estimated_peak_bytes or max((step.estimated_peak_bytes for step in plan.steps), default=0))
        return self._make_cost_estimate(
            hw,
            flops=plan.estimated_flops,
            read_bytes=plan.estimated_read_bytes,
            write_bytes=plan.estimated_write_bytes,
            copy_bytes=plan.estimated_copy_bytes,
            comm_bytes=plan.estimated_comm_bytes,
            workspace_bytes=plan.required_workspace_bytes,
            peak_bytes=peak_bytes,
        )

    def estimate_redistribute(self, src, dst, tensor_shape, hw=None, *, itemsize=8):
        hw = HardwareModel() if hw is None else hw
        tensor_shape = tuple(int(dim) for dim in tensor_shape)
        if any(dim < 0 for dim in tensor_shape):
            raise ValueError("estimate_redistribute tensor_shape dimensions must be non-negative")
        itemsize = int(itemsize)
        if itemsize <= 0:
            raise ValueError("estimate_redistribute itemsize must be positive")
        if isinstance(src, ShardingSpec):
            if tuple(src.global_shape) != tensor_shape:
                raise ValueError("estimate_redistribute tensor_shape must match source global_shape")
        if isinstance(dst, ShardingSpec):
            if tuple(dst.global_shape) != tensor_shape:
                raise ValueError("estimate_redistribute tensor_shape must match destination global_shape")
        if isinstance(src, ShardingSpec) and isinstance(dst, ShardingSpec):
            if tuple(src.modes) != tuple(dst.modes):
                raise ValueError("estimate_redistribute source and destination modes must match")
        nbytes = self._prod_shape(tensor_shape) * itemsize
        movement_bytes = 0 if self._sharding_specs_equivalent(src, dst) else nbytes
        src_local_bytes = (
            self._local_nbytes_for_sharding(src, itemsize)
            if isinstance(src, ShardingSpec)
            else nbytes
        )
        dst_local_bytes = (
            self._local_nbytes_for_sharding(dst, itemsize)
            if isinstance(dst, ShardingSpec)
            else nbytes
        )
        local_movement_bytes = 0 if movement_bytes == 0 else max(
            int(src_local_bytes or nbytes),
            int(dst_local_bytes or nbytes),
        )
        self._validate_cost_model_peak_memory_limit(hw, local_movement_bytes)
        copy_s = self._rate_seconds(local_movement_bytes, self._hardware_copy_bandwidth(hw))
        comm_s = self._rate_seconds(local_movement_bytes, self._hardware_comm_bandwidth(hw))
        if local_movement_bytes and hw.latency_s:
            comm_s += float(hw.latency_s)
        total_s = copy_s + comm_s
        return CostEstimate(
            copy_bytes=int(movement_bytes),
            comm_bytes=int(movement_bytes),
            peak_bytes=int(local_movement_bytes),
            copy_s=copy_s,
            comm_s=comm_s,
            total_s=total_s,
        )

    def _local_nbytes_for_sharding(self, sharding, itemsize):
        if sharding is None:
            return 0
        local_slices = sharding.local_slices.get(sharding.mesh.local_rank)
        if local_slices is None:
            return 0
        local_shape = self._local_shape_for_slices(sharding.global_shape, local_slices)
        return self._prod_shape(local_shape) * int(itemsize)

    def _output_communication_local_bytes(self, output_sharding, itemsize, total_bytes):
        if output_sharding is None:
            return int(total_bytes)
        local_bytes = self._local_nbytes_for_sharding(output_sharding, itemsize)
        return int(local_bytes or total_bytes)

    @staticmethod
    def _local_shape_for_slices(global_shape, local_slices):
        local_shape = []
        for local_slice, dim in zip(local_slices, global_shape):
            start, stop, step = local_slice.indices(dim)
            if step != 1:
                local_shape.append(len(range(start, stop, step)))
            else:
                local_shape.append(max(0, stop - start))
        return tuple(local_shape)

    def _local_shape_for_sharding(self, sharding):
        local_slices = sharding.local_slices.get(sharding.mesh.local_rank)
        if local_slices is None:
            return ()
        return self._local_shape_for_slices(sharding.global_shape, local_slices)

    def _local_shape_for_activate_distribution_operand(self, operand, output_sharding):
        if isinstance(operand.array, DistributedTensor):
            return tuple(operand.array.local_shape)
        sharding = self._placement_sharding_for_operand(operand, output_sharding)
        if sharding is None:
            return tuple(int(dim) for dim in getattr(operand.array, "shape", ()))
        return self._local_shape_for_sharding(sharding)

    def _local_step_for_activate_distribution(self, path, output_sharding):
        step = path.steps[0]
        if len(path.input_specs) != 2:
            return step
        local_input_shapes = tuple(
            self._local_shape_for_activate_distribution_operand(operand, output_sharding)
            for operand in path.input_specs
        )
        if not all(local_input_shapes):
            return step
        size_map = {}
        for operand, local_shape in zip(path.input_specs, local_input_shapes):
            for mode, dim in zip(operand.modes, local_shape):
                dim = int(dim)
                if mode in size_map and size_map[mode] != dim:
                    return step
                size_map[mode] = dim
        output_shape = self._local_shape_for_sharding(output_sharding)
        if not output_shape:
            output_shape = tuple(size_map[mode] for mode in path.output_modes)
        input_mode_sets = tuple(set(operand.modes) for operand in path.input_specs)
        output_set = set(path.output_modes)
        batch_modes = tuple(mode for mode in path.output_modes if mode in input_mode_sets[0] and mode in input_mode_sets[1])
        contracted_modes = tuple(mode for mode in path.input_specs[0].modes if mode in input_mode_sets[1] and mode not in output_set)
        left_only_modes = tuple(mode for mode in path.output_modes if mode in input_mode_sets[0] and mode not in input_mode_sets[1])
        right_only_modes = tuple(mode for mode in path.output_modes if mode in input_mode_sets[1] and mode not in input_mode_sets[0])
        batch_shape = tuple(size_map[mode] for mode in batch_modes)
        m = self._prod_shape(tuple(size_map[mode] for mode in left_only_modes))
        n = self._prod_shape(tuple(size_map[mode] for mode in right_only_modes))
        k = self._prod_shape(tuple(size_map[mode] for mode in contracted_modes))
        itemsize = max((self._operand_itemsize(operand.array) for operand in path.input_specs), default=0)
        read_bytes = sum(
            self._prod_shape(local_shape) * self._operand_itemsize(operand.array)
            for operand, local_shape in zip(path.input_specs, local_input_shapes)
        )
        write_bytes = self._prod_shape(output_shape) * int(itemsize or 0)
        flops = int(2 * self._prod_shape(batch_shape) * m * n * k)
        local_plan = step.plan
        if isinstance(local_plan, MatmulPlan) and len(local_plan.descs) == 1:
            desc = local_plan.descs[0]
            local_left = self._shape_only_operand(
                local_input_shapes[0],
                path.input_specs[0].modes,
                path.input_specs[0],
                path.input_specs[1],
                name="local_operand0",
            )
            local_right = self._shape_only_operand(
                local_input_shapes[1],
                path.input_specs[1].modes,
                path.input_specs[0],
                path.input_specs[1],
                name="local_operand1",
            )
            local_desc = MatmulDesc(
                A=local_left.array,
                B=local_right.array,
                C=None,
                m=m,
                n=n,
                k=k,
                batch_shape=batch_shape,
                trans_a=desc.trans_a,
                trans_b=desc.trans_b,
                conj_a=desc.conj_a,
                conj_b=desc.conj_b,
                alpha=desc.alpha,
                beta=desc.beta,
                dtype_compute=desc.dtype_compute,
                dtype_output=desc.dtype_output,
                layout_a=local_left.layout,
                layout_b=local_right.layout,
                layout_c=layout_from_modes_shape(path.output_modes, output_shape),
                estimated_flops=flops,
                estimated_read_bytes=read_bytes,
                estimated_write_bytes=write_bytes,
            )
            local_copy_bytes = sum(
                int(getattr(layout, "estimated_copy_bytes", 0) or 0)
                for layout in (local_desc.layout_a, local_desc.layout_b, local_desc.layout_c)
            )
            local_desc = replace(local_desc, estimated_workspace_bytes=local_copy_bytes)
            # The distributed activation step rewrites the contraction around
            # already-materialized local shards.  The source plan's transforms
            # describe the global/distributed operands and may have incompatible
            # shapes, so the local plan must carry only local layout transforms.
            local_plan = MatmulPlan(
                kind=local_plan.kind,
                descs=(local_desc,),
                pre_ops=(),
                post_ops=(),
                output_shape=output_shape,
                copy_bytes=local_copy_bytes,
                workspace_bytes=local_copy_bytes,
                estimated_flops=flops,
                estimated_time_s=local_plan.estimated_time_s,
                reason=local_plan.reason,
                fallback_reason=local_plan.fallback_reason,
            )
        return ContractionStep(
            kind=step.kind,
            inputs=step.inputs,
            output=step.output,
            input_modes=step.input_modes,
            output_modes=step.output_modes,
            plan=local_plan,
            estimated_flops=flops,
            estimated_read_bytes=read_bytes,
            estimated_write_bytes=write_bytes,
            estimated_copy_bytes=getattr(local_plan, "copy_bytes", step.estimated_copy_bytes),
            estimated_peak_bytes=max(write_bytes, getattr(local_plan, "workspace_bytes", step.required_workspace_bytes)),
            estimated_comm_bytes=step.estimated_comm_bytes,
            required_workspace_bytes=getattr(local_plan, "workspace_bytes", step.required_workspace_bytes),
            reason=step.reason,
            fallback_reason=step.fallback_reason,
        )

    def _peak_local_bytes_for_distributed_plan(self, plan):
        peaks = []
        for step in plan.steps:
            peaks.extend(int(state.local_nbytes) for state in step.input_states)
            if step.output_state is not None:
                peaks.append(int(step.output_state.local_nbytes))
            peaks.append(int(getattr(step.local_step, "estimated_peak_bytes", 0) or 0))
            peaks.append(int(getattr(step.local_step, "required_workspace_bytes", 0) or 0))
            peaks.extend(int(getattr(item, "local_bytes", 0) or 0) for item in step.communication)
        return max(peaks, default=0)

    @staticmethod
    def _communication_totals(steps):
        total_comm_bytes = 0
        total_redistribute_bytes = 0
        total_allreduce_bytes = 0
        total_gather_bytes = 0
        total_point_to_point_bytes = 0
        total_broadcast_bytes = 0
        total_reduce_scatter_bytes = 0
        total_alltoall_bytes = 0
        total_allgather_bytes = 0
        for step in steps:
            for communication in step.communication:
                nbytes = int(communication.bytes)
                total_comm_bytes += nbytes
                if communication.kind == "broadcast":
                    total_broadcast_bytes += nbytes
                elif communication.kind == "reduce_scatter":
                    total_reduce_scatter_bytes += nbytes
                elif communication.kind == "alltoall":
                    total_alltoall_bytes += nbytes
                elif communication.kind == "allgather":
                    total_allgather_bytes += nbytes
                if communication.kind in ("redistribute", "alltoall", "activate_distribution"):
                    total_redistribute_bytes += nbytes
                elif communication.kind in ("allreduce", "reduce_scatter"):
                    total_allreduce_bytes += nbytes
                elif communication.kind in ("gather", "allgather"):
                    total_gather_bytes += nbytes
                elif communication.kind == "point_to_point":
                    total_point_to_point_bytes += nbytes
        return (
            total_comm_bytes,
            total_redistribute_bytes,
            total_allreduce_bytes,
            total_gather_bytes,
            total_point_to_point_bytes,
            total_broadcast_bytes,
            total_reduce_scatter_bytes,
            total_alltoall_bytes,
            total_allgather_bytes,
        )

    def _with_distributed_plan_totals(self, plan):
        (
            total_comm_bytes,
            total_redistribute_bytes,
            total_allreduce_bytes,
            total_gather_bytes,
            total_point_to_point_bytes,
            total_broadcast_bytes,
            total_reduce_scatter_bytes,
            total_alltoall_bytes,
            total_allgather_bytes,
        ) = (
            self._communication_totals(plan.steps)
        )
        return DistributedContractionPlan(
            path=plan.path,
            steps=plan.steps,
            output_sharding=plan.output_sharding,
            estimated_comm_bytes=total_comm_bytes,
            equation=plan.equation,
            peak_local_bytes=self._peak_local_bytes_for_distributed_plan(plan),
            total_flops=int(plan.path.estimated_flops),
            total_comm_bytes=total_comm_bytes,
            total_redistribute_bytes=total_redistribute_bytes,
            total_allreduce_bytes=total_allreduce_bytes,
            total_gather_bytes=total_gather_bytes,
            total_point_to_point_bytes=total_point_to_point_bytes,
            total_broadcast_bytes=total_broadcast_bytes,
            total_reduce_scatter_bytes=total_reduce_scatter_bytes,
            total_alltoall_bytes=total_alltoall_bytes,
            total_allgather_bytes=total_allgather_bytes,
        )

    def _estimate_communication_sequence_s(self, communication, hw):
        hw = HardwareModel() if hw is None else hw
        total = 0.0
        for item in communication:
            bandwidth = self._hardware_bandwidth_for_communication(item, hw)
            nbytes = int(getattr(item, "local_bytes", item.bytes))
            total += self._rate_seconds(nbytes, bandwidth)
            if nbytes and hw.latency_s:
                total += int(item.num_messages) * float(hw.latency_s)
        return total

    def _with_distributed_step_timing(self, step, hw):
        hw = HardwareModel() if hw is None else hw
        compute_s = self._rate_seconds(
            int(step.local_step.estimated_flops),
            self._hardware_flop_rate(hw),
        )
        comm_s = self._estimate_communication_sequence_s(step.communication, hw)
        return DistributedStepPlan(
            local_step=step.local_step,
            input_states=step.input_states,
            output_sharding=step.output_sharding,
            output_state=step.output_state,
            communication=step.communication,
            estimated_compute_s=compute_s,
            estimated_comm_s=comm_s,
            estimated_total_s=compute_s + comm_s,
            kind=step.kind,
        )

    def _output_shape_for_contraction_plan(self, path):
        input_modes = tuple(operand.modes for operand in path.input_specs)
        operands = tuple(operand.array for operand in path.input_specs)
        sizes = self._mode_sizes_from_equation(input_modes, operands)
        return self._output_shape_from_sizes(path.output_modes, sizes)

    def _auto_output_sharding_for_path(self, path, mesh):
        output_shape = self._output_shape_for_contraction_plan(path)
        if not path.output_modes:
            raise BackendFeatureError("cannot distribute scalar contraction output automatically")
        candidates = sorted(
            list(zip(path.output_modes, output_shape)),
            key=lambda item: (int(item[1]) >= 2, int(item[1])),
            reverse=True,
        )
        shardable_candidates = [
            item for item in candidates
            if int(item[1]) > 1
        ]
        assignments = {}
        used_modes = set()
        mesh_axes = sorted(
            zip(mesh.axis_names, mesh.shape),
            key=lambda item: int(item[1]),
            reverse=True,
        )
        for axis_name, axis_size in mesh_axes:
            if int(axis_size) <= 1:
                continue
            axis_size = int(axis_size)
            candidate = next(
                (
                    item for item in shardable_candidates
                    if item[0] not in used_modes and int(item[1]) >= axis_size
                ),
                None,
            )
            if candidate is None:
                continue
            mode, _ = candidate
            used_modes.add(mode)
            assignments[mode] = (axis_name, axis_size)
        if not assignments:
            raise BackendFeatureError("cannot distribute contraction automatically: no shardable output modes")
        sharded_modes = tuple(mode for mode in path.output_modes if mode in assignments)
        return ShardingSpec(
            global_shape=output_shape,
            modes=path.output_modes,
            mesh=mesh,
            ranks_per_mode={mode: parts for mode, (_, parts) in assignments.items()},
            mode_to_mesh_axis={mode: axis_name for mode, (axis_name, _) in assignments.items()},
            sharded_modes=sharded_modes,
            replicated_modes=tuple(mode for mode in path.output_modes if mode not in sharded_modes),
        )

    def _auto_distributed_plan_from_dense_path(self, path, mesh):
        if not isinstance(path, ContractionPlan) or not path.steps:
            raise BackendFeatureError("automatic distribution requires a non-empty ContractionPlan")
        output_sharding = self._auto_output_sharding_for_path(path, mesh)
        activation_bytes = int(path.estimated_read_bytes or sum(
            self._array_nbytes(operand.array)
            for operand in path.input_specs
        ))
        states = tuple(
            self._distribution_state_for_operand(index, operand.array, operand.modes)
            for index, operand in enumerate(path.input_specs)
        )
        local_step = self._local_step_for_activate_distribution(path, output_sharding)
        activation_local_bytes = int(
            local_step.estimated_read_bytes or activation_bytes
        )
        output_shape = self._output_shape_for_contraction_plan(path)
        itemsize = max((self._operand_itemsize(operand.array) for operand in path.input_specs), default=0)
        step_plan = DistributedStepPlan(
            local_step=local_step,
            input_states=states,
            output_sharding=output_sharding,
            output_state=self._distribution_state_for_output(
                local_step.output,
                path.output_modes,
                output_shape,
                output_sharding,
                itemsize,
            ),
            communication=(
                CommunicationPlan(
                    kind="activate_distribution",
                    bytes=activation_bytes,
                    local_bytes=activation_local_bytes,
                    modes=output_sharding.sharded_modes,
                    reason="activate automatic output-mode distribution for dense contraction plan",
                ),
            ),
            kind="activate_distribution",
        )
        return self._with_distributed_plan_totals(DistributedContractionPlan(
            path=path,
            steps=(step_plan,),
            output_sharding=output_sharding,
            estimated_comm_bytes=activation_bytes,
            equation=self._einsum_equation_from_plan(path),
        ))

    @staticmethod
    def _effective_memory_limit_per_device(memory_limit_per_device, cost_model):
        if memory_limit_per_device is not None:
            return int(memory_limit_per_device)
        if cost_model is not None and getattr(cost_model, "max_memory_bytes", None) is not None:
            return int(cost_model.max_memory_bytes)
        return None

    def plan_distributed_contraction_path(
        self,
        path,
        mesh,
        *,
        memory_limit_per_device=None,
        cost_model=None,
    ):
        if isinstance(path, DistributedContractionPlan):
            distributed_plan = path
        elif isinstance(path, ContractionPlan) and len(path.steps) == 1 and isinstance(path.steps[0].plan, DistributedContractionPlan):
            distributed_plan = path.steps[0].plan
        elif isinstance(path, ContractionPlan):
            distributed_plan = self._auto_distributed_plan_from_dense_path(path, mesh)
        else:
            raise BackendFeatureError("plan_distributed_contraction_path requires a distributed ContractionPlan")
        if cost_model is not None:
            distributed_plan = DistributedContractionPlan(
                path=distributed_plan.path,
                steps=tuple(
                    self._with_distributed_step_timing(step, cost_model)
                    for step in distributed_plan.steps
                ),
                output_sharding=distributed_plan.output_sharding,
                estimated_comm_bytes=distributed_plan.estimated_comm_bytes,
                equation=distributed_plan.equation,
                peak_local_bytes=distributed_plan.peak_local_bytes,
                total_flops=distributed_plan.total_flops,
                total_comm_bytes=distributed_plan.total_comm_bytes,
                total_redistribute_bytes=distributed_plan.total_redistribute_bytes,
                total_allreduce_bytes=distributed_plan.total_allreduce_bytes,
                total_gather_bytes=distributed_plan.total_gather_bytes,
                total_point_to_point_bytes=distributed_plan.total_point_to_point_bytes,
                total_broadcast_bytes=distributed_plan.total_broadcast_bytes,
                total_reduce_scatter_bytes=distributed_plan.total_reduce_scatter_bytes,
                total_alltoall_bytes=distributed_plan.total_alltoall_bytes,
                total_allgather_bytes=distributed_plan.total_allgather_bytes,
            )
        result = self._with_distributed_plan_totals(distributed_plan)
        effective_memory_limit = self._effective_memory_limit_per_device(memory_limit_per_device, cost_model)
        if effective_memory_limit is not None and result.peak_local_bytes > effective_memory_limit:
            raise BackendFeatureError(
                "distributed contraction peak local bytes {0} exceeds memory limit {1}"
                .format(result.peak_local_bytes, effective_memory_limit)
            )
        return result

    def default_stream(self):
        return None

    def new_stream(self):
        return None

    def record_event(self, stream=None):
        return StreamEvent(device=self.current_device(), stream=stream)

    def wait_event(self, event, stream=None):
        self._validate_stream_event(event)
        return None

    def _stream_context(self, stream):
        return contextlib.nullcontext()

    def allocate_workspace(self, nbytes, *, device=None):
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("workspace nbytes must be non-negative")
        spec = parse_device_spec(device) if device is not None else self.current_device()
        if spec is None:
            spec = self.current_device()
        buffer = self.to_backend(_np.empty((nbytes,), dtype=_np.uint8), device=spec)
        return Workspace(device=spec, nbytes=nbytes, buffer=buffer)

    def release_workspace(self, workspace):
        if not isinstance(workspace, Workspace):
            raise TypeError("workspace must be a Workspace instance")
        if getattr(workspace, "released", False):
            return None
        current = self.current_device()
        if not self._device_specs_compatible(workspace.device, current):
            raise BackendFeatureError(
                "workspace device {0!r} is incompatible with current device {1!r}"
                .format(workspace.device, current)
            )
        workspace.buffer = None
        workspace.released = True
        return None

    def _validate_stream_event(self, event):
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent instance")
        current = self.current_device()
        if not self._device_specs_compatible(event.device, current):
            raise BackendFeatureError(
                "stream event device {0!r} is incompatible with current device {1!r}"
                .format(event.device, current)
            )
        return event

    @staticmethod
    def _device_specs_compatible(actual, expected):
        if actual.kind != expected.kind:
            return False
        for attr in ("index", "local_rank", "global_rank", "visible_id"):
            actual_value = getattr(actual, attr)
            expected_value = getattr(expected, attr)
            if actual_value is not None and expected_value is not None and actual_value != expected_value:
                return False
        return True

    def _plan_required_workspace_bytes(self, plan):
        if isinstance(plan, ContractionPlan):
            distributed_plan = self._distributed_plan_from_contraction_plan(plan)
            if distributed_plan is not None:
                return self._plan_required_workspace_bytes(distributed_plan)
            return int(plan.required_workspace_bytes)
        if isinstance(plan, MatmulPlan):
            return int(plan.workspace_bytes)
        if isinstance(plan, GroupedGemmPlan):
            return int(plan.estimated_workspace_bytes)
        if isinstance(plan, DistributedContractionPlan):
            return max(
                (
                    int(getattr(step.local_step, "required_workspace_bytes", 0) or 0)
                    for step in plan.steps
                ),
                default=0,
            )
        return 0

    def _validate_workspace(self, workspace, *, required_bytes=0):
        if workspace is None:
            return
        if not isinstance(workspace, Workspace):
            raise TypeError("workspace must be a Workspace instance")
        if getattr(workspace, "released", False):
            raise BackendFeatureError("released workspace cannot be used")
        current = self.current_device()
        if not self._device_specs_compatible(workspace.device, current):
            raise BackendFeatureError(
                "workspace device {0!r} is incompatible with current device {1!r}"
                .format(workspace.device, current)
            )
        required_bytes = int(required_bytes or 0)
        if int(workspace.nbytes) < required_bytes:
            raise BackendFeatureError(
                "workspace requires at least {0} bytes, got {1}"
                .format(required_bytes, int(workspace.nbytes))
            )
        buffer_nbytes = self._array_nbytes(workspace.buffer)
        if buffer_nbytes < int(workspace.nbytes):
            raise BackendFeatureError(
                "workspace buffer has {0} bytes, less than declared {1}"
                .format(buffer_nbytes, int(workspace.nbytes))
            )

    @staticmethod
    def _einsum_equation_from_plan(plan):
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        all_modes = tuple(mode for operand in plan.input_specs for mode in operand.modes) + tuple(plan.output_modes)
        if all(
            isinstance(mode, str) and len(mode) == 1 and mode in labels
            for mode in all_modes
        ):
            inputs = ["".join(operand.modes) for operand in plan.input_specs]
            return "{0}->{1}".format(",".join(inputs), "".join(plan.output_modes))

        mapping = {}

        def label_for(mode):
            if mode not in mapping:
                if len(mapping) >= len(labels):
                    raise BackendFeatureError("cannot synthesize einsum equation for more than {0} modes".format(len(labels)))
                mapping[mode] = labels[len(mapping)]
            return mapping[mode]

        inputs = []
        for operand in plan.input_specs:
            inputs.append("".join(label_for(mode) for mode in operand.modes))
        output = "".join(label_for(mode) for mode in plan.output_modes)
        return "{0}->{1}".format(",".join(inputs), output)

    def _distributed_spec_from_plan(self, plan):
        equation = plan.equation or self._einsum_equation_from_plan(plan.path)
        operands = tuple(operand.array for operand in plan.path.input_specs)
        return DistributedContractionSpec(
            equation=equation,
            operands=operands,
            output_sharding=plan.output_sharding,
        )

    @staticmethod
    def _distributed_plan_from_contraction_plan(plan):
        if isinstance(plan, DistributedContractionPlan):
            return plan
        if isinstance(plan, ContractionPlan) and len(plan.steps) == 1 and isinstance(plan.steps[0].plan, DistributedContractionPlan):
            return plan.steps[0].plan
        return None

    def _placement_sharding_for_operand(self, operand, output_sharding):
        operand_modes = tuple(operand.modes)
        sharded_modes = tuple(
            mode for mode in operand_modes
            if mode in output_sharding.sharded_modes
        )
        if not sharded_modes:
            return None
        ranks_per_mode = {
            mode: output_sharding.ranks_per_mode[mode]
            for mode in sharded_modes
            if mode in output_sharding.ranks_per_mode
        }
        mode_to_mesh_axis = {
            mode: output_sharding.mode_to_mesh_axis[mode]
            for mode in sharded_modes
            if mode in output_sharding.mode_to_mesh_axis
        }
        return ShardingSpec(
            global_shape=tuple(int(dim) for dim in getattr(operand.array, "shape", ())),
            modes=operand_modes,
            mesh=output_sharding.mesh,
            ranks_per_mode=ranks_per_mode,
            mode_to_mesh_axis=mode_to_mesh_axis,
            sharded_modes=sharded_modes,
            replicated_modes=tuple(mode for mode in operand_modes if mode not in sharded_modes),
        )

    def _placed_operands_for_activate_distribution(self, plan):
        if plan.output_sharding is None:
            raise BackendFeatureError("activate_distribution plan requires output_sharding")
        placed = []
        for operand in plan.path.input_specs:
            if isinstance(operand.array, DistributedTensor):
                placed.append(operand.array)
                continue
            sharding = self._placement_sharding_for_operand(operand, plan.output_sharding)
            if sharding is None:
                placed.append(self.replicate_tensor(
                    operand.array,
                    plan.output_sharding.mesh,
                    modes=operand.modes,
                ))
            else:
                placed.append(self.shard_tensor(operand.array, sharding))
        return tuple(placed)

    def _execute_activate_distribution_plan(self, plan, *, stream=None, workspace=None):
        del stream
        self._validate_workspace(workspace, required_bytes=self._plan_required_workspace_bytes(plan))
        if plan.output_sharding is None:
            raise BackendFeatureError("activate_distribution plan requires output_sharding")
        spec = DistributedContractionSpec(
            equation=plan.equation or self._einsum_equation_from_plan(plan.path),
            operands=self._placed_operands_for_activate_distribution(plan),
            output_sharding=plan.output_sharding,
        )
        return self.distributed_contract(spec, plan=plan)

    def _execute_sliced_contraction_plan(self, plan, *, stream=None, workspace=None):
        base_plan = plan.base_plan
        equation = self._einsum_equation_from_plan(base_plan)
        output_shape = self._output_shape_for_contraction_plan(base_plan)
        result = None
        for output_slice, operand_slices in zip(plan.output_slices, plan.operand_slices):
            operands = tuple(
                operand.array[tuple(slices)]
                for operand, slices in zip(base_plan.input_specs, operand_slices)
            )
            local_result = self._execute_local_contraction_plan(
                equation,
                operands,
                stream=stream,
                workspace=workspace,
                record_plan_profile=False,
            )
            if result is None:
                result = self._zeros_backend(output_shape, getattr(local_result, "dtype", None))
            result = self._slice_set(result, output_slice, local_result)
        if result is None:
            raise BackendFeatureError("sliced contraction plan has no slices to execute")
        return result

    def _record_sliced_contraction_execute(self, plan, result, wall_s, *, stream=None, workspace=None):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            step = plan.steps[0] if plan.steps else None
            sliced_plan = step.plan if step is not None else None
            base_plan = sliced_plan.base_plan if isinstance(sliced_plan, SlicedContractionPlan) else None
            base_step = base_plan.steps[0] if base_plan is not None and base_plan.steps else None
            base_lowering = getattr(getattr(base_step, "plan", None), "kind", getattr(base_step, "kind", None))
            equation, input_modes, output_modes = self._contraction_plan_profile_metadata(plan)
            largest_intermediate_bytes = int(getattr(result, "nbytes", 0) or 0)
            largest_intermediate_elements = self._array_size(result)

            profiling.record(
                "contraction_execute",
                backend=self.name,
                **profiling.contraction_execute_compute_payload("slice"),
                equation=equation,
                lowering="slice",
                plan_hash=plan.plan_hash,
                input_modes=input_modes,
                output_modes=output_modes,
                input_shapes=[
                    tuple(getattr(operand.array, "shape", ()))
                    for operand in plan.input_specs
                ],
                operands=[self._profile_tensor_operand(operand) for operand in plan.input_specs],
                input_dtypes=[
                    str(getattr(operand.array, "dtype", None))
                    for operand in plan.input_specs
                ],
                output_shape=tuple(getattr(result, "shape", ())),
                dtype=str(getattr(result, "dtype", None)),
                **self._profile_device_execution(self.current_device()),
                **self._profile_execution_resources(stream=stream, workspace=workspace),
                flops=plan.estimated_flops,
                read_bytes=plan.estimated_read_bytes,
                write_bytes=plan.estimated_write_bytes,
                copy_bytes=plan.estimated_copy_bytes,
                workspace_bytes=plan.required_workspace_bytes,
                peak_bytes=plan.estimated_peak_bytes,
                largest_intermediate=largest_intermediate_bytes,
                largest_intermediate_elements=largest_intermediate_elements,
                largest_intermediate_bytes=largest_intermediate_bytes,
                num_gemm=len(sliced_plan.output_slices) if base_lowering == "gemm" else 0,
                num_batched_gemm=len(sliced_plan.output_slices) if base_lowering in ("batched_gemm", "strided_batched_gemm") else 0,
                num_grouped_tasks=0,
                num_blocks=0,
                num_shape_buckets=0,
                fallback_reason=getattr(step, "fallback_reason", None),
                sliced_modes=[str(mode) for mode in plan.sliced_modes],
                num_slices=len(sliced_plan.output_slices),
                base_lowering=base_lowering,
                slice_output_axis=sliced_plan.output_axis,
                slice_output_shapes=[
                    self._local_shape_for_slices(tuple(getattr(result, "shape", ())), output_slice)
                    for output_slice in sliced_plan.output_slices
                ],
                wall_s=wall_s,
            )
        except Exception:
            pass

    @staticmethod
    def _matmul_plan_with_operands(plan, left, right):
        descs = tuple(replace(desc, A=left, B=right) for desc in plan.descs)
        return replace(plan, descs=descs)

    def _record_multi_step_contraction_execute(self, plan, result, wall_s, *, stream=None, workspace=None):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            equation, input_modes, output_modes = self._contraction_plan_profile_metadata(plan)
            step_lowerings = [
                getattr(step.plan, "kind", step.kind)
                for step in plan.steps
            ]
            largest_intermediate_bytes = max(
                [int(getattr(result, "nbytes", 0) or 0)]
                + [int(step.estimated_write_bytes or 0) for step in plan.steps]
            )
            largest_intermediate_elements = self._elements_for_nbytes(largest_intermediate_bytes, result)

            profiling.record(
                "contraction_execute",
                backend=self.name,
                **profiling.contraction_execute_compute_payload("multi_step"),
                equation=equation,
                lowering="multi_step",
                step_lowerings=step_lowerings,
                step_count=len(plan.steps),
                contraction_steps=self._profile_contraction_steps(plan.steps),
                plan_hash=plan.plan_hash,
                input_modes=input_modes,
                output_modes=output_modes,
                input_shapes=[
                    tuple(getattr(operand.array, "shape", ()))
                    for operand in plan.input_specs
                ],
                operands=[self._profile_tensor_operand(operand) for operand in plan.input_specs],
                input_dtypes=[
                    str(getattr(operand.array, "dtype", None))
                    for operand in plan.input_specs
                ],
                output_shape=tuple(getattr(result, "shape", ())),
                dtype=str(getattr(result, "dtype", None)),
                **self._profile_device_execution(self.current_device()),
                **self._profile_execution_resources(stream=stream, workspace=workspace),
                flops=plan.estimated_flops,
                read_bytes=plan.estimated_read_bytes,
                write_bytes=plan.estimated_write_bytes,
                copy_bytes=plan.estimated_copy_bytes,
                workspace_bytes=plan.required_workspace_bytes,
                peak_bytes=plan.estimated_peak_bytes,
                largest_intermediate=largest_intermediate_bytes,
                largest_intermediate_elements=largest_intermediate_elements,
                largest_intermediate_bytes=largest_intermediate_bytes,
                num_gemm=sum(1 for lowering in step_lowerings if lowering == "gemm"),
                num_batched_gemm=sum(
                    1
                    for lowering in step_lowerings
                    if lowering in ("batched_gemm", "strided_batched_gemm")
                ),
                num_grouped_tasks=sum(
                    len(getattr(step.plan, "descs", ()) or ())
                    for step, lowering in zip(plan.steps, step_lowerings)
                    if lowering == "grouped_gemm"
                ),
                num_blocks=0,
                num_shape_buckets=0,
                fallback_reason=next((step.fallback_reason for step in plan.steps if step.fallback_reason), None),
                wall_s=wall_s,
            )
        except Exception:
            pass

    def _execute_multi_step_contraction_plan(self, plan, *, stream=None, workspace=None):
        try:
            from renormalizer.utils import profiling

            should_profile = profiling.should_record_op()
        except Exception:
            should_profile = False
        if should_profile:
            import time

            started = time.perf_counter()
        else:
            started = None
        operands = [operand.array for operand in plan.input_specs]
        for step in plan.steps:
            step_operands = tuple(operands[int(index)] for index in step.inputs)
            if len(step_operands) != 2:
                raise BackendFeatureError("multi-step contraction execution expects pairwise steps")
            if isinstance(step.plan, MatmulPlan):
                step_plan = self._matmul_plan_with_operands(step.plan, step_operands[0], step_operands[1])
                result = self.execute_matmul_plan(
                    step_plan,
                    stream=stream,
                    workspace=workspace,
                    plan_hash=plan.plan_hash,
                    record_profile=False,
                    equation=self._equation_from_modes(step.input_modes, step.output_modes),
                    input_modes=step.input_modes,
                    output_modes=step.output_modes,
                )
            elif step.plan is not None:
                result = self.execute(step.plan, stream=stream, workspace=workspace)
            else:
                equation = self._equation_from_modes(step.input_modes, step.output_modes)
                result = self._execute_einsum(equation, step_operands)
            for index in sorted((int(index) for index in step.inputs), reverse=True):
                del operands[index]
            operands.append(result)
        if len(operands) != 1:
            raise BackendFeatureError("multi-step contraction execution did not reduce to one output")
        result = operands[0]
        if started is not None:
            self._record_multi_step_contraction_execute(
                plan,
                result,
                time.perf_counter() - started,
                stream=stream,
                workspace=workspace,
            )
        return result

    @staticmethod
    def _mode_token(mode):
        return str(mode)

    @classmethod
    def _equation_from_modes(cls, input_modes, output_modes):
        lhs = ",".join("".join(cls._mode_token(mode) for mode in modes) for modes in input_modes)
        rhs = "".join(cls._mode_token(mode) for mode in output_modes)
        return "{0}->{1}".format(lhs, rhs)

    @classmethod
    def _contraction_plan_profile_metadata(cls, plan):
        input_modes = tuple(tuple(operand.modes) for operand in plan.input_specs)
        output_modes = tuple(plan.output_modes)
        return cls._equation_from_modes(input_modes, output_modes), input_modes, output_modes

    def execute(self, plan, *, stream=None, workspace=None):
        self._validate_workspace(workspace, required_bytes=self._plan_required_workspace_bytes(plan))
        if isinstance(plan, ContractionPlan):
            if len(plan.steps) != 1:
                return self._execute_multi_step_contraction_plan(plan, stream=stream, workspace=workspace)
            inner_plan = plan.steps[0].plan
            if isinstance(inner_plan, MatmulPlan):
                equation, input_modes, output_modes = self._contraction_plan_profile_metadata(plan)
                return self.execute_matmul_plan(
                    inner_plan,
                    stream=stream,
                    workspace=workspace,
                    plan_hash=plan.plan_hash,
                    equation=equation,
                    input_modes=input_modes,
                    output_modes=output_modes,
                )
            if isinstance(inner_plan, SlicedContractionPlan):
                try:
                    from renormalizer.utils import profiling

                    should_profile = profiling.should_record_op()
                except Exception:
                    should_profile = False
                if should_profile:
                    import time

                    started = time.perf_counter()
                    result = self._execute_sliced_contraction_plan(
                        inner_plan,
                        stream=stream,
                        workspace=workspace,
                    )
                    wall_s = time.perf_counter() - started
                    self._record_sliced_contraction_execute(
                        plan,
                        result,
                        wall_s,
                        stream=stream,
                        workspace=workspace,
                    )
                    return result
                return self._execute_sliced_contraction_plan(
                    inner_plan,
                    stream=stream,
                    workspace=workspace,
                )
            return self.execute(inner_plan, stream=stream, workspace=workspace)
        if isinstance(plan, MatmulPlan):
            return self.execute_matmul_plan(plan, stream=stream, workspace=workspace)
        if isinstance(plan, GroupedGemmPlan):
            return self.execute_grouped_gemm_plan(plan, stream=stream, workspace=workspace)
        if isinstance(plan, DistributedContractionPlan):
            if plan.steps and plan.steps[0].kind == "activate_distribution":
                return self._execute_activate_distribution_plan(plan, stream=stream, workspace=workspace)
            spec = self._distributed_spec_from_plan(plan)
            return self.distributed_contract(spec, plan=plan, stream=stream, workspace=workspace)
        raise BackendFeatureError("Unknown backend execution plan {0!r}".format(type(plan).__name__))

    def synchronize(self, device=None, stream=None):
        if device is not None:
            spec = parse_device_spec(device)
            if spec is None:
                spec = self.current_device()
            current = self.current_device()
            if not self._device_specs_compatible(spec, current):
                raise BackendFeatureError(
                    "synchronize device {0!r} is incompatible with current device {1!r}"
                    .format(spec, current)
                )
        return self.sync()

    def unpack_masked_vectors(self, x: Any, spec):
        mask = self._validated_packed_mask(spec)
        shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if shape == (spec.packed_dim,):
            if spec.nrhs != 1:
                raise ValueError(
                    "batched packed vector shape must be ({0}, {1}); got {2}"
                    .format(spec.packed_dim, spec.nrhs, shape)
                )
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
            if spec.nrhs != 1:
                raise ValueError(
                    "batched center tensor shape must include RHS axis; expected ({0}, {1}) layout; got {2}"
                    .format(spec.center_shape, spec.nrhs, shape)
                )
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

    def shard_tensor(self, x, spec: ShardingSpec):
        if self.is_distributed_array(x):
            x = self.gather_tensor(x)
        input_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if input_shape != tuple(spec.global_shape):
            raise ValueError(
                "shard_tensor input shape {0} must match ShardingSpec global_shape {1}"
                .format(input_shape, tuple(spec.global_shape))
            )
        if self.is_distributed and int(spec.mesh.world_size) == int(self.size):
            rank = int(self.rank)
            local_slice = spec.local_slices[rank]
            local_array = x[tuple(local_slice)]
            return DistributedTensor(
                local_array=local_array,
                global_shape=spec.global_shape,
                modes=spec.modes,
                sharding=spec,
                mesh=spec.mesh,
                dtype=getattr(local_array, "dtype", None),
                local_shape=tuple(getattr(local_array, "shape", ())),
                local_nbytes=self._array_nbytes(local_array),
                rank_local_arrays=None,
            )
        rank_local_arrays = {
            int(rank): x[tuple(local_slice)]
            for rank, local_slice in spec.local_slices.items()
        }
        local_array = rank_local_arrays[spec.mesh.local_rank]
        return DistributedTensor(
            local_array=local_array,
            global_shape=spec.global_shape,
            modes=spec.modes,
            sharding=spec,
            mesh=spec.mesh,
            dtype=getattr(local_array, "dtype", None),
            local_shape=tuple(getattr(local_array, "shape", ())),
            local_nbytes=self._array_nbytes(local_array),
            rank_local_arrays=rank_local_arrays,
        )

    def gather_tensor(self, x, root=None):
        if not self.is_distributed_array(x):
            return x
        if root is not None:
            root = int(root)
            world_size = int(x.mesh.world_size)
            if root < 0 or root >= world_size:
                raise ValueError(
                    "gather_tensor root {0} is out of range for world size {1}"
                    .format(root, world_size)
                )
        if x.rank_local_arrays is None:
            if self.is_distributed and int(x.mesh.world_size) == int(self.size):
                if root is None:
                    gathered = self.allgather(x.local_array)
                else:
                    gathered = self.gather(x.local_array, root=root)
                    if int(self.rank) != root:
                        return None
                result = self._zeros_backend(x.global_shape, x.dtype)
                for rank, value in enumerate(gathered):
                    result = self._slice_set(result, x.sharding.local_slices[rank], value)
                return result
            if x.sharding.local_slices.get(x.mesh.local_rank) == tuple(slice(None) for _ in x.global_shape):
                if root is not None and int(x.mesh.local_rank) != root:
                    return None
                return x.local_array
            raise BackendFeatureError("cannot gather distributed tensor without rank-local arrays")
        if root is not None and int(x.mesh.local_rank) != root:
            return None
        result = self._zeros_backend(x.global_shape, x.dtype)
        for rank in sorted(x.rank_local_arrays):
            result = self._slice_set(result, x.sharding.local_slices[rank], x.rank_local_arrays[rank])
        return result

    def _redistribute_same_mesh_alltoall(self, x, new_spec: ShardingSpec):
        if not self.is_distributed_array(x):
            return None
        if x.rank_local_arrays is not None:
            return None
        if not self.is_distributed or int(x.mesh.world_size) != int(self.size):
            return None
        old_spec = x.sharding
        if old_spec.mesh != new_spec.mesh:
            return None
        if tuple(old_spec.modes) != tuple(new_spec.modes):
            return None
        if tuple(old_spec.global_shape) != tuple(new_spec.global_shape):
            return None
        if len(old_spec.sharded_modes) != 1 or len(new_spec.sharded_modes) != 1:
            return None
        source_mode = old_spec.sharded_modes[0]
        target_mode = new_spec.sharded_modes[0]
        if source_mode == target_mode:
            return None
        source_mesh_axis = old_spec.mode_to_mesh_axis.get(source_mode)
        target_mesh_axis = new_spec.mode_to_mesh_axis.get(target_mode)
        if source_mesh_axis is None or target_mesh_axis is None or source_mesh_axis != target_mesh_axis:
            return None
        split_axis = old_spec.modes.index(target_mode)
        concat_axis = old_spec.modes.index(source_mode)
        local_array = self.alltoall(x.local_array, split_axis=split_axis, concat_axis=concat_axis)
        return DistributedTensor(
            local_array=local_array,
            global_shape=new_spec.global_shape,
            modes=new_spec.modes,
            sharding=new_spec,
            mesh=new_spec.mesh,
            dtype=getattr(local_array, "dtype", None),
            local_shape=tuple(getattr(local_array, "shape", ())),
            local_nbytes=self._array_nbytes(local_array),
            rank_local_arrays=None,
        )

    def redistribute(self, x, new_spec: ShardingSpec):
        redistributed = self._redistribute_same_mesh_alltoall(x, new_spec)
        if redistributed is not None:
            return redistributed
        dense = self.gather_tensor(x)
        return self.shard_tensor(dense, new_spec)

    def replicate_tensor(self, x, mesh, *, modes=None):
        if self.is_distributed_array(x):
            x = self.gather_tensor(x)
        modes = tuple(range(len(getattr(x, "shape", ())))) if modes is None else tuple(modes)
        spec = ShardingSpec(
            global_shape=tuple(int(dim) for dim in getattr(x, "shape", ())),
            modes=modes,
            mesh=mesh,
            ranks_per_mode={},
            mode_to_mesh_axis={},
            replicated_modes=modes,
            sharded_modes=(),
        )
        if self.is_distributed and int(mesh.world_size) == int(self.size):
            return DistributedTensor(
                local_array=x,
                global_shape=spec.global_shape,
                modes=modes,
                sharding=spec,
                mesh=mesh,
                dtype=getattr(x, "dtype", None),
                local_shape=tuple(getattr(x, "shape", ())),
                local_nbytes=self._array_nbytes(x),
                rank_local_arrays=None,
            )
        rank_local_arrays = {rank: x for rank in spec.local_slices}
        return DistributedTensor(
            local_array=rank_local_arrays[mesh.local_rank],
            global_shape=spec.global_shape,
            modes=modes,
            sharding=spec,
            mesh=mesh,
            dtype=getattr(x, "dtype", None),
            local_shape=tuple(getattr(x, "shape", ())),
            local_nbytes=self._array_nbytes(x),
            rank_local_arrays=rank_local_arrays,
        )

    def _local_operand_for_rank(self, operand, rank):
        if not self.is_distributed_array(operand):
            return operand
        if operand.rank_local_arrays is None:
            if int(rank) == operand.mesh.local_rank:
                return operand.local_array
            raise BackendFeatureError("cannot execute distributed contraction without rank-local arrays")
        return operand.rank_local_arrays[int(rank)]

    def _execute_einsum(self, equation, operands):
        xp = self.array_namespace or _np
        einsum = getattr(xp, "einsum", None)
        if einsum is None:
            return _np.einsum(equation, *operands)
        return einsum(equation, *operands)

    def _record_local_contraction_fallback_execute(
        self,
        equation,
        operands,
        result,
        wall_s,
        fallback_reason,
        *,
        stream=None,
        workspace=None,
    ):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            input_modes, output_modes = parse_einsum_equation(equation)
            result_nbytes = self._array_nbytes(result)
            result_elements = self._array_size(result)
            profiling.record(
                "contraction_execute",
                backend=self.name,
                **profiling.contraction_execute_compute_payload("block_grouped_gemm"),
                equation=equation,
                lowering="fallback_einsum",
                input_modes=[[str(mode) for mode in modes] for modes in input_modes],
                output_modes=[str(mode) for mode in output_modes],
                input_shapes=[tuple(getattr(operand, "shape", ())) for operand in operands],
                operands=[
                    self._profile_array_operand(
                        "operand{0}".format(index),
                        operand,
                        input_modes[index],
                    )
                    for index, operand in enumerate(operands)
                ],
                input_dtypes=[str(getattr(operand, "dtype", None)) for operand in operands],
                output_shape=tuple(getattr(result, "shape", ())),
                dtype=str(getattr(result, "dtype", None)),
                **self._profile_device_execution(self.current_device()),
                **self._profile_execution_resources(stream=stream, workspace=workspace),
                flops=0,
                read_bytes=sum(self._array_nbytes(operand) for operand in operands),
                write_bytes=result_nbytes,
                copy_bytes=0,
                workspace_bytes=0,
                peak_bytes=result_nbytes,
                largest_intermediate=result_nbytes,
                largest_intermediate_elements=result_elements,
                largest_intermediate_bytes=result_nbytes,
                num_gemm=0,
                num_batched_gemm=0,
                num_grouped_tasks=0,
                num_blocks=0,
                num_shape_buckets=0,
                fallback_reason=fallback_reason,
                fallback_from="local_contraction_plan",
                fallback_to="einsum",
                fallback_policy=self.fallback_policy.value,
                wall_s=wall_s,
            )
        except Exception:
            pass

    def _execute_local_contraction_plan(
        self,
        equation,
        operands,
        *,
        stream=None,
        workspace=None,
        record_plan_profile=True,
    ):
        try:
            spec = self.parse_einsum(equation, *operands)
            plan = self._plan_einsum_contraction(spec, record_profile=record_plan_profile)
        except BackendFeatureError as exc:
            reason = "local contraction planning failed; used einsum fallback: {0}".format(exc)
            if self.fallback_policy is FallbackPolicy.FORBID:
                raise BackendFeatureError(reason) from exc
            if self.fallback_policy is FallbackPolicy.WARN:
                import warnings

                warnings.warn(reason, RuntimeWarning, stacklevel=3)
            try:
                from renormalizer.utils import profiling

                should_profile = profiling.should_record_op()
            except Exception:
                should_profile = False
            if should_profile:
                import time

                started = time.perf_counter()
                result = self._execute_einsum(equation, operands)
                self._record_local_contraction_fallback_execute(
                    equation,
                    operands,
                    result,
                    time.perf_counter() - started,
                    reason,
                    stream=stream,
                    workspace=workspace,
                )
                return result
            return self._execute_einsum(equation, operands)
        if (
            isinstance(plan, ContractionPlan)
            and len(plan.steps) == 1
            and isinstance(plan.steps[0].plan, MatmulPlan)
        ):
            return self.execute_matmul_plan(
                plan.steps[0].plan,
                stream=stream,
                workspace=workspace,
                plan_hash=plan.plan_hash,
                record_profile=False,
            )
        return self.execute(plan, stream=stream, workspace=workspace)

    def _sum_rank_local_arrays(self, rank_local_arrays):
        total = None
        for rank in sorted(rank_local_arrays):
            value = rank_local_arrays[rank]
            total = value if total is None else total + value
        return total

    def _reduce_scatter_axis_for_output(self, reduced_distributed_modes, output_sharding, execution_sharding, output_modes):
        if not reduced_distributed_modes:
            return None
        if output_sharding is None or execution_sharding is None:
            return None
        if self._sharding_specs_equivalent(output_sharding, execution_sharding):
            return None
        if len(output_sharding.sharded_modes) != 1:
            return None
        target_mode = output_sharding.sharded_modes[0]
        if target_mode not in output_modes:
            return None
        if target_mode in execution_sharding.sharded_modes:
            return None
        return tuple(output_modes).index(target_mode)

    @staticmethod
    def _local_contraction_profile(plan, step=None):
        def step_metric(name, default=0):
            if step is None:
                return int(default or 0)
            return int(getattr(step, name, default) or 0)

        def with_execution_items(profile, execution_items):
            return {
                **profile,
                "local_execution_primitives": list(execution_items.get("execution_primitives", ())),
                "local_execution_policies": list(execution_items.get("execution_policies", ())),
                "local_fallback_reasons": list(execution_items.get("fallback_reasons", ())),
            }

        if plan is None:
            return with_execution_items({
                "local_lowering": None,
                "num_gemm": 0,
                "num_batched_gemm": 0,
                "num_grouped_tasks": 0,
                "num_blocks": 0,
                "num_shape_buckets": 0,
                "local_flops": step_metric("estimated_flops"),
                "local_read_bytes": step_metric("estimated_read_bytes"),
                "local_write_bytes": step_metric("estimated_write_bytes"),
                "local_copy_bytes": step_metric("estimated_copy_bytes"),
                "local_workspace_bytes": step_metric("required_workspace_bytes"),
                "local_peak_bytes": step_metric("estimated_peak_bytes"),
                "fallback_reason": None,
            }, {})
        if isinstance(plan, MatmulPlan):
            read_bytes = sum(int(desc.estimated_read_bytes) for desc in plan.descs)
            write_bytes = sum(int(desc.estimated_write_bytes) for desc in plan.descs)
            workspace_bytes = int(plan.workspace_bytes)
            execution_items = AbstractBackend._matmul_plan_execution_items(
                plan,
                fallback_reason=getattr(step, "fallback_reason", None),
            )
            return with_execution_items({
                "local_lowering": plan.kind,
                "num_gemm": 1 if plan.kind == "gemm" else 0,
                "num_batched_gemm": 1 if plan.kind in ("batched_gemm", "strided_batched_gemm") else 0,
                "num_grouped_tasks": len(plan.descs) if plan.kind == "grouped_gemm" else 0,
                "num_blocks": 0,
                "num_shape_buckets": 0,
                "local_flops": step_metric("estimated_flops", plan.estimated_flops),
                "local_read_bytes": step_metric("estimated_read_bytes", read_bytes),
                "local_write_bytes": step_metric("estimated_write_bytes", write_bytes),
                "local_copy_bytes": step_metric("estimated_copy_bytes", plan.copy_bytes),
                "local_workspace_bytes": step_metric("required_workspace_bytes", workspace_bytes),
                "local_peak_bytes": step_metric("estimated_peak_bytes", max(write_bytes, workspace_bytes)),
                "fallback_reason": plan.fallback_reason,
            }, execution_items)
        if isinstance(plan, GroupedGemmPlan):
            execution_items = {
                "execution_primitives": ["grouped_gemm"],
                "execution_policies": ["backend_grouped_gemm"],
                "fallback_reasons": [],
            }
            return with_execution_items({
                "local_lowering": "grouped_gemm",
                "num_gemm": 0,
                "num_batched_gemm": 0,
                "num_grouped_tasks": len(plan.tasks),
                "num_blocks": len(plan.output_blocks),
                "num_shape_buckets": len(plan.bucketed_by_shape),
                "local_flops": step_metric("estimated_flops", plan.estimated_flops),
                "local_read_bytes": step_metric("estimated_read_bytes", plan.estimated_read_bytes),
                "local_write_bytes": step_metric("estimated_write_bytes", plan.estimated_write_bytes),
                "local_copy_bytes": step_metric("estimated_copy_bytes", plan.estimated_copy_bytes),
                "local_workspace_bytes": step_metric("required_workspace_bytes", plan.estimated_workspace_bytes),
                "local_peak_bytes": step_metric(
                    "estimated_peak_bytes",
                    plan.estimated_write_bytes + plan.estimated_copy_bytes + plan.estimated_workspace_bytes,
                ),
                "fallback_reason": None,
            }, execution_items)
        return with_execution_items({
            "local_lowering": type(plan).__name__,
            "num_gemm": 0,
            "num_batched_gemm": 0,
            "num_grouped_tasks": 0,
            "num_blocks": 0,
            "num_shape_buckets": 0,
            "local_flops": step_metric("estimated_flops"),
            "local_read_bytes": step_metric("estimated_read_bytes"),
            "local_write_bytes": step_metric("estimated_write_bytes"),
            "local_copy_bytes": step_metric("estimated_copy_bytes"),
            "local_workspace_bytes": step_metric("required_workspace_bytes"),
            "local_peak_bytes": step_metric("estimated_peak_bytes"),
            "fallback_reason": None,
        }, {})

    @staticmethod
    def _distributed_execution_items(local_profile):
        return {
            "execution_primitives": [
                "distributed_contract",
                *list(local_profile.get("local_execution_primitives", ())),
            ],
            "execution_policies": [
                "backend_distributed_contract",
                *list(local_profile.get("local_execution_policies", ())),
            ],
            "fallback_reasons": list(local_profile.get("local_fallback_reasons", ())),
        }

    @staticmethod
    def _profile_slice(local_slice):
        return [
            [
                item.start,
                item.stop,
                item.step,
            ]
            for item in tuple(local_slice)
        ]

    @staticmethod
    def _profile_device(device):
        from renormalizer.utils import profiling

        return profiling.device_payload(device)

    @staticmethod
    def _profile_device_execution(device):
        from renormalizer.utils import profiling

        return profiling.device_execution_payload(device)

    @staticmethod
    def _profile_execution_resources(stream=None, workspace=None):
        workspace_device = getattr(workspace, "device", None)
        return {
            "stream_provided": stream is not None,
            "stream_type": type(stream).__name__ if stream is not None else None,
            "workspace_provided": workspace is not None,
            "workspace_nbytes": int(getattr(workspace, "nbytes", 0) or 0) if workspace is not None else None,
            "workspace_released": bool(getattr(workspace, "released", False)) if workspace is not None else None,
            "workspace_device_kind": getattr(workspace_device, "kind", None),
            "workspace_device_index": getattr(workspace_device, "index", None),
            "workspace_device_local_rank": getattr(workspace_device, "local_rank", None),
            "workspace_device_global_rank": getattr(workspace_device, "global_rank", None),
        }

    def _profile_tensor_operand(self, operand):
        from renormalizer.utils import profiling

        return profiling.array_operand_payload(
            self,
            operand.name,
            operand.array,
            operand.modes,
        )

    def _profile_array_operand(self, name, array, modes):
        from renormalizer.utils import profiling

        return profiling.array_operand_payload(self, name, array, modes)

    @staticmethod
    def _profile_scalar(value):
        if value is None or isinstance(value, (str, int, float, bool, complex)):
            return value
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                pass
        return repr(value)

    @staticmethod
    def _profile_gemm_task_modes(task, side):
        if side == "A":
            return ("k", "m") if task.trans_a else ("m", "k")
        if side == "B":
            return ("n", "k") if task.trans_b else ("k", "n")
        raise ValueError("unknown GEMM task side {0!r}".format(side))

    def _profile_gemm_task_operands(self, task, index):
        return [
            self._profile_array_operand(
                "task{0}.A".format(index),
                task.A,
                self._profile_gemm_task_modes(task, "A"),
            ),
            self._profile_array_operand(
                "task{0}.B".format(index),
                task.B,
                self._profile_gemm_task_modes(task, "B"),
            ),
        ]

    def _profile_gemm_task_spec(self, task, index, *, xp=None):
        key = gemm_task_key(task, xp=xp)
        spec = {
            "index": int(index),
            "m": int(key[3]),
            "n": int(key[4]),
            "k": int(key[5]),
            "trans_a": bool(task.trans_a),
            "trans_b": bool(task.trans_b),
            "conj_a": bool(task.conj_a),
            "conj_b": bool(task.conj_b),
            "alpha": self._profile_scalar(task.alpha),
            "beta": self._profile_scalar(task.beta),
            "tag": self._profile_scalar(task.tag),
        }
        batch_shape = tuple(int(dim) for dim in key[2])
        if batch_shape:
            spec["batch_shape"] = batch_shape
        return spec

    @staticmethod
    def _profile_layout_spec(layout):
        if layout is None:
            return None
        return {
            "logical_shape": [int(dim) for dim in layout.logical_shape],
            "physical_shape": (
                [int(dim) for dim in layout.physical_shape]
                if layout.physical_shape is not None
                else None
            ),
            "logical_modes": [str(mode) for mode in layout.logical_modes],
            "strides": [int(stride) for stride in layout.strides] if layout.strides is not None else None,
            "order": layout.order,
            "contiguous_groups": [
                [int(axis) for axis in group]
                for group in layout.contiguous_groups
            ],
            "requires_transpose": bool(layout.requires_transpose),
            "transpose_perm": (
                [int(axis) for axis in layout.transpose_perm]
                if layout.transpose_perm is not None
                else None
            ),
            "estimated_copy_bytes": int(layout.estimated_copy_bytes),
        }

    @staticmethod
    def _profile_layout_transform(transform):
        return {
            "kind": transform.kind,
            "input_shape": [int(dim) for dim in transform.input_shape],
            "output_shape": [int(dim) for dim in transform.output_shape],
            "copy_bytes": int(transform.copy_bytes),
            "reason": transform.reason,
        }

    @classmethod
    def _profile_layout_transforms(cls, transforms):
        return [
            cls._profile_layout_transform(transform)
            for transform in tuple(transforms)
        ]

    def _profile_matmul_desc_operands(self, desc, index):
        left_modes = tuple(getattr(desc.layout_a, "logical_modes", ()) or ())
        right_modes = tuple(getattr(desc.layout_b, "logical_modes", ()) or ())
        return [
            self._profile_array_operand("task{0}.A".format(index), desc.A, left_modes),
            self._profile_array_operand("task{0}.B".format(index), desc.B, right_modes),
        ]

    def _profile_matmul_desc_spec(self, desc, index):
        return {
            "index": int(index),
            "m": int(desc.m),
            "n": int(desc.n),
            "k": int(desc.k),
            "batch_shape": [int(dim) for dim in desc.batch_shape],
            "trans_a": bool(desc.trans_a),
            "trans_b": bool(desc.trans_b),
            "conj_a": bool(desc.conj_a),
            "conj_b": bool(desc.conj_b),
            "alpha": self._profile_scalar(desc.alpha),
            "beta": self._profile_scalar(desc.beta),
            "dtype_compute": str(desc.dtype_compute),
            "dtype_output": str(desc.dtype_output),
            "estimated_flops": int(desc.estimated_flops),
            "estimated_read_bytes": int(desc.estimated_read_bytes),
            "estimated_write_bytes": int(desc.estimated_write_bytes),
            "estimated_workspace_bytes": int(desc.estimated_workspace_bytes),
            "layout_a": self._profile_layout_spec(desc.layout_a),
            "layout_b": self._profile_layout_spec(desc.layout_b),
            "layout_c": self._profile_layout_spec(desc.layout_c),
        }

    @staticmethod
    def _profile_matmul_desc_group_key(desc):
        dtype_a = str(getattr(desc.A, "dtype", None))
        dtype_b = str(getattr(desc.B, "dtype", None))
        dtype = dtype_a if dtype_a == dtype_b else "{0},{1}".format(dtype_a, dtype_b)
        return {
            "dtype": dtype,
            "trans_a": "C" if desc.conj_a and desc.trans_a else ("T" if desc.trans_a else "N"),
            "trans_b": "C" if desc.conj_b and desc.trans_b else ("T" if desc.trans_b else "N"),
            "conj_a": bool(desc.conj_a),
            "conj_b": bool(desc.conj_b),
            "batch_shape": tuple(int(dim) for dim in desc.batch_shape),
            "m": int(desc.m),
            "n": int(desc.n),
            "k": int(desc.k),
            "lda": int(desc.k),
            "ldb": int(desc.n),
            "ldc": int(desc.n),
        }

    @classmethod
    def _profile_matmul_desc_group_keys(cls, descs):
        group_keys = []
        seen = set()
        for desc in descs:
            payload = cls._profile_matmul_desc_group_key(desc)
            key = (
                payload["dtype"],
                payload["trans_a"],
                payload["trans_b"],
                payload["conj_a"],
                payload["conj_b"],
                payload["batch_shape"],
                payload["m"],
                payload["n"],
                payload["k"],
                payload["lda"],
                payload["ldb"],
                payload["ldc"],
            )
            if key in seen:
                continue
            seen.add(key)
            group_keys.append(payload)
        return group_keys

    @classmethod
    def _profile_matmul_desc_group_key_buckets(cls, descs):
        buckets = {}
        order = []
        for index, desc in enumerate(descs):
            payload = cls._profile_matmul_desc_group_key(desc)
            key = (
                payload["dtype"],
                payload["trans_a"],
                payload["trans_b"],
                payload["conj_a"],
                payload["conj_b"],
                payload["batch_shape"],
                payload["m"],
                payload["n"],
                payload["k"],
                payload["lda"],
                payload["ldb"],
                payload["ldc"],
            )
            if key not in buckets:
                buckets[key] = {**payload, "task_indices": []}
                order.append(key)
            buckets[key]["task_indices"].append(int(index))

        group_key_buckets = []
        for key in order:
            item = dict(buckets[key])
            item["task_count"] = len(item["task_indices"])
            group_key_buckets.append(item)
        return group_key_buckets

    @classmethod
    def _profile_sharding(cls, sharding):
        if sharding is None:
            return None
        local_slice = sharding.local_slices.get(sharding.mesh.global_rank)
        return {
            "global_shape": list(sharding.global_shape),
            "modes": [str(mode) for mode in sharding.modes],
            "sharded_modes": [str(mode) for mode in sharding.sharded_modes],
            "replicated_modes": [str(mode) for mode in sharding.replicated_modes],
            "ranks_per_mode": {str(mode): int(count) for mode, count in sharding.ranks_per_mode.items()},
            "mode_to_mesh_axis": {str(mode): str(axis) for mode, axis in sharding.mode_to_mesh_axis.items()},
            "mesh_shape": list(sharding.mesh.shape),
            "mesh_axis_names": [str(axis) for axis in sharding.mesh.axis_names],
            "mesh_backend": sharding.mesh.backend,
            "mesh_world_size": int(sharding.mesh.world_size),
            "mesh_local_rank": int(sharding.mesh.local_rank),
            "mesh_global_rank": int(sharding.mesh.global_rank),
            "local_slice": cls._profile_slice(local_slice) if local_slice is not None else None,
        }

    @classmethod
    def _profile_distribution_state(cls, state):
        if state is None:
            return None
        return {
            "operand_index": int(state.operand_index),
            "tensor_id": int(state.tensor_id),
            "modes": [str(mode) for mode in state.modes],
            "shape": list(state.shape),
            "distributed_modes": [str(mode) for mode in state.distributed_modes],
            "replicated_modes": [str(mode) for mode in state.replicated_modes],
            "local_shape": list(state.local_shape),
            "local_nbytes": int(state.local_nbytes),
            "sharding": cls._profile_sharding(state.sharding),
        }

    def _profile_communication_plan(self, item, *, wall_s=None):
        primitive = str(item.kind)
        is_collective = primitive in {
            "broadcast",
            "allreduce",
            "reduce_scatter",
            "gather",
            "allgather",
            "alltoall",
        }
        payload = {
            "kind": primitive,
            "primitive": primitive,
            "collective": primitive,
            "is_collective": is_collective,
            "is_point_to_point": primitive == "point_to_point",
            "bytes": int(item.bytes),
            "local_bytes": int(getattr(item, "local_bytes", item.bytes)),
            "modes": [str(mode) for mode in item.modes],
            "num_messages": int(item.num_messages),
            "block_size": int(item.block_size),
            "wall_s": float(0.0 if wall_s is None else wall_s),
        }
        return payload

    def _profile_distributed_step(self, step, index, *, communication_payloads=None):
        if communication_payloads is None:
            communication_payloads = [
                self._profile_communication_plan(item)
                for item in step.communication
            ]
        return {
            "index": int(index),
            "kind": str(step.kind),
            "local_step": self._profile_contraction_steps((step.local_step,))[0],
            "input_states": [
                self._profile_distribution_state(state)
                for state in step.input_states
            ],
            "output_state": self._profile_distribution_state(step.output_state),
            "communication": list(communication_payloads),
            "estimated_compute_s": float(step.estimated_compute_s),
            "estimated_comm_s": float(step.estimated_comm_s),
            "estimated_total_s": float(step.estimated_total_s),
        }

    def _record_distributed_contraction_plan(self, spec, plan):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            distributed_plan = self._distributed_plan_from_contraction_plan(plan)
            if distributed_plan is None:
                return
            distributed_step = distributed_plan.steps[0] if distributed_plan.steps else None
            communication = distributed_step.communication if distributed_step is not None else ()
            input_states = distributed_step.input_states if distributed_step is not None else ()
            output_state = distributed_step.output_state if distributed_step is not None else None
            local_profile = self._local_contraction_profile(
                distributed_step.local_contraction_plan if distributed_step is not None else None,
                distributed_step.local_step if distributed_step is not None else None,
            )
            input_modes, output_modes = parse_einsum_equation(spec.equation)
            sizes = self._mode_sizes_from_equation(input_modes, spec.operands)
            output_shape = self._output_shape_from_sizes(output_modes, sizes)
            distributed_modes = tuple(getattr(plan, "distributed_modes", ()))
            if not distributed_modes and output_state is not None:
                distributed_modes = tuple(getattr(output_state, "distributed_modes", ()))
            communication_payload = [
                self._profile_communication_plan(item)
                for item in communication
            ]
            dtype = None
            if distributed_step is not None:
                dtype = self._matmul_plan_output_dtype(distributed_step.local_contraction_plan)
            largest_intermediate_bytes = int(local_profile["local_write_bytes"] or 0)
            largest_intermediate_elements = 0
            if output_state is not None:
                largest_intermediate_bytes = max(
                    largest_intermediate_bytes,
                    int(getattr(output_state, "local_nbytes", 0) or 0),
                )
                largest_intermediate_elements = self._prod_shape(getattr(output_state, "local_shape", ()))
            elif largest_intermediate_bytes:
                dtype_itemsize = max(
                    (int(getattr(self._operand_dtype(operand), "itemsize", 0) or 0) for operand in spec.operands),
                    default=0,
                )
                if dtype_itemsize:
                    largest_intermediate_elements = largest_intermediate_bytes // dtype_itemsize
            rank = None
            world_size = None
            local_shape = None
            global_shape = tuple(output_shape)
            if output_state is not None:
                local_shape = tuple(getattr(output_state, "local_shape", ()) or ())
                global_shape = tuple(getattr(output_state, "shape", ()) or output_shape)
                sharding = getattr(output_state, "sharding", None)
                mesh = getattr(sharding, "mesh", None)
                if mesh is not None:
                    rank = int(getattr(mesh, "global_rank", 0))
                    world_size = int(getattr(mesh, "world_size", 1))

            profiling.record(
                "contraction_plan",
                backend=self.name,
                equation=spec.equation,
                lowering="distributed",
                dtype=dtype,
                is_distributed_runtime=bool(getattr(self, "is_distributed", False)),
                plan_hash=plan.plan_hash,
                local_plan_hash=distributed_plan.path.plan_hash,
                input_modes=[[str(mode) for mode in modes] for modes in input_modes],
                output_modes=[str(mode) for mode in output_modes],
                input_shapes=[tuple(self._operand_global_shape(operand)) for operand in spec.operands],
                operands=[
                    self._profile_array_operand(
                        "operand{0}".format(index),
                        operand,
                        input_modes[index],
                    )
                    for index, operand in enumerate(spec.operands)
                ],
                input_dtypes=[str(self._operand_dtype(operand)) for operand in spec.operands],
                output_shape=output_shape,
                **self._profile_device_execution(self.current_device()),
                flops=int(getattr(plan, "estimated_flops", 0)),
                read_bytes=int(getattr(plan, "estimated_read_bytes", 0)),
                write_bytes=int(getattr(plan, "estimated_write_bytes", 0)),
                copy_bytes=int(getattr(plan, "estimated_copy_bytes", 0)),
                workspace_bytes=int(getattr(plan, "required_workspace_bytes", 0)),
                peak_bytes=int(getattr(plan, "estimated_peak_bytes", 0)),
                largest_intermediate=largest_intermediate_bytes,
                largest_intermediate_elements=largest_intermediate_elements,
                largest_intermediate_bytes=largest_intermediate_bytes,
                local_lowering=local_profile["local_lowering"],
                local_execution_primitives=local_profile["local_execution_primitives"],
                local_execution_policies=local_profile["local_execution_policies"],
                local_fallback_reasons=local_profile["local_fallback_reasons"],
                **self._distributed_execution_items(local_profile),
                num_gemm=local_profile["num_gemm"],
                num_batched_gemm=local_profile["num_batched_gemm"],
                num_grouped_tasks=local_profile["num_grouped_tasks"],
                num_blocks=local_profile["num_blocks"],
                num_shape_buckets=local_profile["num_shape_buckets"],
                local_flops=local_profile["local_flops"],
                local_read_bytes=local_profile["local_read_bytes"],
                local_write_bytes=local_profile["local_write_bytes"],
                local_copy_bytes=local_profile["local_copy_bytes"],
                local_workspace_bytes=local_profile["local_workspace_bytes"],
                local_peak_bytes=local_profile["local_peak_bytes"],
                fallback_reason=local_profile["fallback_reason"],
                distributed_modes=[str(mode) for mode in distributed_modes],
                distributed_step_kind=getattr(distributed_step, "kind", None),
                estimated_compute_s=float(getattr(distributed_step, "estimated_compute_s", 0.0)),
                estimated_comm_s=float(getattr(distributed_step, "estimated_comm_s", 0.0)),
                estimated_total_s=float(getattr(distributed_step, "estimated_total_s", 0.0)),
                rank=rank,
                world_size=world_size,
                local_shape=local_shape,
                global_shape=global_shape,
                input_states=[
                    self._profile_distribution_state(state)
                    for state in input_states
                ],
                output_state=self._profile_distribution_state(output_state),
                communication=communication_payload,
                distributed_step_count=len(distributed_plan.steps),
                distributed_steps=[
                    self._profile_distributed_step(
                        step,
                        index,
                        communication_payloads=communication_payload if index == 0 else None,
                    )
                    for index, step in enumerate(distributed_plan.steps)
                ],
                comm_bytes=int(
                    distributed_plan.total_comm_bytes
                    or distributed_plan.estimated_comm_bytes
                ),
                redistribute_bytes=int(distributed_plan.total_redistribute_bytes),
                allreduce_bytes=int(distributed_plan.total_allreduce_bytes),
                gather_bytes=int(distributed_plan.total_gather_bytes),
                point_to_point_bytes=int(distributed_plan.total_point_to_point_bytes),
                broadcast_bytes=int(distributed_plan.total_broadcast_bytes),
                reduce_scatter_bytes=int(distributed_plan.total_reduce_scatter_bytes),
                alltoall_bytes=int(distributed_plan.total_alltoall_bytes),
                allgather_bytes=int(distributed_plan.total_allgather_bytes),
            )
        except Exception:
            pass

    def _record_distributed_contraction_execute(
        self,
        spec,
        plan,
        result,
        wall_s,
        communication_timings=None,
        *,
        stream=None,
        workspace=None,
    ):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            distributed_plan = self._distributed_plan_from_contraction_plan(plan)
            communication = ()
            input_states = ()
            output_state = None
            distributed_step = None
            local_profile = self._local_contraction_profile(None)
            if distributed_plan is not None and distributed_plan.steps:
                distributed_step = distributed_plan.steps[0]
                communication = distributed_step.communication
                input_states = distributed_step.input_states
                output_state = distributed_step.output_state
                local_profile = self._local_contraction_profile(
                    distributed_step.local_contraction_plan,
                    distributed_step.local_step,
                )
            step = plan.steps[0] if isinstance(plan, ContractionPlan) and plan.steps else None
            plan_hash = ""
            if isinstance(plan, ContractionPlan):
                plan_hash = getattr(plan, "plan_hash", "")
            elif distributed_plan is not None:
                plan_hash = getattr(distributed_plan.path, "plan_hash", "")
            local_plan_hash = (
                getattr(distributed_plan.path, "plan_hash", "")
                if distributed_plan is not None
                else ""
            )
            metric_plan = distributed_plan.path if distributed_plan is not None else plan
            flops = 0
            if distributed_plan is not None:
                flops = int(distributed_plan.total_flops or getattr(distributed_plan.path, "estimated_flops", 0))
            elif step is not None:
                flops = int(step.estimated_flops)
            comm_bytes = 0
            if distributed_plan is not None:
                comm_bytes = int(distributed_plan.total_comm_bytes or distributed_plan.estimated_comm_bytes)
            communication_wall_s = {}
            for timing in communication_timings or ():
                collective = timing.get("collective")
                if collective is None:
                    continue
                communication_wall_s.setdefault(collective, []).append(float(timing.get("wall_s", 0.0)))

            def pop_communication_wall_s(kind):
                timings = communication_wall_s.get(kind)
                if not timings:
                    return 0.0
                return timings.pop(0)

            distributed_modes = tuple(getattr(plan, "distributed_modes", ()))
            if not distributed_modes and output_state is not None:
                distributed_modes = tuple(getattr(output_state, "distributed_modes", ()))
            input_modes, output_modes = parse_einsum_equation(spec.equation)
            communication_payload = [
                self._profile_communication_plan(
                    item,
                    wall_s=pop_communication_wall_s(item.kind),
                )
                for item in communication
            ]
            output_local_array = result.local_array
            output_local_shape = tuple(result.local_shape)
            output_local_strides = profiling.array_strides(output_local_array)
            output_local_order = profiling.array_order(output_local_array)
            output_local_contiguous = profiling.array_contiguous(output_local_array)
            output_local_backend = profiling.array_backend_name(output_local_array)
            output_local_device_kind = profiling.array_device_kind(output_local_array)
            output_local_location = profiling.array_location(output_local_array)

            execute_payload = {
                "event": "contraction_execute",
                "backend": self.name,
            }
            execute_payload.update(
                profiling.contraction_execute_compute_payload("distributed"),
                equation=spec.equation,
                lowering="distributed",
                is_distributed_runtime=bool(getattr(self, "is_distributed", False)),
                plan_hash=plan_hash,
                local_plan_hash=local_plan_hash,
                input_modes=[[str(mode) for mode in modes] for modes in input_modes],
                output_modes=[str(mode) for mode in output_modes],
                input_shapes=[tuple(self._operand_global_shape(operand)) for operand in spec.operands],
                operands=[
                    self._profile_array_operand(
                        "operand{0}".format(index),
                        operand,
                        input_modes[index],
                    )
                    for index, operand in enumerate(spec.operands)
                ],
                input_dtypes=[str(self._operand_dtype(operand)) for operand in spec.operands],
                output_shape=tuple(result.global_shape),
                output_strides=output_local_strides,
                output_order=output_local_order,
                output_contiguous=output_local_contiguous,
                output_backend=self.name,
                output_device_kind="distributed",
                output_location="distributed",
                output_is_host=False,
                output_is_device=False,
                output_is_distributed=True,
                output_local_shape=output_local_shape,
                output_local_strides=output_local_strides,
                output_local_order=output_local_order,
                output_local_contiguous=output_local_contiguous,
                output_local_backend=output_local_backend,
                output_local_device_kind=output_local_device_kind,
                output_local_location=output_local_location,
                output_local_is_host=profiling.array_is_host(output_local_array),
                output_local_is_device=profiling.array_is_device(output_local_array),
                output_local_is_distributed=profiling.array_is_distributed(output_local_array),
                dtype=str(getattr(result, "dtype", None)),
                **self._profile_device_execution(self.current_device()),
                **self._profile_execution_resources(stream=stream, workspace=workspace),
                flops=flops,
                read_bytes=int(getattr(metric_plan, "estimated_read_bytes", 0)),
                write_bytes=int(getattr(metric_plan, "estimated_write_bytes", 0)),
                copy_bytes=int(getattr(metric_plan, "estimated_copy_bytes", 0)),
                workspace_bytes=int(getattr(metric_plan, "required_workspace_bytes", 0)),
                peak_bytes=int(getattr(metric_plan, "estimated_peak_bytes", 0)),
                largest_intermediate=int(result.local_nbytes),
                largest_intermediate_elements=self._prod_shape(result.local_shape),
                largest_intermediate_bytes=int(result.local_nbytes),
                local_lowering=local_profile["local_lowering"],
                local_execution_primitives=local_profile["local_execution_primitives"],
                local_execution_policies=local_profile["local_execution_policies"],
                local_fallback_reasons=local_profile["local_fallback_reasons"],
                **self._distributed_execution_items(local_profile),
                num_gemm=local_profile["num_gemm"],
                num_batched_gemm=local_profile["num_batched_gemm"],
                num_grouped_tasks=local_profile["num_grouped_tasks"],
                num_blocks=local_profile["num_blocks"],
                num_shape_buckets=local_profile["num_shape_buckets"],
                local_flops=local_profile["local_flops"],
                local_read_bytes=local_profile["local_read_bytes"],
                local_write_bytes=local_profile["local_write_bytes"],
                local_copy_bytes=local_profile["local_copy_bytes"],
                local_workspace_bytes=local_profile["local_workspace_bytes"],
                local_peak_bytes=local_profile["local_peak_bytes"],
                fallback_reason=local_profile["fallback_reason"],
                distributed_modes=[str(mode) for mode in distributed_modes],
                distributed_step_kind=getattr(distributed_step, "kind", None),
                estimated_compute_s=float(getattr(distributed_step, "estimated_compute_s", 0.0)),
                estimated_comm_s=float(getattr(distributed_step, "estimated_comm_s", 0.0)),
                estimated_total_s=float(getattr(distributed_step, "estimated_total_s", 0.0)),
                rank=int(result.mesh.global_rank),
                world_size=int(result.mesh.world_size),
                local_shape=tuple(result.local_shape),
                global_shape=tuple(result.global_shape),
                input_states=[
                    self._profile_distribution_state(state)
                    for state in input_states
                ],
                output_state=self._profile_distribution_state(output_state),
                communication=communication_payload,
                distributed_step_count=len(distributed_plan.steps) if distributed_plan is not None else 0,
                distributed_steps=[
                    self._profile_distributed_step(
                        step,
                        index,
                        communication_payloads=communication_payload if index == 0 else None,
                    )
                    for index, step in enumerate(distributed_plan.steps)
                ] if distributed_plan is not None else [],
                comm_bytes=comm_bytes,
                redistribute_bytes=(
                    int(distributed_plan.total_redistribute_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                allreduce_bytes=(
                    int(distributed_plan.total_allreduce_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                gather_bytes=(
                    int(distributed_plan.total_gather_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                point_to_point_bytes=(
                    int(distributed_plan.total_point_to_point_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                broadcast_bytes=(
                    int(distributed_plan.total_broadcast_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                reduce_scatter_bytes=(
                    int(distributed_plan.total_reduce_scatter_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                alltoall_bytes=(
                    int(distributed_plan.total_alltoall_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                allgather_bytes=(
                    int(distributed_plan.total_allgather_bytes)
                    if distributed_plan is not None
                    else 0
                ),
                wall_s=wall_s,
            )
            standardized_payload = profiling.standardize_event_payload(execute_payload)
            self._last_execution_profile = dict(standardized_payload)
            record_payload = dict(standardized_payload)
            record_payload.pop("event", None)
            profiling.record("contraction_execute", **record_payload)
        except Exception:
            pass

    def _time_distributed_communication(self, collective, communication_timings, fn, *args, **kwargs):
        if communication_timings is None:
            return fn(*args, **kwargs)
        import time

        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            communication_timings.append(
                {
                    "collective": collective,
                    "wall_s": time.perf_counter() - started,
                }
            )

    def _distributed_contract_impl(self, spec, *, plan=None, stream=None, workspace=None, communication_timings=None):
        if not isinstance(spec, DistributedContractionSpec):
            raise TypeError("distributed_contract expects a DistributedContractionSpec")
        if plan is None:
            plan = self.plan_contraction(spec, allow_distribution=True)
        input_modes, output_modes = parse_einsum_equation(spec.equation)
        sizes = self._mode_sizes_from_equation(input_modes, spec.operands)
        output_shape = self._output_shape_from_sizes(output_modes, sizes)
        output_sharding = spec.output_sharding
        natural_output_sharding = self._derive_output_sharding(
            spec,
            input_modes,
            output_modes,
            output_shape,
            use_requested=False,
        )
        if output_sharding is None and plan.steps:
            distributed_plan = getattr(plan.steps[0], "plan", None)
            output_sharding = getattr(distributed_plan, "output_sharding", None)
        if output_sharding is None:
            output_sharding = natural_output_sharding
        if output_sharding is None:
            return self._execute_einsum(spec.equation, spec.operands)

        execution_sharding = natural_output_sharding or output_sharding
        mesh = execution_sharding.mesh
        distributed_modes = self._distributed_modes_for_operands(spec.operands)
        reduced_distributed_modes = self._reduced_distributed_modes(distributed_modes, output_modes)
        operands = self._redistribute_incompatible_input_operands(
            spec.operands,
            input_modes,
            execution_sharding,
            reduced_distributed_modes,
        )
        reduce_scatter_axis = self._reduce_scatter_axis_for_output(
            reduced_distributed_modes,
            output_sharding,
            execution_sharding,
            output_modes,
        )
        if self.is_distributed and int(mesh.world_size) == int(self.size):
            local_operands = tuple(
                operand.local_array if self.is_distributed_array(operand) else operand
                for operand in operands
            )
            local_array = self._execute_local_contraction_plan(
                spec.equation,
                local_operands,
                stream=stream,
                workspace=workspace,
            )
            result_sharding = execution_sharding
            if reduced_distributed_modes:
                if reduce_scatter_axis is not None:
                    local_array = self._time_distributed_communication(
                        "reduce_scatter",
                        communication_timings,
                        self.reduce_scatter,
                        local_array,
                        op="sum",
                        axis=reduce_scatter_axis,
                    )
                    result_sharding = output_sharding
                else:
                    local_array = self._time_distributed_communication(
                        "allreduce",
                        communication_timings,
                        self.allreduce,
                        local_array,
                        op="sum",
                    )
            result = DistributedTensor(
                local_array=local_array,
                global_shape=output_shape,
                modes=output_modes,
                sharding=result_sharding,
                mesh=result_sharding.mesh,
                dtype=getattr(local_array, "dtype", None),
                local_shape=tuple(getattr(local_array, "shape", ())),
                local_nbytes=self._array_nbytes(local_array),
                rank_local_arrays=None,
            )
            if not self._sharding_specs_equivalent(output_sharding, result_sharding):
                return self._time_distributed_communication(
                    "alltoall",
                    communication_timings,
                    self.redistribute,
                    result,
                    output_sharding,
                )
            return result
        rank_local_arrays = {}
        for rank in range(mesh.world_size):
            local_operands = tuple(self._local_operand_for_rank(operand, rank) for operand in operands)
            rank_local_arrays[rank] = self._execute_local_contraction_plan(
                spec.equation,
                local_operands,
                stream=stream,
                workspace=workspace,
            )
        result_sharding = execution_sharding
        if reduced_distributed_modes:
            reduced = self._time_distributed_communication(
                "allreduce",
                communication_timings,
                self.allreduce,
                self._sum_rank_local_arrays(rank_local_arrays),
                op="sum",
            )
            if reduce_scatter_axis is not None:
                result_sharding = output_sharding
                rank_local_arrays = {
                    rank: reduced[tuple(output_sharding.local_slices[rank])]
                    for rank in range(mesh.world_size)
                }
            else:
                rank_local_arrays = {rank: reduced for rank in range(mesh.world_size)}
        local_array = rank_local_arrays[mesh.local_rank]
        result = DistributedTensor(
            local_array=local_array,
            global_shape=output_shape,
            modes=output_modes,
            sharding=result_sharding,
            mesh=result_sharding.mesh,
            dtype=getattr(local_array, "dtype", None),
            local_shape=tuple(getattr(local_array, "shape", ())),
            local_nbytes=self._array_nbytes(local_array),
            rank_local_arrays=rank_local_arrays,
        )
        if not self._sharding_specs_equivalent(output_sharding, result_sharding):
            return self._time_distributed_communication(
                "alltoall",
                communication_timings,
                self.redistribute,
                result,
                output_sharding,
            )
        return result

    def distributed_contract(self, spec, *, plan=None, stream=None, workspace=None):
        if plan is None and isinstance(spec, DistributedContractionSpec):
            plan = self.plan_contraction(spec, allow_distribution=True)
        self._validate_workspace(workspace, required_bytes=self._plan_required_workspace_bytes(plan))
        try:
            from renormalizer.utils import profiling

            should_profile = profiling.should_record_op()
        except Exception:
            should_profile = False
        if not should_profile:
            return self._distributed_contract_impl(spec, plan=plan, stream=stream, workspace=workspace)

        import time

        communication_timings = []
        started = time.perf_counter()
        result = self._distributed_contract_impl(
            spec,
            plan=plan,
            stream=stream,
            workspace=workspace,
            communication_timings=communication_timings,
        )
        wall_s = time.perf_counter() - started
        self._record_distributed_contraction_execute(
            spec,
            plan,
            result,
            wall_s,
            communication_timings=communication_timings,
            stream=stream,
            workspace=workspace,
        )
        return result

    def lower_pair_contraction_to_matmul(self, spec, *, record_profile=True):
        plan = lower_pair_contraction_to_matmul(spec, self.capabilities)
        try:
            from renormalizer.utils import profiling

            if record_profile and profiling.should_record_op():
                input_modes = (spec.left.modes, spec.right.modes)
                equation = self._equation_from_modes(input_modes, spec.output_modes)
                step = self._contraction_step_from_matmul_plan(
                    plan,
                    input_modes=input_modes,
                    output_modes=spec.output_modes,
                )
                contraction_plan = self._contraction_plan_from_step(
                    step,
                    input_specs=(spec.left, spec.right),
                    output_modes=spec.output_modes,
                )
                largest_intermediate_bytes, largest_intermediate_elements = (
                    self._matmul_plan_largest_intermediate_memory(plan)
                )

                profiling.record(
                    "contraction_plan",
                    backend=self.name,
                    equation=equation,
                    plan_hash=contraction_plan.plan_hash,
                    lowering=plan.kind,
                    dtype=self._matmul_plan_output_dtype(plan),
                    operands=[
                        self._profile_tensor_operand(spec.left),
                        self._profile_tensor_operand(spec.right),
                    ],
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
                    **self._profile_device_execution(self.current_device()),
                    flops=plan.estimated_flops,
                    read_bytes=sum(desc.estimated_read_bytes for desc in plan.descs),
                    write_bytes=sum(desc.estimated_write_bytes for desc in plan.descs),
                    copy_bytes=plan.copy_bytes,
                    workspace_bytes=plan.workspace_bytes,
                    peak_bytes=(
                        sum(desc.estimated_write_bytes for desc in plan.descs)
                        + plan.copy_bytes
                        + plan.workspace_bytes
                    ),
                    largest_intermediate=largest_intermediate_bytes,
                    largest_intermediate_elements=largest_intermediate_elements,
                    largest_intermediate_bytes=largest_intermediate_bytes,
                    num_gemm=1 if plan.kind == "gemm" else 0,
                    num_batched_gemm=1 if plan.kind in ("batched_gemm", "strided_batched_gemm") else 0,
                    num_grouped_tasks=0,
                    num_blocks=0,
                    num_shape_buckets=0,
                    **self._matmul_plan_execution_items(plan),
                    fallback_reason=plan.fallback_reason,
                )
        except Exception:
            pass
        return plan

    @staticmethod
    def _block_sort_key(item):
        key = item[0]
        qn_left = getattr(key, "qn_left", None)
        qn_right = getattr(key, "qn_right", None)
        extra = getattr(key, "extra", None)
        if qn_left is not None and qn_right is not None:
            return ("block_key", tuple(qn_left), tuple(qn_right), tuple(extra or ()), repr(key))
        return (type(key).__name__, repr(key))

    @staticmethod
    def _profile_block_key(key):
        qn_left = getattr(key, "qn_left", None)
        qn_right = getattr(key, "qn_right", None)
        extra = getattr(key, "extra", None)
        if qn_left is not None and qn_right is not None:
            return {
                "qn_left": list(qn_left),
                "qn_right": list(qn_right),
                "extra": list(extra or ()),
            }
        return {"repr": repr(key)}

    @classmethod
    def _profile_zero_sized_block_skips(cls, plan):
        left_keys = tuple(getattr(plan, "zero_sized_left_block_keys", ()) or ())
        right_keys = tuple(getattr(plan, "zero_sized_right_block_keys", ()) or ())
        return {
            "num_zero_sized_left_blocks_skipped": len(left_keys),
            "num_zero_sized_right_blocks_skipped": len(right_keys),
            "num_zero_sized_blocks_skipped": len(left_keys) + len(right_keys),
            "zero_sized_left_block_keys": [cls._profile_block_key(key) for key in left_keys],
            "zero_sized_right_block_keys": [cls._profile_block_key(key) for key in right_keys],
        }

    @classmethod
    def _profile_output_block_offsets(cls, output_blocks, output_block_offsets, *, unique=False):
        if unique:
            keys = sorted(set(output_blocks), key=lambda key: cls._block_sort_key((key, None)))
        else:
            keys = tuple(output_blocks)
        items = []
        for key in keys:
            offset = output_block_offsets.get(key)
            if offset is None:
                continue
            items.append({
                **cls._profile_block_key(key),
                "offset": [int(dim) for dim in offset],
            })
        return items

    @classmethod
    def _profile_result_block_shape(cls, key, block):
        item = {
            **cls._profile_block_key(key),
            "shape": tuple(block.shape),
        }
        if block.offset is not None:
            item["offset"] = tuple(int(dim) for dim in block.offset)
        return item

    def _profile_result_block_layout(self, key, block):
        from renormalizer.utils import profiling

        array = block.array
        info = self.array_info(array)
        item = {
            **self._profile_block_key(key),
            "shape": tuple(info.shape),
            "strides": info.strides,
            "order": info.order,
            "contiguous": info.contiguous,
            "dtype": str(info.dtype),
            "itemsize": int(info.itemsize),
            "nbytes": int(info.nbytes),
            "backend": info.backend_name,
            "device": str(info.device),
            "device_kind": getattr(info.device, "kind", None),
            "device_index": getattr(info.device, "index", None),
            "location": profiling.array_location(array),
            "is_host": bool(info.is_host),
            "is_device": bool(info.is_device),
            "is_distributed": bool(info.is_distributed),
        }
        if block.offset is not None:
            item["offset"] = tuple(int(dim) for dim in block.offset)
        return item

    @staticmethod
    def _is_zero_sized_block(block):
        shape = tuple(int(dim) for dim in getattr(block, "shape", getattr(block.array, "shape", ())))
        if shape:
            return any(dim == 0 for dim in shape)
        numel = getattr(block.array, "numel", None)
        if callable(numel):
            return int(numel()) == 0
        size = getattr(block.array, "size", None)
        if callable(size):
            size = size()
        if size is None:
            return False
        if isinstance(size, (tuple, list)):
            product = 1
            for dim in size:
                product *= int(dim)
            return product == 0
        return int(size) == 0

    @staticmethod
    def _bucketed_by_desc_shape(descs):
        buckets = {}
        for index, desc in enumerate(descs):
            batch_shape = tuple(int(dim) for dim in getattr(desc, "batch_shape", ()) or ())
            key = (batch_shape, int(desc.m), int(desc.n), int(desc.k)) if batch_shape else (
                int(desc.m),
                int(desc.n),
                int(desc.k),
            )
            buckets.setdefault(key, []).append(index)
        return {key: tuple(indices) for key, indices in buckets.items()}

    @staticmethod
    def _shape_bucket_sort_key(item):
        shape = item[0]
        if len(shape) == 4 and isinstance(shape[0], tuple):
            return (1, shape[0], int(shape[1]), int(shape[2]), int(shape[3]))
        return (0, (), int(shape[0]), int(shape[1]), int(shape[2]))

    @classmethod
    def _profile_shape_buckets(cls, bucketed_by_shape):
        items = []
        for shape, indices in sorted(bucketed_by_shape.items(), key=cls._shape_bucket_sort_key):
            if len(shape) == 4 and isinstance(shape[0], tuple):
                batch_shape = tuple(int(dim) for dim in shape[0])
                m, n, k = shape[1:]
            else:
                batch_shape = ()
                m, n, k = shape[:3]
            item = {
                "m": int(m),
                "n": int(n),
                "k": int(k),
                "task_indices": [int(index) for index in indices],
                "task_count": int(len(indices)),
            }
            if batch_shape:
                item["batch_shape"] = batch_shape
                batch_count = 1
                for dim in batch_shape:
                    batch_count *= int(dim)
                item["batch_count"] = int(batch_count)
            items.append(item)
        return items

    @classmethod
    def _profile_group_boundaries(cls, bucketed_by_shape):
        group_sizes = []
        sorted_indices = []
        for _shape, indices in sorted(bucketed_by_shape.items(), key=cls._shape_bucket_sort_key):
            group_indices = [int(index) for index in indices]
            group_sizes.append(len(group_indices))
            sorted_indices.extend(group_indices)
        gsta = [0]
        for size in group_sizes:
            gsta.append(gsta[-1] + int(size))
        return {
            "group_sizes": group_sizes,
            "gsta": gsta,
            "sorted_indices": sorted_indices,
        }

    @classmethod
    def _profile_output_contributions(cls, output_blocks, output_block_offsets=None, tasks=None):
        groups = {}
        for index, key in enumerate(output_blocks):
            groups.setdefault(key, []).append(int(index))
        output_block_offsets = output_block_offsets or {}
        tasks = tuple(tasks or ())

        contribution_counts = []
        output_contributions = []
        reduction_group_count = 0
        scatter_add_task_count = 0
        max_contributions = 0
        for key, indices in sorted(groups.items(), key=cls._block_sort_key):
            contribution_count = len(indices)
            max_contributions = max(max_contributions, contribution_count)
            if contribution_count > 1:
                reduction_group_count += 1
                scatter_add_task_count += contribution_count
            contribution = {
                **cls._profile_block_key(key),
                "contribution_count": contribution_count,
                "task_indices": [int(index) for index in indices],
            }
            offset = output_block_offsets.get(key)
            if offset is not None:
                contribution["offset"] = [int(dim) for dim in offset]
            contribution_counts.append(
                contribution
            )
            contribution_detail = dict(contribution)
            contributions = []
            total_write_bytes = 0
            for index in indices:
                task = tasks[index] if index < len(tasks) else None
                if task is None:
                    shape = None
                    nbytes = 0
                else:
                    batch_shape = tuple(int(dim) for dim in getattr(task, "batch_shape", ()) or ())
                    shape = batch_shape + (int(task.m), int(task.n))
                    nbytes = int(getattr(task, "estimated_write_bytes", 0) or 0)
                total_write_bytes += int(nbytes)
                contributions.append({
                    "task_index": int(index),
                    "shape": [int(dim) for dim in shape] if shape is not None else None,
                    "nbytes": int(nbytes),
                    "output_offset": [int(dim) for dim in offset] if offset is not None else None,
                })
            contribution_detail["total_write_bytes"] = int(total_write_bytes)
            contribution_detail["contributions"] = contributions
            output_contributions.append(contribution_detail)

        return {
            "reduction_mode": "scatter_add" if reduction_group_count else "overwrite",
            "num_output_reduction_groups": int(reduction_group_count),
            "num_scatter_add_tasks": int(scatter_add_task_count),
            "max_output_contributions": int(max_contributions),
            "output_contribution_counts": contribution_counts,
            "output_contributions": output_contributions,
        }

    @staticmethod
    def _bucketed_by_task_shape(tasks, *, xp=None):
        buckets = {}
        for index, task in enumerate(tasks):
            key = gemm_task_key(task, xp=xp)
            batch_shape = tuple(int(dim) for dim in key[2])
            shape = (batch_shape, int(key[3]), int(key[4]), int(key[5])) if batch_shape else (
                int(key[3]),
                int(key[4]),
                int(key[5]),
            )
            buckets.setdefault(shape, []).append(index)
        return {shape: tuple(indices) for shape, indices in buckets.items()}

    @staticmethod
    def _block_contraction_global_shape(spec):
        left_sizes = dict(zip(spec.left.modes, spec.left.global_shape))
        right_sizes = dict(zip(spec.right.modes, spec.right.global_shape))
        shape = []
        for mode in spec.output_modes:
            left_size = left_sizes.get(mode)
            right_size = right_sizes.get(mode)
            if left_size is not None and right_size is not None and int(left_size) != int(right_size):
                raise ValueError(
                    "BlockContractionSpec output mode {0!r} has inconsistent global sizes {1} and {2}"
                    .format(mode, left_size, right_size)
                )
            size = left_size if left_size is not None else right_size
            shape.append(int(size))
        return tuple(shape)

    @staticmethod
    def _block_offset_map(block):
        if block.offset is None:
            return None
        return {
            mode: int(offset)
            for mode, offset in zip(block.modes, block.offset)
        }

    def _block_contraction_output_offset(self, output_modes, left_block, right_block):
        left_offsets = self._block_offset_map(left_block)
        right_offsets = self._block_offset_map(right_block)
        if left_offsets is None and right_offsets is None:
            return None
        output_offset = []
        for mode in output_modes:
            candidates = []
            if left_offsets is not None and mode in left_offsets:
                candidates.append(left_offsets[mode])
            if right_offsets is not None and mode in right_offsets:
                candidates.append(right_offsets[mode])
            if not candidates:
                return None
            if len(candidates) == 2 and candidates[0] != candidates[1]:
                raise BackendFeatureError(
                    "block contraction output mode {0!r} has incompatible block offsets {1} and {2}"
                    .format(mode, candidates[0], candidates[1])
                )
            output_offset.append(int(candidates[0]))
        return tuple(output_offset)

    def lower_block_contraction(self, spec):
        descs = []
        output_blocks = []
        output_block_offsets = {}
        estimated_copy_bytes = 0
        global_shape = self._block_contraction_global_shape(spec)
        sorted_left_blocks = sorted(spec.left.blocks.items(), key=self._block_sort_key)
        sorted_right_blocks = sorted(spec.right.blocks.items(), key=self._block_sort_key)
        zero_sized_left_block_keys = tuple(
            key
            for key, block in sorted_left_blocks
            if self._is_zero_sized_block(block)
        )
        zero_sized_right_block_keys = tuple(
            key
            for key, block in sorted_right_blocks
            if self._is_zero_sized_block(block)
        )
        zero_sized_left_block_key_set = set(zero_sized_left_block_keys)
        zero_sized_right_block_key_set = set(zero_sized_right_block_keys)
        for left_key, left_block in sorted_left_blocks:
            if left_key in zero_sized_left_block_key_set:
                continue
            for right_key, right_block in sorted_right_blocks:
                if right_key in zero_sized_right_block_key_set:
                    continue
                output_key = spec.qn_rule(left_key, right_key)
                if output_key is None:
                    continue
                pair_spec = PairContractionSpec.from_operands(
                    TensorOperand(left_block.array, tuple(left_block.modes), name="left"),
                    TensorOperand(right_block.array, tuple(right_block.modes), name="right"),
                    output_modes=spec.output_modes,
                )
                plan = self.lower_pair_contraction_to_matmul(
                    pair_spec,
                    record_profile=False,
                )
                if plan.kind == "fallback_tensordot":
                    self._handle_plan_fallback(plan)
                if not plan.descs:
                    raise BackendFeatureError("block contraction lowering produced no GEMM descriptors")
                desc = plan.descs[0]
                descs.append(desc)
                output_blocks.append(output_key)
                output_offset = self._block_contraction_output_offset(
                    spec.output_modes,
                    left_block,
                    right_block,
                )
                if output_offset is not None:
                    previous_offset = output_block_offsets.setdefault(output_key, output_offset)
                    if previous_offset != output_offset:
                        raise BackendFeatureError(
                            "duplicate output block keys require matching output offsets"
                        )
                estimated_copy_bytes += int(plan.copy_bytes)

        repeated_outputs = len(set(output_blocks)) != len(output_blocks)
        if repeated_outputs and not spec.accumulate:
            raise BackendFeatureError(
                "duplicate output block keys require accumulate=True; got accumulate=False"
            )
        plan = GroupedGemmPlan(
            tasks=tuple(descs),
            output_blocks=tuple(output_blocks),
            bucketed_by_shape=self._bucketed_by_desc_shape(descs),
            scatter_add_required=bool(repeated_outputs and spec.accumulate),
            estimated_flops=sum(int(desc.estimated_flops) for desc in descs),
            estimated_read_bytes=sum(int(desc.estimated_read_bytes) for desc in descs),
            estimated_write_bytes=sum(int(desc.estimated_write_bytes) for desc in descs),
            estimated_workspace_bytes=sum(int(desc.estimated_workspace_bytes) for desc in descs),
            estimated_copy_bytes=estimated_copy_bytes,
            output_modes=tuple(spec.output_modes),
            global_shape=global_shape,
            output_block_offsets=output_block_offsets,
            block_axis_meta={
                "left": spec.left.block_axis_meta,
                "right": spec.right.block_axis_meta,
            },
            zero_sized_left_block_keys=zero_sized_left_block_keys,
            zero_sized_right_block_keys=zero_sized_right_block_keys,
            backend=self.name,
        )
        self._record_block_contraction_plan(plan)
        return plan

    def _record_block_contraction_plan(self, plan):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            unique_output_keys = [
                key
                for key in sorted(set(plan.output_blocks), key=lambda key: self._block_sort_key((key, None)))
            ]
            fallback_reason = self._grouped_gemm_fallback_reason(len(plan.tasks))
            grouped_gemm_policy = self._block_grouped_gemm_plan_policy(fallback_reason)
            task_operands = [
                self._profile_matmul_desc_operands(desc, index)
                for index, desc in enumerate(plan.tasks)
            ]

            profiling.record(
                "contraction_plan",
                backend=self.name,
                equation=None,
                lowering="block_grouped_gemm",
                plan_hash=plan.plan_hash,
                input_shapes=[
                    [tuple(getattr(desc.A, "shape", ())), tuple(getattr(desc.B, "shape", ()))]
                    for desc in plan.tasks
                ],
                input_dtypes=[
                    [str(getattr(desc.A, "dtype", None)), str(getattr(desc.B, "dtype", None))]
                    for desc in plan.tasks
                ],
                operands=task_operands,
                task_operands=task_operands,
                task_specs=[
                    self._profile_matmul_desc_spec(desc, index)
                    for index, desc in enumerate(plan.tasks)
                ],
                output_shape=tuple(plan.global_shape),
                dtype=str(getattr(plan.tasks[0].A, "dtype", None)) if plan.tasks else None,
                **self._profile_device_execution(self.current_device()),
                flops=int(plan.estimated_flops),
                read_bytes=int(plan.estimated_read_bytes),
                write_bytes=int(plan.estimated_write_bytes),
                copy_bytes=int(plan.estimated_copy_bytes),
                workspace_bytes=int(plan.estimated_workspace_bytes),
                peak_bytes=(
                    int(plan.estimated_write_bytes or 0)
                    + int(plan.estimated_copy_bytes or 0)
                    + int(plan.estimated_workspace_bytes or 0)
                ),
                num_gemm=0,
                num_batched_gemm=0,
                num_grouped_tasks=len(plan.tasks),
                num_blocks=len(unique_output_keys),
                num_shape_buckets=len(plan.bucketed_by_shape),
                supports_grouped_gemm=bool(self.supports_grouped_gemm),
                grouped_gemm_policy=grouped_gemm_policy,
                grouped_gemm_implementation=self._grouped_gemm_implementation_from_policy(
                    grouped_gemm_policy,
                    fallback_reason,
                ),
                **self._grouped_gemm_execution_items(grouped_gemm_policy, fallback_reason),
                requires_grouped_gemm_fallback=fallback_reason is not None,
                fallback_from="grouped_gemm" if fallback_reason is not None else None,
                fallback_to="bucketed_grouped_gemm" if fallback_reason is not None else None,
                fallback_reason=fallback_reason,
                fallback_policy=self.fallback_policy.value if fallback_reason is not None else None,
                output_modes=[str(mode) for mode in plan.output_modes],
                global_shape=tuple(plan.global_shape),
                scatter_add_required=bool(plan.scatter_add_required),
                **self._profile_output_contributions(
                    plan.output_blocks,
                    plan.output_block_offsets,
                    tasks=plan.tasks,
                ),
                **self._profile_zero_sized_block_skips(plan),
                dense_materialized=False,
                materialized_dense_bytes=0,
                output_block_keys=[self._profile_block_key(key) for key in plan.output_blocks],
                unique_output_block_keys=[self._profile_block_key(key) for key in unique_output_keys],
                output_block_offsets=self._profile_output_block_offsets(
                    plan.output_blocks,
                    plan.output_block_offsets,
                ),
                unique_output_block_offsets=self._profile_output_block_offsets(
                    plan.output_blocks,
                    plan.output_block_offsets,
                    unique=True,
                ),
                bucket_task_counts=[len(indices) for indices in plan.bucketed_by_shape.values()],
                shape_buckets=self._profile_shape_buckets(plan.bucketed_by_shape),
                **self._profile_group_boundaries(plan.bucketed_by_shape),
                group_keys=self._profile_matmul_desc_group_keys(plan.tasks),
                group_key_buckets=self._profile_matmul_desc_group_key_buckets(plan.tasks),
            )
        except Exception:
            pass

    def execute_grouped_gemm_plan(
        self,
        plan,
        *,
        pack_threshold=4,
        stream=None,
        workspace=None,
        policy="auto",
        fallback_policy=None,
    ):
        self._validate_workspace(workspace, required_bytes=self._plan_required_workspace_bytes(plan))
        if len(plan.tasks) != len(plan.output_blocks):
            raise ValueError("GroupedGemmPlan tasks and output_blocks must have the same length")
        execution_policy, fallback_policy = self._resolve_grouped_gemm_call_policies(
            policy,
            fallback_policy,
        )
        try:
            from renormalizer.utils import profiling

            should_profile = profiling.should_record_op()
        except Exception:
            profiling = None
            should_profile = False
        if should_profile:
            import time

            started = time.perf_counter()
        flat_outputs = {}
        task_groups = {}
        tasks = []
        task_output_keys = []
        functional_accumulation = self.name == "jax" or bool(plan.scatter_add_required)
        for desc, output_key in zip(plan.tasks, plan.output_blocks):
            a, b, groups = self._prepare_matmul_desc(desc)
            output_shape = tuple(int(dim) for dim in getattr(desc, "batch_shape", ()) or ()) + (
                int(desc.m),
                int(desc.n),
            )
            if output_key not in flat_outputs:
                dtype = getattr(desc, "dtype_output", None) or getattr(a, "dtype", None)
                flat_outputs[output_key] = self._zeros_backend(output_shape, dtype)
                task_groups[output_key] = groups
            elif tuple(getattr(flat_outputs[output_key], "shape", ())) != output_shape:
                raise BackendFeatureError(
                    "duplicate output block keys require matching contribution shapes"
                )
            tasks.append(
                GemmTask(
                    a,
                    b,
                    C=None if functional_accumulation else flat_outputs[output_key],
                    trans_a=desc.trans_a,
                    trans_b=desc.trans_b,
                    conj_a=desc.conj_a,
                    conj_b=desc.conj_b,
                    alpha=desc.alpha,
                    beta=1.0,
                    tag=output_key,
                )
            )
            task_output_keys.append(output_key)

        if tasks:
            grouped_gemm_kwargs = {
                "pack_threshold": pack_threshold,
                "stream": stream,
                "workspace": workspace,
            }
            if execution_policy != "auto":
                grouped_gemm_kwargs["policy"] = execution_policy
            if fallback_policy is not self.fallback_policy:
                grouped_gemm_kwargs["fallback_policy"] = fallback_policy
            if should_profile:
                with profiling.scope(compute_accounting_override=profiling.COMPUTE_ACCOUNTING_INCLUSIVE):
                    results = self.grouped_gemm(tasks, **grouped_gemm_kwargs)
            else:
                results = self.grouped_gemm(tasks, **grouped_gemm_kwargs)
            if functional_accumulation:
                for output_key, result in zip(task_output_keys, results):
                    flat_outputs[output_key] = flat_outputs[output_key] + result

        blocks = {}
        for output_key in sorted(flat_outputs, key=lambda key: self._block_sort_key((key, None))):
            array = self._finalize_matmul_result(flat_outputs[output_key], task_groups[output_key])
            blocks[output_key] = DenseBlock(
                key=output_key,
                array=array,
                modes=tuple(plan.output_modes),
                shape=tuple(getattr(array, "shape", ())),
                offset=plan.output_block_offsets.get(output_key),
            )
        result = BlockTensor(
            blocks=blocks,
            global_shape=tuple(plan.global_shape),
            modes=tuple(plan.output_modes),
            block_axis_meta=plan.block_axis_meta,
            backend=self.name,
        )
        if should_profile:
            xp = self.array_namespace or _np
            flop_copy_ratio = 0 if self.supports_grouped_gemm else 10
            allow_batched = bool(self.supports_grouped_gemm or self.supports_batched_matmul)
            effective_pack_threshold, effective_flop_copy_ratio, effective_allow_batched = (
                self._grouped_gemm_execution_policy_controls(
                    execution_policy,
                    pack_threshold=pack_threshold,
                    flop_copy_ratio=flop_copy_ratio,
                    allow_batched=allow_batched,
                )
            )
            stats = grouped_gemm_stats(
                tasks,
                xp=xp,
                pack_threshold=effective_pack_threshold,
                flop_copy_ratio=effective_flop_copy_ratio,
                allow_batched=effective_allow_batched,
            )
            workspace_bytes = max(int(plan.estimated_workspace_bytes), int(stats.workspace_bytes))
            fallback_reason = self._grouped_gemm_fallback_reason(len(plan.tasks))
            task_operands = [
                self._profile_matmul_desc_operands(desc, index)
                for index, desc in enumerate(plan.tasks)
            ]
            output_reduction_executor = (
                "post_grouped_gemm"
                if plan.scatter_add_required
                else ("functional_accumulation" if functional_accumulation else "direct_output")
            )
            grouped_gemm_output_write_mode = (
                "workspace_then_reduce"
                if plan.scatter_add_required
                else ("workspace_then_assign" if functional_accumulation else "direct_output")
            )
            grouped_gemm_policy = self._grouped_gemm_execution_policy(stats, fallback_reason)
            execute_payload = {
                "event": "contraction_execute",
                "backend": self.name,
                **profiling.contraction_execute_compute_payload("block_grouped_gemm"),
                "equation": None,
                "lowering": "block_grouped_gemm",
                "plan_hash": plan.plan_hash,
                "input_shapes": [
                    [tuple(getattr(desc.A, "shape", ())), tuple(getattr(desc.B, "shape", ()))]
                    for desc in plan.tasks
                ],
                "input_dtypes": [
                    [str(getattr(desc.A, "dtype", None)), str(getattr(desc.B, "dtype", None))]
                    for desc in plan.tasks
                ],
                "operands": task_operands,
                "task_operands": task_operands,
                "task_specs": [
                    self._profile_matmul_desc_spec(desc, index)
                    for index, desc in enumerate(plan.tasks)
                ],
                "output_shape": tuple(result.global_shape),
                "dtype": str(getattr(next(iter(result.blocks.values())).array, "dtype", None)) if result.blocks else None,
                **self._profile_device_execution(self.current_device()),
                "flops": int(stats.flops),
                "read_bytes": int(stats.read_bytes),
                "write_bytes": int(stats.write_bytes),
                "copy_bytes": int(stats.copy_bytes),
                "workspace_bytes": workspace_bytes,
                **self._profile_execution_resources(stream=stream, workspace=workspace),
                "peak_bytes": int(stats.write_bytes) + int(stats.copy_bytes) + workspace_bytes,
                "largest_intermediate": max((self._array_nbytes(block.array) for block in result.blocks.values()), default=0),
                "largest_intermediate_elements": max((self._array_size(block.array) for block in result.blocks.values()), default=0),
                "largest_intermediate_bytes": max((self._array_nbytes(block.array) for block in result.blocks.values()), default=0),
                "num_gemm": int(stats.loop_task_count),
                "num_batched_gemm": int(stats.batched_bucket_count),
                "num_grouped_tasks": int(stats.task_count),
                "num_blocks": len(result.blocks),
                "num_shape_buckets": int(stats.shape_bucket_count),
                "supports_grouped_gemm": bool(self.supports_grouped_gemm),
                "requested_policy": execution_policy,
                "grouped_gemm_policy": grouped_gemm_policy,
                "grouped_gemm_implementation": self._grouped_gemm_implementation(stats, fallback_reason),
                **self._grouped_gemm_execution_items(grouped_gemm_policy, fallback_reason),
                "requires_grouped_gemm_fallback": fallback_reason is not None,
                "fallback_from": "grouped_gemm" if fallback_reason is not None else None,
                "fallback_to": self._grouped_gemm_fallback_target(stats, fallback_reason),
                "fallback_reason": fallback_reason,
                "fallback_policy": fallback_policy.value if fallback_reason is not None else None,
                "output_modes": [str(mode) for mode in plan.output_modes],
                "global_shape": tuple(result.global_shape),
                "scatter_add_required": bool(plan.scatter_add_required),
                "output_reduction_executor": output_reduction_executor,
                "grouped_gemm_output_write_mode": grouped_gemm_output_write_mode,
                **self._profile_output_contributions(
                    plan.output_blocks,
                    plan.output_block_offsets,
                    tasks=plan.tasks,
                ),
                **self._profile_zero_sized_block_skips(plan),
                "dense_materialized": False,
                "materialized_dense_bytes": 0,
                "output_block_keys": [self._profile_block_key(key) for key in plan.output_blocks],
                "unique_output_block_keys": [
                    self._profile_block_key(key)
                    for key, _block in sorted(result.blocks.items(), key=self._block_sort_key)
                ],
                "output_block_offsets": self._profile_output_block_offsets(
                    plan.output_blocks,
                    plan.output_block_offsets,
                ),
                "unique_output_block_offsets": self._profile_output_block_offsets(
                    result.blocks.keys(),
                    plan.output_block_offsets,
                    unique=True,
                ),
                "result_block_shapes": [
                    self._profile_result_block_shape(key, block)
                    for key, block in sorted(result.blocks.items(), key=self._block_sort_key)
                ],
                "result_block_layouts": [
                    self._profile_result_block_layout(key, block)
                    for key, block in sorted(result.blocks.items(), key=self._block_sort_key)
                ],
                "bucket_task_counts": stats.bucket_task_counts,
                "shape_buckets": self._profile_shape_buckets(plan.bucketed_by_shape),
                **self._profile_group_boundaries(plan.bucketed_by_shape),
                "group_keys": self._profile_matmul_desc_group_keys(plan.tasks),
                "group_key_buckets": self._profile_matmul_desc_group_key_buckets(plan.tasks),
                "batched_bucket_count": int(stats.batched_bucket_count),
                "loop_bucket_count": int(stats.loop_bucket_count),
                "batched_task_count": int(stats.batched_task_count),
                "loop_task_count": int(stats.loop_task_count),
                "pack_threshold": pack_threshold,
                "effective_pack_threshold": int(effective_pack_threshold),
                "effective_allow_batched": bool(effective_allow_batched),
                "wall_s": time.perf_counter() - started,
            }
            self._last_execution_profile = dict(execute_payload)
            record_payload = dict(execute_payload)
            record_payload.pop("event", None)
            profiling.record("contraction_execute", **record_payload)
        return result

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
        for storage_attr in ("untyped_storage", "storage"):
            left_storage = getattr(left, storage_attr, None)
            right_storage = getattr(right, storage_attr, None)
            if callable(left_storage) and callable(right_storage):
                try:
                    return left_storage().data_ptr() == right_storage().data_ptr()
                except Exception:
                    pass
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
        if self.is_array(spec.qn_mask):
            mask_shape = tuple(int(dim) for dim in getattr(spec.qn_mask, "shape", ()))
            if mask_shape != tuple(spec.center_shape):
                raise ValueError(
                    "qn_mask shape must match center_shape {0}; got {1}"
                    .format(spec.center_shape, mask_shape)
                )
            return self.to_backend(spec.qn_mask, dtype=bool)

        mask_host = _np.asarray(spec.qn_mask, dtype=bool)
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

    def _slice_set(self, array, slices, values):
        slices = tuple(slices)
        at = getattr(array, "at", None)
        if at is not None:
            return at[slices].set(values)
        array[slices] = values
        return array

    @staticmethod
    def _array_nbytes(array):
        nbytes = getattr(array, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        numel = getattr(array, "numel", None)
        element_size = getattr(array, "element_size", None)
        if callable(numel) and callable(element_size):
            return int(numel() * element_size())
        dtype = getattr(array, "dtype", None)
        itemsize = int(getattr(dtype, "itemsize", 0) or 0)
        return AbstractBackend._prod_shape(getattr(array, "shape", ())) * itemsize

    @staticmethod
    def _array_size(array):
        numel = getattr(array, "numel", None)
        if callable(numel):
            return int(numel())
        size = getattr(array, "size", None)
        if size is not None:
            if callable(size):
                try:
                    return int(size())
                except (TypeError, ValueError):
                    pass
            else:
                return int(size)
        return AbstractBackend._prod_shape(getattr(array, "shape", ()))

    @classmethod
    def _largest_array_memory(cls, arrays):
        largest_bytes = 0
        largest_elements = 0
        for array in arrays:
            nbytes = cls._array_nbytes(array)
            if nbytes > largest_bytes:
                largest_bytes = nbytes
                largest_elements = cls._array_size(array)
        return int(largest_bytes), int(largest_elements)

    def _elements_for_nbytes(self, nbytes, reference_array):
        itemsize = self._operand_itemsize(reference_array)
        if itemsize <= 0:
            return 0
        return int(nbytes or 0) // int(itemsize)

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
        if self.name == "torch" and hasattr(array, "permute"):
            return array.permute(*perm)
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

    def _accumulate_matmul_desc_output(self, desc, result):
        if desc.C is None:
            return result
        if desc.beta != 0.0:
            result = result + desc.beta * desc.C
        desc.C[...] = result
        return desc.C

    @staticmethod
    def _is_matmul_desc(value):
        return all(hasattr(value, attr) for attr in ("A", "B", "m", "n", "k"))

    @staticmethod
    def _is_buffer_matmul_desc(value):
        return isinstance(value, BufferMatmulDesc)

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

    @staticmethod
    def _buffer_table_array(buffers, name):
        try:
            ref = buffers[str(name)]
        except KeyError as exc:
            raise BackendFeatureError("grouped_gemm buffer {0!r} is not present in BufferTable".format(name)) from exc
        if isinstance(ref, BufferRef):
            return ref.array
        return ref

    @staticmethod
    def _profile_buffer_table(buffers):
        if buffers is None:
            return []
        if hasattr(buffers, "items"):
            items = buffers.items()
        else:
            items = ((name, buffers[name]) for name in buffers)
        profile = []
        for name, ref in sorted(items, key=lambda item: str(item[0])):
            if isinstance(ref, BufferRef):
                profile.append({
                    "name": str(ref.name),
                    "shape": tuple(int(dim) for dim in ref.shape),
                    "dtype": str(getattr(ref.dtype, "name", ref.dtype)),
                    "device": str(ref.device),
                    "nbytes": int(ref.nbytes),
                })
            else:
                array = ref
                profile.append({
                    "name": str(name),
                    "shape": tuple(int(dim) for dim in getattr(array, "shape", ())),
                    "dtype": str(getattr(array, "dtype", None)),
                    "device": "unknown",
                    "nbytes": int(array_nbytes(array)),
                })
        return profile

    @staticmethod
    def _profile_buffer_slice(buffer_slice):
        if buffer_slice is None:
            return None
        return {
            "buffer": str(buffer_slice.buffer),
            "offset": int(buffer_slice.offset),
            "shape": [int(dim) for dim in buffer_slice.shape],
            "leading_dim": None if buffer_slice.leading_dim is None else int(buffer_slice.leading_dim),
        }

    @classmethod
    def _profile_buffer_matmul_desc(cls, desc, index):
        return {
            "index": int(index),
            "A_slice": cls._profile_buffer_slice(desc.A),
            "B_slice": cls._profile_buffer_slice(desc.B),
            "C_slice": cls._profile_buffer_slice(desc.C),
            "trans_a": str(desc.trans_a).upper(),
            "trans_b": str(desc.trans_b).upper(),
            "m": int(desc.m),
            "n": int(desc.n),
            "k": int(desc.k),
            "lda": int(desc.lda),
            "ldb": int(desc.ldb),
            "ldc": int(desc.ldc),
            "alpha": cls._profile_scalar(desc.alpha),
            "beta": cls._profile_scalar(desc.beta),
            "dtype": str(getattr(desc.dtype, "name", desc.dtype)),
            "tag": cls._profile_scalar(desc.tag),
        }

    @classmethod
    def _profile_buffer_matmul_descs(cls, raw_tasks):
        return [
            cls._profile_buffer_matmul_desc(task, index)
            for index, task in enumerate(raw_tasks)
            if cls._is_buffer_matmul_desc(task)
        ]

    @staticmethod
    def _is_buffer_gemv_desc(value):
        return isinstance(value, GemvDesc)

    @classmethod
    def _profile_buffer_gemv_desc(cls, desc, index):
        return {
            "index": int(index),
            "A_slice": cls._profile_buffer_slice(desc.A),
            "x_slice": cls._profile_buffer_slice(desc.x),
            "y_slice": cls._profile_buffer_slice(desc.y),
            "trans_a": str(desc.trans_a).upper(),
            "conj_a": str(desc.trans_a).upper() == "C",
            "m": int(desc.m),
            "n": int(desc.n),
            "lda": int(desc.lda),
            "incx": int(desc.incx),
            "incy": int(desc.incy),
            "alpha": cls._profile_scalar(desc.alpha),
            "beta": cls._profile_scalar(desc.beta),
            "dtype": str(getattr(desc.dtype, "name", desc.dtype)),
            "tag": cls._profile_scalar(desc.tag),
        }

    @classmethod
    def _profile_buffer_gemv_descs(cls, descs):
        return [
            cls._profile_buffer_gemv_desc(desc, index)
            for index, desc in enumerate(descs)
            if cls._is_buffer_gemv_desc(desc)
        ]

    def _buffer_slice_view(self, buffer_slice, buffers):
        xp = self.array_namespace or _np
        array = self._buffer_table_array(buffers, buffer_slice.buffer)
        flat = xp.reshape(array, (-1,))
        shape = tuple(int(dim) for dim in buffer_slice.shape)
        offset = int(buffer_slice.offset)
        leading_dim = buffer_slice.leading_dim
        if offset < 0:
            raise BackendFeatureError("BufferSlice offset must be non-negative")
        flat_size = int(getattr(flat, "shape", (0,))[0])

        def validate_range(required_length):
            required_length = int(required_length)
            end = offset + required_length
            if offset > flat_size or end > flat_size:
                raise BackendFeatureError(
                    "BufferSlice {0!r} range [{1}, {2}) exceeds buffer size {3}".format(
                        buffer_slice.buffer,
                        offset,
                        end,
                        flat_size,
                    )
                )

        if not shape:
            validate_range(1)
            return flat[offset]
        if leading_dim is None or len(shape) < 2:
            length = self._prod_shape(shape)
            validate_range(length)
            return xp.reshape(flat[offset:offset + length], shape)
        leading_dim = int(leading_dim)
        logical_cols = int(shape[-1])
        if leading_dim < logical_cols:
            raise BackendFeatureError(
                "BufferSlice leading_dim {0} is smaller than logical last dimension {1}".format(
                    leading_dim,
                    logical_cols,
                )
            )
        if leading_dim == logical_cols:
            length = self._prod_shape(shape)
            validate_range(length)
            return xp.reshape(flat[offset:offset + length], shape)
        row_count = self._prod_shape(shape[:-1])
        padded_shape = tuple(shape[:-1]) + (leading_dim,)
        padded_length = row_count * leading_dim
        validate_range(padded_length)
        padded = xp.reshape(flat[offset:offset + padded_length], padded_shape)
        return padded[..., :logical_cols]

    def _gemv_batch_from_input(self, descs, buffers):
        if isinstance(descs, GemvBatch):
            gemv_batch = descs
            raw_descs = tuple(descs.descs)
        else:
            raw_descs = tuple(descs)
            gemv_batch = GemvBatch.from_descs(raw_descs)
        if buffers is None and raw_descs:
            raise BackendFeatureError("gemv_batch BufferSlice descriptors require buffers=")
        for desc in raw_descs:
            if not self._is_buffer_gemv_desc(desc):
                raise BackendFeatureError("gemv_batch expects GemvDesc descriptors")
        return gemv_batch, raw_descs

    def _buffer_gemv_vector_view(self, vector, *, increment, logical_length, name):
        xp = self.array_namespace or _np
        increment = int(increment)
        logical_length = int(logical_length)
        if logical_length < 0:
            raise BackendFeatureError("GemvDesc {0} logical length must be non-negative".format(name))
        flat = xp.reshape(vector, (-1,))
        if logical_length == 0:
            return flat[:0]
        required = 1 + (logical_length - 1) * increment
        flat_len = int(getattr(flat, "shape", (0,))[0])
        if flat_len < required:
            raise BackendFeatureError(
                "GemvDesc {0} slice has length {1}, but increment {2} requires {3} elements".format(
                    name,
                    flat_len,
                    increment,
                    required,
                )
            )
        return flat[:required:increment]

    def _execute_buffer_gemv_desc(self, desc, buffers):
        xp = self.array_namespace or _np
        A = self._buffer_slice_view(desc.A, buffers)
        x_raw = self._buffer_slice_view(desc.x, buffers)
        y_raw = self._buffer_slice_view(desc.y, buffers)
        a_shape = tuple(int(dim) for dim in getattr(A, "shape", ()))
        expected_a = (int(desc.m), int(desc.n))
        if a_shape != expected_a:
            raise BackendFeatureError(
                "GemvDesc dimensions do not match BufferSlice shape: expected A{0}; got A{1}".format(
                    expected_a,
                    a_shape,
                )
            )
        trans_a = str(desc.trans_a).upper()
        if trans_a == "N":
            x_len = int(desc.n)
            y_len = int(desc.m)
            op_a = A
        else:
            x_len = int(desc.m)
            y_len = int(desc.n)
            op_a = xp.swapaxes(A, -1, -2)
            if trans_a == "C":
                op_a = xp.conj(op_a)
        x = self._buffer_gemv_vector_view(
            x_raw,
            increment=desc.incx,
            logical_length=x_len,
            name="x",
        )
        y = self._buffer_gemv_vector_view(
            y_raw,
            increment=desc.incy,
            logical_length=y_len,
            name="y",
        )
        result = xp.matmul(op_a, x)
        if desc.alpha != 1.0:
            result = desc.alpha * result
        if desc.beta != 0.0:
            result = result + desc.beta * y
        y[...] = result
        return y, A, x_raw, y_raw

    @staticmethod
    def _gemv_group_key_buckets(gemv_batch):
        items = []
        for group_key, sorted_group_indices in gemv_batch.groups.items():
            item = group_key.to_dict()
            item["task_indices"] = [
                int(gemv_batch.sorted_indices[index])
                for index in sorted_group_indices
            ]
            item["task_count"] = int(len(sorted_group_indices))
            items.append(item)
        return items

    def gemv_batch(
        self,
        descs,
        *,
        buffers=None,
        role=None,
        stream=None,
        workspace=None,
        fallback_policy=None,
        profile_context=None,
    ):
        import time

        self._resolve_fallback_policy(fallback_policy)
        self._validate_workspace(workspace, required_bytes=0)
        gemv_batch, _raw_descs = self._gemv_batch_from_input(descs, buffers)
        buffer_names = sorted(str(name) for name in buffers) if buffers is not None else []
        buffer_table = self._profile_buffer_table(buffers)
        gemv_descriptors = self._profile_buffer_gemv_descs(gemv_batch.descs)
        started = time.perf_counter()
        results = []
        read_bytes = 0
        write_bytes = 0
        with self._stream_context(stream):
            for desc in gemv_batch.descs:
                result, A, x_raw, y_raw = self._execute_buffer_gemv_desc(desc, buffers)
                results.append(result)
                read_bytes += int(array_nbytes(A)) + int(array_nbytes(x_raw))
                if desc.beta != 0.0:
                    read_bytes += int(array_nbytes(y_raw))
                write_bytes += int(array_nbytes(result))
        wall_s = time.perf_counter() - started

        try:
            from renormalizer.utils import profiling

            output_shape = [tuple(int(dim) for dim in getattr(item, "shape", ())) for item in results]
            context_payload = dict(profile_context or {})
            context_payload.pop("event", None)
            payload = {
                "event": "gemv_batch_execute",
                "backend": self.name,
                **profiling.compute_payload(
                    profiling.COMPUTE_CLASS_CONTRACTION_PLAN,
                    "gemv_batch_execute",
                    profiling.COMPUTE_ROLE_KERNEL,
                ),
                "lowering": "gemv",
                "role": None if role is None else str(role),
                "descriptor_source": "focus_buffer_slice",
                "buffer_names": buffer_names,
                "buffer_table": buffer_table,
                "gemv_descriptors": gemv_descriptors,
                **self._profile_device_execution(self.current_device()),
                **self._profile_execution_resources(stream=stream, workspace=workspace),
                "num_tasks": int(len(gemv_batch.descs)),
                "num_groups": int(len(gemv_batch.groups)),
                "group_sizes": [int(size) for size in gemv_batch.group_sizes],
                "group_keys": [key.to_dict() for key in gemv_batch.groups],
                "group_key_buckets": self._gemv_group_key_buckets(gemv_batch),
                "gsta": [int(index) for index in gemv_batch.gsta],
                "sorted_indices": [int(index) for index in gemv_batch.sorted_indices],
                "total_flops": int(gemv_batch.total_flops),
                "flops": int(gemv_batch.total_flops),
                "read_bytes": int(read_bytes),
                "write_bytes": int(write_bytes),
                "copy_bytes": 0,
                "workspace_bytes": 0,
                "peak_bytes": int(write_bytes),
                "largest_intermediate": max((array_nbytes(item) for item in results), default=0),
                "largest_intermediate_elements": max((self._array_size(item) for item in results), default=0),
                "largest_intermediate_bytes": max((array_nbytes(item) for item in results), default=0),
                "max_m": int(gemv_batch.max_m),
                "max_n": int(gemv_batch.max_n),
                "output_shape": output_shape,
                "output_strides": [profiling.array_strides(item) for item in results],
                "output_order": [profiling.array_order(item) for item in results],
                "output_contiguous": [profiling.array_contiguous(item) for item in results],
                "output_backend": [profiling.array_backend_name(item) for item in results],
                "output_device_kind": [profiling.array_device_kind(item) for item in results],
                "output_location": [profiling.array_location(item) for item in results],
                "output_is_host": [profiling.array_is_host(item) for item in results],
                "output_is_device": [profiling.array_is_device(item) for item in results],
                "output_is_distributed": [profiling.array_is_distributed(item) for item in results],
                "execution_primitives": ["gemv"] if results else [],
                "execution_policies": ["loop_gemv"] if results else [],
                "fallback_reasons": [],
                "fallback_reason": None,
                "fallback_from": None,
                "fallback_to": None,
                "fallback_policy": None,
                "wall_s": float(wall_s),
                **context_payload,
            }
            payload = profiling.standardize_event_payload(payload)
            self._last_execution_profile = dict(payload)
            if profiling.should_record_op():
                record_payload = dict(payload)
                record_payload.pop("event", None)
                profiling.record("gemv_batch_execute", **record_payload)
        except Exception:
            self._last_execution_profile = {
                "event": "gemv_batch_execute",
                "backend": self.name,
                "lowering": "gemv",
                "role": None if role is None else str(role),
                "num_tasks": int(len(gemv_batch.descs)),
                "num_groups": int(len(gemv_batch.groups)),
                "group_sizes": [int(size) for size in gemv_batch.group_sizes],
                "total_flops": int(gemv_batch.total_flops),
                "wall_s": float(wall_s),
            }
        return results

    def _buffer_matmul_desc_to_task(self, desc, buffers):
        trans_a = str(desc.trans_a).upper()
        trans_b = str(desc.trans_b).upper()
        a = self._buffer_slice_view(desc.A, buffers)
        b = self._buffer_slice_view(desc.B, buffers)
        c = self._buffer_slice_view(desc.C, buffers)
        self._validate_buffer_matmul_desc_shapes(desc, a, b, c)
        return GemmTask(
            a,
            b,
            C=c,
            trans_a=trans_a in ("T", "C"),
            trans_b=trans_b in ("T", "C"),
            conj_a=trans_a == "C",
            conj_b=trans_b == "C",
            alpha=desc.alpha,
            beta=desc.beta,
            tag=desc.tag,
        )

    @staticmethod
    def _effective_buffer_gemm_shape(array, trans_flag):
        shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
        if len(shape) < 2:
            return shape
        if str(trans_flag).upper() in ("T", "C"):
            return shape[:-2] + (shape[-1], shape[-2])
        return shape

    def _validate_buffer_matmul_desc_shapes(self, desc, a, b, c):
        a_shape = self._effective_buffer_gemm_shape(a, desc.trans_a)
        b_shape = self._effective_buffer_gemm_shape(b, desc.trans_b)
        c_shape = tuple(int(dim) for dim in getattr(c, "shape", ()))
        expected_a = (int(desc.m), int(desc.k))
        expected_b = (int(desc.k), int(desc.n))
        expected_c = (int(desc.m), int(desc.n))
        if a_shape != expected_a or b_shape != expected_b or c_shape != expected_c:
            raise BackendFeatureError(
                "MatmulDesc dimensions do not match BufferSlice shapes: "
                "expected A{0}, B{1}, C{2}; got A{3}, B{4}, C{5}".format(
                    expected_a,
                    expected_b,
                    expected_c,
                    a_shape,
                    b_shape,
                    c_shape,
                )
            )

    @staticmethod
    def _effective_raw_matmul_shape(array, trans=False):
        shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
        if trans and len(shape) >= 2:
            return shape[:-2] + (shape[-1], shape[-2])
        return shape

    @classmethod
    def _raw_matmul_flops(cls, a_shape, output_shape):
        if len(a_shape) < 2 or len(output_shape) < 2:
            return 0
        batch = cls._prod_shape(output_shape[:-2]) if len(output_shape) > 2 else 1
        return int(2 * batch * int(output_shape[-2]) * int(output_shape[-1]) * int(a_shape[-1]))

    @contextlib.contextmanager
    def _suppress_primitive_profile(self):
        depth = int(getattr(self, "_primitive_profile_suppression_depth", 0) or 0)
        self._primitive_profile_suppression_depth = depth + 1
        try:
            yield
        finally:
            if depth:
                self._primitive_profile_suppression_depth = depth
            else:
                try:
                    delattr(self, "_primitive_profile_suppression_depth")
                except AttributeError:
                    pass

    def _primitive_profile_suppressed(self):
        return bool(getattr(self, "_primitive_profile_suppression_depth", 0))

    def _record_raw_matmul_profile(
        self,
        lowering,
        A,
        B,
        result,
        wall_s,
        *,
        C=None,
        trans_a=False,
        trans_b=False,
        stream=None,
        workspace=None,
        fallback_reason=None,
        fallback_from=None,
        fallback_to=None,
        fallback_policy=None,
    ):
        a_shape = self._effective_raw_matmul_shape(A, trans_a)
        output_shape = tuple(int(dim) for dim in getattr(result, "shape", ()))
        write_bytes = self._array_nbytes(result)
        read_bytes = self._array_nbytes(A) + self._array_nbytes(B)
        if C is not None:
            read_bytes += self._array_nbytes(C)
        from renormalizer.utils import profiling

        payload = {
            "event": "matmul_execute",
            "backend": self.name,
            **profiling.compute_payload(
                profiling.COMPUTE_CLASS_CONTRACTION_PLAN,
                "matmul_execute",
                profiling.COMPUTE_ROLE_KERNEL,
            ),
            "lowering": str(lowering),
            "input_shapes": [
                tuple(int(dim) for dim in getattr(A, "shape", ())),
                tuple(int(dim) for dim in getattr(B, "shape", ())),
            ],
            "input_strides": [
                profiling.array_strides(A),
                profiling.array_strides(B),
            ],
            "input_orders": [
                profiling.array_order(A),
                profiling.array_order(B),
            ],
            "input_contiguous": [
                profiling.array_contiguous(A),
                profiling.array_contiguous(B),
            ],
            "input_dtypes": [
                str(getattr(A, "dtype", None)),
                str(getattr(B, "dtype", None)),
            ],
            "input_backends": [
                profiling.array_backend_name(A),
                profiling.array_backend_name(B),
            ],
            "input_device_kinds": [
                profiling.array_device_kind(A),
                profiling.array_device_kind(B),
            ],
            "input_locations": [
                profiling.array_location(A),
                profiling.array_location(B),
            ],
            "output_shape": output_shape,
            "output_strides": profiling.array_strides(result),
            "output_order": profiling.array_order(result),
            "output_contiguous": profiling.array_contiguous(result),
            "output_backend": profiling.array_backend_name(result),
            "output_device_kind": profiling.array_device_kind(result),
            "output_location": profiling.array_location(result),
            "output_is_host": profiling.array_is_host(result),
            "output_is_device": profiling.array_is_device(result),
            "output_is_distributed": profiling.array_is_distributed(result),
            "dtype": str(getattr(result, "dtype", None)),
            "output_dtype": str(getattr(result, "dtype", None)),
            **self._profile_device_execution(self.current_device()),
            **self._profile_execution_resources(stream=stream, workspace=workspace),
            "flops": self._raw_matmul_flops(a_shape, output_shape),
            "read_bytes": int(read_bytes),
            "write_bytes": int(write_bytes),
            "copy_bytes": 0,
            "workspace_bytes": 0,
            "peak_bytes": int(write_bytes),
            "largest_intermediate": int(write_bytes),
            "largest_intermediate_elements": self._array_size(result),
            "largest_intermediate_bytes": int(write_bytes),
            "num_gemm": 1 if str(lowering) == "gemm" else 0,
            "num_batched_gemm": 1 if str(lowering) in ("batched_gemm", "strided_batched_gemm") else 0,
            "num_grouped_tasks": 0,
            "num_blocks": 0,
            "num_shape_buckets": 0,
            "fallback_reason": fallback_reason,
            "fallback_from": fallback_from,
            "fallback_to": fallback_to,
            "fallback_policy": fallback_policy,
            "wall_s": float(wall_s),
        }
        payload = profiling.standardize_event_payload(payload)
        self._last_execution_profile = dict(payload)
        if profiling.should_record_op() and not self._primitive_profile_suppressed():
            record_payload = dict(payload)
            record_payload.pop("event")
            profiling.record("matmul_execute", **record_payload)

    def _grouped_gemm_tasks_from_input(self, tasks, buffers):
        if isinstance(tasks, GemmBatch):
            raw_tasks = tuple(tasks.descs)
            source = "focus_buffer_slice" if any(self._is_buffer_matmul_desc(task) for task in raw_tasks) else "gemm_batch"
        else:
            raw_tasks = tuple(tasks)
            source = "focus_buffer_slice" if any(self._is_buffer_matmul_desc(task) for task in raw_tasks) else "gemm_task"
        if buffers is None and any(self._is_buffer_matmul_desc(task) for task in raw_tasks):
            raise BackendFeatureError("grouped_gemm BufferSlice descriptors require buffers=")
        converted = []
        for task in raw_tasks:
            if self._is_buffer_matmul_desc(task):
                converted.append(self._buffer_matmul_desc_to_task(task, buffers))
            elif self._is_matmul_desc(task):
                converted.append(self._desc_to_task(task))
            else:
                converted.append(task)
        return converted, source, raw_tasks

    def _handle_direct_primitive_fallback(self, reason, *, fallback_policy=None):
        fallback_policy = self._resolve_fallback_policy(fallback_policy)
        if fallback_policy is FallbackPolicy.FORBID:
            raise BackendFeatureError(reason)
        if fallback_policy is FallbackPolicy.WARN:
            import warnings

            warnings.warn(reason, RuntimeWarning, stacklevel=3)
        return fallback_policy

    def _execute_matmul_desc(self, desc, *, stream=None, workspace=None, fallback_policy=None):
        if desc.batch_shape:
            raise BackendFeatureError("matmul received a batched descriptor; use batched_matmul")
        if not self.supports_matmul:
            import time

            reason = "backend lacks matmul"
            policy = self._handle_direct_primitive_fallback(reason, fallback_policy=fallback_policy)
            started = time.perf_counter()
            result = self._tensor_contract_matmul_fallback(desc, stream=stream, workspace=workspace)
            self._record_raw_matmul_profile(
                "fallback_tensordot",
                desc.A,
                desc.B,
                result,
                time.perf_counter() - started,
                C=desc.C,
                trans_a=desc.trans_a,
                trans_b=desc.trans_b,
                stream=stream,
                workspace=workspace,
                fallback_reason=reason,
                fallback_from="gemm",
                fallback_to="tensordot",
                fallback_policy=policy.value,
            )
            return result
        a, b, groups = self._prepare_matmul_desc(desc)
        result = self.matmul(
            a,
            b,
            C=None,
            trans_a=desc.trans_a,
            trans_b=desc.trans_b,
            conj_a=desc.conj_a,
            conj_b=desc.conj_b,
            alpha=desc.alpha,
            beta=0.0,
            stream=stream,
            workspace=workspace,
            fallback_policy=fallback_policy,
        )
        result = self._finalize_matmul_result(result, groups)
        return self._accumulate_matmul_desc_output(desc, result)

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
        fallback_policy=None,
    ):
        self._validate_workspace(workspace, required_bytes=0)
        if B is None and self._is_matmul_desc(A):
            return self._execute_matmul_desc(
                A,
                stream=stream,
                workspace=workspace,
                fallback_policy=fallback_policy,
            )
        xp = self.array_namespace or _np
        import time

        if not self.supports_matmul:
            reason = "backend lacks matmul"
            policy = self._handle_direct_primitive_fallback(reason, fallback_policy=fallback_policy)
            started = time.perf_counter()
            result = self._raw_tensor_contract_matmul_fallback(
                A,
                B,
                C=C,
                trans_a=trans_a,
                trans_b=trans_b,
                conj_a=conj_a,
                conj_b=conj_b,
                alpha=alpha,
                beta=beta,
                stream=stream,
                workspace=workspace,
            )
            self._record_raw_matmul_profile(
                "fallback_tensordot",
                A,
                B,
                result,
                time.perf_counter() - started,
                C=C,
                trans_a=trans_a,
                trans_b=trans_b,
                stream=stream,
                workspace=workspace,
                fallback_reason=reason,
                fallback_from="gemm",
                fallback_to="tensordot",
                fallback_policy=policy.value,
            )
            return result

        self._resolve_fallback_policy(fallback_policy)
        started = time.perf_counter()
        with self._stream_context(stream):
            result = run_gemm_task(
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
        self._record_raw_matmul_profile(
            "gemm",
            A,
            B,
            result,
            time.perf_counter() - started,
            C=C,
            trans_a=trans_a,
            trans_b=trans_b,
            stream=stream,
            workspace=workspace,
        )
        return result

    def _raw_tensor_contract_matmul_fallback(
        self,
        A,
        B,
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
        xp = self.array_namespace or _np
        with self._stream_context(stream):
            if conj_a:
                A = xp.conj(A)
            if conj_b:
                B = xp.conj(B)
            if trans_a:
                A = xp.swapaxes(A, -1, -2)
            if trans_b:
                B = xp.swapaxes(B, -1, -2)
            if len(getattr(A, "shape", ())) > 2 or len(getattr(B, "shape", ())) > 2:
                batch_ndim = max(
                    len(getattr(A, "shape", ())) - 2,
                    len(getattr(B, "shape", ())) - 2,
                )
                result = self._execute_einsum(self._compact_matmul_einsum(batch_ndim), (A, B))
            else:
                tensordot = getattr(xp, "tensordot", None)
                if tensordot is None:
                    result = self._execute_einsum("mk,kn->mn", (A, B))
                else:
                    result = tensordot(A, B, axes=([-1], [-2]))
            if alpha != 1.0:
                result = alpha * result
            if C is not None:
                if beta != 0.0:
                    result = result + beta * C
                C[...] = result
                result = C
        return result

    def _raw_loop_batched_matmul(
        self,
        A,
        B,
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
        xp = self.array_namespace or _np
        with self._stream_context(stream):
            if conj_a:
                A = xp.conj(A)
            if conj_b:
                B = xp.conj(B)
            if trans_a:
                A = xp.swapaxes(A, -1, -2)
            if trans_b:
                B = xp.swapaxes(B, -1, -2)
            a_shape = tuple(getattr(A, "shape", ()))
            if len(a_shape) <= 2:
                result = xp.matmul(A, B)
            else:
                batch_shape = a_shape[:-2]
                flat_batch = int(_np.prod(batch_shape, dtype=_np.int64))
                a_flat = xp.reshape(A, (flat_batch,) + a_shape[-2:])
                b_shape = tuple(getattr(B, "shape", ()))
                b_flat = xp.reshape(B, (flat_batch,) + b_shape[-2:])
                pieces = [xp.matmul(a_flat[index], b_flat[index]) for index in range(flat_batch)]
                result = xp.stack(pieces, axis=0)
                result = xp.reshape(result, batch_shape + tuple(getattr(result, "shape", ())[-2:]))
            if alpha != 1.0:
                result = alpha * result
            if C is not None:
                if beta != 0.0:
                    result = result + beta * C
                C[...] = result
                result = C
        return result

    def _execute_batched_matmul_desc(self, desc, *, stream=None, workspace=None, fallback_policy=None):
        if not self.supports_batched_matmul:
            import time

            reason = "backend lacks batched_matmul for batch shape {0}".format(tuple(desc.batch_shape))
            policy = self._handle_direct_primitive_fallback(
                reason,
                fallback_policy=fallback_policy,
            )
            started = time.perf_counter()
            if not self.supports_matmul:
                result = self._tensor_contract_matmul_fallback(desc, stream=stream, workspace=workspace)
                fallback_to = "tensordot"
            else:
                result = self._loop_matmul(desc, stream=stream, workspace=workspace)
                fallback_to = "loop_matmul"
            self._record_raw_matmul_profile(
                "fallback_tensordot",
                desc.A,
                desc.B,
                result,
                time.perf_counter() - started,
                C=desc.C,
                trans_a=desc.trans_a,
                trans_b=desc.trans_b,
                stream=stream,
                workspace=workspace,
                fallback_reason=reason,
                fallback_from="batched_gemm",
                fallback_to=fallback_to,
                fallback_policy=policy.value,
            )
            return result
        a, b, groups = self._prepare_matmul_desc(desc)
        result = self.batched_matmul(
            a,
            b,
            C=None,
            trans_a=desc.trans_a,
            trans_b=desc.trans_b,
            conj_a=desc.conj_a,
            conj_b=desc.conj_b,
            alpha=desc.alpha,
            beta=0.0,
            stream=stream,
            workspace=workspace,
            fallback_policy=fallback_policy,
        )
        result = self._finalize_matmul_result(result, groups)
        return self._accumulate_matmul_desc_output(desc, result)

    def _execute_grouped_matmul_descs(
        self,
        descs,
        *,
        stream=None,
        workspace=None,
        pack_threshold=None,
        fallback_policy=None,
    ):
        tasks = []
        groups_by_desc = []
        for desc in descs:
            a, b, groups = self._prepare_matmul_desc(desc)
            tasks.append(GemmTask(
                a,
                b,
                C=None,
                trans_a=desc.trans_a,
                trans_b=desc.trans_b,
                conj_a=desc.conj_a,
                conj_b=desc.conj_b,
                alpha=desc.alpha,
                beta=0.0,
            ))
            groups_by_desc.append((desc, groups))
        threshold = 4 if pack_threshold is None else int(pack_threshold)
        packed_results = self.grouped_gemm(
            tasks,
            pack_threshold=threshold,
            stream=stream,
            workspace=workspace,
            fallback_policy=fallback_policy,
        )
        results = []
        for packed_result, (desc, groups) in zip(packed_results, groups_by_desc):
            result = self._finalize_matmul_result(packed_result, groups)
            results.append(self._accumulate_matmul_desc_output(desc, result))
        return results

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
        fallback_policy=None,
    ):
        self._validate_workspace(workspace, required_bytes=0)
        if B is None and self._is_matmul_desc(A):
            return self._execute_batched_matmul_desc(
                A,
                stream=stream,
                workspace=workspace,
                fallback_policy=fallback_policy,
            )
        xp = self.array_namespace or _np
        import time

        raw_a = A
        raw_b = B
        if not self.supports_batched_matmul:
            batch_shape = tuple(int(dim) for dim in getattr(A, "shape", ())[:-2])
            reason = "backend lacks batched_matmul for batch shape {0}".format(batch_shape)
            policy = self._handle_direct_primitive_fallback(
                reason,
                fallback_policy=fallback_policy,
            )
            started = time.perf_counter()
            if self.supports_matmul:
                result = self._raw_loop_batched_matmul(
                    A,
                    B,
                    C=C,
                    trans_a=trans_a,
                    trans_b=trans_b,
                    conj_a=conj_a,
                    conj_b=conj_b,
                    alpha=alpha,
                    beta=beta,
                    stream=stream,
                    workspace=workspace,
                )
                fallback_to = "loop_matmul"
            else:
                result = self._raw_tensor_contract_matmul_fallback(
                    A,
                    B,
                    C=C,
                    trans_a=trans_a,
                    trans_b=trans_b,
                    conj_a=conj_a,
                    conj_b=conj_b,
                    alpha=alpha,
                    beta=beta,
                    stream=stream,
                    workspace=workspace,
                )
                fallback_to = "tensordot"
            self._record_raw_matmul_profile(
                "fallback_tensordot",
                raw_a,
                raw_b,
                result,
                time.perf_counter() - started,
                C=C,
                trans_a=trans_a,
                trans_b=trans_b,
                stream=stream,
                workspace=workspace,
                fallback_reason=reason,
                fallback_from="batched_gemm",
                fallback_to=fallback_to,
                fallback_policy=policy.value,
            )
            return result

        self._resolve_fallback_policy(fallback_policy)
        started = time.perf_counter()
        with self._stream_context(stream):
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
                result = C
        self._record_raw_matmul_profile(
            "batched_gemm",
            raw_a,
            raw_b,
            result,
            time.perf_counter() - started,
            C=C,
            trans_a=trans_a,
            trans_b=trans_b,
            stream=stream,
            workspace=workspace,
        )
        return result

    def _loop_matmul(self, desc, *, stream=None, workspace=None):
        xp = self.array_namespace or _np
        a, b, groups = self._prepare_matmul_desc(desc)
        with self._stream_context(stream):
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
        result = self._finalize_matmul_result(result, groups)
        return self._accumulate_matmul_desc_output(desc, result)

    @staticmethod
    def _compact_matmul_einsum(batch_ndim):
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        batch_ndim = int(batch_ndim)
        if batch_ndim + 3 > len(labels):
            raise BackendFeatureError(
                "cannot synthesize fallback einsum equation for {0} batch dimensions".format(batch_ndim)
            )
        batch = labels[:batch_ndim]
        m_label = labels[batch_ndim]
        k_label = labels[batch_ndim + 1]
        n_label = labels[batch_ndim + 2]
        return "{0}{1}{2},{0}{2}{3}->{0}{1}{3}".format(batch, m_label, k_label, n_label)

    def _tensor_contract_matmul_fallback(self, desc, *, stream=None, workspace=None):
        xp = self.array_namespace or _np
        a, b, groups = self._prepare_matmul_desc(desc)
        with self._stream_context(stream):
            if desc.conj_a:
                a = xp.conj(a)
            if desc.conj_b:
                b = xp.conj(b)
            if desc.trans_a:
                a = xp.swapaxes(a, -1, -2)
            if desc.trans_b:
                b = xp.swapaxes(b, -1, -2)
            if desc.batch_shape:
                result = self._execute_einsum(self._compact_matmul_einsum(len(desc.batch_shape)), (a, b))
            else:
                tensordot = getattr(xp, "tensordot", None)
                if tensordot is None:
                    result = self._execute_einsum("mk,kn->mn", (a, b))
                else:
                    result = tensordot(a, b, axes=([-1], [-2]))
            if desc.alpha != 1.0:
                result = desc.alpha * result
        result = self._finalize_matmul_result(result, groups)
        return self._accumulate_matmul_desc_output(desc, result)

    def grouped_gemm(
        self,
        tasks,
        *,
        buffers=None,
        pack_threshold=4,
        stream=None,
        workspace=None,
        policy="auto",
        fallback_policy=None,
        profile_context=None,
    ):
        import time

        self._validate_workspace(workspace, required_bytes=0)
        xp = self.array_namespace or _np
        converted, descriptor_source, raw_tasks = self._grouped_gemm_tasks_from_input(tasks, buffers)
        buffer_names = sorted(str(name) for name in buffers) if buffers is not None else []
        buffer_table = self._profile_buffer_table(buffers)
        buffer_slice_descriptors = self._profile_buffer_matmul_descs(raw_tasks)
        execution_policy, fallback_policy = self._resolve_grouped_gemm_call_policies(
            policy,
            fallback_policy,
        )
        fallback_reason = self._handle_grouped_gemm_fallback(len(converted), policy=fallback_policy)
        flop_copy_ratio = 0 if self.supports_grouped_gemm else 10
        allow_batched = bool(self.supports_grouped_gemm or self.supports_batched_matmul)
        effective_pack_threshold, effective_flop_copy_ratio, effective_allow_batched = (
            self._grouped_gemm_execution_policy_controls(
                execution_policy,
                pack_threshold=pack_threshold,
                flop_copy_ratio=flop_copy_ratio,
                allow_batched=allow_batched,
            )
        )
        stats = grouped_gemm_stats(
            converted,
            xp=xp,
            pack_threshold=effective_pack_threshold,
            flop_copy_ratio=effective_flop_copy_ratio,
            allow_batched=effective_allow_batched,
        )
        batched_matmul = None
        if self.supports_batched_matmul:
            batched_matmul = lambda a, b: self.batched_matmul(
                a,
                b,
                stream=stream,
                workspace=workspace,
            )
        batched_matmul_provider = (
            self._batched_matmul_provider_name()
            if batched_matmul is not None
            else None
        )
        try:
            from renormalizer.utils import profiling

            should_profile = profiling.should_record_op()
        except Exception:
            profiling = None
            should_profile = False

        started = time.perf_counter()
        pointer_setup_started = time.perf_counter()
        gemm_batch = GemmBatch.from_tasks(converted, xp=xp)
        pointer_setup_s = time.perf_counter() - pointer_setup_started
        compute_started = time.perf_counter()
        with self._stream_context(stream):
            with self._suppress_primitive_profile():
                result, bucket_execution_profile = grouped_gemm_bucketed_profiled(
                    converted,
                    xp=xp,
                    pack_threshold=effective_pack_threshold,
                    flop_copy_ratio=effective_flop_copy_ratio,
                    allow_batched=effective_allow_batched,
                    batched_matmul=batched_matmul,
                    batched_matmul_provider=batched_matmul_provider,
                )
        compute_s = time.perf_counter() - compute_started
        wall_s = time.perf_counter() - started
        grouped_gemm_policy = self._grouped_gemm_execution_policy(stats, fallback_reason)
        context_payload = dict(profile_context or {})
        context_payload.pop("event", None)
        if should_profile:
            output_shape = [tuple(getattr(item, "shape", ())) for item in result]
            output_strides = [profiling.array_strides(item) for item in result]
            output_order = [profiling.array_order(item) for item in result]
            output_contiguous = [profiling.array_contiguous(item) for item in result]
            output_backend = [profiling.array_backend_name(item) for item in result]
            output_device_kind = [profiling.array_device_kind(item) for item in result]
            output_location = [profiling.array_location(item) for item in result]
            output_is_host = [profiling.array_is_host(item) for item in result]
            output_is_device = [profiling.array_is_device(item) for item in result]
            output_is_distributed = [profiling.array_is_distributed(item) for item in result]
        else:
            output_shape = [tuple(getattr(item, "shape", ())) for item in result]
            output_strides = None
            output_order = None
            output_contiguous = None
            output_backend = None
            output_device_kind = None
            output_location = None
            output_is_host = None
            output_is_device = None
            output_is_distributed = None

        group_key_buckets = self._grouped_gemm_execute_group_key_buckets(gemm_batch)
        if should_profile:
            group_keys = [key.to_dict() for key in gemm_batch.groups]
            group_descriptors = self._grouped_gemm_group_descriptors(
                gemm_batch,
                bucket_execution_profile.bucket_execution_profiles,
            )
            shape_buckets = self._grouped_gemm_execute_shape_buckets(gemm_batch, stats)
        else:
            group_keys = [key.to_dict() for key in gemm_batch.groups]
            group_descriptors = None
            shape_buckets = [dict(bucket) for bucket in stats.shape_buckets]

        execute_payload = {
            "event": "grouped_gemm_execute",
            "backend": self.name,
            **self._profile_device_execution(self.current_device()),
            **self._profile_execution_resources(stream=stream, workspace=workspace),
            "descriptor_source": descriptor_source,
            "buffer_names": buffer_names,
            "buffer_table": buffer_table,
            "buffer_slice_descriptors": buffer_slice_descriptors,
            "requested_policy": execution_policy,
            "policy": grouped_gemm_policy,
            "supports_grouped_gemm": bool(self.supports_grouped_gemm),
            "grouped_gemm_implementation": self._grouped_gemm_implementation(stats, fallback_reason),
            **self._grouped_gemm_execution_items(grouped_gemm_policy, fallback_reason),
            "requires_grouped_gemm_fallback": fallback_reason is not None,
            "num_tasks": len(gemm_batch.descs),
            "num_groups": len(gemm_batch.groups),
            "group_sizes": list(gemm_batch.group_sizes),
            "group_keys": group_keys,
            "group_key_buckets": group_key_buckets,
            "shape_buckets": shape_buckets,
            "gsta": list(gemm_batch.gsta),
            "sorted_indices": list(gemm_batch.sorted_indices),
            "total_flops": gemm_batch.total_flops,
            "max_m": gemm_batch.max_m,
            "max_n": gemm_batch.max_n,
            "max_k": gemm_batch.max_k,
            "read_bytes": stats.read_bytes,
            "write_bytes": stats.write_bytes,
            "copy_bytes": stats.copy_bytes,
            "workspace_bytes": stats.workspace_bytes,
            "output_shape": output_shape,
            "output_strides": output_strides,
            "output_order": output_order,
            "output_contiguous": output_contiguous,
            "output_backend": output_backend,
            "output_device_kind": output_device_kind,
            "output_location": output_location,
            "output_is_host": output_is_host,
            "output_is_device": output_is_device,
            "output_is_distributed": output_is_distributed,
            "num_batched_gemm": stats.batched_bucket_count,
            "num_gemm": stats.loop_task_count,
            "batched_matmul_provider": batched_matmul_provider if stats.batched_bucket_count else None,
            "pointer_setup_s": pointer_setup_s,
            "compute_s": compute_s,
            "wall_s": wall_s,
            **bucket_execution_profile.to_dict(),
            "fallback_from": "grouped_gemm" if fallback_reason is not None else None,
            "fallback_reason": fallback_reason,
            "fallback_policy": fallback_policy.value if fallback_reason is not None else None,
            "fallback_to": self._grouped_gemm_fallback_target(stats, fallback_reason),
            "pack_threshold": pack_threshold,
            "effective_pack_threshold": int(effective_pack_threshold),
            "effective_allow_batched": bool(effective_allow_batched),
            **context_payload,
        }
        if group_descriptors is not None:
            execute_payload["group_descriptors"] = group_descriptors
        if profiling is not None:
            execute_payload = profiling.standardize_event_payload(execute_payload)
        self._last_execution_profile = dict(execute_payload)

        if should_profile:
            try:
                task_operands = [
                    self._profile_gemm_task_operands(task, index)
                    for index, task in enumerate(converted)
                ]
                profiling.record(
                    "contraction_execute",
                    backend=self.name,
                    **profiling.contraction_execute_compute_payload("grouped_gemm"),
                    equation=None,
                    lowering="grouped_gemm",
                    descriptor_source=descriptor_source,
                    buffer_names=buffer_names,
                    buffer_table=buffer_table,
                    buffer_slice_descriptors=buffer_slice_descriptors,
                    input_shapes=[
                        [tuple(getattr(task.A, "shape", ())), tuple(getattr(task.B, "shape", ()))]
                        for task in converted
                    ],
                    input_dtypes=[
                        [str(getattr(task.A, "dtype", None)), str(getattr(task.B, "dtype", None))]
                        for task in converted
                    ],
                    operands=task_operands,
                    task_operands=task_operands,
                    task_specs=[
                        self._profile_gemm_task_spec(task, index, xp=xp)
                        for index, task in enumerate(converted)
                    ],
                    output_shape=[tuple(getattr(item, "shape", ())) for item in result],
                    dtype=str(getattr(result[0], "dtype", None)) if result else None,
                    **self._profile_device_execution(self.current_device()),
                    **self._profile_execution_resources(stream=stream, workspace=workspace),
                    flops=stats.flops,
                    read_bytes=stats.read_bytes,
                    write_bytes=stats.write_bytes,
                    copy_bytes=stats.copy_bytes,
                    workspace_bytes=stats.workspace_bytes,
                    peak_bytes=stats.write_bytes + stats.copy_bytes + stats.workspace_bytes,
                    largest_intermediate=max((array_nbytes(item) for item in result), default=0),
                    largest_intermediate_elements=max((self._array_size(item) for item in result), default=0),
                    largest_intermediate_bytes=max((array_nbytes(item) for item in result), default=0),
                    num_gemm=stats.loop_task_count,
                    num_batched_gemm=stats.batched_bucket_count,
                    num_grouped_tasks=stats.task_count,
                    num_blocks=stats.task_count,
                    num_shape_buckets=stats.shape_bucket_count,
                    batched_matmul_provider=(
                        batched_matmul_provider
                        if stats.batched_bucket_count
                        else None
                    ),
                    supports_grouped_gemm=bool(self.supports_grouped_gemm),
                    grouped_gemm_policy=grouped_gemm_policy,
                    grouped_gemm_implementation=self._grouped_gemm_implementation(stats, fallback_reason),
                    **self._grouped_gemm_execution_items(grouped_gemm_policy, fallback_reason),
                    requires_grouped_gemm_fallback=fallback_reason is not None,
                    fallback_from="grouped_gemm" if fallback_reason is not None else None,
                    fallback_to=self._grouped_gemm_fallback_target(stats, fallback_reason),
                    fallback_reason=fallback_reason,
                    fallback_policy=fallback_policy.value if fallback_reason is not None else None,
                    bucket_task_counts=stats.bucket_task_counts,
                    shape_buckets=self._profile_shape_buckets(self._bucketed_by_task_shape(converted, xp=xp)),
                    group_sizes=list(gemm_batch.group_sizes),
                    gsta=list(gemm_batch.gsta),
                    sorted_indices=list(gemm_batch.sorted_indices),
                    group_keys=[key.to_dict() for key in gemm_batch.groups],
                    group_key_buckets=self._grouped_gemm_execute_group_key_buckets(gemm_batch),
                    group_descriptors=self._grouped_gemm_group_descriptors(
                        gemm_batch,
                        bucket_execution_profile.bucket_execution_profiles,
                    ),
                    batched_bucket_count=stats.batched_bucket_count,
                    loop_bucket_count=stats.loop_bucket_count,
                    batched_task_count=stats.batched_task_count,
                    loop_task_count=stats.loop_task_count,
                    pack_threshold=pack_threshold,
                    requested_policy=execution_policy,
                    effective_pack_threshold=int(effective_pack_threshold),
                    effective_allow_batched=bool(effective_allow_batched),
                    **bucket_execution_profile.to_dict(),
                    **context_payload,
                    wall_s=wall_s,
                )
                record_payload = dict(execute_payload)
                record_payload.pop("event")
                profiling.record("grouped_gemm_execute", **record_payload)
            except Exception:
                pass
        return result

    def prepack_grouped_gemm(
        self,
        tasks,
        *,
        buffers=None,
        pack_threshold=4,
        stream=None,
        workspace=None,
        policy="auto",
        fallback_policy=None,
    ):
        import time

        self._validate_workspace(workspace, required_bytes=0)
        xp = self.array_namespace or _np
        converted, descriptor_source, raw_tasks = self._grouped_gemm_tasks_from_input(tasks, buffers)
        buffer_table = self._profile_buffer_table(buffers)
        buffer_slice_descriptors = self._profile_buffer_matmul_descs(raw_tasks)
        execution_policy, fallback_policy = self._resolve_grouped_gemm_call_policies(
            policy,
            fallback_policy,
        )
        fallback_reason = self._handle_grouped_gemm_fallback(len(converted), policy=fallback_policy)
        flop_copy_ratio = 0 if self.supports_grouped_gemm else 10
        allow_batched = bool(self.supports_grouped_gemm or self.supports_batched_matmul)
        effective_pack_threshold, effective_flop_copy_ratio, effective_allow_batched = (
            self._grouped_gemm_execution_policy_controls(
                execution_policy,
                pack_threshold=pack_threshold,
                flop_copy_ratio=flop_copy_ratio,
                allow_batched=allow_batched,
            )
        )
        started = time.perf_counter()
        with self._stream_context(stream):
            plan = prepack_grouped_gemm_plan(
                converted,
                xp=xp,
                pack_threshold=effective_pack_threshold,
                flop_copy_ratio=effective_flop_copy_ratio,
                allow_batched=effective_allow_batched,
                batched_matmul_provider=self._batched_matmul_provider_name() if effective_allow_batched else None,
            )
        wall_s = time.perf_counter() - started
        task_specs = [
            self._profile_gemm_task_spec(task, index, xp=xp)
            for index, task in enumerate(converted)
        ]
        task_operands = [
            self._profile_gemm_task_operands(task, index)
            for index, task in enumerate(converted)
        ]
        gemm_batch = GemmBatch.from_tasks(converted, xp=xp)
        group_descriptors = self._grouped_gemm_group_descriptors(
            gemm_batch,
            plan.profile.get("bucket_execution_profiles", ()),
        )
        input_dtypes = [
            [str(getattr(task.A, "dtype", None)), str(getattr(task.B, "dtype", None))]
            for task in converted
        ]
        output_dtype = None
        if converted:
            dtype_values = [
                getattr(array, "dtype", None)
                for task in converted
                for array in (task.A, task.B)
                if getattr(array, "dtype", None) is not None
            ]
            result_type = getattr(xp, "result_type", None)
            if callable(result_type) and dtype_values:
                try:
                    output_dtype = str(result_type(*dtype_values))
                except Exception:
                    output_dtype = None
            if output_dtype is None:
                output_dtype = str(dtype_values[0]) if dtype_values else None
        profile = dict(plan.profile)
        grouped_gemm_policy = self._grouped_gemm_execution_policy(plan.stats, fallback_reason)
        profile.update({
            "backend": self.name,
            **self._profile_device_execution(self.current_device()),
            **self._profile_execution_resources(stream=stream, workspace=workspace),
            "descriptor_source": descriptor_source,
            "buffer_names": sorted(str(name) for name in buffers) if buffers is not None else [],
            "buffer_table": buffer_table,
            "buffer_slice_descriptors": buffer_slice_descriptors,
            "input_shapes": [
                [tuple(getattr(task.A, "shape", ())), tuple(getattr(task.B, "shape", ()))]
                for task in converted
            ],
            "input_dtypes": input_dtypes,
            "task_operands": task_operands,
            "task_specs": task_specs,
            "group_descriptors": group_descriptors,
            "output_shape": [
                tuple(spec.get("batch_shape", ())) + (int(spec["m"]), int(spec["n"]))
                for spec in task_specs
            ],
            "dtype": output_dtype,
            "requested_policy": execution_policy,
            "policy": grouped_gemm_policy,
            "supports_grouped_gemm": bool(self.supports_grouped_gemm),
            "grouped_gemm_implementation": (
                "backend_prepacked_bucketed_matmul"
                if fallback_reason is None
                else "fallback_prepacked_bucketed_matmul"
            ),
            "requires_grouped_gemm_fallback": fallback_reason is not None,
            "fallback_from": "grouped_gemm" if fallback_reason is not None else None,
            "fallback_to": "prepacked_bucketed_matmul" if fallback_reason is not None else None,
            "fallback_reason": fallback_reason,
            "fallback_policy": fallback_policy.value if fallback_reason is not None else None,
            **self._grouped_gemm_execution_items(
                "prepacked_bucketed_matmul",
                fallback_reason,
            ),
            "pack_threshold": int(pack_threshold),
            "effective_pack_threshold": int(effective_pack_threshold),
            "effective_allow_batched": bool(effective_allow_batched),
            "wall_s": float(wall_s),
        })
        try:
            from renormalizer.utils import profiling

            profile = profiling.standardize_event_payload(profile)
            if profiling.should_record_op():
                record_payload = dict(profile)
                record_payload.pop("event", None)
                profiling.record("grouped_gemm_prepack", **record_payload)
        except Exception:
            pass
        return replace(plan, profile=profile)

    def execute_prepacked_grouped_gemm(self, plan, *, stream=None, workspace=None, fallback_policy=None):
        import time

        self._validate_workspace(workspace, required_bytes=0)
        xp = self.array_namespace or _np
        fallback_policy = self._resolve_fallback_policy(fallback_policy)
        fallback_reason = self._handle_grouped_gemm_fallback(len(plan.tasks), policy=fallback_policy)
        batched_matmul = None
        if self.supports_batched_matmul:
            batched_matmul = lambda a, b: self.batched_matmul(
                a,
                b,
                stream=stream,
                workspace=workspace,
            )
        batched_matmul_provider = (
            self._batched_matmul_provider_name()
            if batched_matmul is not None
            else None
        )
        started = time.perf_counter()
        pointer_setup_s = 0.0
        with self._stream_context(stream):
            with self._suppress_primitive_profile():
                result, execution_profile = execute_prepacked_grouped_gemm_plan(
                    plan,
                    xp=xp,
                    batched_matmul=batched_matmul,
                    batched_matmul_provider=batched_matmul_provider,
                )
        wall_s = time.perf_counter() - started
        stats = plan.stats
        gemm_batch = GemmBatch.from_tasks(plan.tasks, xp=xp)
        group_descriptors = self._grouped_gemm_group_descriptors(
            gemm_batch,
            execution_profile.get("bucket_execution_profiles", ()),
        )
        try:
            from renormalizer.utils import profiling

            output_shape = [tuple(getattr(item, "shape", ())) for item in result]
            output_strides = [profiling.array_strides(item) for item in result]
            output_order = [profiling.array_order(item) for item in result]
            output_contiguous = [profiling.array_contiguous(item) for item in result]
            output_backend = [profiling.array_backend_name(item) for item in result]
            output_device_kind = [profiling.array_device_kind(item) for item in result]
            output_location = [profiling.array_location(item) for item in result]
            output_is_host = [profiling.array_is_host(item) for item in result]
            output_is_device = [profiling.array_is_device(item) for item in result]
            output_is_distributed = [profiling.array_is_distributed(item) for item in result]
        except Exception:
            profiling = None
            output_shape = [tuple(getattr(item, "shape", ())) for item in result]
            output_strides = None
            output_order = None
            output_contiguous = None
            output_backend = None
            output_device_kind = None
            output_location = None
            output_is_host = None
            output_is_device = None
            output_is_distributed = None
        payload = {
            "event": "grouped_gemm_execute",
            "backend": self.name,
            **self._profile_device_execution(self.current_device()),
            **self._profile_execution_resources(stream=stream, workspace=workspace),
            "descriptor_source": plan.profile.get("descriptor_source", "prepacked_gemm_task"),
            "buffer_names": list(plan.profile.get("buffer_names", [])),
            "buffer_table": list(plan.profile.get("buffer_table", [])),
            "buffer_slice_descriptors": list(plan.profile.get("buffer_slice_descriptors", [])),
            "policy": "prepacked_bucketed_matmul",
            "supports_grouped_gemm": bool(self.supports_grouped_gemm),
            "grouped_gemm_implementation": (
                "backend_prepacked_bucketed_matmul"
                if fallback_reason is None
                else "fallback_prepacked_bucketed_matmul"
            ),
            **self._grouped_gemm_execution_items(
                "prepacked_bucketed_matmul",
                fallback_reason,
            ),
            "requires_grouped_gemm_fallback": fallback_reason is not None,
            "num_tasks": int(stats.task_count),
            "num_groups": int(stats.shape_bucket_count),
            "group_sizes": list(stats.bucket_task_counts),
            "group_descriptors": group_descriptors,
            "shape_buckets": [
                {
                    **dict(bucket),
                    "batch_shape": [int(dim) for dim in bucket.get("batch_shape", ())],
                }
                for bucket in stats.shape_buckets
            ],
            "total_flops": int(stats.flops),
            "read_bytes": int(stats.read_bytes),
            "write_bytes": int(stats.write_bytes),
            "copy_bytes": int(stats.copy_bytes),
            "workspace_bytes": int(stats.workspace_bytes),
            "output_shape": output_shape,
            "output_strides": output_strides,
            "output_order": output_order,
            "output_contiguous": output_contiguous,
            "output_backend": output_backend,
            "output_device_kind": output_device_kind,
            "output_location": output_location,
            "output_is_host": output_is_host,
            "output_is_device": output_is_device,
            "output_is_distributed": output_is_distributed,
            "num_batched_gemm": int(stats.batched_bucket_count),
            "num_gemm": int(stats.loop_task_count),
            "batched_matmul_provider": batched_matmul_provider if stats.batched_bucket_count else None,
            "pointer_setup_s": float(pointer_setup_s),
            "compute_s": float(wall_s),
            "wall_s": float(wall_s),
            **execution_profile,
            "fallback_from": "grouped_gemm" if fallback_reason is not None else None,
            "fallback_to": "prepacked_bucketed_matmul" if fallback_reason is not None else None,
            "fallback_reason": fallback_reason,
            "fallback_policy": fallback_policy.value if fallback_reason is not None else None,
            "pack_threshold": int(plan.pack_threshold),
        }
        try:
            payload = profiling.standardize_event_payload(payload)
            self._last_execution_profile = dict(payload)
            if profiling.should_record_op():
                contraction_payload = dict(payload)
                contraction_payload.update(
                    {
                        "event": "contraction_execute",
                        **profiling.contraction_execute_compute_payload("grouped_gemm"),
                        "lowering": "grouped_gemm",
                        "input_shapes": list(plan.profile.get("input_shapes", [])),
                        "input_dtypes": list(plan.profile.get("input_dtypes", [])),
                        "operands": list(plan.profile.get("task_operands", [])),
                        "task_operands": list(plan.profile.get("task_operands", [])),
                        "task_specs": list(plan.profile.get("task_specs", [])),
                        "dtype": plan.profile.get("dtype"),
                        "flops": int(stats.flops),
                        "peak_bytes": (
                            int(stats.write_bytes)
                            + int(stats.copy_bytes)
                            + int(stats.workspace_bytes)
                        ),
                        "largest_intermediate": max(
                            (array_nbytes(item) for item in result),
                            default=0,
                        ),
                        "largest_intermediate_elements": max(
                            (self._array_size(item) for item in result),
                            default=0,
                        ),
                        "largest_intermediate_bytes": max(
                            (array_nbytes(item) for item in result),
                            default=0,
                        ),
                        "num_grouped_tasks": int(stats.task_count),
                        "num_blocks": int(stats.task_count),
                        "num_shape_buckets": int(stats.shape_bucket_count),
                        "grouped_gemm_policy": "prepacked_bucketed_matmul",
                    }
                )
                contraction_payload = profiling.standardize_event_payload(contraction_payload)
                record_payload = dict(contraction_payload)
                record_payload.pop("event", None)
                profiling.record("contraction_execute", **record_payload)
                record_payload = dict(payload)
                record_payload.pop("event", None)
                profiling.record("grouped_gemm_execute", **record_payload)
        except Exception:
            self._last_execution_profile = dict(payload)
            pass
        return result

    def _grouped_gemm_fallback_reason(self, task_count=None):
        if task_count is not None and int(task_count) == 0:
            return None
        if self.supports_grouped_gemm:
            return None
        reason = "backend-owned grouped_gemm unavailable; used bucketed fallback"
        return reason

    def _resolve_fallback_policy(self, policy=None):
        if policy is None:
            return self.fallback_policy
        if isinstance(policy, str) and policy.lower() == "auto":
            return self.fallback_policy
        return FallbackPolicy.from_value(policy)

    @staticmethod
    def _fallback_policy_value_set():
        return {item.value for item in FallbackPolicy}

    def _resolve_grouped_gemm_call_policies(self, policy="auto", fallback_policy=None):
        if isinstance(policy, FallbackPolicy):
            if fallback_policy is None:
                fallback_policy = policy
            policy = "auto"
        elif fallback_policy is None and policy is not None:
            value = str(policy).lower()
            if value in self._fallback_policy_value_set():
                fallback_policy = value
                policy = "auto"
        return (
            self._normalize_grouped_gemm_execution_policy(policy),
            self._resolve_fallback_policy(fallback_policy),
        )

    @staticmethod
    def _normalize_grouped_gemm_execution_policy(policy):
        if policy is None:
            return "auto"
        value = str(policy).lower()
        aliases = {
            "loop": "bucketed_loop_matmul",
            "bucketed_loop": "bucketed_loop_matmul",
            "batched": "bucketed_batched_matmul",
            "bucketed_batched": "bucketed_batched_matmul",
            "mixed": "bucketed_mixed_matmul",
            "bucketed_mixed": "bucketed_mixed_matmul",
        }
        value = aliases.get(value, value)
        allowed = {
            "auto",
            "bucketed_loop_matmul",
            "bucketed_batched_matmul",
            "bucketed_mixed_matmul",
        }
        if value not in allowed:
            raise BackendFeatureError(
                "unsupported grouped_gemm execution policy {0!r}".format(policy)
            )
        return value

    @staticmethod
    def _grouped_gemm_execution_policy_controls(
        policy,
        *,
        pack_threshold,
        flop_copy_ratio,
        allow_batched,
    ):
        if policy == "bucketed_loop_matmul":
            return int(pack_threshold), int(flop_copy_ratio), False
        if policy == "bucketed_batched_matmul":
            return 0, 0, bool(allow_batched)
        return int(pack_threshold), int(flop_copy_ratio), bool(allow_batched)

    def _handle_grouped_gemm_fallback(self, task_count=None, *, policy=None):
        policy = self._resolve_fallback_policy(policy)
        reason = self._grouped_gemm_fallback_reason(task_count)
        if reason is None:
            return None
        if policy is FallbackPolicy.FORBID:
            raise BackendFeatureError(reason)
        if policy is FallbackPolicy.WARN:
            import warnings

            warnings.warn(reason, RuntimeWarning, stacklevel=3)
        return reason

    @staticmethod
    def _grouped_gemm_fallback_target(stats, fallback_reason):
        if fallback_reason is None:
            return None
        if int(getattr(stats, "batched_task_count", 0) or 0) and int(getattr(stats, "loop_task_count", 0) or 0):
            return "bucketed_mixed_matmul"
        if int(getattr(stats, "batched_task_count", 0) or 0):
            return "bucketed_batched_matmul"
        return "bucketed_loop_matmul"

    @staticmethod
    def _grouped_gemm_execution_policy(stats, fallback_reason):
        fallback_target = AbstractBackend._grouped_gemm_fallback_target(stats, fallback_reason)
        if fallback_target is not None:
            return fallback_target
        if not int(getattr(stats, "task_count", 0) or 0):
            return "empty"
        if int(getattr(stats, "batched_task_count", 0) or 0) and int(getattr(stats, "loop_task_count", 0) or 0):
            return "bucketed_mixed_matmul"
        if int(getattr(stats, "batched_task_count", 0) or 0):
            return "bucketed_batched_matmul"
        return "bucketed_loop_matmul"

    @staticmethod
    def _grouped_gemm_execution_items(policy, fallback_reason):
        return {
            "execution_primitives": ["grouped_gemm"],
            "execution_policies": [str(policy)],
            "fallback_reasons": [] if fallback_reason is None else [str(fallback_reason)],
        }

    @staticmethod
    def _grouped_gemm_execute_shape_buckets(gemm_batch, stats):
        stats_by_key = {}
        for bucket in getattr(stats, "shape_buckets", ()):
            key = (
                tuple(int(dim) for dim in bucket.get("batch_shape", ())),
                int(bucket.get("m") or 0),
                int(bucket.get("n") or 0),
                int(bucket.get("k") or 0),
                bool(bucket.get("trans_a")),
                bool(bucket.get("trans_b")),
                bool(bucket.get("conj_a")),
                bool(bucket.get("conj_b")),
            )
            stats_by_key[key] = bucket

        items = []
        for group_key, indices in gemm_batch.groups.items():
            lookup_key = (
                tuple(int(dim) for dim in group_key.batch_shape),
                int(group_key.m),
                int(group_key.n),
                int(group_key.k),
                str(group_key.trans_a).upper() in ("T", "C"),
                str(group_key.trans_b).upper() in ("T", "C"),
                bool(group_key.conj_a),
                bool(group_key.conj_b),
            )
            bucket = stats_by_key.get(lookup_key)
            if bucket is None:
                item = {
                    "dtype_a": group_key.dtype,
                    "dtype_b": group_key.dtype,
                    "batch_shape": [int(dim) for dim in group_key.batch_shape],
                    "m": int(group_key.m),
                    "n": int(group_key.n),
                    "k": int(group_key.k),
                    "trans_a": str(group_key.trans_a).upper() in ("T", "C"),
                    "trans_b": str(group_key.trans_b).upper() in ("T", "C"),
                    "conj_a": bool(group_key.conj_a),
                    "conj_b": bool(group_key.conj_b),
                    "execution": "unknown",
                    "task_count": int(len(indices)),
                }
            else:
                item = dict(bucket)
                item["batch_shape"] = [int(dim) for dim in item.get("batch_shape", ())]
                item["task_count"] = int(len(indices))
            items.append(item)
        return items

    @staticmethod
    def _grouped_gemm_execute_group_key_buckets(gemm_batch):
        items = []
        for group_key, sorted_group_indices in gemm_batch.groups.items():
            item = group_key.to_dict()
            item["batch_shape"] = [int(dim) for dim in item.get("batch_shape", ())]
            item["task_indices"] = [
                int(gemm_batch.sorted_indices[index])
                for index in sorted_group_indices
            ]
            item["task_count"] = int(len(sorted_group_indices))
            items.append(item)
        return items

    @staticmethod
    def _grouped_gemm_profile_key_from_group_key(group_key):
        return (
            str(group_key.dtype),
            tuple(int(dim) for dim in group_key.batch_shape),
            int(group_key.m),
            int(group_key.n),
            int(group_key.k),
            str(group_key.trans_a).upper() in ("T", "C"),
            str(group_key.trans_b).upper() in ("T", "C"),
            bool(group_key.conj_a),
            bool(group_key.conj_b),
        )

    @staticmethod
    def _grouped_gemm_profile_key_from_bucket(bucket):
        dtype_a = str(bucket.get("dtype_a"))
        dtype_b = str(bucket.get("dtype_b"))
        dtype = dtype_a if dtype_a == dtype_b else "{0},{1}".format(dtype_a, dtype_b)
        return (
            dtype,
            tuple(int(dim) for dim in bucket.get("batch_shape", ())),
            int(bucket.get("m") or 0),
            int(bucket.get("n") or 0),
            int(bucket.get("k") or 0),
            bool(bucket.get("trans_a")),
            bool(bucket.get("trans_b")),
            bool(bucket.get("conj_a")),
            bool(bucket.get("conj_b")),
        )

    @staticmethod
    def _array_element_count(array):
        size = getattr(array, "size", None)
        if size is not None:
            try:
                return int(size)
            except Exception:
                pass
        numel = getattr(array, "numel", None)
        if callable(numel):
            return int(numel())
        count = 1
        for dim in getattr(array, "shape", ()):
            count *= int(dim)
        return int(count)

    @classmethod
    def _array_itemsize_for_profile(cls, array):
        dtype = getattr(array, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is not None:
            return int(itemsize)
        element_size = getattr(array, "element_size", None)
        if callable(element_size):
            return int(element_size())
        count = cls._array_element_count(array)
        return int(array_nbytes(array) // max(count, 1))

    @classmethod
    def _grouped_gemm_result_itemsize(cls, task):
        dtypes = [
            getattr(array, "dtype", None)
            for array in (task.A, task.B, task.C)
            if array is not None and getattr(array, "dtype", None) is not None
        ]
        if dtypes:
            try:
                return int(_np.dtype(_np.result_type(*dtypes)).itemsize)
            except Exception:
                pass
        return max(cls._array_itemsize_for_profile(task.A), cls._array_itemsize_for_profile(task.B))

    @classmethod
    def _grouped_gemm_group_descriptors(cls, gemm_batch, bucket_execution_profiles=()):
        profile_by_key = {}
        for profile in bucket_execution_profiles or ():
            profile_by_key[cls._grouped_gemm_profile_key_from_bucket(profile)] = profile

        descriptors = []
        for group_index, (group_key, sorted_group_indices) in enumerate(gemm_batch.groups.items()):
            task_indices = [
                int(gemm_batch.sorted_indices[index])
                for index in sorted_group_indices
            ]
            tasks = [gemm_batch.descs[index] for index in sorted_group_indices]
            batch_count = 1
            for dim in group_key.batch_shape:
                batch_count *= int(dim)

            flops = int(len(tasks) * max(batch_count, 1) * 2 * int(group_key.m) * int(group_key.n) * int(group_key.k))
            read_bytes = int(sum(array_nbytes(task.A) + array_nbytes(task.B) for task in tasks))
            write_bytes = 0
            for task in tasks:
                if task.C is not None:
                    write_bytes += array_nbytes(task.C)
                else:
                    write_bytes += int(max(batch_count, 1) * int(group_key.m) * int(group_key.n) * cls._grouped_gemm_result_itemsize(task))

            profile = profile_by_key.get(cls._grouped_gemm_profile_key_from_group_key(group_key), {})
            item = group_key.to_dict()
            item["batch_shape"] = [int(dim) for dim in item.get("batch_shape", ())]
            item.update({
                "group_index": int(group_index),
                "task_indices": task_indices,
                "task_count": int(len(task_indices)),
                "execution": str(profile.get("execution", "unknown")),
                "flops": flops,
                "read_bytes": read_bytes,
                "write_bytes": int(write_bytes),
                "copy_bytes": int(profile.get("pack_bytes", 0) or 0),
                "pack_strategy": str(profile.get("pack_strategy", "unknown")),
                "kernel_calls": int(profile.get("kernel_calls", 0) or 0),
                "batched_matmul_provider": profile.get("batched_matmul_provider"),
            })
            descriptors.append(item)
        return descriptors

    @staticmethod
    def _grouped_gemm_implementation(stats, fallback_reason):
        policy = AbstractBackend._grouped_gemm_execution_policy(stats, fallback_reason)
        return AbstractBackend._grouped_gemm_implementation_from_policy(policy, fallback_reason)

    @staticmethod
    def _block_grouped_gemm_plan_policy(fallback_reason):
        return "bucketed_grouped_gemm" if fallback_reason is not None else "backend_grouped_gemm"

    @staticmethod
    def _grouped_gemm_implementation_from_policy(policy, fallback_reason):
        if policy == "empty":
            return "empty"
        prefix = "fallback" if fallback_reason is not None else "backend"
        return "{0}_{1}".format(prefix, policy)

    def _batched_matmul_provider_name(self):
        namespace = self.array_namespace or _np
        module_name = getattr(namespace, "__name__", None)
        if not module_name:
            module_name = type(namespace).__module__
        return "{0}.matmul".format(module_name)

    def _handle_plan_fallback(self, plan, *, fallback_policy=None):
        if plan.fallback_reason is None:
            return
        fallback_policy = self._resolve_fallback_policy(fallback_policy)
        if fallback_policy is FallbackPolicy.FORBID:
            raise BackendFeatureError(plan.fallback_reason)
        if fallback_policy is FallbackPolicy.WARN:
            import warnings

            warnings.warn(plan.fallback_reason, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _fallback_source(plan):
        kind = str(getattr(plan, "kind", ""))
        if kind == "grouped_gemm" and getattr(plan, "fallback_reason", None):
            return "grouped_gemm"
        if not kind.startswith("fallback_"):
            return None
        reason = str(getattr(plan, "fallback_reason", "") or getattr(plan, "reason", ""))
        if "batched_matmul" in reason:
            return "batched_gemm"
        if "matmul" in reason:
            return "gemm"
        return None

    def _fallback_target(self, plan):
        kind = str(getattr(plan, "kind", ""))
        reason = getattr(plan, "fallback_reason", None)
        if not reason:
            return None
        if kind == "grouped_gemm":
            return "bucketed_grouped_gemm"
        if kind in ("fallback_tensordot", "fallback_einsum"):
            if not self.supports_matmul:
                if any(getattr(desc, "batch_shape", ()) for desc in getattr(plan, "descs", ())):
                    return "einsum"
                return "tensordot"
            return "loop_matmul"
        return None

    def _primitive_plan_fallback_metadata(self, plan):
        kind = str(getattr(plan, "kind", ""))
        descs = tuple(getattr(plan, "descs", ()))
        desc = descs[0] if descs else None
        if kind == "gemm" and not self.supports_matmul:
            return {
                "lowering": "fallback_tensordot",
                "fallback_reason": "backend lacks matmul",
                "fallback_from": "gemm",
                "fallback_to": "tensordot",
                "execution_primitive": "tensordot",
                "execution_policy": "fallback_tensordot",
            }
        if kind in ("batched_gemm", "strided_batched_gemm") and not self.supports_batched_matmul:
            batch_shape = tuple(getattr(desc, "batch_shape", ()) or ()) if desc is not None else ()
            if not batch_shape and desc is not None:
                batch_shape = tuple(getattr(desc.A, "shape", ())[:-2])
            fallback_to = "loop_matmul" if self.supports_matmul else "tensordot"
            return {
                "lowering": "fallback_tensordot",
                "fallback_reason": "backend lacks batched_matmul for batch shape {0}".format(batch_shape),
                "fallback_from": "batched_gemm",
                "fallback_to": fallback_to,
                "execution_primitive": fallback_to,
                "execution_policy": "fallback_{0}".format(fallback_to),
            }
        return None

    @staticmethod
    def _matmul_plan_execution_primitive(kind):
        kind = str(kind)
        if kind == "gemm":
            return "matmul"
        if kind in ("batched_gemm", "strided_batched_gemm"):
            return "batched_matmul"
        if kind == "grouped_gemm":
            return "grouped_gemm"
        if kind == "fallback_tensordot":
            return "tensordot"
        if kind == "fallback_einsum":
            return "einsum"
        return kind

    @classmethod
    def _matmul_plan_execution_policy(cls, kind, *, fallback_reason=None, execution_policy=None):
        if execution_policy is not None:
            return str(execution_policy)
        kind = str(kind)
        if kind in ("fallback_tensordot", "fallback_einsum"):
            return kind
        if kind == "grouped_gemm" and fallback_reason is not None:
            return "bucketed_grouped_gemm"
        return "backend_{0}".format(cls._matmul_plan_execution_primitive(kind))

    @classmethod
    def _matmul_plan_execution_items(
        cls,
        plan,
        *,
        fallback_reason=None,
        execution_policy=None,
        execution_primitive=None,
    ):
        kind = str(getattr(plan, "kind", ""))
        reason = fallback_reason
        if reason is None:
            reason = getattr(plan, "fallback_reason", None)
        return {
            "execution_primitives": [
                str(execution_primitive)
                if execution_primitive is not None
                else cls._matmul_plan_execution_primitive(kind)
            ],
            "execution_policies": [
                cls._matmul_plan_execution_policy(
                    kind,
                    fallback_reason=reason,
                    execution_policy=execution_policy,
                )
            ],
            "fallback_reasons": [] if reason is None else [str(reason)],
        }

    @classmethod
    def _matmul_plan_largest_intermediate_memory(cls, plan):
        if getattr(plan, "kind", None) == "grouped_gemm":
            candidates = [
                (
                    int(desc.estimated_write_bytes),
                    cls._prod_shape(tuple(desc.batch_shape) + (desc.m, desc.n)),
                )
                for desc in tuple(getattr(plan, "descs", ()))
            ]
            return max(candidates, default=(0, 0), key=lambda item: item[0])
        return (
            sum(int(desc.estimated_write_bytes) for desc in tuple(getattr(plan, "descs", ()))),
            cls._prod_shape(getattr(plan, "output_shape", ())),
        )

    def _execute_plan_impl(
        self,
        plan,
        *,
        stream=None,
        workspace=None,
        pack_threshold=None,
        fallback_policy=None,
    ):
        if not plan.descs:
            raise ValueError("MatmulPlan has no descriptors to execute")
        if plan.kind == "gemm":
            return self.matmul(
                plan.descs[0],
                stream=stream,
                workspace=workspace,
                fallback_policy=fallback_policy,
            )
        if plan.kind in ("batched_gemm", "strided_batched_gemm"):
            return self.batched_matmul(
                plan.descs[0],
                stream=stream,
                workspace=workspace,
                fallback_policy=fallback_policy,
            )
        if plan.kind == "grouped_gemm":
            return self._execute_grouped_matmul_descs(
                plan.descs,
                stream=stream,
                workspace=workspace,
                pack_threshold=pack_threshold,
                fallback_policy=fallback_policy,
            )
        if plan.kind in ("fallback_tensordot", "fallback_einsum"):
            self._handle_plan_fallback(plan, fallback_policy=fallback_policy)
            if len(plan.descs) != 1:
                raise BackendFeatureError("fallback execution expects exactly one descriptor")
            if not self.supports_matmul:
                return self._tensor_contract_matmul_fallback(plan.descs[0], stream=stream, workspace=workspace)
            return self._loop_matmul(plan.descs[0], stream=stream, workspace=workspace)
        raise BackendFeatureError("Unknown MatmulPlan kind {0!r}".format(plan.kind))

    def _record_contraction_execute(
        self,
        plan,
        result,
        wall_s,
        plan_hash=None,
        equation=None,
        input_modes=None,
        output_modes=None,
        pack_threshold=None,
        stream=None,
        workspace=None,
        fallback_policy=None,
    ):
        try:
            from renormalizer.utils import profiling

            should_log_profile = bool(profiling.should_record_op())
            descs = tuple(plan.descs)
            desc = descs[0] if descs else None
            grouped = plan.kind == "grouped_gemm"
            profile_lowering = plan.kind
            primitive_fallback = None
            if grouped:
                grouped_pack_threshold = 4 if pack_threshold is None else int(pack_threshold)
                xp = self.array_namespace or _np
                fallback_reason = plan.fallback_reason or self._grouped_gemm_fallback_reason(len(descs))
                flop_copy_ratio = 0 if self.supports_grouped_gemm else 10
                allow_batched = bool(self.supports_grouped_gemm or self.supports_batched_matmul)
                stats = grouped_gemm_stats(
                    descs,
                    xp=xp,
                    pack_threshold=grouped_pack_threshold,
                    flop_copy_ratio=flop_copy_ratio,
                    allow_batched=allow_batched,
                )
                input_shapes = [
                    [tuple(getattr(item.A, "shape", ())), tuple(getattr(item.B, "shape", ()))]
                    for item in descs
                ]
                input_dtypes = [
                    [str(getattr(item.A, "dtype", None)), str(getattr(item.B, "dtype", None))]
                    for item in descs
                ]
                operands = [
                    self._profile_matmul_desc_operands(item, index)
                    for index, item in enumerate(descs)
                ]
                task_specs = [
                    self._profile_matmul_desc_spec(item, index)
                    for index, item in enumerate(descs)
                ]
                if isinstance(result, (tuple, list)):
                    output_shape = [tuple(getattr(item, "shape", ())) for item in result]
                    dtype = str(getattr(result[0], "dtype", None)) if result else None
                    largest_intermediate, largest_intermediate_elements = self._largest_array_memory(result)
                else:
                    output_shape = tuple(getattr(result, "shape", plan.output_shape))
                    dtype = str(getattr(result, "dtype", None))
                    largest_intermediate = self._array_nbytes(result)
                    largest_intermediate_elements = self._array_size(result)
                shape_buckets = self._bucketed_by_desc_shape(descs)
                num_blocks = len(descs)
                num_shape_buckets = len(shape_buckets)
                bucket_task_counts = [len(indices) for _, indices in sorted(shape_buckets.items())]
                shape_bucket_payload = self._profile_shape_buckets(shape_buckets)
                group_boundaries = self._profile_group_boundaries(shape_buckets)
                group_keys = self._profile_matmul_desc_group_keys(descs)
                group_key_buckets = self._profile_matmul_desc_group_key_buckets(descs)
                profile_flops = int(stats.flops)
                profile_read_bytes = int(stats.read_bytes)
                profile_write_bytes = int(stats.write_bytes)
                profile_copy_bytes = int(stats.copy_bytes)
                profile_workspace_bytes = max(int(plan.workspace_bytes), int(stats.workspace_bytes))
                profile_num_gemm = int(stats.loop_task_count)
                profile_num_batched_gemm = int(stats.batched_bucket_count)
                grouped_policy = self._grouped_gemm_execution_policy(stats, fallback_reason)
                grouped_implementation = self._grouped_gemm_implementation(stats, fallback_reason)
                grouped_fallback_from = "grouped_gemm" if fallback_reason is not None else None
                grouped_fallback_to = self._grouped_gemm_fallback_target(stats, fallback_reason)
            else:
                primitive_fallback = self._primitive_plan_fallback_metadata(plan)
                fallback_reason = (
                    primitive_fallback["fallback_reason"]
                    if primitive_fallback is not None
                    else plan.fallback_reason
                )
                profile_lowering = (
                    primitive_fallback["lowering"]
                    if primitive_fallback is not None
                    else plan.kind
                )
                input_shapes = [
                    tuple(getattr(desc.A, "shape", ())),
                    tuple(getattr(desc.B, "shape", ())),
                ] if desc is not None else []
                input_dtypes = [
                    str(getattr(desc.A, "dtype", None)),
                    str(getattr(desc.B, "dtype", None)),
                ] if desc is not None else []
                operands = [
                    self._profile_array_operand(
                        "operand0",
                        desc.A,
                        input_modes[0] if input_modes else (),
                    ),
                    self._profile_array_operand(
                        "operand1",
                        desc.B,
                        input_modes[1] if input_modes else (),
                    ),
                ] if desc is not None else []
                task_specs = None
                output_shape = tuple(getattr(result, "shape", plan.output_shape))
                dtype = str(getattr(result, "dtype", None))
                largest_intermediate = self._array_nbytes(result)
                largest_intermediate_elements = self._array_size(result)
                num_blocks = 0
                num_shape_buckets = 0
                bucket_task_counts = None
                shape_bucket_payload = None
                group_boundaries = {"group_sizes": None, "gsta": None, "sorted_indices": None}
                group_keys = None
                group_key_buckets = None
                profile_flops = int(plan.estimated_flops)
                profile_read_bytes = sum(desc.estimated_read_bytes for desc in plan.descs)
                profile_write_bytes = sum(desc.estimated_write_bytes for desc in plan.descs)
                profile_copy_bytes = int(plan.copy_bytes)
                profile_workspace_bytes = int(plan.workspace_bytes)
                if primitive_fallback is None:
                    profile_num_gemm = 1 if plan.kind == "gemm" else 0
                    profile_num_batched_gemm = 1 if plan.kind in ("batched_gemm", "strided_batched_gemm") else 0
                else:
                    profile_num_gemm = 0
                    profile_num_batched_gemm = 0
                grouped_pack_threshold = None
                grouped_policy = None
                grouped_implementation = None
                grouped_fallback_from = (
                    primitive_fallback["fallback_from"]
                    if primitive_fallback is not None
                    else self._fallback_source(plan)
                )
                grouped_fallback_to = (
                    primitive_fallback["fallback_to"]
                    if primitive_fallback is not None
                    else None
                )
            pre_ops_profile = self._profile_layout_transforms(plan.pre_ops)
            post_ops_profile = self._profile_layout_transforms(plan.post_ops)
            input_layout_transform_copy_bytes = sum(
                int(transform.copy_bytes) for transform in tuple(plan.pre_ops)
            )
            output_layout_transform_copy_bytes = sum(
                int(transform.copy_bytes) for transform in tuple(plan.post_ops)
            )
            layout_transform_copy_bytes = (
                input_layout_transform_copy_bytes + output_layout_transform_copy_bytes
            )
            if isinstance(result, (tuple, list)):
                output_strides = [profiling.array_strides(item) for item in result]
                output_order = [profiling.array_order(item) for item in result]
                output_contiguous = [profiling.array_contiguous(item) for item in result]
                output_backend = [profiling.array_backend_name(item) for item in result]
                output_device_kind = [profiling.array_device_kind(item) for item in result]
                output_location = [profiling.array_location(item) for item in result]
                output_is_host = [profiling.array_is_host(item) for item in result]
                output_is_device = [profiling.array_is_device(item) for item in result]
                output_is_distributed = [profiling.array_is_distributed(item) for item in result]
            else:
                output_strides = profiling.array_strides(result)
                output_order = profiling.array_order(result)
                output_contiguous = profiling.array_contiguous(result)
                output_backend = profiling.array_backend_name(result)
                output_device_kind = profiling.array_device_kind(result)
                output_location = profiling.array_location(result)
                output_is_host = profiling.array_is_host(result)
                output_is_device = profiling.array_is_device(result)
                output_is_distributed = profiling.array_is_distributed(result)

            payload = {
                "event": "contraction_execute",
                "backend": self.name,
                **profiling.contraction_execute_compute_payload("grouped_gemm" if grouped else profile_lowering),
                "equation": equation,
                "lowering": profile_lowering,
                "plan_hash": plan_hash or getattr(plan, "plan_hash", ""),
                "input_modes": input_modes,
                "output_modes": output_modes,
                "input_shapes": input_shapes,
                "input_dtypes": input_dtypes,
                "operands": operands,
                "task_specs": task_specs,
                "pre_ops": pre_ops_profile,
                "post_ops": post_ops_profile,
                "num_pre_ops": len(pre_ops_profile),
                "num_post_ops": len(post_ops_profile),
                "layout_transform_copy_bytes": layout_transform_copy_bytes,
                "input_layout_transform_copy_bytes": input_layout_transform_copy_bytes,
                "output_layout_transform_copy_bytes": output_layout_transform_copy_bytes,
                "output_shape": output_shape,
                "output_strides": output_strides,
                "output_order": output_order,
                "output_contiguous": output_contiguous,
                "output_backend": output_backend,
                "output_device_kind": output_device_kind,
                "output_location": output_location,
                "output_is_host": output_is_host,
                "output_is_device": output_is_device,
                "output_is_distributed": output_is_distributed,
                "dtype": dtype,
                "output_dtype": dtype,
                **self._profile_device_execution(self.current_device()),
                **self._profile_execution_resources(stream=stream, workspace=workspace),
                "flops": profile_flops,
                "read_bytes": profile_read_bytes,
                "write_bytes": profile_write_bytes,
                "copy_bytes": profile_copy_bytes,
                "workspace_bytes": profile_workspace_bytes,
                "peak_bytes": (
                    profile_write_bytes
                    + profile_copy_bytes
                    + profile_workspace_bytes
                ),
                "largest_intermediate": largest_intermediate,
                "largest_intermediate_elements": int(largest_intermediate_elements or 0),
                "largest_intermediate_bytes": int(largest_intermediate or 0),
                "num_gemm": profile_num_gemm,
                "num_batched_gemm": profile_num_batched_gemm,
                "num_grouped_tasks": len(plan.descs) if plan.kind == "grouped_gemm" else 0,
                "num_blocks": num_blocks,
                "num_shape_buckets": num_shape_buckets,
                "bucket_task_counts": bucket_task_counts,
                "shape_buckets": shape_bucket_payload,
                **group_boundaries,
                "group_keys": group_keys,
                "group_key_buckets": group_key_buckets,
                "supports_grouped_gemm": bool(self.supports_grouped_gemm) if grouped else None,
                "grouped_gemm_policy": grouped_policy,
                "grouped_gemm_implementation": grouped_implementation,
                "requires_grouped_gemm_fallback": fallback_reason is not None if grouped else None,
                "batched_bucket_count": profile_num_batched_gemm if grouped else None,
                "loop_task_count": profile_num_gemm if grouped else None,
                "pack_threshold": grouped_pack_threshold,
                **self._matmul_plan_execution_items(
                    plan,
                    fallback_reason=fallback_reason,
                    execution_policy=(
                        grouped_policy
                        if grouped
                        else (
                            primitive_fallback["execution_policy"]
                            if primitive_fallback is not None
                            else None
                        )
                    ),
                    execution_primitive=(
                        primitive_fallback["execution_primitive"]
                        if primitive_fallback is not None
                        else None
                    ),
                ),
                "fallback_reason": fallback_reason,
                "fallback_from": grouped_fallback_from,
                "fallback_to": grouped_fallback_to if (grouped or primitive_fallback is not None) else self._fallback_target(plan),
                "fallback_policy": (
                    self._resolve_fallback_policy(fallback_policy).value
                    if fallback_reason is not None
                    else None
                ),
                "wall_s": wall_s,
            }
            payload = profiling.standardize_event_payload(payload)
            self._last_execution_profile = dict(payload)
            if should_log_profile:
                record_payload = dict(payload)
                record_payload.pop("event")
                profiling.record("contraction_execute", **record_payload)
        except Exception:
            pass

    def execute_matmul_plan(
        self,
        plan,
        *,
        stream=None,
        workspace=None,
        pack_threshold=None,
        plan_hash=None,
        record_profile=True,
        equation=None,
        input_modes=None,
        output_modes=None,
        fallback_policy=None,
    ):
        fallback_policy = self._resolve_fallback_policy(fallback_policy)
        self._validate_workspace(workspace, required_bytes=self._plan_required_workspace_bytes(plan))
        if record_profile:
            import time

            start = time.perf_counter()
            with self._suppress_primitive_profile():
                result = self._execute_plan_impl(
                    plan,
                    stream=stream,
                    workspace=workspace,
                    pack_threshold=pack_threshold,
                    fallback_policy=fallback_policy,
                )
            wall_s = time.perf_counter() - start
            self._record_contraction_execute(
                plan,
                result,
                wall_s,
                plan_hash=plan_hash,
                equation=equation,
                input_modes=input_modes,
                output_modes=output_modes,
                pack_threshold=pack_threshold,
                stream=stream,
                workspace=workspace,
                fallback_policy=fallback_policy,
            )
            return result
        with self._suppress_primitive_profile():
            return self._execute_plan_impl(
                plan,
                stream=stream,
                workspace=workspace,
                pack_threshold=pack_threshold,
                fallback_policy=fallback_policy,
            )

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
