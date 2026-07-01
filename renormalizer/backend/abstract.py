# -*- coding: utf-8 -*-

import os
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
    MatmulDesc,
    MatmulPlan,
    PairContractionSpec,
    ShardingSpec,
    StreamEvent,
    TensorOperand,
    Workspace,
    array_info_for_backend,
    layout_from_array,
    legacy_device_kind,
    lower_pair_contraction_to_matmul,
    parse_einsum,
    parse_device_spec,
    parse_einsum_equation,
)
from renormalizer.backend.gemm import GemmTask, array_nbytes, grouped_gemm_bucketed, grouped_gemm_stats, run_gemm_task
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

    def tensordot(self, a, b, axes=2):
        xp = self.array_namespace or _np
        a, b = self._promote_tensordot_operands(a, b)
        return xp.tensordot(a, b, axes)

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
        return isinstance(x, DistributedTensor)

    def array_info(self, x: Any):
        if self.is_distributed_array(x):
            local_info = array_info_for_backend(self, x.local_array, self.current_device())
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

    @staticmethod
    def _contraction_step_from_matmul_plan(plan, input_modes, output_modes):
        return ContractionStep(
            kind=plan.kind,
            inputs=(0, 1),
            output=2,
            input_modes=tuple(tuple(modes) for modes in input_modes),
            output_modes=tuple(output_modes),
            plan=plan,
            estimated_flops=plan.estimated_flops,
            estimated_read_bytes=sum(desc.estimated_read_bytes for desc in plan.descs),
            estimated_write_bytes=sum(desc.estimated_write_bytes for desc in plan.descs),
            estimated_copy_bytes=plan.copy_bytes,
            estimated_peak_bytes=max((desc.estimated_write_bytes for desc in plan.descs), default=0),
            estimated_comm_bytes=0,
            required_workspace_bytes=plan.workspace_bytes,
            reason=plan.reason,
            fallback_reason=plan.fallback_reason,
        )

    def _plan_einsum_contraction(self, spec, *, sliced_modes=(), distributed_modes=(), comm_bytes=0):
        if len(spec.operands) != 2:
            raise BackendFeatureError("plan_contraction currently supports two-operand explicit einsum specs")
        pair_spec = PairContractionSpec.from_operands(spec.operands[0], spec.operands[1], spec.output_modes)
        matmul_plan = self.lower_pair_contraction_to_matmul(pair_spec)
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
        return ContractionPlan(
            steps=(step,),
            input_specs=spec.operands,
            output_modes=spec.output_modes,
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

    @staticmethod
    def _operand_global_shape(operand):
        if isinstance(operand, DistributedTensor):
            return tuple(operand.global_shape)
        return tuple(int(dim) for dim in getattr(operand, "shape", ()))

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
    ):
        del memory_limit, prefer, allow_slicing, target_devices
        if isinstance(spec, DistributedContractionSpec):
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
                active_distributed_modes = tuple(
                    mode for mode in distributed_modes if mode not in set(input_redistribution_modes)
                )
                reduce_scatter_required = bool(reduced_distributed_modes and output_redistribution_required)
                reduce_scatter_modes = tuple(dict.fromkeys(
                    tuple(reduced_distributed_modes) + tuple(output_sharding.sharded_modes if output_sharding is not None else ())
                ))
                output_comm_bytes = plan.estimated_comm_bytes
                total_comm_bytes = output_comm_bytes + input_redistribution_bytes
                states = tuple(
                    DistributionState(
                        operand_index=index,
                        modes=tuple(modes),
                        sharding=operand.sharding if isinstance(operand, DistributedTensor) else None,
                        distributed_modes=tuple(mode for mode in modes if isinstance(operand, DistributedTensor) and mode in operand.sharding.sharded_modes),
                        replicated_modes=tuple(mode for mode in modes if not isinstance(operand, DistributedTensor) or mode not in operand.sharding.sharded_modes),
                    )
                    for index, (operand, modes) in enumerate(zip(spec.operands, input_modes))
                )
                step_plan = DistributedStepPlan(
                    local_step=plan.steps[0],
                    input_states=states,
                    output_sharding=output_sharding,
                    communication=tuple(
                        (
                            CommunicationPlan(
                                kind="redistribute",
                                bytes=input_redistribution_bytes,
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
                return ContractionPlan(
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
            return plan
        return self._plan_einsum_contraction(spec)

    @staticmethod
    def _rate_seconds(amount, rate):
        if not amount or not rate:
            return 0.0
        return float(amount) / float(rate)

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
            return self._make_cost_estimate(
                hw,
                flops=desc.estimated_flops,
                read_bytes=sum(item.estimated_read_bytes for item in desc.descs),
                write_bytes=sum(item.estimated_write_bytes for item in desc.descs),
                copy_bytes=desc.copy_bytes,
                workspace_bytes=desc.workspace_bytes,
                peak_bytes=max((item.estimated_write_bytes for item in desc.descs), default=0) + desc.workspace_bytes,
            )
        batch = self._prod_shape(desc.batch_shape)
        flops = desc.estimated_flops or int(2 * batch * desc.m * desc.n * desc.k)
        read_bytes = desc.estimated_read_bytes or (self._array_nbytes(desc.A) + self._array_nbytes(desc.B))
        itemsize = max(self._operand_itemsize(desc.A), self._operand_itemsize(desc.B), 1)
        write_bytes = desc.estimated_write_bytes or int(batch * desc.m * desc.n * itemsize)
        workspace_bytes = int(desc.estimated_workspace_bytes)
        return self._make_cost_estimate(
            hw,
            flops=flops,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            workspace_bytes=workspace_bytes,
            peak_bytes=write_bytes + workspace_bytes,
        )

    def estimate_contraction(self, plan, hw=None):
        if isinstance(plan, MatmulPlan):
            return self.estimate_matmul(plan, hw)
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
        nbytes = self._prod_shape(tensor_shape) * int(itemsize)
        movement_bytes = 0 if src == dst else nbytes
        return self._make_cost_estimate(
            hw,
            copy_bytes=movement_bytes,
            comm_bytes=movement_bytes,
            peak_bytes=movement_bytes,
        )

    def _local_nbytes_for_sharding(self, sharding, itemsize):
        if sharding is None:
            return 0
        local_slices = sharding.local_slices.get(sharding.mesh.local_rank)
        if local_slices is None:
            return 0
        local_shape = self._local_shape_for_slices(sharding.global_shape, local_slices)
        return self._prod_shape(local_shape) * int(itemsize)

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
            local_desc = MatmulDesc(
                A=desc.A,
                B=desc.B,
                C=desc.C,
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
                layout_a=desc.layout_a,
                layout_b=desc.layout_b,
                layout_c=desc.layout_c,
                estimated_flops=flops,
                estimated_read_bytes=read_bytes,
                estimated_write_bytes=write_bytes,
                estimated_workspace_bytes=desc.estimated_workspace_bytes,
            )
            local_plan = MatmulPlan(
                kind=local_plan.kind,
                descs=(local_desc,),
                pre_ops=local_plan.pre_ops,
                post_ops=local_plan.post_ops,
                output_shape=output_shape,
                copy_bytes=local_plan.copy_bytes,
                workspace_bytes=local_plan.workspace_bytes,
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
            estimated_copy_bytes=step.estimated_copy_bytes,
            estimated_peak_bytes=max(write_bytes, step.required_workspace_bytes),
            estimated_comm_bytes=step.estimated_comm_bytes,
            required_workspace_bytes=step.required_workspace_bytes,
            reason=step.reason,
            fallback_reason=step.fallback_reason,
        )

    def _peak_local_bytes_for_distributed_plan(self, plan):
        itemsize = max(
            (self._operand_itemsize(operand.array) for operand in plan.path.input_specs),
            default=1,
        )
        operand_peak = max(
            (
                operand.array.local_nbytes
                if isinstance(operand.array, DistributedTensor)
                else self._array_nbytes(operand.array)
                for operand in plan.path.input_specs
            ),
            default=0,
        )
        output_peak = self._local_nbytes_for_sharding(plan.output_sharding, itemsize)
        return max(operand_peak, output_peak, int(plan.path.required_workspace_bytes))

    @staticmethod
    def _communication_totals(steps):
        total_comm_bytes = 0
        total_redistribute_bytes = 0
        total_allreduce_bytes = 0
        total_gather_bytes = 0
        for step in steps:
            for communication in step.communication:
                nbytes = int(communication.bytes)
                total_comm_bytes += nbytes
                if communication.kind in ("redistribute", "alltoall", "activate_distribution"):
                    total_redistribute_bytes += nbytes
                elif communication.kind in ("allreduce", "reduce_scatter"):
                    total_allreduce_bytes += nbytes
                elif communication.kind in ("gather", "allgather"):
                    total_gather_bytes += nbytes
        return total_comm_bytes, total_redistribute_bytes, total_allreduce_bytes, total_gather_bytes

    def _with_distributed_plan_totals(self, plan):
        total_comm_bytes, total_redistribute_bytes, total_allreduce_bytes, total_gather_bytes = (
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
        )

    def _estimate_communication_sequence_s(self, communication, hw):
        hw = HardwareModel() if hw is None else hw
        bandwidth = self._hardware_comm_bandwidth(hw)
        total = 0.0
        for item in communication:
            nbytes = int(item.bytes)
            total += self._rate_seconds(nbytes, bandwidth)
            if nbytes and hw.latency_s:
                total += float(hw.latency_s)
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
        assignments = {}
        used_modes = set()
        for axis_name, axis_size in zip(mesh.axis_names, mesh.shape):
            if int(axis_size) <= 1:
                continue
            candidate = next((item for item in candidates if item[0] not in used_modes), None)
            if candidate is None:
                break
            mode, _ = candidate
            used_modes.add(mode)
            assignments[mode] = (axis_name, int(axis_size))
        if not assignments:
            selected_mode, _ = candidates[0]
            assignments[selected_mode] = (mesh.axis_names[0], int(mesh.shape[0]))
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
            DistributionState(
                operand_index=index,
                modes=operand.modes,
                sharding=None,
                distributed_modes=(),
                replicated_modes=operand.modes,
            )
            for index, operand in enumerate(path.input_specs)
        )
        step_plan = DistributedStepPlan(
            local_step=self._local_step_for_activate_distribution(path, output_sharding),
            input_states=states,
            output_sharding=output_sharding,
            communication=(
                CommunicationPlan(
                    kind="activate_distribution",
                    bytes=activation_bytes,
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

    def plan_distributed_contraction_path(
        self,
        path,
        mesh,
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
            )
        result = self._with_distributed_plan_totals(distributed_plan)
        if memory_limit_per_device is not None and result.peak_local_bytes > int(memory_limit_per_device):
            raise BackendFeatureError(
                "distributed contraction peak local bytes {0} exceeds memory limit {1}"
                .format(result.peak_local_bytes, int(memory_limit_per_device))
            )
        return result

    def default_stream(self):
        return None

    def new_stream(self):
        return None

    def record_event(self, stream=None):
        return StreamEvent(device=self.current_device(), stream=stream)

    def wait_event(self, event, stream=None):
        return None

    def allocate_workspace(self, nbytes, *, device=None):
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("workspace nbytes must be non-negative")
        spec = parse_device_spec(device) if device is not None else self.current_device()
        if spec is None:
            spec = self.current_device()
        buffer = self.to_backend(_np.empty((nbytes,), dtype=_np.uint8))
        return Workspace(device=spec, nbytes=nbytes, buffer=buffer)

    def release_workspace(self, workspace):
        return None

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
        del stream, workspace
        if plan.output_sharding is None:
            raise BackendFeatureError("activate_distribution plan requires output_sharding")
        spec = DistributedContractionSpec(
            equation=plan.equation or self._einsum_equation_from_plan(plan.path),
            operands=self._placed_operands_for_activate_distribution(plan),
            output_sharding=plan.output_sharding,
        )
        return self.distributed_contract(spec)

    def execute(self, plan, *, stream=None, workspace=None):
        if isinstance(plan, ContractionPlan):
            if len(plan.steps) != 1:
                raise BackendFeatureError("execute currently supports single-step ContractionPlan objects")
            return self.execute(plan.steps[0].plan, stream=stream, workspace=workspace)
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

    def shard_tensor(self, x, spec: ShardingSpec):
        if self.is_distributed_array(x):
            x = self.gather_tensor(x)
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
        if x.rank_local_arrays is None:
            if self.is_distributed and int(x.mesh.world_size) == int(self.size):
                gathered = self.allgather(x.local_array)
                result = self._zeros_backend(x.global_shape, x.dtype)
                for rank, value in enumerate(gathered):
                    result = self._slice_set(result, x.sharding.local_slices[rank], value)
                return result
            if x.sharding.local_slices.get(x.mesh.local_rank) == tuple(slice(None) for _ in x.global_shape):
                return x.local_array
            raise BackendFeatureError("cannot gather distributed tensor without rank-local arrays")
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

    def _sum_rank_local_arrays(self, rank_local_arrays):
        total = None
        for rank in sorted(rank_local_arrays):
            value = rank_local_arrays[rank]
            total = value if total is None else total + value
        return total

    def _record_distributed_contraction_execute(self, spec, plan, result, wall_s):
        try:
            from renormalizer.utils import profiling

            if not profiling.should_record_op():
                return
            distributed_plan = self._distributed_plan_from_contraction_plan(plan)
            communication = ()
            if distributed_plan is not None and distributed_plan.steps:
                communication = distributed_plan.steps[0].communication
            step = plan.steps[0] if isinstance(plan, ContractionPlan) and plan.steps else None
            flops = 0
            if distributed_plan is not None:
                flops = int(distributed_plan.total_flops or getattr(distributed_plan.path, "estimated_flops", 0))
            elif step is not None:
                flops = int(step.estimated_flops)
            comm_bytes = 0
            if distributed_plan is not None:
                comm_bytes = int(distributed_plan.total_comm_bytes or distributed_plan.estimated_comm_bytes)
            profiling.record(
                "contraction_execute",
                backend=self.name,
                equation=spec.equation,
                lowering="distributed",
                input_shapes=[tuple(self._operand_global_shape(operand)) for operand in spec.operands],
                output_shape=tuple(result.global_shape),
                dtype=str(getattr(result, "dtype", None)),
                device=str(self.current_device()),
                flops=flops,
                read_bytes=int(getattr(step, "estimated_read_bytes", 0)),
                write_bytes=int(getattr(step, "estimated_write_bytes", 0)),
                copy_bytes=int(getattr(step, "estimated_copy_bytes", 0)),
                workspace_bytes=int(getattr(step, "required_workspace_bytes", 0)),
                largest_intermediate=int(result.local_nbytes),
                num_gemm=0,
                num_batched_gemm=0,
                num_grouped_tasks=0,
                num_blocks=0,
                num_shape_buckets=0,
                fallback_reason=None,
                distributed_modes=[str(mode) for mode in getattr(plan, "distributed_modes", ())],
                rank=int(result.mesh.global_rank),
                world_size=int(result.mesh.world_size),
                local_shape=tuple(result.local_shape),
                global_shape=tuple(result.global_shape),
                communication=[
                    {
                        "collective": item.kind,
                        "bytes": int(item.bytes),
                        "modes": [str(mode) for mode in item.modes],
                        "num_messages": 1,
                        "block_size": int(item.bytes),
                    }
                    for item in communication
                ],
                comm_bytes=comm_bytes,
                wall_s=wall_s,
            )
        except Exception:
            pass

    def _distributed_contract_impl(self, spec, *, plan=None, stream=None, workspace=None):
        del stream, workspace
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
        if self.is_distributed and int(mesh.world_size) == int(self.size):
            local_operands = tuple(
                operand.local_array if self.is_distributed_array(operand) else operand
                for operand in operands
            )
            local_array = self._execute_einsum(spec.equation, local_operands)
            if reduced_distributed_modes:
                local_array = self.allreduce(local_array, op="sum")
            result = DistributedTensor(
                local_array=local_array,
                global_shape=output_shape,
                modes=output_modes,
                sharding=execution_sharding,
                mesh=mesh,
                dtype=getattr(local_array, "dtype", None),
                local_shape=tuple(getattr(local_array, "shape", ())),
                local_nbytes=self._array_nbytes(local_array),
                rank_local_arrays=None,
            )
            if not self._sharding_specs_equivalent(output_sharding, execution_sharding):
                return self.redistribute(result, output_sharding)
            return result
        rank_local_arrays = {}
        for rank in range(mesh.world_size):
            local_operands = tuple(self._local_operand_for_rank(operand, rank) for operand in operands)
            rank_local_arrays[rank] = self._execute_einsum(spec.equation, local_operands)
        if reduced_distributed_modes:
            reduced = self.allreduce(self._sum_rank_local_arrays(rank_local_arrays), op="sum")
            rank_local_arrays = {rank: reduced for rank in range(mesh.world_size)}
        local_array = rank_local_arrays[mesh.local_rank]
        result = DistributedTensor(
            local_array=local_array,
            global_shape=output_shape,
            modes=output_modes,
            sharding=execution_sharding,
            mesh=mesh,
            dtype=getattr(local_array, "dtype", None),
            local_shape=tuple(getattr(local_array, "shape", ())),
            local_nbytes=self._array_nbytes(local_array),
            rank_local_arrays=rank_local_arrays,
        )
        if not self._sharding_specs_equivalent(output_sharding, execution_sharding):
            return self.redistribute(result, output_sharding)
        return result

    def distributed_contract(self, spec, *, plan=None, stream=None, workspace=None):
        if plan is None and isinstance(spec, DistributedContractionSpec):
            plan = self.plan_contraction(spec, allow_distribution=True)
        try:
            from renormalizer.utils import profiling

            should_profile = profiling.should_record_op()
        except Exception:
            should_profile = False
        if not should_profile:
            return self._distributed_contract_impl(spec, plan=plan, stream=stream, workspace=workspace)

        import time

        started = time.perf_counter()
        result = self._distributed_contract_impl(spec, plan=plan, stream=stream, workspace=workspace)
        wall_s = time.perf_counter() - started
        self._record_distributed_contraction_execute(spec, plan, result, wall_s)
        return result

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
            key = (int(desc.m), int(desc.n), int(desc.k))
            buckets.setdefault(key, []).append(index)
        return {key: tuple(indices) for key, indices in buckets.items()}

    def lower_block_contraction(self, spec):
        descs = []
        output_blocks = []
        global_shape = None
        for left_key, left_block in sorted(spec.left.blocks.items(), key=self._block_sort_key):
            if self._is_zero_sized_block(left_block):
                continue
            for right_key, right_block in sorted(spec.right.blocks.items(), key=self._block_sort_key):
                if self._is_zero_sized_block(right_block):
                    continue
                output_key = spec.qn_rule(left_key, right_key)
                if output_key is None:
                    continue
                pair_spec = PairContractionSpec.from_operands(
                    TensorOperand(left_block.array, tuple(left_block.modes), name="left"),
                    TensorOperand(right_block.array, tuple(right_block.modes), name="right"),
                    output_modes=spec.output_modes,
                )
                plan = lower_pair_contraction_to_matmul(pair_spec, self.capabilities)
                if plan.kind == "fallback_tensordot":
                    self._handle_plan_fallback(plan)
                if not plan.descs:
                    raise BackendFeatureError("block contraction lowering produced no GEMM descriptors")
                desc = plan.descs[0]
                descs.append(desc)
                output_blocks.append(output_key)
                if global_shape is None:
                    global_shape = tuple(plan.output_shape)

        repeated_outputs = len(set(output_blocks)) != len(output_blocks)
        return GroupedGemmPlan(
            tasks=tuple(descs),
            output_blocks=tuple(output_blocks),
            bucketed_by_shape=self._bucketed_by_desc_shape(descs),
            scatter_add_required=bool(repeated_outputs and spec.accumulate),
            estimated_flops=sum(int(desc.estimated_flops) for desc in descs),
            estimated_read_bytes=sum(int(desc.estimated_read_bytes) for desc in descs),
            estimated_write_bytes=sum(int(desc.estimated_write_bytes) for desc in descs),
            estimated_workspace_bytes=sum(int(desc.estimated_workspace_bytes) for desc in descs),
            output_modes=tuple(spec.output_modes),
            global_shape=global_shape or (),
            block_axis_meta={
                "left": spec.left.block_axis_meta,
                "right": spec.right.block_axis_meta,
            },
            backend=self.name,
        )

    def execute_grouped_gemm_plan(self, plan, *, pack_threshold=4, stream=None, workspace=None):
        if len(plan.tasks) != len(plan.output_blocks):
            raise ValueError("GroupedGemmPlan tasks and output_blocks must have the same length")
        flat_outputs = {}
        task_groups = {}
        tasks = []
        task_output_keys = []
        functional_accumulation = self.name == "jax"
        for desc, output_key in zip(plan.tasks, plan.output_blocks):
            a, b, groups = self._prepare_matmul_desc(desc)
            if output_key not in flat_outputs:
                dtype = getattr(desc, "dtype_output", None) or getattr(a, "dtype", None)
                flat_outputs[output_key] = self._zeros_backend((desc.m, desc.n), dtype)
                task_groups[output_key] = groups
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
            results = self.grouped_gemm(tasks, pack_threshold=pack_threshold, stream=stream, workspace=workspace)
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
            )
        return BlockTensor(
            blocks=blocks,
            global_shape=tuple(plan.global_shape),
            modes=tuple(plan.output_modes),
            block_axis_meta=plan.block_axis_meta,
            backend=self.name,
        )

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
            result = grouped_gemm_bucketed(converted, xp=xp, pack_threshold=pack_threshold)
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
        return grouped_gemm_bucketed(converted, xp=xp, pack_threshold=pack_threshold)

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
