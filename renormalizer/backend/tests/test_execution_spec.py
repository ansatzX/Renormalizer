# -*- coding: utf-8 -*-

import json

import numpy as np
import pytest


def test_backend_protocol_planning_and_execution_signatures_are_explicit():
    import inspect

    from renormalizer.backend.protocol import BackendProtocol

    def assert_keyword_only(signature, name):
        assert name in signature.parameters
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    plan_signature = inspect.signature(BackendProtocol.plan_contraction)
    for name in (
        "memory_limit",
        "prefer",
        "allow_slicing",
        "allow_distribution",
        "target_devices",
        "record_profile",
    ):
        assert_keyword_only(plan_signature, name)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in plan_signature.parameters.values()
    )

    distributed_plan_signature = inspect.signature(BackendProtocol.plan_distributed_contraction_path)
    for name in ("memory_limit_per_device", "cost_model"):
        assert_keyword_only(distributed_plan_signature, name)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in distributed_plan_signature.parameters.values()
    )

    from renormalizer.backend.numpy_backend import NumpyBackend

    concrete_plan_signature = inspect.signature(NumpyBackend.plan_contraction)
    for name in (
        "memory_limit",
        "prefer",
        "allow_slicing",
        "allow_distribution",
        "target_devices",
        "record_profile",
    ):
        assert_keyword_only(concrete_plan_signature, name)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in concrete_plan_signature.parameters.values()
    )

    concrete_distributed_signature = inspect.signature(NumpyBackend.plan_distributed_contraction_path)
    for name in ("memory_limit_per_device", "cost_model"):
        assert_keyword_only(concrete_distributed_signature, name)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in concrete_distributed_signature.parameters.values()
    )

    workspace_signature = inspect.signature(BackendProtocol.allocate_workspace)
    assert_keyword_only(workspace_signature, "device")
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in workspace_signature.parameters.values()
    )

    execute_signature = inspect.signature(BackendProtocol.execute)
    assert_keyword_only(execute_signature, "stream")
    assert_keyword_only(execute_signature, "workspace")
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in execute_signature.parameters.values()
    )

    profile_signature = inspect.signature(BackendProtocol.last_execution_profile)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in profile_signature.parameters.values()
    )


def test_backend_protocol_execution_primitive_signatures_are_explicit():
    import inspect

    from renormalizer.backend.protocol import BackendProtocol

    def assert_keyword_only(signature, name):
        assert name in signature.parameters
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    def assert_no_var_keyword(signature):
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    redistribute_signature = inspect.signature(BackendProtocol.estimate_redistribute)
    assert_keyword_only(redistribute_signature, "itemsize")
    assert_no_var_keyword(redistribute_signature)

    distributed_signature = inspect.signature(BackendProtocol.distributed_contract)
    for name in ("plan", "stream", "workspace"):
        assert_keyword_only(distributed_signature, name)
    assert_no_var_keyword(distributed_signature)

    for method_name in ("shard_tensor", "gather_tensor", "redistribute"):
        assert_no_var_keyword(inspect.signature(getattr(BackendProtocol, method_name)))

    replicate_signature = inspect.signature(BackendProtocol.replicate_tensor)
    assert_keyword_only(replicate_signature, "modes")
    assert_no_var_keyword(replicate_signature)

    execute_matmul_signature = inspect.signature(BackendProtocol.execute_matmul_plan)
    for name in (
        "stream",
        "workspace",
        "pack_threshold",
        "plan_hash",
        "record_profile",
        "equation",
        "input_modes",
        "output_modes",
        "fallback_policy",
    ):
        assert_keyword_only(execute_matmul_signature, name)
    assert_no_var_keyword(execute_matmul_signature)

    execute_grouped_signature = inspect.signature(BackendProtocol.execute_grouped_gemm_plan)
    for name in ("pack_threshold", "stream", "workspace", "policy", "fallback_policy"):
        assert_keyword_only(execute_grouped_signature, name)
    assert_no_var_keyword(execute_grouped_signature)

    lower_pair_signature = inspect.signature(BackendProtocol.lower_pair_contraction_to_matmul)
    assert_keyword_only(lower_pair_signature, "record_profile")
    assert_no_var_keyword(lower_pair_signature)

    for method_name in ("matmul", "batched_matmul"):
        signature = inspect.signature(getattr(BackendProtocol, method_name))
        for name in (
            "C",
            "trans_a",
            "trans_b",
            "conj_a",
            "conj_b",
            "alpha",
            "beta",
            "stream",
            "workspace",
            "fallback_policy",
        ):
            assert_keyword_only(signature, name)
        assert_no_var_keyword(signature)

    grouped_signature = inspect.signature(BackendProtocol.grouped_gemm)
    for name in ("pack_threshold", "stream", "workspace", "policy", "fallback_policy"):
        assert_keyword_only(grouped_signature, name)
    assert_no_var_keyword(grouped_signature)

    send_signature = inspect.signature(BackendProtocol.send)
    for name in ("dst", "tag"):
        assert_keyword_only(send_signature, name)
    assert_no_var_keyword(send_signature)

    recv_signature = inspect.signature(BackendProtocol.recv)
    for name in ("src", "tag", "like"):
        assert_keyword_only(recv_signature, name)
    assert_no_var_keyword(recv_signature)

    gemv_signature = inspect.signature(BackendProtocol.gemv_batch)
    for name in ("buffers", "role", "stream", "workspace", "fallback_policy", "profile_context"):
        assert_keyword_only(gemv_signature, name)
    assert_no_var_keyword(gemv_signature)

    prepack_signature = inspect.signature(BackendProtocol.prepack_grouped_gemm)
    for name in ("pack_threshold", "stream", "workspace", "policy", "fallback_policy"):
        assert_keyword_only(prepack_signature, name)
    assert_no_var_keyword(prepack_signature)

    execute_prepacked_signature = inspect.signature(BackendProtocol.execute_prepacked_grouped_gemm)
    for name in ("stream", "workspace", "fallback_policy"):
        assert_keyword_only(execute_prepacked_signature, name)
    assert_no_var_keyword(execute_prepacked_signature)

    tensordot_signature = inspect.signature(BackendProtocol.tensordot)
    for name in ("stream", "workspace"):
        assert_keyword_only(tensordot_signature, name)
    assert_no_var_keyword(tensordot_signature)


def test_backend_protocol_copy_and_layout_signatures_are_explicit():
    import inspect

    from renormalizer.backend.protocol import BackendProtocol

    def assert_keyword_only(signature, name):
        assert name in signature.parameters
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    def assert_no_var_keyword(signature):
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    to_host_signature = inspect.signature(BackendProtocol.to_host)
    assert_keyword_only(to_host_signature, "copy")
    assert_no_var_keyword(to_host_signature)

    to_backend_signature = inspect.signature(BackendProtocol.to_backend)
    for name in ("device", "dtype", "copy"):
        assert_keyword_only(to_backend_signature, name)
    assert_no_var_keyword(to_backend_signature)

    astype_signature = inspect.signature(BackendProtocol.astype)
    assert_keyword_only(astype_signature, "copy")
    assert_no_var_keyword(astype_signature)

    ascontiguous_signature = inspect.signature(BackendProtocol.ascontiguousarray)
    assert_keyword_only(ascontiguous_signature, "copy")
    assert_no_var_keyword(ascontiguous_signature)

    permute_signature = inspect.signature(BackendProtocol.permute)
    assert_keyword_only(permute_signature, "copy_policy")
    assert_no_var_keyword(permute_signature)

    make_contiguous_signature = inspect.signature(BackendProtocol.make_contiguous)
    for name in ("mode_groups", "copy_policy"):
        assert_keyword_only(make_contiguous_signature, name)
    assert_no_var_keyword(make_contiguous_signature)

    parse_einsum_signature = inspect.signature(BackendProtocol.parse_einsum)
    for name in ("constants", "optimize"):
        assert_keyword_only(parse_einsum_signature, name)
    assert_no_var_keyword(parse_einsum_signature)


def test_backend_protocol_capability_flags_are_explicit():
    from renormalizer.backend.protocol import BackendProtocol

    required = (
        "supports_cpu",
        "supports_gpu",
        "supports_autodiff",
        "supports_jit",
        "supports_sparse",
        "supports_functional_update",
        "supports_complex64",
        "supports_complex128",
        "supports_fp32",
        "supports_fp64",
        "supports_mixed_precision",
        "supports_device_index",
        "supports_matmul",
        "supports_batched_matmul",
        "supports_grouped_gemm",
        "supports_strided_batched_gemm",
        "supports_einsum",
        "supports_contract_expression",
        "supports_contraction_path",
        "supports_custom_contraction_plan",
        "supports_streams",
        "supports_events",
        "supports_memory_pool",
        "supports_block_sparse",
        "supports_packed_blocks",
        "supports_scatter_add",
        "supports_distributed",
        "supports_distributed_array",
        "supports_allreduce",
        "supports_broadcast",
        "supports_allgather",
        "supports_reduce_scatter",
        "supports_alltoall",
        "supports_point_to_point",
    )

    for name in required:
        assert name in BackendProtocol.__annotations__


def test_backend_proxy_explicitly_delegates_execution_profile_api():
    from renormalizer.backend.proxy import BackendProxy

    assert "last_execution_profile" in BackendProxy.__dict__
    assert BackendProxy.last_execution_profile.__name__ == "last_execution_profile"


def test_backend_proxy_explicitly_delegates_point_to_point_api():
    from renormalizer.backend.proxy import BackendProxy

    assert "send" in BackendProxy.__dict__
    assert BackendProxy.send.__name__ == "send"
    assert "recv" in BackendProxy.__dict__
    assert BackendProxy.recv.__name__ == "recv"


def test_backend_capability_flags_mirror_capability_fields():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    assert backend.supports_complex64 is backend.capabilities.complex64
    assert backend.supports_complex128 is backend.capabilities.complex128
    assert backend.supports_fp32 is backend.capabilities.fp32
    assert backend.supports_fp64 is backend.capabilities.fp64
    assert backend.supports_mixed_precision is backend.capabilities.mixed_precision
    assert backend.supports_device_index is backend.capabilities.device_index
    assert backend.supports_matmul is backend.capabilities.matmul
    assert backend.supports_einsum is backend.capabilities.einsum
    assert backend.supports_contract_expression is backend.capabilities.contract_expression
    assert backend.supports_contraction_path is backend.capabilities.contraction_path
    assert backend.supports_custom_contraction_plan is backend.capabilities.custom_contraction_plan
    assert backend.supports_distributed is backend.capabilities.distributed
    assert backend.capabilities.distributed_array is bool(
        backend.supports_distributed_array and backend.is_distributed
    )
    assert backend.supports_strided_batched_gemm is backend.capabilities.strided_batched_gemm
    assert backend.supports_point_to_point is backend.capabilities.point_to_point


def test_numpy_single_process_capabilities_do_not_advertise_distributed_arrays():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    assert backend.size == 1
    assert backend.is_distributed is False
    assert backend.capabilities.distributed is False
    assert backend.capabilities.distributed_array is False


def test_backend_collective_capability_flags_are_size_gated():
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CollectiveBackend(NumpyBackend):
        @property
        def size(self):
            return 2

        @property
        def is_distributed(self):
            return True

    single = NumpyBackend()
    collective = CollectiveBackend()

    assert single.supports_allreduce is True
    assert single.capabilities.allreduce is False
    assert single.capabilities.broadcast is False
    assert single.capabilities.allgather is False
    assert single.capabilities.reduce_scatter is False
    assert single.capabilities.alltoall is False

    assert collective.capabilities.allreduce is collective.supports_allreduce
    assert collective.capabilities.broadcast is collective.supports_broadcast
    assert collective.capabilities.allgather is collective.supports_allgather
    assert collective.capabilities.reduce_scatter is collective.supports_reduce_scatter
    assert collective.capabilities.alltoall is collective.supports_alltoall


def test_backend_capabilities_reject_incoherent_capability_sets():
    from renormalizer.backend.execution import BackendCapabilities

    with pytest.raises(ValueError, match="distributed_array requires distributed"):
        BackendCapabilities(distributed=False, distributed_array=True)

    for field in (
        "allreduce",
        "broadcast",
        "allgather",
        "reduce_scatter",
        "alltoall",
        "point_to_point",
    ):
        with pytest.raises(ValueError, match="distributed collectives require distributed"):
            BackendCapabilities(distributed=False, **{field: True})

    with pytest.raises(ValueError, match="events require streams"):
        BackendCapabilities(streams=False, events=True)

    with pytest.raises(ValueError, match="strided_batched_gemm requires batched_matmul"):
        BackendCapabilities(batched_matmul=False, strided_batched_gemm=True)

    with pytest.raises(ValueError, match="packed_blocks requires block_sparse"):
        BackendCapabilities(block_sparse=False, packed_blocks=True)

    capabilities = BackendCapabilities(
        distributed=True,
        distributed_array=True,
        allreduce=True,
        broadcast=True,
        allgather=True,
        reduce_scatter=True,
        alltoall=True,
        point_to_point=True,
        streams=True,
        events=True,
    )

    assert capabilities.distributed_array is True
    assert capabilities.allreduce is True
    assert capabilities.events is True


def test_backend_instance_capabilities_gate_packed_blocks_on_block_sparse():
    from renormalizer.backend.abstract import AbstractBackend

    class PackedWithoutBlockSparseBackend(AbstractBackend):
        supports_block_sparse = False
        supports_packed_blocks = True

    backend = PackedWithoutBlockSparseBackend()

    assert backend.capabilities.block_sparse is False
    assert backend.capabilities.packed_blocks is False


def test_device_spec_parses_cpu_and_indexed_cuda_aliases():
    from renormalizer.backend.execution import DeviceSpec, parse_device_spec

    assert parse_device_spec("cpu") == DeviceSpec(kind="cpu")
    assert parse_device_spec("gpu") == DeviceSpec(kind="cuda")
    assert parse_device_spec("cuda:1") == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert parse_device_spec("gpu:2") == DeviceSpec(kind="cuda", index=2, visible_id="2")

    with pytest.raises(ValueError, match="Unknown backend device"):
        parse_device_spec("quantum:0")


def test_device_spec_rejects_unknown_kind_and_canonicalizes_gpu_alias():
    from renormalizer.backend.execution import DeviceSpec

    assert DeviceSpec(kind="GPU", index=1) == DeviceSpec(kind="cuda", index=1)
    assert DeviceSpec(kind="distributed", local_rank=0, global_rank=1).kind == "distributed"

    with pytest.raises(ValueError, match="Unknown DeviceSpec kind"):
        DeviceSpec(kind="quantum")


def test_device_spec_rejects_negative_device_metadata():
    from renormalizer.backend.execution import DeviceSpec, parse_device_spec

    with pytest.raises(ValueError, match="DeviceSpec index must be non-negative"):
        DeviceSpec(kind="cuda", index=-1)
    with pytest.raises(ValueError, match="DeviceSpec local_rank must be non-negative"):
        DeviceSpec(kind="cuda", local_rank=-1)
    with pytest.raises(ValueError, match="DeviceSpec global_rank must be non-negative"):
        DeviceSpec(kind="cuda", global_rank=-1)
    with pytest.raises(ValueError, match="DeviceSpec index must be non-negative"):
        parse_device_spec("cuda:-1")


def test_backend_config_keeps_legacy_device_kind_and_exposes_device_spec():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec, FallbackPolicy

    config = BackendConfig(device="cuda:1", fallback_policy="record")

    assert config.device == "gpu"
    assert config.device_spec == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert config.fallback_policy is FallbackPolicy.RECORD


def test_backend_config_rejects_silent_fallback_without_explicit_opt_in():
    from renormalizer.backend import BackendConfig

    with pytest.raises(ValueError, match="silent fallback policy requires allow_silent_fallback=True"):
        BackendConfig(fallback_policy="silent")


def test_backend_config_allows_silent_fallback_with_explicit_opt_in():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import FallbackPolicy

    config = BackendConfig(fallback_policy="silent", allow_silent_fallback=True)

    assert config.fallback_policy is FallbackPolicy.SILENT
    assert config.options["allow_silent_fallback"] is True


def test_numpy_backend_capabilities_and_array_info_are_explicit():
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.backend.execution import DeviceSpec

    backend = NumpyBackend()
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    info = backend.array_info(x)

    assert backend.capabilities.matmul is True
    assert backend.capabilities.batched_matmul is True
    assert backend.capabilities.grouped_gemm is False
    assert backend.capabilities.block_sparse is True
    assert backend.capabilities.packed_blocks is False
    assert backend.capabilities.scatter_add is True
    assert backend.capabilities.distributed_array is False
    assert backend.supports_distributed_array is True
    assert backend.supports_batched_matmul is True
    assert backend.supports_grouped_gemm is False
    assert backend.current_device() == DeviceSpec(kind="cpu")
    assert backend.device_count() == 1

    assert info.shape == (2, 3)
    assert info.dtype == np.dtype("float64")
    assert info.itemsize == 8
    assert info.nbytes == 48
    assert info.device == DeviceSpec(kind="cpu")
    assert info.is_host is True
    assert info.is_device is False
    assert info.order == "C"
    assert info.contiguous is True
    assert info.backend_name == "numpy"


def test_only_cupy_advertises_grouped_gemm_in_np2_cupy_phase():
    from renormalizer.backend.cupy_backend import CupyBackend
    from renormalizer.backend.jax_backend import JaxBackend
    from renormalizer.backend.torch_backend import TorchBackend

    assert TorchBackend.supports_grouped_gemm is False
    assert JaxBackend.supports_grouped_gemm is False
    assert CupyBackend.supports_grouped_gemm is True


def _valid_array_info_kwargs():
    from renormalizer.backend import DeviceSpec

    return {
        "shape": (2, 3),
        "dtype": np.dtype("float64"),
        "itemsize": 8,
        "ndim": 2,
        "size": 6,
        "nbytes": 48,
        "device": DeviceSpec(kind="cpu"),
        "is_host": True,
        "is_device": False,
        "is_distributed": False,
        "strides": (24, 8),
        "order": "C",
        "contiguous": True,
        "writeable": True,
        "owns_data": False,
        "backend_name": "numpy",
    }


def test_array_info_rejects_inconsistent_shape_metadata():
    from renormalizer.backend import ArrayInfo

    kwargs = _valid_array_info_kwargs()
    kwargs["shape"] = (2, -1)
    with pytest.raises(ValueError, match="ArrayInfo shape dimensions must be non-negative"):
        ArrayInfo(**kwargs)

    kwargs = _valid_array_info_kwargs()
    kwargs["ndim"] = 1
    with pytest.raises(ValueError, match="ArrayInfo ndim must match shape rank"):
        ArrayInfo(**kwargs)

    kwargs = _valid_array_info_kwargs()
    kwargs["size"] = 7
    with pytest.raises(ValueError, match="ArrayInfo size must match shape product"):
        ArrayInfo(**kwargs)

    kwargs = _valid_array_info_kwargs()
    kwargs["nbytes"] = 40
    with pytest.raises(ValueError, match="ArrayInfo nbytes must match size and itemsize"):
        ArrayInfo(**kwargs)


def test_array_info_rejects_invalid_stride_rank():
    from renormalizer.backend import ArrayInfo

    kwargs = _valid_array_info_kwargs()
    kwargs["strides"] = (8,)

    with pytest.raises(ValueError, match="ArrayInfo strides must match shape rank"):
        ArrayInfo(**kwargs)


def test_array_info_rejects_unknown_order_metadata():
    from renormalizer.backend import ArrayInfo

    kwargs = _valid_array_info_kwargs()
    kwargs["order"] = "row_majorish"

    with pytest.raises(ValueError, match="ArrayInfo order must be one of"):
        ArrayInfo(**kwargs)


def test_array_info_rejects_inconsistent_distributed_location_metadata():
    from renormalizer.backend import ArrayInfo, DeviceSpec

    kwargs = _valid_array_info_kwargs()
    kwargs["is_distributed"] = True
    with pytest.raises(ValueError, match="ArrayInfo distributed arrays must use a distributed device"):
        ArrayInfo(**kwargs)

    kwargs = _valid_array_info_kwargs()
    kwargs["device"] = DeviceSpec(kind="distributed")
    with pytest.raises(ValueError, match="ArrayInfo distributed device requires is_distributed"):
        ArrayInfo(**kwargs)


def test_array_info_rejects_ambiguous_residency_metadata():
    from renormalizer.backend import ArrayInfo

    kwargs = _valid_array_info_kwargs()
    kwargs["is_host"] = True
    kwargs["is_device"] = True
    with pytest.raises(ValueError, match="ArrayInfo array cannot be both host and device resident"):
        ArrayInfo(**kwargs)

    kwargs = _valid_array_info_kwargs()
    kwargs["is_host"] = False
    kwargs["is_device"] = False
    with pytest.raises(ValueError, match="ArrayInfo non-distributed arrays must be host or device resident"):
        ArrayInfo(**kwargs)


def _valid_layout_spec_kwargs():
    return {
        "logical_shape": (2, 3),
        "physical_shape": (2, 3),
        "logical_modes": ("i", "j"),
        "strides": (24, 8),
        "order": "C",
        "contiguous_groups": ((0, 1),),
        "requires_transpose": False,
        "transpose_perm": None,
        "estimated_copy_bytes": 0,
    }


def test_layout_spec_rejects_inconsistent_rank_metadata():
    from renormalizer.backend import LayoutSpec

    kwargs = _valid_layout_spec_kwargs()
    kwargs["logical_shape"] = (2, -1)
    with pytest.raises(ValueError, match="LayoutSpec logical_shape dimensions must be non-negative"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["logical_modes"] = ("i",)
    with pytest.raises(ValueError, match="LayoutSpec logical_modes must match logical_shape rank"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["logical_modes"] = ("i", "i")
    with pytest.raises(ValueError, match="LayoutSpec logical_modes must be unique"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["strides"] = (8,)
    with pytest.raises(ValueError, match="LayoutSpec strides must match physical shape rank"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["order"] = "row_majorish"
    with pytest.raises(ValueError, match="LayoutSpec order must be one of"):
        LayoutSpec(**kwargs)


def test_layout_spec_rejects_invalid_transform_metadata():
    from renormalizer.backend import LayoutSpec

    kwargs = _valid_layout_spec_kwargs()
    kwargs["contiguous_groups"] = ((0, 2),)
    with pytest.raises(ValueError, match="LayoutSpec contiguous_groups indices out of range"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["contiguous_groups"] = ((0, 0),)
    with pytest.raises(ValueError, match="LayoutSpec contiguous_groups axes must be unique"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["contiguous_groups"] = ((0,), (0, 1))
    with pytest.raises(ValueError, match="LayoutSpec contiguous_groups axes must be unique"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["transpose_perm"] = (0, 0)
    with pytest.raises(ValueError, match="LayoutSpec transpose_perm must be a permutation"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["requires_transpose"] = True
    kwargs["transpose_perm"] = None
    with pytest.raises(ValueError, match="LayoutSpec transpose_perm is required when requires_transpose is true"):
        LayoutSpec(**kwargs)

    kwargs = _valid_layout_spec_kwargs()
    kwargs["estimated_copy_bytes"] = -1
    with pytest.raises(ValueError, match="LayoutSpec estimated_copy_bytes must be non-negative"):
        LayoutSpec(**kwargs)


def test_numpy_copy_policy_rejects_required_copy_and_allows_explicit_copy():
    from renormalizer.backend.execution import BackendCopyError, CopyPolicy
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.array([1.0, 2.0])

    y = backend.to_backend(x, copy=CopyPolicy.NEVER)
    assert y is x

    z = backend.to_backend(x, copy=CopyPolicy.ALWAYS)
    assert np.array_equal(z, x)
    assert not np.shares_memory(z, x)

    with pytest.raises(BackendCopyError):
        backend.to_backend([1.0, 2.0], copy=CopyPolicy.NEVER)


def test_numpy_to_host_never_copy_policy_rejects_before_materializing_array_like():
    from renormalizer.backend.execution import BackendCopyError, CopyPolicy
    from renormalizer.backend.numpy_backend import NumpyBackend

    class ArrayLike:
        def __init__(self):
            self.array_calls = 0

        def __array__(self, dtype=None):
            self.array_calls += 1
            return np.asarray([1.0, 2.0], dtype=dtype)

    backend = NumpyBackend()
    source = ArrayLike()

    with pytest.raises(BackendCopyError, match="to_host would require"):
        backend.to_host(source, copy=CopyPolicy.NEVER)

    assert source.array_calls == 0


def test_numpy_layout_transform_api_tracks_view_and_contiguous_copy():
    from renormalizer.backend.execution import BackendCopyError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.arange(12, dtype=np.float64).reshape(3, 4)

    transposed = backend.permute(x, (1, 0))
    assert np.array_equal(transposed, x.T)
    assert np.shares_memory(transposed, x)

    assert backend.can_reshape_view(x, (4, 3)) is True
    reshaped = backend.reshape_view(x, (4, 3))
    assert reshaped.shape == (4, 3)
    assert np.shares_memory(reshaped, x)

    strided = x[::2, ::2]
    assert backend.can_reshape_view(strided, (4,)) is False
    with pytest.raises(BackendCopyError, match="reshape would require a copy"):
        backend.reshape_view(strided, (4,))

    contiguous = backend.make_contiguous(transposed)
    assert contiguous.flags["C_CONTIGUOUS"]
    assert np.array_equal(contiguous, transposed)
    assert not np.shares_memory(contiguous, transposed)


def test_permute_never_copy_policy_rejects_copying_backend_transpose():
    from renormalizer.backend.execution import BackendCopyError, CopyPolicy
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CopyingTransposeNamespace:
        @staticmethod
        def transpose(x, perm):
            return np.array(np.transpose(x, perm), copy=True)

    backend = NumpyBackend()
    backend.array_namespace = CopyingTransposeNamespace
    x = np.arange(12, dtype=np.float64).reshape(3, 4)

    with pytest.raises(BackendCopyError, match="permute would require a copy"):
        backend.permute(x, (1, 0), copy_policy=CopyPolicy.NEVER)


def test_make_contiguous_accepts_mode_groups_for_gemm_layout_boundary():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    transposed = backend.permute(x, (1, 0, 2))

    contiguous = backend.make_contiguous(transposed, mode_groups=((0, 1), (2,)))

    assert contiguous.flags["C_CONTIGUOUS"]
    assert np.array_equal(contiguous, transposed)
    assert not np.shares_memory(contiguous, transposed)


def test_make_contiguous_rejects_invalid_mode_groups():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    with pytest.raises(ValueError, match="repeat"):
        backend.make_contiguous(x, mode_groups=((0, 1), (1, 2)))

    with pytest.raises(ValueError, match="valid.*rank"):
        backend.make_contiguous(x, mode_groups=((3,),))

    with pytest.raises(ValueError, match="adjacent"):
        backend.make_contiguous(x, mode_groups=((0, 2),))


def test_backend_synchronize_alias_delegates_to_sync():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    calls = []

    def fake_sync():
        calls.append("sync")
        return "synced"

    backend.sync = fake_sync

    assert backend.synchronize(device=None, stream=None) == "synced"
    assert calls == ["sync"]


def test_backend_synchronize_rejects_wrong_device():
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    assert backend.synchronize(device="cpu") is None
    with pytest.raises(BackendFeatureError, match="synchronize device"):
        backend.synchronize(device="cuda")


def test_device_mesh_and_sharding_spec_compute_local_slices():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    spec = ShardingSpec(
        global_shape=(5, 3),
        modes=("x", "y"),
        mesh=mesh,
        ranks_per_mode={"x": 2},
        mode_to_mesh_axis={"x": "rank"},
    )

    assert mesh.world_size == 2
    assert spec.sharded_modes == ("x",)
    assert spec.replicated_modes == ("y",)
    assert spec.local_slices == {
        0: (slice(0, 3), slice(None)),
        1: (slice(3, 5), slice(None)),
    }


def test_device_mesh_rejects_duplicate_axis_names():
    from renormalizer.backend import DeviceMesh, DeviceSpec

    with pytest.raises(ValueError, match="DeviceMesh axis_names must be unique"):
        DeviceMesh(
            devices=(
                DeviceSpec("cpu", global_rank=0),
                DeviceSpec("cpu", global_rank=1),
                DeviceSpec("cpu", global_rank=2),
                DeviceSpec("cpu", global_rank=3),
            ),
            shape=(2, 2),
            axis_names=("rank", "rank"),
            backend="numpy",
        )


def test_device_mesh_rejects_duplicate_device_descriptors():
    from renormalizer.backend import DeviceMesh, DeviceSpec

    with pytest.raises(ValueError, match="DeviceMesh devices must be unique"):
        DeviceMesh(
            devices=(DeviceSpec("cuda", index=0), DeviceSpec("cuda", index=0)),
            shape=(2,),
            axis_names=("rank",),
            backend="cupy",
        )

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
    )
    assert mesh.world_size == 2


def test_sharding_spec_rejects_invalid_global_shape_and_duplicate_modes():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    kwargs = {
        "global_shape": (5, 3),
        "modes": ("x", "y"),
        "mesh": mesh,
        "ranks_per_mode": {"x": 2},
        "mode_to_mesh_axis": {"x": "rank"},
    }

    with pytest.raises(ValueError, match="ShardingSpec global_shape dimensions must be non-negative"):
        ShardingSpec(**{**kwargs, "global_shape": (-5, 3)})

    with pytest.raises(ValueError, match="ShardingSpec modes must be unique"):
        ShardingSpec(**{**kwargs, "modes": ("x", "x"), "ranks_per_mode": {}, "mode_to_mesh_axis": {}})


def test_sharding_spec_rejects_replicated_mode_with_sharded_rank_count():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )

    with pytest.raises(ValueError, match="replicated modes must have ranks_per_mode equal to 1"):
        ShardingSpec(
            global_shape=(5, 3),
            modes=("x", "y"),
            mesh=mesh,
            ranks_per_mode={"x": 2, "y": 2},
            mode_to_mesh_axis={"y": "rank"},
            sharded_modes=("y",),
            replicated_modes=("x",),
        )


def test_sharding_spec_rejects_more_shards_than_mode_extent():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(4)),
        shape=(4,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )

    with pytest.raises(ValueError, match="ranks_per_mode.*must not exceed global dimension"):
        ShardingSpec(
            global_shape=(2,),
            modes=("x",),
            mesh=mesh,
            ranks_per_mode={"x": 4},
            mode_to_mesh_axis={"x": "rank"},
        )


def test_sharding_spec_rejects_invalid_explicit_local_slices():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    kwargs = {
        "global_shape": (5, 3),
        "modes": ("x", "y"),
        "mesh": mesh,
        "ranks_per_mode": {"x": 2},
        "mode_to_mesh_axis": {"x": "rank"},
    }

    with pytest.raises(ValueError, match="ShardingSpec local_slices ranks must match mesh ranks"):
        ShardingSpec(**{**kwargs, "local_slices": {0: (slice(None), slice(None))}})

    with pytest.raises(ValueError, match="ShardingSpec local_slices entries must match global_shape rank"):
        ShardingSpec(**{**kwargs, "local_slices": {0: (slice(0, 3),), 1: (slice(3, 5),)}})

    with pytest.raises(ValueError, match="ShardingSpec local_slices entries must be slice objects"):
        ShardingSpec(**{**kwargs, "local_slices": {0: (slice(0, 3), slice(None)), 1: (3, slice(None))}})

    with pytest.raises(ValueError, match="ShardingSpec local_slices entries must have unit step"):
        ShardingSpec(
            **{
                **kwargs,
                "local_slices": {
                    0: (slice(0, 3, 2), slice(None)),
                    1: (slice(3, 5), slice(None)),
                },
            }
        )

    with pytest.raises(ValueError, match="ShardingSpec local_slices entries must stay within global_shape"):
        ShardingSpec(
            **{
                **kwargs,
                "local_slices": {
                    0: (slice(-1, 3), slice(None)),
                    1: (slice(3, 6), slice(None)),
                },
            }
        )

    with pytest.raises(ValueError, match="replicated local_slices must cover the full replicated mode"):
        ShardingSpec(
            **{
                **kwargs,
                "local_slices": {
                    0: (slice(0, 3), slice(0, 2)),
                    1: (slice(3, 5), slice(0, 2)),
                },
            }
        )

    with pytest.raises(ValueError, match="sharded local_slices must exactly cover global_shape"):
        ShardingSpec(
            **{
                **kwargs,
                "local_slices": {
                    0: (slice(0, 4), slice(None)),
                    1: (slice(3, 5), slice(None)),
                },
            }
        )

    with pytest.raises(ValueError, match="sharded local_slices must exactly cover global_shape"):
        ShardingSpec(
            **{
                **kwargs,
                "local_slices": {
                    0: (slice(0, 2), slice(None)),
                    1: (slice(3, 5), slice(None)),
                },
            }
        )

    with pytest.raises(ValueError, match="sharded local_slices must assign non-empty shards"):
        ShardingSpec(
            **{
                **kwargs,
                "local_slices": {
                    0: (slice(0, 0), slice(None)),
                    1: (slice(0, 5), slice(None)),
                },
            }
        )


def test_numpy_shard_gather_and_redistribute_roundtrip():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    row_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    col_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"col": 2},
        mode_to_mesh_axis={"col": "rank"},
    )
    x = np.arange(20, dtype=np.float64).reshape(5, 4)

    distributed = backend.shard_tensor(x, row_spec)
    redistributed = backend.redistribute(distributed, col_spec)

    assert backend.is_distributed_array(distributed) is True
    assert distributed.global_shape == (5, 4)
    assert distributed.modes == ("row", "col")
    assert distributed.local_shape == (3, 4)
    assert np.array_equal(distributed.local_array, x[:3, :])
    assert np.array_equal(backend.gather_tensor(distributed), x)
    assert redistributed.sharding == col_spec
    assert redistributed.local_shape == (5, 2)
    assert np.array_equal(redistributed.local_array, x[:, :2])
    assert np.array_equal(backend.gather_tensor(redistributed), x)


def test_shard_tensor_rejects_dense_input_shape_mismatch():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    x = np.arange(24, dtype=np.float64).reshape(6, 4)

    with pytest.raises(ValueError, match="shard_tensor input shape .* must match ShardingSpec global_shape"):
        backend.shard_tensor(x, spec)


def test_real_distributed_shard_and_gather_use_rank_local_collectives():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CollectiveNumpyBackend(NumpyBackend):
        @property
        def rank(self):
            return 1

        @property
        def size(self):
            return 2

        @property
        def is_distributed(self):
            return True

        def allgather(self, x):
            return [
                np.arange(20, dtype=np.float64).reshape(5, 4)[:3, :],
                x,
            ]

    backend = CollectiveNumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=1,
        global_rank=1,
    )
    x = np.arange(20, dtype=np.float64).reshape(5, 4)
    spec = ShardingSpec(
        global_shape=x.shape,
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )

    distributed = backend.shard_tensor(x, spec)
    gathered = backend.gather_tensor(distributed)

    assert distributed.rank_local_arrays is None
    assert distributed.local_shape == (2, 4)
    assert np.array_equal(distributed.local_array, x[3:, :])
    assert np.array_equal(gathered, x)


def test_real_distributed_redistribute_same_mesh_axes_uses_alltoall():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CollectiveNumpyBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.alltoall_calls = []

        @property
        def rank(self):
            return 1

        @property
        def size(self):
            return 2

        @property
        def is_distributed(self):
            return True

        def allgather(self, x):
            raise AssertionError("redistribute should not gather the full tensor")

        def alltoall(self, x, split_axis=0, concat_axis=0):
            self.alltoall_calls.append((tuple(x.shape), split_axis, concat_axis))
            assert np.array_equal(x, np.arange(20, dtype=np.float64).reshape(5, 4)[3:, :])
            return np.arange(20, dtype=np.float64).reshape(5, 4)[:, 2:]

    backend = CollectiveNumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=1,
        global_rank=1,
    )
    x = np.arange(20, dtype=np.float64).reshape(5, 4)
    row_spec = ShardingSpec(
        global_shape=x.shape,
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    col_spec = ShardingSpec(
        global_shape=x.shape,
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"col": 2},
        mode_to_mesh_axis={"col": "rank"},
    )
    distributed = backend.shard_tensor(x, row_spec)

    redistributed = backend.redistribute(distributed, col_spec)

    assert backend.alltoall_calls == [((2, 4), 1, 0)]
    assert redistributed.rank_local_arrays is None
    assert redistributed.sharding == col_spec
    assert redistributed.local_shape == (5, 2)
    assert np.array_equal(redistributed.local_array, x[:, 2:])


def test_torch_alltoall_fallback_handles_empty_uneven_rank_chunks():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))

    from renormalizer.backend.mpi import TorchDistributedMixin

    class FallbackAlltoallBackend(TorchDistributedMixin):
        @property
        def array_namespace(self):
            return torch

        @property
        def rank(self):
            return 3

        @property
        def size(self):
            return 4

        def _distributed_ready(self):
            return True

        def allgather(self, x, axis=None):
            assert axis is None
            return [
                torch.arange(rank * 6, rank * 6 + 6, dtype=torch.float64).reshape(2, 3)
                for rank in range(self.size)
            ]

    backend = FallbackAlltoallBackend()
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    result = backend.alltoall(x, split_axis=0, concat_axis=1)

    assert tuple(result.shape) == (0, 12)
    assert result.dtype == x.dtype


def test_replicate_tensor_and_single_process_collectives():
    from renormalizer.backend import DeviceMesh, DeviceSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    x = np.arange(6, dtype=np.float64).reshape(2, 3)

    distributed = backend.replicate_tensor(x, mesh, modes=("i", "j"))

    assert distributed.sharding.sharded_modes == ()
    assert distributed.sharding.replicated_modes == ("i", "j")
    assert np.array_equal(distributed.local_array, x)
    assert np.array_equal(distributed.rank_local_arrays[0], x)
    assert np.array_equal(distributed.rank_local_arrays[1], x)
    assert np.array_equal(backend.gather_tensor(distributed), x)
    assert backend.reduce_scatter(x) is x
    assert backend.alltoall(x) is x
    assert backend.allgather(x, axis=0) is x


def test_single_process_point_to_point_validates_peer_and_returns_like_buffer():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.arange(6, dtype=np.float64).reshape(2, 3)

    assert backend.send(x, dst=0) is None
    assert backend.recv(src=0, like=x) is x

    with pytest.raises(ValueError, match="send dst 1 is out of range for world size 1"):
        backend.send(x, dst=1)
    with pytest.raises(ValueError, match="recv src 1 is out of range for world size 1"):
        backend.recv(src=1, like=x)
    with pytest.raises(ValueError, match="recv requires a like buffer in single-process mode"):
        backend.recv(src=0)


def test_torch_point_to_point_dispatches_tensor_send_recv_to_distributed_runtime():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))

    from renormalizer.backend.mpi import TorchDistributedMixin
    from renormalizer.backend.torch_backend import TorchBackend

    assert TorchBackend.supports_point_to_point is True

    class FakeDistributed:
        def __init__(self):
            self.sent = []
            self.recv_calls = []

        def send(self, tensor, dst, tag=0):
            self.sent.append((tensor.clone(), int(dst), int(tag)))

        def recv(self, tensor, src, tag=0):
            self.recv_calls.append((tuple(tensor.shape), int(src), int(tag)))
            tensor.copy_(torch.full_like(tensor, 7.0))

    class FakeTorchBackend(TorchDistributedMixin):
        array_namespace = torch
        rank = 1
        size = 2

        def __init__(self):
            self._distributed = FakeDistributed()

        def _distributed_ready(self):
            return True

    backend = FakeTorchBackend()
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    assert backend.send(x, dst=0, tag=5) is None
    received = backend.recv(src=0, tag=6, like=x)

    assert len(backend._distributed.sent) == 1
    sent_tensor, dst, send_tag = backend._distributed.sent[0]
    assert torch.equal(sent_tensor, x)
    assert dst == 0
    assert send_tag == 5
    assert backend._distributed.recv_calls == [((2, 3), 0, 6)]
    assert torch.equal(received, torch.full_like(x, 7.0))
    assert received is not x


def test_torch_collective_roots_are_validated_before_runtime_dispatch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))

    from renormalizer.backend.mpi import TorchDistributedMixin

    class FakeDistributed:
        def broadcast(self, tensor, src):
            raise AssertionError("invalid root should be rejected before torch.distributed.broadcast")

        def broadcast_object_list(self, values, src):
            raise AssertionError("invalid root should be rejected before torch.distributed.broadcast_object_list")

    class FakeTorchBackend(TorchDistributedMixin):
        array_namespace = torch
        rank = 0
        size = 2

        def __init__(self):
            self._distributed = FakeDistributed()

        def _distributed_ready(self):
            return True

    backend = FakeTorchBackend()
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    with pytest.raises(ValueError, match="broadcast root 2 is out of range for world size 2"):
        backend.broadcast(x, root=2)
    with pytest.raises(ValueError, match="gather root -1 is out of range for world size 2"):
        backend.gather(x, root=-1)


def test_torch_nonready_collectives_validate_axis_and_reduce_op_before_fallback():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))

    from renormalizer.backend.mpi import TorchDistributedMixin

    class NonReadyTorchBackend(TorchDistributedMixin):
        array_namespace = torch

        def _distributed_ready(self):
            return False

    backend = NonReadyTorchBackend()
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    assert backend.allreduce(x, op="sum") is x
    assert backend.allgather(x, axis=-1) is x
    assert backend.reduce_scatter(x, op="sum", axis=-1) is x
    assert backend.alltoall(x, split_axis=-1, concat_axis=0) is x

    with pytest.raises(ValueError, match="unsupported distributed reduction op 'median'"):
        backend.allreduce(x, op="median")
    with pytest.raises(ValueError, match="allgather axis 2 is out of range for rank 2"):
        backend.allgather(x, axis=2)
    with pytest.raises(ValueError, match="unsupported distributed reduction op 'median'"):
        backend.reduce_scatter(x, op="median", axis=0)
    with pytest.raises(ValueError, match="reduce_scatter axis 2 is out of range for rank 2"):
        backend.reduce_scatter(x, op="sum", axis=2)
    with pytest.raises(ValueError, match="alltoall split_axis 2 is out of range for rank 2"):
        backend.alltoall(x, split_axis=2, concat_axis=0)
    with pytest.raises(ValueError, match="alltoall concat_axis 2 is out of range for rank 2"):
        backend.alltoall(x, split_axis=0, concat_axis=2)


def test_torch_tensor_gather_uses_native_gather_instead_of_allgather():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))

    from renormalizer.backend.mpi import TorchDistributedMixin

    class FakeDistributed:
        def __init__(self):
            self.gather_calls = []
            self.broadcast_calls = []

        def gather(self, tensor, gather_list=None, dst=0):
            self.gather_calls.append((tensor.clone(), gather_list, int(dst)))
            if gather_list is not None:
                if tensor.dtype == torch.long:
                    gather_list[0].copy_(torch.tensor([2, 3], dtype=torch.long, device=tensor.device))
                    gather_list[1].copy_(torch.tensor([2, 3], dtype=torch.long, device=tensor.device))
                else:
                    gather_list[0].copy_(torch.zeros_like(tensor))
                    gather_list[1].copy_(tensor)

        def broadcast(self, tensor, src):
            self.broadcast_calls.append((tensor.clone(), int(src)))
            tensor.copy_(torch.tensor([2, 3], dtype=tensor.dtype, device=tensor.device))

        def all_gather(self, *args, **kwargs):
            raise AssertionError("tensor gather should not dispatch through all_gather")

    class RootTorchBackend(TorchDistributedMixin):
        array_namespace = torch
        rank = 1
        size = 2

        def __init__(self):
            self._distributed = FakeDistributed()

        def _distributed_ready(self):
            return True

    backend = RootTorchBackend()
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    gathered = backend.gather(x, root=1)

    assert len(backend._distributed.gather_calls) == 2
    shape_tensor, shape_gather_list, shape_dst = backend._distributed.gather_calls[0]
    assert torch.equal(shape_tensor, torch.tensor([2, 3], dtype=torch.long))
    assert shape_gather_list is not None
    assert shape_dst == 1
    sent_tensor, gather_list, dst = backend._distributed.gather_calls[1]
    assert torch.equal(sent_tensor, x)
    assert dst == 1
    assert gather_list is not None
    assert len(backend._distributed.broadcast_calls) == 1
    broadcast_tensor, broadcast_src = backend._distributed.broadcast_calls[0]
    assert torch.equal(broadcast_tensor, torch.tensor([2, 3], dtype=torch.long))
    assert broadcast_src == 1
    assert len(gathered) == 2
    assert torch.equal(gathered[0], torch.zeros_like(x))
    assert torch.equal(gathered[1], x)


def test_torch_tensor_gather_preserves_uneven_local_shapes_with_native_gather():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))

    from renormalizer.backend.mpi import TorchDistributedMixin

    class FakeDistributed:
        def __init__(self):
            self.gather_calls = []
            self.broadcast_calls = []

        def gather(self, tensor, gather_list=None, dst=0):
            self.gather_calls.append((tuple(tensor.shape), tensor.dtype, gather_list is None, int(dst)))
            if tensor.dtype == torch.long:
                assert tuple(tensor.shape) == (2,)
                assert gather_list is not None
                gather_list[0].copy_(torch.tensor([2, 3], dtype=torch.long, device=tensor.device))
                gather_list[1].copy_(torch.tensor([1, 3], dtype=torch.long, device=tensor.device))
                return

            assert tuple(tensor.shape) == (2, 3), "gather should pad uneven tensors before native gather"
            assert gather_list is not None
            gather_list[0].copy_(torch.arange(6, dtype=tensor.dtype, device=tensor.device).reshape(2, 3))
            gather_list[1].zero_()
            gather_list[1][:1, :3].copy_(torch.full((1, 3), 7.0, dtype=tensor.dtype, device=tensor.device))

        def broadcast(self, tensor, src):
            self.broadcast_calls.append((tuple(tensor.shape), int(src)))
            tensor.copy_(torch.tensor([2, 3], dtype=tensor.dtype, device=tensor.device))

        def all_gather(self, *args, **kwargs):
            raise AssertionError("tensor gather should not dispatch through all_gather")

    class RootTorchBackend(TorchDistributedMixin):
        array_namespace = torch
        rank = 1
        size = 2

        def __init__(self):
            self._distributed = FakeDistributed()

        def _distributed_ready(self):
            return True

    backend = RootTorchBackend()
    x = torch.full((1, 3), 7.0, dtype=torch.float64)

    gathered = backend.gather(x, root=1)

    assert [tuple(item.shape) for item in gathered] == [(2, 3), (1, 3)]
    assert torch.equal(gathered[0], torch.arange(6, dtype=torch.float64).reshape(2, 3))
    assert torch.equal(gathered[1], x)
    assert [(shape, gather_list_is_none, dst) for shape, _, gather_list_is_none, dst in backend._distributed.gather_calls] == [
        ((2,), False, 1),
        ((2, 3), False, 1),
    ]
    assert backend._distributed.broadcast_calls == [((2,), 1)]


def test_torch_tensor_gather_nonroot_sends_without_gather_list():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))

    from renormalizer.backend.mpi import TorchDistributedMixin

    class FakeDistributed:
        def __init__(self):
            self.gather_calls = []
            self.broadcast_calls = []

        def gather(self, tensor, gather_list=None, dst=0):
            self.gather_calls.append((tensor.clone(), gather_list, int(dst)))
            if gather_list is not None:
                raise AssertionError("non-root gather must not allocate a gather_list")

        def broadcast(self, tensor, src):
            self.broadcast_calls.append((tuple(tensor.shape), int(src)))
            tensor.copy_(torch.tensor([2, 3], dtype=tensor.dtype, device=tensor.device))

        def all_gather(self, *args, **kwargs):
            raise AssertionError("tensor gather should not dispatch through all_gather")

    class NonRootTorchBackend(TorchDistributedMixin):
        array_namespace = torch
        rank = 0
        size = 2

        def __init__(self):
            self._distributed = FakeDistributed()

        def _distributed_ready(self):
            return True

    backend = NonRootTorchBackend()
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    gathered = backend.gather(x, root=1)

    assert gathered is None
    assert len(backend._distributed.gather_calls) == 2
    shape_tensor, shape_gather_list, shape_dst = backend._distributed.gather_calls[0]
    assert torch.equal(shape_tensor, torch.tensor([2, 3], dtype=torch.long))
    assert shape_gather_list is None
    assert shape_dst == 1
    sent_tensor, gather_list, dst = backend._distributed.gather_calls[1]
    assert torch.equal(sent_tensor, x)
    assert gather_list is None
    assert dst == 1
    assert backend._distributed.broadcast_calls == [((2,), 1)]


def test_gather_tensor_respects_explicit_root_for_rank_local_arrays():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=1,
        global_rank=1,
    )
    x = np.arange(20, dtype=np.float64).reshape(5, 4)
    spec = ShardingSpec(
        global_shape=x.shape,
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    distributed = backend.shard_tensor(x, spec)

    assert np.array_equal(backend.gather_tensor(distributed), x)
    assert np.array_equal(backend.gather_tensor(distributed, root=1), x)
    assert backend.gather_tensor(distributed, root=0) is None

    with pytest.raises(ValueError, match="gather_tensor root 2 is out of range for world size 2"):
        backend.gather_tensor(distributed, root=2)


def test_single_process_collectives_validate_axis_metadata():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.arange(6, dtype=np.float64).reshape(2, 3)

    assert backend.allgather(x, axis=-1) is x
    assert backend.reduce_scatter(x, axis=-1) is x
    assert backend.alltoall(x, split_axis=-1, concat_axis=0) is x

    with pytest.raises(ValueError, match="allgather axis 2 is out of range for rank 2"):
        backend.allgather(x, axis=2)
    with pytest.raises(ValueError, match="reduce_scatter axis 2 is out of range for rank 2"):
        backend.reduce_scatter(x, axis=2)
    with pytest.raises(ValueError, match="alltoall split_axis 2 is out of range for rank 2"):
        backend.alltoall(x, split_axis=2)
    with pytest.raises(ValueError, match="alltoall concat_axis 2 is out of range for rank 2"):
        backend.alltoall(x, concat_axis=2)


def test_single_process_collectives_validate_root_and_reduction_metadata():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.arange(6, dtype=np.float64).reshape(2, 3)

    assert backend.broadcast(x, root=0) is x
    assert backend.gather(x, root=0) == [x]
    assert backend.allreduce(x, op="sum") is x
    assert backend.reduce_scatter(x, op="max") is x

    with pytest.raises(ValueError, match="broadcast root 1 is out of range for world size 1"):
        backend.broadcast(x, root=1)
    with pytest.raises(ValueError, match="gather root 1 is out of range for world size 1"):
        backend.gather(x, root=1)
    with pytest.raises(ValueError, match="unsupported distributed reduction op 'median'"):
        backend.allreduce(x, op="median")
    with pytest.raises(ValueError, match="unsupported distributed reduction op 'median'"):
        backend.reduce_scatter(x, op="median")


def test_replicate_tensor_gathers_distributed_input_before_replicating():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedTensor, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    sharding = ShardingSpec(
        global_shape=x.shape,
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    sharded = backend.shard_tensor(x, sharding)

    replicated = backend.replicate_tensor(sharded, mesh, modes=("i", "j"))

    assert isinstance(replicated.local_array, np.ndarray)
    assert not isinstance(replicated.local_array, DistributedTensor)
    assert replicated.sharding.sharded_modes == ()
    assert replicated.sharding.replicated_modes == ("i", "j")
    assert np.array_equal(replicated.local_array, x)
    assert np.array_equal(replicated.rank_local_arrays[0], x)
    assert np.array_equal(replicated.rank_local_arrays[1], x)
    assert np.array_equal(backend.gather_tensor(replicated), x)


def test_distributed_tensor_is_backend_array_and_reports_distributed_info():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    sharding = ShardingSpec(
        global_shape=x.shape,
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )

    distributed = backend.shard_tensor(x, sharding)
    info = backend.array_info(distributed)

    assert backend.is_array(distributed) is True
    assert backend.is_distributed_array(distributed) is True
    assert info.shape == x.shape
    assert info.is_distributed is True
    assert info.device == DeviceSpec(kind="distributed", local_rank=0, global_rank=0)
    assert info.nbytes == distributed.local_nbytes


def test_distributed_tensor_layout_reports_logical_and_local_physical_shape():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    sharding = ShardingSpec(
        global_shape=x.shape,
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )

    distributed = backend.shard_tensor(x, sharding)
    layout = backend.layout(distributed)

    assert layout.logical_shape == x.shape
    assert layout.physical_shape == distributed.local_shape
    assert layout.logical_modes == ("i", "j")
    assert layout.order == "distributed"
    assert layout.estimated_copy_bytes == 0


def _valid_distributed_tensor_kwargs():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    sharding = ShardingSpec(
        global_shape=(4, 3),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    return {
        "local_array": np.arange(6, dtype=np.float64).reshape(2, 3),
        "global_shape": (4, 3),
        "modes": ("i", "j"),
        "sharding": sharding,
        "mesh": mesh,
    }


def test_distributed_tensor_rejects_inconsistent_global_metadata():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedTensor, ShardingSpec

    kwargs = _valid_distributed_tensor_kwargs()

    with pytest.raises(ValueError, match="DistributedTensor global_shape dimensions must be non-negative"):
        DistributedTensor(**{**kwargs, "global_shape": (-1, 3)})

    with pytest.raises(ValueError, match="DistributedTensor modes must match global_shape rank"):
        DistributedTensor(**{**kwargs, "modes": ("i",)})

    mesh = kwargs["mesh"]
    shape_mismatch = ShardingSpec(
        global_shape=(5, 3),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    with pytest.raises(ValueError, match="DistributedTensor sharding global_shape must match global_shape"):
        DistributedTensor(**{**kwargs, "sharding": shape_mismatch})

    mode_mismatch = ShardingSpec(
        global_shape=(4, 3),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    with pytest.raises(ValueError, match="DistributedTensor sharding modes must match modes"):
        DistributedTensor(**{**kwargs, "sharding": mode_mismatch})

    other_mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="other",
        local_rank=0,
        global_rank=0,
    )
    with pytest.raises(ValueError, match="DistributedTensor mesh must match sharding mesh"):
        DistributedTensor(**{**kwargs, "mesh": other_mesh})


def test_distributed_tensor_rejects_invalid_local_metadata():
    from renormalizer.backend import DistributedTensor

    kwargs = _valid_distributed_tensor_kwargs()

    with pytest.raises(ValueError, match="DistributedTensor local_shape dimensions must be non-negative"):
        DistributedTensor(**{**kwargs, "local_shape": (-1, 3)})

    with pytest.raises(ValueError, match="DistributedTensor local_shape must match global_shape rank"):
        DistributedTensor(**{**kwargs, "local_shape": (2,)})

    with pytest.raises(ValueError, match="DistributedTensor local_nbytes must be non-negative"):
        DistributedTensor(**{**kwargs, "local_nbytes": -1})

    with pytest.raises(ValueError, match="DistributedTensor rank_local_arrays ranks must be present in sharding local_slices"):
        DistributedTensor(**{**kwargs, "rank_local_arrays": {2: np.zeros((2, 3))}})


def test_distributed_tensor_rejects_local_shape_that_disagrees_with_sharding_slice():
    from renormalizer.backend import DistributedTensor

    kwargs = _valid_distributed_tensor_kwargs()

    with pytest.raises(ValueError, match="DistributedTensor local_shape must match sharding local slice"):
        DistributedTensor(**{**kwargs, "local_array": np.zeros((3, 3))})

    with pytest.raises(ValueError, match="DistributedTensor local_shape must match sharding local slice"):
        DistributedTensor(**{**kwargs, "local_shape": (1, 3)})


def test_distributed_tensor_rejects_rank_local_array_shape_mismatch():
    from renormalizer.backend import DistributedTensor

    kwargs = _valid_distributed_tensor_kwargs()

    with pytest.raises(ValueError, match="DistributedTensor rank_local_arrays shapes must match sharding local_slices"):
        DistributedTensor(
            **{
                **kwargs,
                "rank_local_arrays": {
                    0: np.zeros((2, 3)),
                    1: np.zeros((3, 3)),
                },
            }
        )


def test_distributed_capabilities_advertise_collective_primitives():
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CollectiveBackend(NumpyBackend):
        @property
        def size(self):
            return 2

        @property
        def is_distributed(self):
            return True

    capabilities = CollectiveBackend().capabilities

    assert capabilities.distributed is True
    assert capabilities.allreduce is True
    assert capabilities.broadcast is True
    assert capabilities.allgather is True
    assert capabilities.reduce_scatter is True
    assert capabilities.alltoall is True


def test_jax_layout_transform_api_allows_same_size_device_reshape_when_available():
    try:
        __import__("jax")
    except (ImportError, OSError) as exc:
        pytest.skip("jax unavailable: {0}".format(exc))

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.jax_backend import JaxBackend

    try:
        backend = JaxBackend(config=BackendConfig(device="cpu"))
    except (ImportError, ValueError, RuntimeError) as exc:
        pytest.skip("jax backend unavailable: {0}".format(exc))
    x = backend.to_backend(np.arange(12, dtype=np.float64).reshape(3, 4))

    assert backend.can_reshape_view(x, (4, 3)) is True
    reshaped = backend.reshape_view(x, (4, 3))
    assert tuple(reshaped.shape) == (4, 3)


def test_packed_vector_spec_roundtrips_single_and_batched_rhs():
    from renormalizer.backend.execution import PackedVectorSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mask = np.array([[True, False, True], [False, True, False]])

    single_spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=3, nrhs=1)
    single = np.array([1.0, 2.0, 3.0])
    single_struct = backend.unpack_masked_vectors(single, single_spec)
    assert single_struct.shape == mask.shape
    assert np.array_equal(single_struct[mask], single)
    assert np.array_equal(single_struct[~mask], np.zeros(np.count_nonzero(~mask)))
    assert np.array_equal(backend.pack_masked_vectors(single_struct, single_spec), single)

    batched_spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=3, nrhs=2)
    batched = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    batched_struct = backend.unpack_masked_vectors(batched, batched_spec)
    assert batched_struct.shape == mask.shape + (2,)
    assert np.array_equal(batched_struct[mask], batched)
    assert np.array_equal(backend.pack_masked_vectors(batched_struct, batched_spec), batched)


def test_packed_vector_spec_supports_non_last_batch_axis():
    from renormalizer.backend.execution import PackedVectorSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mask = np.array([[True, False, True], [False, True, False]])
    spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, batch_axis=0, packed_dim=3, nrhs=2)
    packed = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    struct = backend.unpack_masked_vectors(packed, spec)

    assert struct.shape == (2,) + mask.shape
    assert np.array_equal(np.moveaxis(struct, 0, -1)[mask], packed)
    assert np.array_equal(backend.pack_masked_vectors(struct, spec), packed)


def test_packed_vector_backend_mask_does_not_roundtrip_through_host():
    from renormalizer.backend.execution import PackedVectorSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    class BackendResidentMask(NumpyBackend):
        host_array_types = ()
        device_array_types = (np.ndarray,)

        def __init__(self):
            super().__init__()
            self.forbidden_host_conversions = 0

        def to_numpy(self, x):
            if getattr(x, "_forbid_host_conversion", False):
                self.forbidden_host_conversions += 1
                raise AssertionError("backend-resident qn_mask must not be converted through host")
            return super().to_numpy(x)

    class MaskArray(np.ndarray):
        pass

    backend = BackendResidentMask()
    mask = np.array([[True, False, True], [False, True, False]]).view(MaskArray)
    mask._forbid_host_conversion = True
    spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=3, nrhs=2)
    packed = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    struct = backend.unpack_masked_vectors(packed, spec)
    roundtrip = backend.pack_masked_vectors(struct, spec)

    assert backend.forbidden_host_conversions == 0
    assert np.array_equal(roundtrip, packed)


def test_packed_vector_spec_rejects_wrong_packed_shapes():
    from renormalizer.backend.execution import PackedVectorSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mask = np.array([[True, False, True], [False, True, False]])
    spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=3, nrhs=2)

    with pytest.raises(ValueError, match="packed vector shape"):
        backend.unpack_masked_vectors(np.ones((4, 2)), spec)

    with pytest.raises(ValueError, match="center tensor shape"):
        backend.pack_masked_vectors(np.ones(mask.shape + (3,)), spec)

    with pytest.raises(ValueError, match="batched packed vector shape"):
        backend.unpack_masked_vectors(np.ones(3), spec)

    with pytest.raises(ValueError, match="batched center tensor shape"):
        backend.pack_masked_vectors(np.ones(mask.shape), spec)


def test_packed_vector_spec_rejects_inconsistent_mask_metadata():
    from renormalizer.backend.execution import PackedVectorSpec

    mask = np.array([[True, False, True], [False, True, False]])

    with pytest.raises(ValueError, match="PackedVectorSpec qn_mask shape must match center_shape"):
        PackedVectorSpec(qn_mask=mask, center_shape=(3, 2), packed_dim=3)

    with pytest.raises(ValueError, match="PackedVectorSpec packed_dim must match qn_mask true count"):
        PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=2)

    with pytest.raises(ValueError, match="PackedVectorSpec batch_axis 3 is out of bounds for ndim 3"):
        PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, batch_axis=3, packed_dim=3)


def test_tensor_operand_rejects_inconsistent_layout_metadata():
    from renormalizer.backend import LayoutSpec, TensorOperand

    array = np.ones((2, 3))
    wrong_modes = LayoutSpec(
        logical_shape=(2, 3),
        physical_shape=(2, 3),
        logical_modes=("x", "y"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1),),
    )
    wrong_shape = LayoutSpec(
        logical_shape=(2, 4),
        physical_shape=(2, 4),
        logical_modes=("i", "k"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1),),
    )

    with pytest.raises(ValueError, match="TensorOperand modes must match array rank"):
        TensorOperand(array, ("i", "k", "extra"))

    with pytest.raises(ValueError, match="TensorOperand modes must be unique"):
        TensorOperand(array, ("i", "i"))

    with pytest.raises(ValueError, match="TensorOperand layout modes must match operand modes"):
        TensorOperand(array, ("i", "k"), layout=wrong_modes)

    with pytest.raises(ValueError, match="TensorOperand layout shape must match array shape"):
        TensorOperand(array, ("i", "k"), layout=wrong_shape)


def test_einsum_spec_rejects_invalid_metadata():
    from renormalizer.backend import EinsumSpec, TensorOperand

    left = TensorOperand(np.ones((2, 3)), ("i", "k"))
    right = TensorOperand(np.ones((3, 4)), ("k", "j"))

    with pytest.raises(ValueError, match="EinsumSpec operands must be non-empty"):
        EinsumSpec(operands=(), output_modes=("i", "j"))

    with pytest.raises(ValueError, match="EinsumSpec output_modes must be unique"):
        EinsumSpec(operands=(left, right), output_modes=("i", "i"))

    with pytest.raises(ValueError, match="EinsumSpec output_modes are not present in operands"):
        EinsumSpec(operands=(left, right), output_modes=("i", "missing"))

    with pytest.raises(ValueError, match="EinsumSpec constant operand index -1 is out of range"):
        EinsumSpec(operands=(left, right), output_modes=("i", "j"), constants=(-1,))

    with pytest.raises(ValueError, match="EinsumSpec constant operand index 2 is out of range"):
        EinsumSpec(operands=(left, right), output_modes=("i", "j"), constants=(2,))

    with pytest.raises(ValueError, match="EinsumSpec constants must be unique"):
        EinsumSpec(operands=(left, right), output_modes=("i", "j"), constants=(1, 1))


def test_einsum_spec_rejects_inconsistent_shared_mode_sizes():
    from renormalizer.backend import EinsumSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    left = TensorOperand(np.ones((2, 3)), ("i", "k"))
    wrong_right = TensorOperand(np.ones((4, 5)), ("k", "j"))

    with pytest.raises(ValueError, match="EinsumSpec mode 'k' has inconsistent dimensions"):
        EinsumSpec(operands=(left, wrong_right), output_modes=("i", "j"))

    backend = NumpyBackend()
    with pytest.raises(ValueError, match="EinsumSpec mode 'k' has inconsistent dimensions"):
        backend.parse_einsum("ik,kj->ij", np.ones((2, 3)), np.ones((4, 5)))


def test_pair_contraction_lowering_reports_gemm_shape_and_costs():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    spec = PairContractionSpec.from_operands(
        TensorOperand(a, ("i", "k"), name="A"),
        TensorOperand(b, ("k", "j"), name="B"),
        output_modes=("i", "j"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert plan.kind == "gemm"
    assert plan.output_shape == (2, 4)
    assert len(plan.descs) == 1
    desc = plan.descs[0]
    assert (desc.m, desc.n, desc.k) == (2, 4, 3)
    assert desc.batch_shape == ()
    assert plan.estimated_flops == 2 * 2 * 4 * 3
    assert plan.copy_bytes == 0
    assert plan.fallback_reason is None


def test_pair_contraction_spec_rejects_invalid_metadata():
    from renormalizer.backend import LayoutSpec, PairContractionSpec, TensorOperand

    left = TensorOperand(np.ones((2, 3)), ("i", "k"), name="left")
    right = TensorOperand(np.ones((3, 4)), ("k", "j"), name="right")
    output_layout = LayoutSpec(
        logical_shape=(2, 4, 1),
        physical_shape=(2, 4, 1),
        logical_modes=("i", "j", "extra"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1, 2),),
    )
    wrong_output_shape_layout = LayoutSpec(
        logical_shape=(2, 5),
        physical_shape=(2, 5),
        logical_modes=("i", "j"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1),),
    )

    valid = {
        "left": left,
        "right": right,
        "output_modes": ("i", "j"),
        "left_batch_modes": (),
        "right_batch_modes": (),
        "contracted_modes": ("k",),
        "left_only_modes": ("i",),
        "right_only_modes": ("j",),
    }

    with pytest.raises(ValueError, match="PairContractionSpec output_modes must be unique"):
        PairContractionSpec(**{**valid, "output_modes": ("i", "i")})

    with pytest.raises(ValueError, match="PairContractionSpec output_modes are not present in operands"):
        PairContractionSpec(**{**valid, "output_modes": ("i", "missing")})

    with pytest.raises(ValueError, match="PairContractionSpec batch modes must match"):
        PairContractionSpec(**{**valid, "left_batch_modes": ("i",), "right_batch_modes": ("j",)})

    with pytest.raises(ValueError, match="PairContractionSpec left_batch_modes are not present in left operand"):
        PairContractionSpec(**{**valid, "left_batch_modes": ("j",), "right_batch_modes": ("j",)})

    with pytest.raises(ValueError, match="PairContractionSpec contracted_modes are not present in both operands"):
        PairContractionSpec(**{**valid, "contracted_modes": ("i",)})

    with pytest.raises(ValueError, match="PairContractionSpec left_only_modes must be unique"):
        PairContractionSpec(**{**valid, "left_only_modes": ("i", "i")})

    batch_left = TensorOperand(np.ones((5, 2, 3)), ("b", "i", "k"), name="batch_left")
    batch_right = TensorOperand(np.ones((5, 3, 4)), ("b", "k", "j"), name="batch_right")
    batch_valid = {
        "left": batch_left,
        "right": batch_right,
        "output_modes": ("b", "i", "j"),
        "left_batch_modes": ("b",),
        "right_batch_modes": ("b",),
        "contracted_modes": ("k",),
        "left_only_modes": ("i",),
        "right_only_modes": ("j",),
    }
    with pytest.raises(ValueError, match="PairContractionSpec mode groups must be disjoint"):
        PairContractionSpec(**{**batch_valid, "left_only_modes": ("b", "i")})

    with pytest.raises(ValueError, match="PairContractionSpec mode groups must be disjoint"):
        PairContractionSpec(**{**valid, "left_only_modes": ("k",)})

    with pytest.raises(ValueError, match="PairContractionSpec output_layout rank must match output_modes"):
        PairContractionSpec(**{**valid, "output_layout": output_layout})

    with pytest.raises(ValueError, match="PairContractionSpec output_layout shape must match output modes"):
        PairContractionSpec(**{**valid, "output_layout": wrong_output_shape_layout})

    with pytest.raises(ValueError, match="PairContractionSpec left_only_modes must not be present in right operand"):
        PairContractionSpec(**{**batch_valid, "left_batch_modes": (), "right_batch_modes": (), "left_only_modes": ("b", "i")})

    with pytest.raises(ValueError, match="PairContractionSpec right_only_modes must not be present in left operand"):
        PairContractionSpec(**{**batch_valid, "left_batch_modes": (), "right_batch_modes": (), "right_only_modes": ("b", "j")})


def test_pair_contraction_spec_rejects_inconsistent_shared_mode_sizes():
    from renormalizer.backend import PairContractionSpec, TensorOperand

    left = TensorOperand(np.ones((5, 2, 3)), ("b", "i", "k"), name="left")
    wrong_batch_right = TensorOperand(np.ones((6, 3, 4)), ("b", "k", "j"), name="wrong_batch")
    wrong_contract_right = TensorOperand(np.ones((5, 4, 4)), ("b", "k", "j"), name="wrong_contract")

    valid = {
        "left": left,
        "output_modes": ("b", "i", "j"),
        "left_batch_modes": ("b",),
        "right_batch_modes": ("b",),
        "contracted_modes": ("k",),
        "left_only_modes": ("i",),
        "right_only_modes": ("j",),
    }

    with pytest.raises(ValueError, match="PairContractionSpec mode 'b' has inconsistent dimensions"):
        PairContractionSpec(**{**valid, "right": wrong_batch_right})

    with pytest.raises(ValueError, match="PairContractionSpec mode 'k' has inconsistent dimensions"):
        PairContractionSpec(**{**valid, "right": wrong_contract_right})


@pytest.mark.parametrize("field", ["m", "n", "k"])
def test_matmul_desc_rejects_negative_dimensions(field):
    from renormalizer.backend import MatmulDesc

    values = {"m": 2, "n": 3, "k": 4}
    values[field] = -1

    with pytest.raises(ValueError, match="MatmulDesc {0} must be non-negative".format(field)):
        MatmulDesc(None, None, None, **values)


def test_matmul_desc_rejects_negative_batch_dimensions():
    from renormalizer.backend import MatmulDesc

    with pytest.raises(ValueError, match="MatmulDesc batch_shape dimensions must be non-negative"):
        MatmulDesc(None, None, None, 2, 3, 4, batch_shape=(5, -1))

    with pytest.raises(ValueError, match="MatmulDesc batch_shape dimensions must be non-negative"):
        MatmulDesc(
            np.ones((5, 2, 3), dtype=np.float64),
            np.ones((5, 3, 4), dtype=np.float64),
            np.zeros((5, 2, 4), dtype=np.float64),
            2,
            4,
            3,
            batch_shape=(5, -1),
        )


@pytest.mark.parametrize(
    "field",
    [
        "estimated_flops",
        "estimated_read_bytes",
        "estimated_write_bytes",
        "estimated_workspace_bytes",
    ],
)
def test_matmul_desc_rejects_negative_estimates(field):
    from renormalizer.backend import MatmulDesc

    kwargs = {field: -1}

    with pytest.raises(ValueError, match="MatmulDesc {0} must be non-negative".format(field)):
        MatmulDesc(None, None, None, 2, 3, 4, **kwargs)


def test_matmul_desc_rejects_matrix_shapes_inconsistent_with_descriptor():
    from renormalizer.backend import MatmulDesc

    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    out = np.zeros((2, 4), dtype=np.float64)

    MatmulDesc(left, right, out, 2, 4, 3)
    MatmulDesc(left.T, right, out, 2, 4, 3, trans_a=True)
    MatmulDesc(
        np.ones((5, 2, 3), dtype=np.float64),
        np.ones((5, 3, 4), dtype=np.float64),
        np.zeros((5, 2, 4), dtype=np.float64),
        2,
        4,
        3,
        batch_shape=(5,),
    )

    with pytest.raises(ValueError, match="MatmulDesc A shape must match descriptor"):
        MatmulDesc(np.ones((2, 5), dtype=np.float64), right, None, 2, 4, 3)

    with pytest.raises(ValueError, match="MatmulDesc B shape must match descriptor"):
        MatmulDesc(left, np.ones((5, 4), dtype=np.float64), None, 2, 4, 3)

    with pytest.raises(ValueError, match="MatmulDesc C shape must match descriptor"):
        MatmulDesc(left, right, np.zeros((3, 4), dtype=np.float64), 2, 4, 3)

    with pytest.raises(ValueError, match="MatmulDesc C shape must match descriptor"):
        MatmulDesc(
            np.ones((5, 2, 3), dtype=np.float64),
            np.ones((5, 3, 4), dtype=np.float64),
            np.zeros((2, 4), dtype=np.float64),
            2,
            4,
            3,
            batch_shape=(5,),
        )


def test_matmul_desc_rejects_layout_semantics_inconsistent_with_descriptor():
    from renormalizer.backend import LayoutSpec, MatmulDesc

    def layout(shape, modes):
        return LayoutSpec(
            logical_shape=shape,
            physical_shape=shape,
            logical_modes=modes,
            strides=None,
            order="C",
            contiguous_groups=(tuple(range(len(shape))),),
        )

    left = np.ones((2, 3, 4), dtype=np.float64)
    right = np.ones((2, 4, 5), dtype=np.float64)
    out = np.zeros((2, 3, 5), dtype=np.float64)
    layout_a = layout((2, 3, 4), ("batch", "i", "k"))
    layout_b = layout((2, 4, 5), ("batch", "k", "j"))
    layout_c = layout((2, 3, 5), ("batch", "i", "j"))

    MatmulDesc(
        left,
        right,
        out,
        3,
        5,
        4,
        batch_shape=(2,),
        layout_a=layout_a,
        layout_b=layout_b,
        layout_c=layout_c,
    )

    with pytest.raises(ValueError, match="MatmulDesc dimensions must match layout metadata"):
        MatmulDesc(
            left,
            right,
            out,
            4,
            5,
            4,
            batch_shape=(2,),
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
        )

    with pytest.raises(ValueError, match="MatmulDesc batch_shape must match layout metadata"):
        MatmulDesc(
            left,
            right,
            out,
            3,
            5,
            4,
            batch_shape=(3,),
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
        )


def test_matmul_desc_validates_actual_array_shapes_even_with_layout_metadata():
    from renormalizer.backend import LayoutSpec, MatmulDesc

    def layout(shape, modes):
        return LayoutSpec(
            logical_shape=shape,
            physical_shape=shape,
            logical_modes=modes,
            strides=None,
            order="C",
            contiguous_groups=(tuple(range(len(shape))),),
        )

    right = np.ones((2, 4, 5), dtype=np.float64)
    out = np.zeros((2, 3, 5), dtype=np.float64)
    layout_a = layout((2, 3, 4), ("batch", "i", "k"))
    layout_b = layout((2, 4, 5), ("batch", "k", "j"))
    layout_c = layout((2, 3, 5), ("batch", "i", "j"))

    with pytest.raises(ValueError, match="MatmulDesc A shape must match layout metadata"):
        MatmulDesc(
            np.ones((2, 5, 4), dtype=np.float64),
            right,
            out,
            3,
            5,
            4,
            batch_shape=(2,),
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
        )

    MatmulDesc(
        np.ones((2, 4, 3), dtype=np.float64),
        right,
        out,
        3,
        5,
        4,
        batch_shape=(2,),
        trans_a=True,
        layout_a=layout((2, 4, 3), ("batch", "k", "i")),
        layout_b=layout_b,
        layout_c=layout_c,
    )


def test_layout_transform_rejects_negative_shape_dimensions():
    from renormalizer.backend import LayoutTransform

    with pytest.raises(ValueError, match="LayoutTransform input_shape dimensions must be non-negative"):
        LayoutTransform(kind="reshape", input_shape=(2, -1), output_shape=(2,))
    with pytest.raises(ValueError, match="LayoutTransform output_shape dimensions must be non-negative"):
        LayoutTransform(kind="reshape", input_shape=(2,), output_shape=(2, -1))


def test_layout_transform_rejects_negative_copy_bytes():
    from renormalizer.backend import LayoutTransform

    with pytest.raises(ValueError, match="LayoutTransform copy_bytes must be non-negative"):
        LayoutTransform(kind="transpose", input_shape=(2, 3), output_shape=(3, 2), copy_bytes=-1)


def test_layout_transform_rejects_unknown_kind():
    from renormalizer.backend import LayoutTransform

    with pytest.raises(ValueError, match="Unknown LayoutTransform kind"):
        LayoutTransform(kind="mystery", input_shape=(2, 3), output_shape=(3, 2))


def _valid_matmul_plan_kwargs():
    from renormalizer.backend import MatmulDesc

    return {
        "kind": "gemm",
        "descs": (MatmulDesc(None, None, None, 2, 3, 4),),
        "pre_ops": (),
        "post_ops": (),
        "output_shape": (2, 3),
        "copy_bytes": 0,
        "workspace_bytes": 0,
        "estimated_flops": 0,
        "estimated_time_s": None,
        "reason": "test plan",
    }


def test_matmul_plan_rejects_unknown_kind():
    from renormalizer.backend import MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["kind"] = "mystery"

    with pytest.raises(ValueError, match="Unknown MatmulPlan kind"):
        MatmulPlan(**kwargs)


def test_matmul_plan_rejects_negative_output_dimensions():
    from renormalizer.backend import MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["output_shape"] = (2, -1)

    with pytest.raises(ValueError, match="MatmulPlan output_shape dimensions must be non-negative"):
        MatmulPlan(**kwargs)


def test_matmul_plan_rejects_empty_descriptors():
    from renormalizer.backend import MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["descs"] = ()

    with pytest.raises(ValueError, match="MatmulPlan descs must be non-empty"):
        MatmulPlan(**kwargs)


def test_matmul_plan_gemm_rejects_batched_descriptor():
    from renormalizer.backend import MatmulDesc, MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["descs"] = (MatmulDesc(None, None, None, 2, 3, 4, batch_shape=(5,)),)

    with pytest.raises(ValueError, match="gemm plan requires unbatched descriptor"):
        MatmulPlan(**kwargs)


@pytest.mark.parametrize("kind", ["batched_gemm", "strided_batched_gemm"])
def test_matmul_plan_batched_kind_requires_batched_descriptor(kind):
    from renormalizer.backend import MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["kind"] = kind

    with pytest.raises(ValueError, match="{0} plan requires a batched descriptor".format(kind)):
        MatmulPlan(**kwargs)


@pytest.mark.parametrize("kind", ["gemm", "batched_gemm", "strided_batched_gemm", "fallback_tensordot", "fallback_einsum"])
def test_matmul_plan_single_descriptor_kinds_reject_multiple_descriptors(kind):
    from renormalizer.backend import MatmulDesc, MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["kind"] = kind
    batch_shape = (5,) if kind in ("batched_gemm", "strided_batched_gemm") else ()
    kwargs["descs"] = (
        MatmulDesc(None, None, None, 2, 3, 4, batch_shape=batch_shape),
        MatmulDesc(None, None, None, 2, 3, 4, batch_shape=batch_shape),
    )

    with pytest.raises(ValueError, match="{0} plan requires exactly one descriptor".format(kind)):
        MatmulPlan(**kwargs)


def test_matmul_plan_grouped_gemm_rejects_batched_descriptors():
    from renormalizer.backend import MatmulDesc, MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["kind"] = "grouped_gemm"
    kwargs["descs"] = (
        MatmulDesc(None, None, None, 2, 3, 4),
        MatmulDesc(None, None, None, 2, 3, 4, batch_shape=(5,)),
    )

    with pytest.raises(ValueError, match="grouped_gemm plan requires unbatched descriptors"):
        MatmulPlan(**kwargs)


@pytest.mark.parametrize("kind", ["fallback_tensordot", "fallback_einsum"])
def test_matmul_plan_fallback_kinds_require_explicit_fallback_reason(kind):
    from renormalizer.backend import MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["kind"] = kind

    with pytest.raises(ValueError, match="{0} plan requires fallback_reason".format(kind)):
        MatmulPlan(**kwargs)


@pytest.mark.parametrize("field", ["copy_bytes", "workspace_bytes", "estimated_flops"])
def test_matmul_plan_rejects_negative_estimates(field):
    from renormalizer.backend import MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs[field] = -1

    with pytest.raises(ValueError, match="MatmulPlan {0} must be non-negative".format(field)):
        MatmulPlan(**kwargs)


@pytest.mark.parametrize(
    "kind,descs_factory,output_shape",
    [
        ("gemm", lambda MatmulDesc: (MatmulDesc(None, None, None, 2, 3, 4),), (2, 4)),
        ("batched_gemm", lambda MatmulDesc: (MatmulDesc(None, None, None, 2, 3, 4, batch_shape=(5,)),), (5, 2, 4)),
        (
            "grouped_gemm",
            lambda MatmulDesc: (
                MatmulDesc(None, None, None, 2, 3, 4),
                MatmulDesc(None, None, None, 2, 3, 4),
            ),
            (2, 4),
        ),
    ],
)
def test_matmul_plan_rejects_output_shape_inconsistent_with_descriptors(kind, descs_factory, output_shape):
    from renormalizer.backend import MatmulDesc, MatmulPlan

    kwargs = _valid_matmul_plan_kwargs()
    kwargs["kind"] = kind
    kwargs["descs"] = descs_factory(MatmulDesc)
    kwargs["output_shape"] = output_shape

    with pytest.raises(ValueError, match="MatmulPlan output_shape must match descriptor output"):
        MatmulPlan(**kwargs)


def test_matmul_plan_copy_bytes_must_cover_layout_transform_copies():
    from renormalizer.backend import LayoutTransform, MatmulPlan

    transforms = (
        LayoutTransform(kind="transpose", input_shape=(2, 3), output_shape=(3, 2), copy_bytes=32),
        LayoutTransform(kind="reshape", input_shape=(3, 2), output_shape=(6,), copy_bytes=16),
    )
    kwargs = _valid_matmul_plan_kwargs()
    kwargs["pre_ops"] = transforms[:1]
    kwargs["post_ops"] = transforms[1:]
    kwargs["copy_bytes"] = 47

    with pytest.raises(ValueError, match="MatmulPlan copy_bytes must include LayoutTransform copy bytes"):
        MatmulPlan(**kwargs)

    kwargs["copy_bytes"] = 48
    plan = MatmulPlan(**kwargs)

    assert plan.copy_bytes == 48


def test_matmul_plan_estimates_must_cover_descriptor_estimates():
    from renormalizer.backend import MatmulDesc, MatmulPlan

    desc = MatmulDesc(
        None,
        None,
        None,
        2,
        3,
        4,
        estimated_flops=48,
        estimated_workspace_bytes=16,
    )
    kwargs = _valid_matmul_plan_kwargs()
    kwargs["descs"] = (desc,)
    kwargs["estimated_flops"] = 47
    kwargs["workspace_bytes"] = 16

    with pytest.raises(ValueError, match="MatmulPlan estimated_flops must cover descriptor estimates"):
        MatmulPlan(**kwargs)

    kwargs["estimated_flops"] = 48
    kwargs["workspace_bytes"] = 15
    with pytest.raises(ValueError, match="MatmulPlan workspace_bytes must cover descriptor estimates"):
        MatmulPlan(**kwargs)

    kwargs["workspace_bytes"] = 16
    plan = MatmulPlan(**kwargs)

    assert plan.estimated_flops == 48
    assert plan.workspace_bytes == 16


def test_matmul_plan_hash_includes_descriptor_semantics():
    from dataclasses import replace

    from renormalizer.backend import MatmulDesc, MatmulPlan

    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    out = np.zeros((2, 4), dtype=np.float64)
    desc = MatmulDesc(
        left,
        right,
        out,
        m=2,
        n=4,
        k=3,
        alpha=1.0,
        beta=0.0,
        dtype_compute=np.float64,
        dtype_output=np.float64,
    )

    def plan_for(item):
        return MatmulPlan(
            kind="gemm",
            descs=(item,),
            pre_ops=(),
            post_ops=(),
            output_shape=(2, 4),
            copy_bytes=0,
            workspace_bytes=0,
            estimated_flops=48,
            estimated_time_s=None,
            reason="hash test",
        )

    base_hash = plan_for(desc).plan_hash

    for changed_desc in (
        replace(desc, alpha=2.0),
        replace(desc, beta=1.0),
        replace(desc, dtype_compute=np.float32),
        replace(desc, dtype_output=np.float32),
    ):
        assert plan_for(changed_desc).plan_hash != base_hash


def test_matmul_plan_hash_includes_descriptor_layouts():
    from dataclasses import replace

    from renormalizer.backend import LayoutSpec, MatmulDesc, MatmulPlan

    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    row_major = LayoutSpec(
        logical_shape=(2, 3),
        physical_shape=(2, 3),
        logical_modes=("i", "k"),
        strides=(24, 8),
        order="C",
        contiguous_groups=((0, 1),),
    )
    transposed = LayoutSpec(
        logical_shape=(2, 3),
        physical_shape=(3, 2),
        logical_modes=("i", "k"),
        strides=(8, 24),
        order="C",
        contiguous_groups=((0,), (1,)),
        requires_transpose=True,
        transpose_perm=(1, 0),
        estimated_copy_bytes=left.nbytes,
    )
    desc = MatmulDesc(
        left,
        right,
        None,
        m=2,
        n=4,
        k=3,
        layout_a=row_major,
    )

    def plan_for(item):
        return MatmulPlan(
            kind="gemm",
            descs=(item,),
            pre_ops=(),
            post_ops=(),
            output_shape=(2, 4),
            copy_bytes=0,
            workspace_bytes=0,
            estimated_flops=48,
            estimated_time_s=None,
            reason="layout hash test",
        )

    assert plan_for(replace(desc, layout_a=transposed)).plan_hash != plan_for(desc).plan_hash


def _valid_contraction_step_kwargs():
    return {
        "kind": "gemm",
        "inputs": (0, 1),
        "output": 2,
        "input_modes": (("i", "k"), ("k", "j")),
        "output_modes": ("i", "j"),
    }


def test_contraction_step_rejects_unknown_kind():
    from renormalizer.backend import ContractionStep

    kwargs = _valid_contraction_step_kwargs()
    kwargs["kind"] = "mystery"

    with pytest.raises(ValueError, match="Unknown ContractionStep kind"):
        ContractionStep(**kwargs)


def test_contraction_step_rejects_kind_that_disagrees_with_matmul_plan():
    from renormalizer.backend import ContractionStep, MatmulDesc, MatmulPlan

    matmul_plan = MatmulPlan(
        kind="batched_gemm",
        descs=(MatmulDesc(None, None, None, 2, 4, 3, batch_shape=(5,)),),
        pre_ops=(),
        post_ops=(),
        output_shape=(5, 2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=240,
        estimated_time_s=None,
        reason="batched plan",
    )
    kwargs = _valid_contraction_step_kwargs()
    kwargs["kind"] = "gemm"
    kwargs["plan"] = matmul_plan

    with pytest.raises(ValueError, match="ContractionStep kind must match MatmulPlan kind"):
        ContractionStep(**kwargs)


def test_contraction_step_allows_distributed_wrapper_around_local_matmul_plan():
    from renormalizer.backend import ContractionStep, MatmulDesc, MatmulPlan

    matmul_plan = MatmulPlan(
        kind="gemm",
        descs=(MatmulDesc(None, None, None, 2, 4, 3),),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=48,
        estimated_time_s=None,
        reason="local plan",
    )
    kwargs = _valid_contraction_step_kwargs()
    kwargs["kind"] = "distributed_contract"
    kwargs["plan"] = matmul_plan

    step = ContractionStep(**kwargs)

    assert step.kind == "distributed_contract"
    assert step.plan is matmul_plan


def test_contraction_step_rejects_negative_tensor_ids():
    from renormalizer.backend import ContractionStep

    kwargs = _valid_contraction_step_kwargs()
    kwargs["inputs"] = (0, -1)
    with pytest.raises(ValueError, match="ContractionStep inputs must be non-negative"):
        ContractionStep(**kwargs)

    kwargs = _valid_contraction_step_kwargs()
    kwargs["output"] = -1
    with pytest.raises(ValueError, match="ContractionStep output must be non-negative"):
        ContractionStep(**kwargs)


def test_contraction_step_rejects_inconsistent_mode_metadata():
    from renormalizer.backend import ContractionStep

    kwargs = _valid_contraction_step_kwargs()
    kwargs["input_modes"] = (("i", "k"),)
    with pytest.raises(ValueError, match="ContractionStep input_modes must match inputs length"):
        ContractionStep(**kwargs)

    kwargs = _valid_contraction_step_kwargs()
    kwargs["output_modes"] = ("i", "i")
    with pytest.raises(ValueError, match="ContractionStep output_modes must be unique"):
        ContractionStep(**kwargs)


@pytest.mark.parametrize("kind", ["fallback_tensordot", "fallback_einsum"])
def test_contraction_step_fallback_kinds_require_explicit_fallback_reason(kind):
    from renormalizer.backend import ContractionStep

    kwargs = _valid_contraction_step_kwargs()
    kwargs["kind"] = kind

    with pytest.raises(ValueError, match="{0} step requires fallback_reason".format(kind)):
        ContractionStep(**kwargs)


def test_contraction_step_inherits_fallback_reason_from_inner_matmul_plan():
    from renormalizer.backend import ContractionStep, MatmulDesc, MatmulPlan

    matmul_plan = MatmulPlan(
        kind="fallback_tensordot",
        descs=(MatmulDesc(None, None, None, 2, 4, 3),),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=48,
        estimated_time_s=None,
        reason="backend lacks matmul",
        fallback_reason="backend lacks matmul",
    )
    kwargs = _valid_contraction_step_kwargs()
    kwargs["kind"] = "fallback_tensordot"
    kwargs["plan"] = matmul_plan

    step = ContractionStep(**kwargs)

    assert step.fallback_reason == "backend lacks matmul"


@pytest.mark.parametrize(
    "field",
    [
        "estimated_flops",
        "estimated_read_bytes",
        "estimated_write_bytes",
        "estimated_copy_bytes",
        "estimated_peak_bytes",
        "estimated_comm_bytes",
        "required_workspace_bytes",
    ],
)
def test_contraction_step_rejects_negative_estimates(field):
    from renormalizer.backend import ContractionStep

    kwargs = _valid_contraction_step_kwargs()
    kwargs[field] = -1

    with pytest.raises(ValueError, match="ContractionStep {0} must be non-negative".format(field)):
        ContractionStep(**kwargs)


def test_contraction_step_estimates_must_cover_nested_matmul_plan():
    from renormalizer.backend import ContractionStep, MatmulDesc, MatmulPlan

    matmul_plan = MatmulPlan(
        kind="gemm",
        descs=(MatmulDesc(None, None, None, 2, 4, 3, estimated_flops=48),),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=32,
        workspace_bytes=16,
        estimated_flops=48,
        estimated_time_s=None,
        reason="nested estimate test",
    )
    kwargs = {
        **_valid_contraction_step_kwargs(),
        "plan": matmul_plan,
        "estimated_copy_bytes": 32,
        "required_workspace_bytes": 16,
    }

    with pytest.raises(ValueError, match="ContractionStep estimated_flops must cover nested plan estimates"):
        ContractionStep(**{**kwargs, "estimated_flops": 47})

    with pytest.raises(ValueError, match="ContractionStep estimated_copy_bytes must cover nested plan estimates"):
        ContractionStep(**{**kwargs, "estimated_flops": 48, "estimated_copy_bytes": 31})

    with pytest.raises(ValueError, match="ContractionStep required_workspace_bytes must cover nested plan estimates"):
        ContractionStep(**{**kwargs, "estimated_flops": 48, "required_workspace_bytes": 15})

    step = ContractionStep(**{**kwargs, "estimated_flops": 48})

    assert step.estimated_flops == 48
    assert step.estimated_copy_bytes == 32
    assert step.required_workspace_bytes == 16


def test_contraction_step_read_write_estimates_cover_nested_matmul_descriptors():
    from renormalizer.backend import ContractionStep, MatmulDesc, MatmulPlan

    matmul_plan = MatmulPlan(
        kind="gemm",
        descs=(
            MatmulDesc(
                None,
                None,
                None,
                2,
                4,
                3,
                estimated_flops=48,
                estimated_read_bytes=80,
                estimated_write_bytes=32,
            ),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=48,
        estimated_time_s=None,
        reason="nested descriptor read/write test",
    )
    kwargs = {
        **_valid_contraction_step_kwargs(),
        "plan": matmul_plan,
        "estimated_flops": 48,
    }

    with pytest.raises(ValueError, match="ContractionStep estimated_read_bytes must cover nested plan estimates"):
        ContractionStep(**{**kwargs, "estimated_read_bytes": 79})

    with pytest.raises(ValueError, match="ContractionStep estimated_write_bytes must cover nested plan estimates"):
        ContractionStep(**{**kwargs, "estimated_write_bytes": 31})

    step = ContractionStep(**kwargs)

    assert step.estimated_read_bytes == 80
    assert step.estimated_write_bytes == 32


def test_contraction_step_peak_estimate_covers_nested_matmul_output_and_workspace():
    from renormalizer.backend import ContractionStep, MatmulDesc, MatmulPlan

    matmul_plan = MatmulPlan(
        kind="gemm",
        descs=(
            MatmulDesc(
                None,
                None,
                None,
                2,
                4,
                3,
                estimated_flops=48,
                estimated_write_bytes=64,
                estimated_workspace_bytes=32,
            ),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=32,
        estimated_flops=48,
        estimated_time_s=None,
        reason="nested peak estimate test",
    )
    kwargs = {
        **_valid_contraction_step_kwargs(),
        "plan": matmul_plan,
        "estimated_flops": 48,
        "estimated_write_bytes": 64,
        "required_workspace_bytes": 32,
    }

    with pytest.raises(ValueError, match="ContractionStep estimated_peak_bytes must cover nested plan estimates"):
        ContractionStep(**{**kwargs, "estimated_peak_bytes": 63})

    step = ContractionStep(**kwargs)

    assert step.estimated_peak_bytes == 64


def _valid_contraction_plan_kwargs():
    from renormalizer.backend import ContractionStep, TensorOperand

    step = ContractionStep(**_valid_contraction_step_kwargs())
    return {
        "steps": (step,),
        "input_specs": (
            TensorOperand(np.ones((2, 3)), ("i", "k")),
            TensorOperand(np.ones((3, 4)), ("k", "j")),
        ),
        "output_modes": ("i", "j"),
    }


def test_contraction_plan_rejects_empty_steps():
    from renormalizer.backend import ContractionPlan

    kwargs = _valid_contraction_plan_kwargs()
    kwargs["steps"] = ()

    with pytest.raises(ValueError, match="ContractionPlan steps must be non-empty"):
        ContractionPlan(**kwargs)


def test_contraction_plan_rejects_inconsistent_step_metadata():
    from renormalizer.backend import ContractionPlan

    kwargs = _valid_contraction_plan_kwargs()
    kwargs["input_specs"] = ()
    with pytest.raises(ValueError, match="ContractionPlan input_specs must be non-empty"):
        ContractionPlan(**kwargs)

    kwargs = _valid_contraction_plan_kwargs()
    kwargs["output_modes"] = ("j", "i")
    with pytest.raises(ValueError, match="ContractionPlan output_modes must match final step output_modes"):
        ContractionPlan(**kwargs)


def test_contraction_plan_hash_includes_nested_step_plan_identity():
    from dataclasses import replace

    from renormalizer.backend import MatmulDesc, MatmulPlan, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    desc = MatmulDesc(left, right, None, m=2, n=4, k=3, alpha=1.0)

    def nested_plan_for(item):
        return NumpyBackend._contraction_plan_from_step(
            NumpyBackend._contraction_step_from_matmul_plan(
                MatmulPlan(
                    kind="gemm",
                    descs=(item,),
                    pre_ops=(),
                    post_ops=(),
                    output_shape=(2, 4),
                    copy_bytes=0,
                    workspace_bytes=0,
                    estimated_flops=48,
                    estimated_time_s=None,
                    reason="nested hash test",
                ),
                (("i", "k"), ("k", "j")),
                ("i", "j"),
            ),
            (
                TensorOperand(left, ("i", "k")),
                TensorOperand(right, ("k", "j")),
            ),
            ("i", "j"),
        )

    base_plan = nested_plan_for(desc)
    changed_plan = nested_plan_for(replace(desc, alpha=2.0))

    assert changed_plan.steps[0].plan.plan_hash != base_plan.steps[0].plan.plan_hash
    assert changed_plan.plan_hash != base_plan.plan_hash


def test_contraction_plan_hash_includes_operand_layout():
    from renormalizer.backend import ContractionPlan, TensorOperand

    c_order = np.ascontiguousarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    f_order = np.asfortranarray(c_order)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert c_order.strides != f_order.strides

    def plan_for(left):
        kwargs = _valid_contraction_plan_kwargs()
        kwargs["input_specs"] = (
            TensorOperand(left, ("i", "k")),
            TensorOperand(right, ("k", "j")),
        )
        return ContractionPlan(**kwargs)

    assert plan_for(c_order).input_specs[0].layout.order == "C"
    assert plan_for(f_order).input_specs[0].layout.order == "F"
    assert plan_for(c_order).plan_hash != plan_for(f_order).plan_hash


def _valid_sliced_contraction_plan_kwargs():
    from renormalizer.backend import ContractionPlan

    return {
        "base_plan": ContractionPlan(**_valid_contraction_plan_kwargs()),
        "sliced_mode": "i",
        "output_axis": 0,
        "output_slices": ((slice(0, 1), slice(None)),),
        "operand_slices": (((slice(0, 1), slice(None)), (slice(None), slice(None))),),
    }


def test_sliced_contraction_plan_rejects_invalid_slice_metadata():
    from renormalizer.backend import SlicedContractionPlan

    kwargs = _valid_sliced_contraction_plan_kwargs()
    kwargs["output_axis"] = 2
    with pytest.raises(ValueError, match="SlicedContractionPlan output_axis is out of bounds"):
        SlicedContractionPlan(**kwargs)

    kwargs = _valid_sliced_contraction_plan_kwargs()
    kwargs["output_slices"] = ()
    with pytest.raises(ValueError, match="SlicedContractionPlan output_slices must be non-empty"):
        SlicedContractionPlan(**kwargs)

    kwargs = _valid_sliced_contraction_plan_kwargs()
    kwargs["operand_slices"] = ()
    with pytest.raises(ValueError, match="SlicedContractionPlan operand_slices must match output_slices length"):
        SlicedContractionPlan(**kwargs)

    kwargs = _valid_sliced_contraction_plan_kwargs()
    kwargs["operand_slices"] = (((slice(None), slice(None)),),)
    with pytest.raises(ValueError, match="SlicedContractionPlan operand slice groups must match input_specs length"):
        SlicedContractionPlan(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "estimated_flops",
        "estimated_peak_bytes",
        "estimated_read_bytes",
        "estimated_write_bytes",
        "estimated_copy_bytes",
        "estimated_comm_bytes",
        "required_workspace_bytes",
    ],
)
def test_contraction_plan_rejects_negative_estimates(field):
    from renormalizer.backend import ContractionPlan

    kwargs = _valid_contraction_plan_kwargs()
    kwargs[field] = -1

    with pytest.raises(ValueError, match="ContractionPlan {0} must be non-negative".format(field)):
        ContractionPlan(**kwargs)


def test_contraction_plan_estimates_must_cover_step_costs():
    from renormalizer.backend import ContractionPlan, ContractionStep, TensorOperand

    steps = (
        ContractionStep(
            kind="gemm",
            inputs=(0, 1),
            output=2,
            input_modes=(("i", "k"), ("k", "x")),
            output_modes=("i", "x"),
            estimated_flops=10,
            estimated_read_bytes=11,
            estimated_write_bytes=12,
            estimated_copy_bytes=13,
            estimated_comm_bytes=14,
            estimated_peak_bytes=30,
            required_workspace_bytes=20,
        ),
        ContractionStep(
            kind="tensordot",
            inputs=(2, 3),
            output=4,
            input_modes=(("i", "x"), ("x", "j")),
            output_modes=("i", "j"),
            estimated_flops=15,
            estimated_read_bytes=16,
            estimated_write_bytes=17,
            estimated_copy_bytes=18,
            estimated_comm_bytes=19,
            estimated_peak_bytes=35,
            required_workspace_bytes=25,
        ),
    )
    kwargs = {
        "steps": steps,
        "input_specs": (
            TensorOperand(np.ones((2, 3)), ("i", "k")),
            TensorOperand(np.ones((3, 4)), ("k", "x")),
            TensorOperand(np.ones((4, 5)), ("x", "j")),
        ),
        "output_modes": ("i", "j"),
        "estimated_flops": 25,
        "estimated_read_bytes": 27,
        "estimated_write_bytes": 29,
        "estimated_copy_bytes": 31,
        "estimated_comm_bytes": 33,
        "estimated_peak_bytes": 35,
        "required_workspace_bytes": 25,
    }

    plan = ContractionPlan(**kwargs)
    assert plan.estimated_flops == 25
    assert plan.estimated_peak_bytes == 35
    assert plan.required_workspace_bytes == 25

    for field, too_small in (
        ("estimated_flops", 24),
        ("estimated_read_bytes", 26),
        ("estimated_write_bytes", 28),
        ("estimated_copy_bytes", 30),
        ("estimated_comm_bytes", 32),
        ("estimated_peak_bytes", 34),
        ("required_workspace_bytes", 24),
    ):
        bad_kwargs = dict(kwargs)
        bad_kwargs[field] = too_small
        with pytest.raises(ValueError, match="ContractionPlan {0} must cover step".format(field)):
            ContractionPlan(**bad_kwargs)


def test_pair_contraction_lowering_reports_strided_batched_gemm_for_same_shape_batch():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.ones((5, 2, 3))
    b = np.ones((5, 3, 4))
    spec = PairContractionSpec.from_operands(
        TensorOperand(a, ("batch", "i", "k")),
        TensorOperand(b, ("batch", "k", "j")),
        output_modes=("batch", "i", "j"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert backend.capabilities.strided_batched_gemm is True
    assert plan.kind == "strided_batched_gemm"
    assert plan.descs[0].batch_shape == (5,)
    assert plan.fallback_reason is None


def test_pair_contraction_lowering_uses_plain_batched_kind_without_strided_support():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NonStridedBatchedNumpyBackend(NumpyBackend):
        supports_strided_batched_gemm = False

    backend = NonStridedBatchedNumpyBackend()
    a = np.ones((5, 2, 3))
    b = np.ones((5, 3, 4))
    spec = PairContractionSpec.from_operands(
        TensorOperand(a, ("batch", "i", "k")),
        TensorOperand(b, ("batch", "k", "j")),
        output_modes=("batch", "i", "j"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert backend.capabilities.strided_batched_gemm is False
    assert plan.kind == "batched_gemm"
    assert plan.descs[0].batch_shape == (5,)
    assert plan.fallback_reason is None


def test_pair_contraction_lowering_uses_strided_batched_kind_when_backend_declares_support():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    class StridedBatchedNumpyBackend(NumpyBackend):
        supports_strided_batched_gemm = True

    backend = StridedBatchedNumpyBackend()
    a = np.ones((5, 2, 3))
    b = np.ones((5, 3, 4))
    spec = PairContractionSpec.from_operands(
        TensorOperand(a, ("batch", "i", "k")),
        TensorOperand(b, ("batch", "k", "j")),
        output_modes=("batch", "i", "j"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert backend.capabilities.strided_batched_gemm is True
    assert plan.kind == "strided_batched_gemm"
    assert plan.descs[0].batch_shape == (5,)
    assert plan.fallback_reason is None


def test_pair_contraction_lowering_reports_layout_copy_bytes():
    from renormalizer.backend import LayoutSpec
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    left_layout = LayoutSpec(
        logical_shape=left.shape,
        physical_shape=left.shape,
        logical_modes=("i", "k"),
        strides=left.strides,
        order="C",
        contiguous_groups=((0,), (1,)),
        requires_transpose=True,
        transpose_perm=(1, 0),
        estimated_copy_bytes=left.nbytes,
    )
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("i", "k"), layout=left_layout),
        TensorOperand(right, ("k", "j")),
        output_modes=("i", "j"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert plan.copy_bytes == left.nbytes
    assert plan.descs[0].estimated_workspace_bytes == left.nbytes
    assert len(plan.pre_ops) == 1
    assert plan.pre_ops[0].kind == "transpose"
    assert plan.pre_ops[0].input_shape == left.shape
    assert plan.pre_ops[0].output_shape == (3, 2)
    assert plan.pre_ops[0].copy_bytes == left.nbytes
    assert plan.pre_ops[0].reason == "left operand requires transpose"
    assert plan.post_ops == ()


def test_execute_matmul_plan_profiles_layout_transforms(tmp_path):
    from renormalizer.backend import LayoutSpec, LayoutTransform, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    output_nbytes = 2 * 4 * left.dtype.itemsize
    pre = LayoutTransform(
        kind="transpose",
        input_shape=left.shape,
        output_shape=(3, 2),
        copy_bytes=left.nbytes,
        reason="left operand requires transpose",
    )
    post = LayoutTransform(
        kind="reshape",
        input_shape=(2, 4),
        output_shape=(8,),
        copy_bytes=output_nbytes,
        reason="flatten output for packed vector",
    )
    left_layout = LayoutSpec(
        logical_shape=left.shape,
        physical_shape=left.shape,
        logical_modes=("i", "k"),
        strides=left.strides,
        order="C",
        contiguous_groups=((0, 1),),
    )
    right_layout = LayoutSpec(
        logical_shape=right.shape,
        physical_shape=right.shape,
        logical_modes=("k", "j"),
        strides=right.strides,
        order="C",
        contiguous_groups=((0, 1),),
    )
    output_layout = LayoutSpec(
        logical_shape=(2, 4),
        physical_shape=(2, 4),
        logical_modes=("i", "j"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1),),
    )
    plan = MatmulPlan(
        kind="gemm",
        descs=(
            MatmulDesc(
                left,
                right,
                None,
                2,
                4,
                3,
                layout_a=left_layout,
                layout_b=right_layout,
                layout_c=output_layout,
                estimated_flops=48,
                estimated_read_bytes=left.nbytes + right.nbytes,
                estimated_write_bytes=output_nbytes,
                estimated_workspace_bytes=left.nbytes + output_nbytes,
            ),
        ),
        pre_ops=(pre,),
        post_ops=(post,),
        output_shape=(2, 4),
        copy_bytes=left.nbytes + output_nbytes,
        workspace_bytes=left.nbytes + output_nbytes,
        estimated_flops=48,
        estimated_time_s=None,
        reason="profile transform metadata",
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_matmul_plan(plan)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["copy_bytes"] == left.nbytes + output_nbytes
    assert event["layout_transform_copy_bytes"] == left.nbytes + output_nbytes
    assert event["input_layout_transform_copy_bytes"] == left.nbytes
    assert event["output_layout_transform_copy_bytes"] == output_nbytes
    assert event["copy_profile"]["copy_kind"] == "layout_transform"
    assert event["copy_profile"]["copy_source"] == "layout_transform_copy_bytes"
    assert event["layout_profile"]["layout_transform_required"] is True
    assert event["layout_profile"]["layout_transform_kind"] == "layout_transform"
    assert event["layout_profile"]["estimated_layout_copy_bytes"] == left.nbytes + output_nbytes
    assert event["num_pre_ops"] == 1
    assert event["num_post_ops"] == 1
    assert event["pre_ops"] == [
        {
            "kind": "transpose",
            "input_shape": [2, 3],
            "output_shape": [3, 2],
            "copy_bytes": left.nbytes,
            "reason": "left operand requires transpose",
        },
    ]
    assert event["post_ops"] == [
        {
            "kind": "reshape",
            "input_shape": [2, 4],
            "output_shape": [8],
            "copy_bytes": output_nbytes,
            "reason": "flatten output for packed vector",
        },
    ]


def test_plan_contraction_forbid_policy_rejects_fallback_lowering():
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="forbid"))
    left = np.ones((5, 2, 3), dtype=np.float64)
    right = np.ones((5, 3, 4), dtype=np.float64)
    spec = backend.parse_einsum("bik,bkj->bij", left, right)

    with pytest.raises(BackendFeatureError, match="backend lacks batched_matmul"):
        backend.plan_contraction(spec)


def test_plan_contraction_warn_policy_warns_for_fallback_lowering():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="warn"))
    left = np.ones((5, 2, 3), dtype=np.float64)
    right = np.ones((5, 3, 4), dtype=np.float64)
    spec = backend.parse_einsum("bik,bkj->bij", left, right)

    with pytest.warns(RuntimeWarning, match="backend lacks batched_matmul"):
        plan = backend.plan_contraction(spec)

    assert plan.steps[0].plan.kind == "fallback_tensordot"
    assert plan.steps[0].fallback_reason == "backend lacks batched_matmul for batch shape (5,)"


def test_matmul_descriptor_call_forbid_policy_rejects_missing_matmul_capability():
    from renormalizer.backend import BackendConfig, BackendFeatureError, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    desc = MatmulDesc(
        np.ones((2, 3), dtype=np.float64),
        np.ones((3, 4), dtype=np.float64),
        None,
        2,
        4,
        3,
    )

    with pytest.raises(BackendFeatureError, match="backend lacks matmul"):
        backend.matmul(desc, fallback_policy="forbid")


def test_batched_matmul_descriptor_call_forbid_policy_rejects_missing_batched_capability():
    from renormalizer.backend import BackendConfig, BackendFeatureError, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    desc = MatmulDesc(
        np.ones((5, 2, 3), dtype=np.float64),
        np.ones((5, 3, 4), dtype=np.float64),
        None,
        2,
        4,
        3,
        batch_shape=(5,),
    )

    with pytest.raises(BackendFeatureError, match="backend lacks batched_matmul"):
        backend.batched_matmul(desc, fallback_policy="forbid")


def test_matmul_descriptor_fallback_records_primitive_profile():
    from renormalizer.backend import BackendConfig, LayoutSpec, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    desc = MatmulDesc(
        left,
        right,
        None,
        2,
        4,
        3,
        layout_a=LayoutSpec(
            logical_shape=(2, 3),
            physical_shape=(2, 3),
            logical_modes=("i", "k"),
            strides=left.strides,
            order="C",
            contiguous_groups=((0, 1),),
        ),
        layout_b=LayoutSpec(
            logical_shape=(3, 4),
            physical_shape=(3, 4),
            logical_modes=("k", "j"),
            strides=right.strides,
            order="C",
            contiguous_groups=((0, 1),),
        ),
        layout_c=LayoutSpec(
            logical_shape=(2, 4),
            physical_shape=(2, 4),
            logical_modes=("i", "j"),
            strides=None,
            order="C",
            contiguous_groups=((0, 1),),
        ),
    )

    result = backend.matmul(desc, fallback_policy="record")

    assert np.allclose(result, left @ right)
    profile = backend.last_execution_profile()
    assert profile["event"] == "matmul_execute"
    assert profile["lowering"] == "fallback_tensordot"
    assert profile["fallback_from"] == "gemm"
    assert profile["fallback_to"] == "tensordot"
    assert profile["fallback_policy"] == "record"
    assert profile["fallback_reason"] == "backend lacks matmul"
    assert profile["lowering_profile"]["silent_fallback"] is False


def test_batched_matmul_descriptor_fallback_records_primitive_profile():
    from renormalizer.backend import BackendConfig, LayoutSpec, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    right = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4)
    desc = MatmulDesc(
        left,
        right,
        None,
        2,
        4,
        3,
        batch_shape=(5,),
        layout_a=LayoutSpec(
            logical_shape=(5, 2, 3),
            physical_shape=(5, 2, 3),
            logical_modes=("b", "i", "k"),
            strides=left.strides,
            order="C",
            contiguous_groups=((0, 1, 2),),
        ),
        layout_b=LayoutSpec(
            logical_shape=(5, 3, 4),
            physical_shape=(5, 3, 4),
            logical_modes=("b", "k", "j"),
            strides=right.strides,
            order="C",
            contiguous_groups=((0, 1, 2),),
        ),
        layout_c=LayoutSpec(
            logical_shape=(5, 2, 4),
            physical_shape=(5, 2, 4),
            logical_modes=("b", "i", "j"),
            strides=None,
            order="C",
            contiguous_groups=((0, 1, 2),),
        ),
    )

    result = backend.batched_matmul(desc, fallback_policy="record")

    assert np.allclose(result, np.matmul(left, right))
    profile = backend.last_execution_profile()
    assert profile["event"] == "matmul_execute"
    assert profile["lowering"] == "fallback_tensordot"
    assert profile["fallback_from"] == "batched_gemm"
    assert profile["fallback_to"] == "loop_matmul"
    assert profile["fallback_policy"] == "record"
    assert profile["fallback_reason"] == "backend lacks batched_matmul for batch shape (5,)"
    assert profile["lowering_profile"]["silent_fallback"] is False


def test_raw_matmul_call_forbid_policy_rejects_missing_matmul_capability():
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    with pytest.raises(BackendFeatureError, match="backend lacks matmul"):
        backend.matmul(left, right, fallback_policy="forbid")


def test_raw_matmul_fallback_records_primitive_profile():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    result = backend.matmul(left, right, fallback_policy="record")

    assert np.allclose(result, left @ right)
    profile = backend.last_execution_profile()
    assert profile["event"] == "matmul_execute"
    assert profile["lowering"] == "fallback_tensordot"
    assert profile["fallback_from"] == "gemm"
    assert profile["fallback_to"] == "tensordot"
    assert profile["fallback_policy"] == "record"
    assert profile["fallback_reason"] == "backend lacks matmul"
    assert profile["lowering_profile"]["silent_fallback"] is False


def test_raw_batched_matmul_call_forbid_policy_rejects_missing_batched_capability():
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    right = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4)

    with pytest.raises(BackendFeatureError, match="backend lacks batched_matmul"):
        backend.batched_matmul(left, right, fallback_policy="forbid")


def test_raw_batched_matmul_fallback_records_primitive_profile():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    right = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4)

    result = backend.batched_matmul(left, right, fallback_policy="record")

    assert np.allclose(result, np.matmul(left, right))
    profile = backend.last_execution_profile()
    assert profile["event"] == "matmul_execute"
    assert profile["lowering"] == "fallback_tensordot"
    assert profile["fallback_from"] == "batched_gemm"
    assert profile["fallback_to"] == "loop_matmul"
    assert profile["fallback_policy"] == "record"
    assert profile["fallback_reason"] == "backend lacks batched_matmul for batch shape (5,)"
    assert profile["lowering_profile"]["silent_fallback"] is False


def test_backend_parse_einsum_builds_explicit_ir_operands():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)

    spec = backend.parse_einsum("ab,bc->ac", left, right, optimize="greedy", constants=(1,))

    assert spec.output_modes == ("a", "c")
    assert spec.optimize == "greedy"
    assert spec.constants == (1,)
    assert [operand.modes for operand in spec.operands] == [("a", "b"), ("b", "c")]
    assert [operand.name for operand in spec.operands] == ["operand0", "operand1"]
    assert spec.operands[0].array is left
    assert spec.operands[1].array is right
    assert spec.operands[0].layout.logical_modes == ("a", "b")
    assert spec.operands[1].layout.logical_modes == ("b", "c")


def test_backend_parse_einsum_rejects_implicit_output():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(ValueError, match="explicit output"):
        backend.parse_einsum("ab,bc", np.ones((2, 3)), np.ones((3, 4)))


def test_backend_parse_einsum_rejects_duplicate_constants():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(ValueError, match="einsum constants must be unique"):
        backend.parse_einsum(
            "ab,bc,cd->ad",
            np.ones((2, 3)),
            np.ones((3, 4)),
            np.ones((4, 5)),
            constants=(1, 1),
        )


def test_backend_parse_einsum_rejects_operand_rank_mismatch():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(ValueError, match="rank"):
        backend.parse_einsum("ab,bc->ac", np.ones((2, 3, 1)), np.ones((3, 4)))


def test_backend_raw_matmul_and_stacked_batched_matmul():
    from renormalizer.backend.execution import DeviceSpec, Workspace
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    batch_a = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    batch_b = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)
    stream = object()
    workspace = Workspace(device=DeviceSpec(kind="cpu"), nbytes=64, buffer=np.empty(64, dtype=np.uint8))

    assert np.allclose(backend.matmul(a, b, stream=stream, workspace=workspace), a @ b)
    profile = backend.last_execution_profile()
    assert profile["event"] == "matmul_execute"
    assert profile["lowering"] == "gemm"
    assert profile["device_kind"] == "cpu"
    assert profile["stream_provided"] is True
    assert profile["workspace_provided"] is True
    assert profile["workspace_nbytes"] == 64
    assert profile["workspace_released"] is False
    assert profile["flops"] == 48
    assert profile["num_gemm"] == 1
    assert profile["num_batched_gemm"] == 0
    assert profile["fallback_reason"] is None
    assert profile["compute_profile"]["compute_class"] == "contraction_plan"
    assert profile["compute_profile"]["compute_subclass"] == "matmul_execute"
    assert profile["dtype_profile"]["output_dtype"] == "float64"
    assert profile["copy_profile"]["copy_required"] is False
    assert profile["workspace_profile"]["workspace_provided_bytes"] == 64
    assert profile["workspace_profile"]["workspace_released"] is False
    assert profile["async_profile"]["async_requested"] is True
    assert profile["lowering_profile"]["lowering"] == "gemm"
    assert profile["layout_profile"]["output_layout"] == {
        "shape": (2, 4),
        "strides": (32, 8),
        "order": "C",
        "contiguous": True,
    }

    assert np.allclose(
        backend.batched_matmul(batch_a, batch_b, stream=stream, workspace=workspace),
        np.matmul(batch_a, batch_b),
    )
    profile = backend.last_execution_profile()
    assert profile["event"] == "matmul_execute"
    assert profile["lowering"] == "batched_gemm"
    assert profile["device_kind"] == "cpu"
    assert profile["stream_provided"] is True
    assert profile["workspace_provided"] is True
    assert profile["workspace_nbytes"] == 64
    assert profile["workspace_released"] is False
    assert profile["flops"] == 240
    assert profile["num_gemm"] == 0
    assert profile["num_batched_gemm"] == 1
    assert profile["fallback_reason"] is None
    assert profile["compute_profile"]["compute_class"] == "contraction_plan"
    assert profile["compute_profile"]["compute_subclass"] == "matmul_execute"
    assert profile["dtype_profile"]["output_dtype"] == "float64"
    assert profile["copy_profile"]["copy_required"] is False
    assert profile["workspace_profile"]["workspace_provided_bytes"] == 64
    assert profile["workspace_profile"]["workspace_released"] is False
    assert profile["async_profile"]["async_requested"] is True
    assert profile["lowering_profile"]["lowering"] == "batched_gemm"
    assert profile["layout_profile"]["output_layout"] == {
        "shape": (4, 2, 5),
        "strides": (80, 40, 8),
        "order": "C",
        "contiguous": True,
    }


def test_raw_matmul_rejects_workspace_on_wrong_device():
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec, Workspace
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    workspace = Workspace(device=DeviceSpec(kind="cuda"), nbytes=0, buffer=np.empty(0, dtype=np.uint8))

    with pytest.raises(BackendFeatureError, match="workspace device"):
        backend.matmul(a, b, workspace=workspace)


def test_raw_batched_matmul_rejects_workspace_on_wrong_device():
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec, Workspace
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    b = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)
    workspace = Workspace(device=DeviceSpec(kind="cuda"), nbytes=0, buffer=np.empty(0, dtype=np.uint8))

    with pytest.raises(BackendFeatureError, match="workspace device"):
        backend.batched_matmul(a, b, workspace=workspace)


def test_grouped_gemm_buckets_same_shape_tasks_and_preserves_order():
    from renormalizer.backend.gemm import GemmTask, group_tasks_by_shape
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    tasks = [
        GemmTask(
            np.arange(6, dtype=np.float64).reshape(2, 3),
            np.arange(12, dtype=np.float64).reshape(3, 4),
            tag="same-0",
        ),
        GemmTask(
            np.arange(6, 12, dtype=np.float64).reshape(2, 3),
            np.arange(12, 24, dtype=np.float64).reshape(3, 4),
            tag="same-1",
        ),
        GemmTask(
            np.arange(12, dtype=np.float64).reshape(3, 4),
            np.arange(8, dtype=np.float64).reshape(4, 2),
            tag="ragged",
        ),
    ]

    buckets = group_tasks_by_shape(tasks, xp=np)
    results = backend.grouped_gemm(tasks, pack_threshold=2)

    assert sorted(len(bucket_tasks) for bucket_tasks in buckets.values()) == [1, 2]
    assert len(results) == len(tasks)
    assert np.allclose(results[0], tasks[0].A @ tasks[0].B)
    assert np.allclose(results[1], tasks[1].A @ tasks[1].B)
    assert np.allclose(results[2], tasks[2].A @ tasks[2].B)


def test_gemm_batch_builds_focus_style_groups_and_gsta():
    from renormalizer.backend.gemm import BufferRef, BufferSlice, GemmBatch, GemmTask

    large_a = np.ones((128, 128), dtype=np.float64)
    large_b = np.ones((128, 128), dtype=np.float64)
    small_a = np.ones((3, 4), dtype=np.float64)
    small_b = np.ones((4, 5), dtype=np.float64)
    zero_a = np.ones((0, 4), dtype=np.float64)
    zero_b = np.ones((4, 5), dtype=np.float64)
    tasks = [
        GemmTask(large_a, large_b, tag="large-0"),
        GemmTask(zero_a, zero_b, tag="zero"),
        GemmTask(large_a * 2.0, large_b, tag="large-1"),
        GemmTask(small_a, small_b, tag="small"),
    ]

    ref = BufferRef.from_array("A", large_a, device="cpu")
    slice_ref = BufferSlice("A", offset=3, shape=(4, 5), leading_dim=5)
    batch = GemmBatch.from_tasks(tasks, xp=np)

    assert ref.name == "A"
    assert ref.device == "cpu"
    assert ref.dtype == np.dtype("float64")
    assert ref.shape == (128, 128)
    assert ref.nbytes == large_a.nbytes
    assert slice_ref.offset == 3
    assert slice_ref.shape == (4, 5)
    assert [task.tag for task in batch.descs] == ["large-0", "large-1", "small"]
    assert batch.sorted_indices == (0, 2, 3)
    assert batch.gsta == (0, 2, 3)
    assert [len(indices) for indices in batch.groups.values()] == [2, 1]
    assert [key.m for key in batch.groups] == [128, 3]
    assert [key.n for key in batch.groups] == [128, 5]
    assert [key.k for key in batch.groups] == [128, 4]
    assert batch.group_sizes == (2, 1)
    assert batch.total_flops == 2 * 128 * 128 * 128 * 2 + 2 * 3 * 5 * 4
    assert batch.max_m == 128
    assert batch.max_n == 128
    assert batch.max_k == 128


def test_matmul_desc_batch_builds_focus_buffer_slice_groups_and_gsta():
    from renormalizer.backend.gemm import BufferSlice, GemmBatch, MatmulDesc

    descs = [
        MatmulDesc(
            A=BufferSlice("A", offset=0, shape=(16, 8), leading_dim=8),
            B=BufferSlice("B", offset=0, shape=(8, 4), leading_dim=4),
            C=BufferSlice("C", offset=0, shape=(16, 4), leading_dim=4),
            trans_a="N",
            trans_b="N",
            m=16,
            n=4,
            k=8,
            lda=8,
            ldb=4,
            ldc=4,
            dtype=np.dtype("float64"),
            tag="same-0",
        ),
        MatmulDesc(
            A=BufferSlice("A", offset=128, shape=(0, 8), leading_dim=8),
            B=BufferSlice("B", offset=32, shape=(8, 4), leading_dim=4),
            C=BufferSlice("C", offset=64, shape=(0, 4), leading_dim=4),
            trans_a="N",
            trans_b="N",
            m=0,
            n=4,
            k=8,
            lda=8,
            ldb=4,
            ldc=4,
            dtype=np.dtype("float64"),
            tag="zero",
        ),
        MatmulDesc(
            A=BufferSlice("A", offset=256, shape=(16, 8), leading_dim=8),
            B=BufferSlice("B", offset=64, shape=(8, 4), leading_dim=4),
            C=BufferSlice("C", offset=128, shape=(16, 4), leading_dim=4),
            trans_a="N",
            trans_b="N",
            m=16,
            n=4,
            k=8,
            lda=8,
            ldb=4,
            ldc=4,
            dtype=np.dtype("float64"),
            tag="same-1",
        ),
        MatmulDesc(
            A=BufferSlice("A", offset=384, shape=(3, 5), leading_dim=5),
            B=BufferSlice("B", offset=96, shape=(5, 2), leading_dim=2),
            C=BufferSlice("C", offset=192, shape=(3, 2), leading_dim=2),
            trans_a="N",
            trans_b="N",
            m=3,
            n=2,
            k=5,
            lda=5,
            ldb=2,
            ldc=2,
            dtype=np.dtype("float64"),
            tag="ragged",
        ),
    ]

    batch = GemmBatch.from_descs(descs)

    assert [desc.tag for desc in batch.descs] == ["same-0", "same-1", "ragged"]
    assert batch.sorted_indices == (0, 2, 3)
    assert batch.gsta == (0, 2, 3)
    assert [len(indices) for indices in batch.groups.values()] == [2, 1]
    assert [key.m for key in batch.groups] == [16, 3]
    assert [key.n for key in batch.groups] == [4, 2]
    assert [key.k for key in batch.groups] == [8, 5]
    assert [key.lda for key in batch.groups] == [8, 5]
    assert [key.ldb for key in batch.groups] == [4, 2]
    assert [key.ldc for key in batch.groups] == [4, 2]
    assert batch.group_sizes == (2, 1)
    assert batch.total_flops == 2 * 16 * 4 * 8 * 2 + 2 * 3 * 2 * 5
    assert batch.max_m == 16
    assert batch.max_n == 4
    assert batch.max_k == 8


def test_grouped_gemm_executes_focus_buffer_slice_batch_with_offsets(tmp_path):
    from renormalizer.backend.gemm import BufferRef, BufferSlice, GemmBatch, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    a0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    a1 = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
    b0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    b1 = np.arange(12, 24, dtype=np.float64).reshape(3, 4)
    a_buffer = np.concatenate([a0.ravel(), a1.ravel()])
    b_buffer = np.concatenate([b0.ravel(), b1.ravel()])
    c_buffer = np.full(16, -1.0, dtype=np.float64)
    buffers = {
        "A": BufferRef.from_array("A", a_buffer, device="cpu"),
        "B": BufferRef.from_array("B", b_buffer, device="cpu"),
        "C": BufferRef.from_array("C", c_buffer, device="cpu"),
    }
    batch = GemmBatch.from_descs(
        [
            MatmulDesc(
                A=BufferSlice("A", offset=0, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=0, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=0, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
                tag="first",
            ),
            MatmulDesc(
                A=BufferSlice("A", offset=6, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=12, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=8, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
                tag="second",
            ),
        ]
    )
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = backend.grouped_gemm(batch, buffers=buffers, pack_threshold=2)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], a0 @ b0)
    assert np.allclose(results[1], a1 @ b1)
    assert np.allclose(c_buffer[:8].reshape(2, 4), a0 @ b0)
    assert np.allclose(c_buffer[8:].reshape(2, 4), a1 @ b1)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "grouped_gemm_execute")
    assert event["descriptor_source"] == "focus_buffer_slice"
    assert event["buffer_names"] == ["A", "B", "C"]
    assert event["buffer_table"] == [
        {
            "name": "A",
            "shape": [12],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": a_buffer.nbytes,
        },
        {
            "name": "B",
            "shape": [24],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": b_buffer.nbytes,
        },
        {
            "name": "C",
            "shape": [16],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": c_buffer.nbytes,
        },
    ]
    assert event["buffer_slice_descriptors"][0]["A_slice"] == {
        "buffer": "A",
        "offset": 0,
        "shape": [2, 3],
        "leading_dim": 3,
    }
    assert event["buffer_slice_descriptors"][1]["B_slice"] == {
        "buffer": "B",
        "offset": 12,
        "shape": [3, 4],
        "leading_dim": 4,
    }
    assert event["group_sizes"] == [2]
    assert event["gsta"] == [0, 2]
    assert event["group_key_buckets"][0]["task_indices"] == [0, 1]
    assert event["buffer_slice_descriptors"] == [
        {
            "index": 0,
            "A_slice": {"buffer": "A", "offset": 0, "shape": [2, 3], "leading_dim": 3},
            "B_slice": {"buffer": "B", "offset": 0, "shape": [3, 4], "leading_dim": 4},
            "C_slice": {"buffer": "C", "offset": 0, "shape": [2, 4], "leading_dim": 4},
            "trans_a": "N",
            "trans_b": "N",
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
            "alpha": 1.0,
            "beta": 0.0,
            "dtype": "float64",
            "tag": "first",
        },
        {
            "index": 1,
            "A_slice": {"buffer": "A", "offset": 6, "shape": [2, 3], "leading_dim": 3},
            "B_slice": {"buffer": "B", "offset": 12, "shape": [3, 4], "leading_dim": 4},
            "C_slice": {"buffer": "C", "offset": 8, "shape": [2, 4], "leading_dim": 4},
            "trans_a": "N",
            "trans_b": "N",
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
            "alpha": 1.0,
            "beta": 0.0,
            "dtype": "float64",
            "tag": "second",
        },
    ]


def test_grouped_gemm_accepts_explicit_focus_buffer_table():
    from renormalizer.backend.gemm import BufferRef, BufferSlice, BufferTable, GemmBatch, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    c = np.full(8, -1.0, dtype=np.float64)
    buffers = BufferTable.from_refs(
        [
            BufferRef.from_array("A", a.ravel(), device="cpu"),
            BufferRef.from_array("B", b.ravel(), device="cpu"),
            BufferRef.from_array("C", c.ravel(), device="cpu"),
        ]
    )
    batch = GemmBatch.from_descs(
        [
            MatmulDesc(
                A=BufferSlice("A", offset=0, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=0, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=0, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
            ),
        ]
    )

    results = backend.grouped_gemm(batch, buffers=buffers)

    assert list(buffers) == ["A", "B", "C"]
    assert buffers["A"].array is not None
    assert np.allclose(results[0], a @ b)
    assert np.allclose(c.reshape(2, 4), a @ b)


def test_grouped_gemm_rejects_buffer_slice_descriptor_dimension_mismatch():
    from renormalizer.backend import BackendFeatureError
    from renormalizer.backend.gemm import BufferRef, BufferSlice, GemmBatch, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a_buffer = np.arange(6, dtype=np.float64)
    b_buffer = np.arange(12, dtype=np.float64)
    c_buffer = np.zeros(8, dtype=np.float64)
    buffers = {
        "A": BufferRef.from_array("A", a_buffer, device="cpu"),
        "B": BufferRef.from_array("B", b_buffer, device="cpu"),
        "C": BufferRef.from_array("C", c_buffer, device="cpu"),
    }
    batch = GemmBatch.from_descs(
        [
            MatmulDesc(
                A=BufferSlice("A", offset=0, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=0, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=0, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=3,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
            ),
        ]
    )

    with pytest.raises(BackendFeatureError, match="MatmulDesc dimensions.*BufferSlice"):
        backend.grouped_gemm(batch, buffers=buffers)


def test_grouped_gemm_rejects_buffer_slice_range_outside_buffer():
    from renormalizer.backend import BackendFeatureError
    from renormalizer.backend.gemm import BufferRef, BufferSlice, GemmBatch, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    buffers = {
        "A": BufferRef.from_array("A", np.arange(6, dtype=np.float64), device="cpu"),
        "B": BufferRef.from_array("B", np.arange(12, dtype=np.float64), device="cpu"),
        "C": BufferRef.from_array("C", np.zeros(8, dtype=np.float64), device="cpu"),
    }
    batch = GemmBatch.from_descs(
        [
            MatmulDesc(
                A=BufferSlice("A", offset=1, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=0, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=0, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
            ),
        ]
    )

    with pytest.raises(BackendFeatureError, match="BufferSlice 'A'.*exceeds buffer size"):
        backend.grouped_gemm(batch, buffers=buffers)


def test_grouped_gemm_buffer_slice_descriptor_supports_complex_conjugate_transpose_and_beta():
    from renormalizer.backend.gemm import BufferRef, BufferSlice, GemmBatch, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.array(
        [
            [1.0 + 2.0j, 3.0 - 1.0j],
            [2.0 - 1.0j, -4.0 + 0.5j],
            [0.5 + 1.5j, 2.0 + 3.0j],
        ],
        dtype=np.complex128,
    )
    b = np.array(
        [
            [1.0 - 0.5j, 2.0 + 1.0j],
            [3.0 + 2.0j, -1.0 + 4.0j],
            [0.25 - 2.0j, 5.0 + 0.5j],
        ],
        dtype=np.complex128,
    )
    c = np.full((2, 2), 2.0 - 1.0j, dtype=np.complex128)
    buffers = {
        "A": BufferRef.from_array("A", a.ravel(), device="cpu"),
        "B": BufferRef.from_array("B", b.ravel(), device="cpu"),
        "C": BufferRef.from_array("C", c.ravel(), device="cpu"),
    }
    batch = GemmBatch.from_descs(
        [
            MatmulDesc(
                A=BufferSlice("A", offset=0, shape=(3, 2), leading_dim=2),
                B=BufferSlice("B", offset=0, shape=(3, 2), leading_dim=2),
                C=BufferSlice("C", offset=0, shape=(2, 2), leading_dim=2),
                trans_a="C",
                trans_b="N",
                m=2,
                n=2,
                k=3,
                lda=2,
                ldb=2,
                ldc=2,
                alpha=0.5 + 0.25j,
                beta=2.0,
                dtype=np.dtype("complex128"),
            ),
        ]
    )
    expected = (0.5 + 0.25j) * (a.conj().T @ b) + 2.0 * c.copy()

    results = backend.grouped_gemm(batch, buffers=buffers)

    assert np.allclose(results[0], expected)
    assert np.allclose(c, expected)


def test_gemv_batch_builds_focus_style_groups_and_gsta():
    from renormalizer.backend.gemm import BufferSlice, GemvBatch, GemvDesc

    descs = [
        GemvDesc(
            A=BufferSlice("A", offset=0, shape=(128, 64), leading_dim=64),
            x=BufferSlice("X", offset=0, shape=(64,)),
            y=BufferSlice("Y", offset=0, shape=(128,)),
            trans_a="N",
            m=128,
            n=64,
            lda=64,
            dtype=np.dtype("float64"),
            tag="large-0",
        ),
        GemvDesc(
            A=BufferSlice("A", offset=128 * 64, shape=(0, 64), leading_dim=64),
            x=BufferSlice("X", offset=64, shape=(64,)),
            y=BufferSlice("Y", offset=128, shape=(0,)),
            trans_a="N",
            m=0,
            n=64,
            lda=64,
            dtype=np.dtype("float64"),
            tag="zero",
        ),
        GemvDesc(
            A=BufferSlice("A", offset=2 * 128 * 64, shape=(128, 64), leading_dim=64),
            x=BufferSlice("X", offset=128, shape=(64,)),
            y=BufferSlice("Y", offset=256, shape=(128,)),
            trans_a="N",
            m=128,
            n=64,
            lda=64,
            dtype=np.dtype("float64"),
            tag="large-1",
        ),
        GemvDesc(
            A=BufferSlice("A", offset=3 * 128 * 64, shape=(3, 4), leading_dim=4),
            x=BufferSlice("X", offset=192, shape=(4,)),
            y=BufferSlice("Y", offset=384, shape=(3,)),
            trans_a="N",
            m=3,
            n=4,
            lda=4,
            dtype=np.dtype("float64"),
            tag="small",
        ),
    ]

    batch = GemvBatch.from_descs(descs)

    assert [desc.tag for desc in batch.descs] == ["large-0", "large-1", "small"]
    assert batch.sorted_indices == (0, 2, 3)
    assert batch.gsta == (0, 2, 3)
    assert [len(indices) for indices in batch.groups.values()] == [2, 1]
    assert [key.m for key in batch.groups] == [128, 3]
    assert [key.n for key in batch.groups] == [64, 4]
    assert batch.group_sizes == (2, 1)
    assert batch.total_flops == 2 * 128 * 64 * 2 + 2 * 3 * 4
    assert batch.max_m == 128
    assert batch.max_n == 64


def test_backend_executes_focus_gemv_batch_from_buffer_table_and_profiles(tmp_path):
    from renormalizer.backend.gemm import BufferRef, BufferSlice, BufferTable, GemvBatch, GemvDesc
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    a_buffer = np.arange(1, 13, dtype=np.float64)
    x_buffer = np.array([1.0, -1.0, 2.0, 3.0, 0.5, -0.5], dtype=np.float64)
    y_buffer = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    buffers = BufferTable.from_refs(
        [
            BufferRef.from_array("A", a_buffer, device="cpu"),
            BufferRef.from_array("X", x_buffer, device="cpu"),
            BufferRef.from_array("Y", y_buffer, device="cpu"),
        ]
    )
    batch = GemvBatch.from_descs(
        [
            GemvDesc(
                A=BufferSlice("A", offset=0, shape=(2, 3), leading_dim=3),
                x=BufferSlice("X", offset=0, shape=(3,)),
                y=BufferSlice("Y", offset=0, shape=(2,)),
                trans_a="N",
                m=2,
                n=3,
                lda=3,
                alpha=2.0,
                beta=0.5,
                dtype=np.dtype("float64"),
                tag="reduce-0",
            ),
            GemvDesc(
                A=BufferSlice("A", offset=6, shape=(2, 3), leading_dim=3),
                x=BufferSlice("X", offset=3, shape=(3,)),
                y=BufferSlice("Y", offset=2, shape=(2,)),
                trans_a="N",
                m=2,
                n=3,
                lda=3,
                alpha=-1.0,
                beta=1.0,
                dtype=np.dtype("float64"),
                tag="reduce-1",
            ),
        ]
    )
    expected0 = 2.0 * a_buffer[:6].reshape(2, 3) @ x_buffer[:3] + 0.5 * y_buffer[:2].copy()
    expected1 = -1.0 * a_buffer[6:12].reshape(2, 3) @ x_buffer[3:6] + y_buffer[2:4].copy()
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = backend.gemv_batch(batch, buffers=buffers, role="reduce")
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], expected0)
    assert np.allclose(results[1], expected1)
    assert np.allclose(y_buffer, np.concatenate([expected0, expected1]))

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in events if payload["event"] == "gemv_batch_execute")
    assert event["descriptor_source"] == "focus_buffer_slice"
    assert event["role"] == "reduce"
    assert event["buffer_names"] == ["A", "X", "Y"]
    assert event["num_tasks"] == 2
    assert event["num_groups"] == 1
    assert event["group_sizes"] == [2]
    assert event["gsta"] == [0, 2]
    assert event["sorted_indices"] == [0, 1]
    assert event["execution_primitives"] == ["gemv"]
    assert event["execution_policies"] == ["loop_gemv"]
    assert event["fallback_reason"] is None
    assert event["total_flops"] == 2 * 2 * 3 * 2
    assert event["output_shape"] == [[2], [2]]
    assert event["gemv_descriptors"][0]["tag"] == "reduce-0"


def test_gemv_batch_profile_records_complex_conjugate_transpose_explicitly(tmp_path):
    from renormalizer.backend.gemm import BufferRef, BufferSlice, BufferTable, GemvBatch, GemvDesc
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    a = np.array(
        [
            [1.0 + 2.0j, 3.0 - 1.0j],
            [2.0 - 1.0j, -4.0 + 0.5j],
            [0.5 + 1.5j, 2.0 + 3.0j],
        ],
        dtype=np.complex128,
    )
    x = np.array([1.0 - 0.5j, 2.0 + 1.0j, -3.0 + 2.0j], dtype=np.complex128)
    y = np.array([10.0 + 1.0j, -2.0 + 0.5j], dtype=np.complex128)
    buffers = BufferTable.from_refs(
        [
            BufferRef.from_array("A", a.ravel(), device="cpu"),
            BufferRef.from_array("X", x, device="cpu"),
            BufferRef.from_array("Y", y, device="cpu"),
        ]
    )
    batch = GemvBatch.from_descs(
        [
            GemvDesc(
                A=BufferSlice("A", offset=0, shape=(3, 2), leading_dim=2),
                x=BufferSlice("X", offset=0, shape=(3,)),
                y=BufferSlice("Y", offset=0, shape=(2,)),
                trans_a="C",
                m=3,
                n=2,
                lda=2,
                alpha=0.5 + 0.25j,
                beta=2.0 - 1.0j,
                dtype=np.dtype("complex128"),
                tag="conjugate-reduce",
            ),
        ]
    )
    expected = (0.5 + 0.25j) * (a.conj().T @ x) + (2.0 - 1.0j) * y.copy()
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = backend.gemv_batch(batch, buffers=buffers, role="reduce")
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], expected)
    assert np.allclose(y, expected)
    event = next(
        payload
        for payload in (
            json.loads(line)
            for line in event_path.read_text().splitlines()
            if line.strip()
        )
        if payload["event"] == "gemv_batch_execute"
    )
    assert event["gemv_descriptors"][0]["trans_a"] == "C"
    assert event["gemv_descriptors"][0]["conj_a"] is True
    assert event["group_keys"][0]["trans_a"] == "C"
    assert event["group_keys"][0]["conj_a"] is True
    assert event["group_key_buckets"][0]["conj_a"] is True


def test_execute_hmm_gemv_batch_records_hmm_phase_context(tmp_path):
    from renormalizer.backend.gemm import (
        BufferRef,
        BufferSlice,
        BufferTable,
        GemvDesc,
        HxBlock,
        HxBlockList,
        build_hmm_task,
        execute_hmm_gemv_batch,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    a_buffer = np.arange(1, 7, dtype=np.float64)
    x_buffer = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    y_buffer = np.array([10.0, 20.0], dtype=np.float64)
    buffers = BufferTable.from_refs(
        [
            BufferRef.from_array("A", a_buffer, device="cpu"),
            BufferRef.from_array("X", x_buffer, device="cpu"),
            BufferRef.from_array("Y", y_buffer, device="cpu"),
        ]
    )
    desc = GemvDesc(
        A=BufferSlice("A", offset=0, shape=(2, 3), leading_dim=3),
        x=BufferSlice("X", offset=0, shape=(3,)),
        y=BufferSlice("Y", offset=0, shape=(2,)),
        trans_a="N",
        m=2,
        n=3,
        lda=3,
        alpha=1.5,
        beta=0.25,
        dtype=np.dtype("float64"),
        tag=("hmm", "inter"),
    )
    block = HxBlock(
        block_id=5,
        input_slice=BufferSlice("X", offset=0, shape=(3,)),
        output_slice=BufferSlice("Y", offset=0, shape=(2,)),
        coefficient=1.0,
        inter_slice=BufferSlice("Y", offset=0, shape=(2,)),
        matmul_stages=(),
        gemv_inter=(desc,),
        gemv_reduce=(),
        qn_key=("q0",),
        term_key=("term0",),
        cost=12.0,
    )
    task = build_hmm_task(
        HxBlockList.from_blocks(
            [block],
            center_kind="onedot",
            direct_intermediate=True,
            qn_adapted=True,
            output_shape=(2,),
        ),
        batch_size=1,
    )
    expected = 1.5 * a_buffer.reshape(2, 3) @ x_buffer + 0.25 * y_buffer.copy()
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = execute_hmm_gemv_batch(
            backend,
            task,
            batch_index=0,
            role="inter",
            buffers=buffers,
            fallback_policy="forbid",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], expected)
    assert np.allclose(y_buffer, expected)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in events if payload["event"] == "gemv_batch_execute")
    assert event["role"] == "inter_gemv"
    assert event["hmm_phase"] == "inter_gemv"
    assert event["hmm_batch"] == 0
    assert event["hmm_center_kind"] == "onedot"
    assert event["hmm_direct_intermediate"] is True
    assert event["hmm_qn_adapted"] is True
    assert event["hmm_num_batches"] == 1
    assert event["task_block_ids"] == [5]
    assert event["task_block_gemv_indices"] == [0]
    assert event["fallback_policy"] is None
    assert event["fallback_reason"] is None
    assert event["hmm_trace_item"]["phase"] == "inter_gemv"
    assert event["hmm_trace_item"]["task_block_ids"] == [5]
    assert event["hmm_trace_item"]["task_descriptors"][0]["tag"] == ["hmm", "inter"]


def test_execute_hmm_task_runs_buffer_gemv_gemm_reduce_pipeline_and_profiles(tmp_path):
    from renormalizer.backend.gemm import (
        BufferRef,
        BufferSlice,
        BufferTable,
        GemvDesc,
        HxBlock,
        HxBlockList,
        MatmulDesc,
        build_hmm_task,
        execute_hmm_task,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    inter_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64).ravel()
    x = np.array([1.0, -1.0, 2.0], dtype=np.float64)
    inter = np.zeros(2, dtype=np.float64)
    gemm_b = np.array([[2.0, -1.0]], dtype=np.float64).ravel()
    c = np.zeros(4, dtype=np.float64)
    reduce_x = np.array([0.5, 2.0], dtype=np.float64)
    y = np.array([1.0, -2.0], dtype=np.float64)
    buffers = BufferTable.from_refs(
        [
            BufferRef.from_array("INTER_A", inter_a, device="cpu"),
            BufferRef.from_array("X", x, device="cpu"),
            BufferRef.from_array("INTER", inter, device="cpu"),
            BufferRef.from_array("B", gemm_b, device="cpu"),
            BufferRef.from_array("C", c, device="cpu"),
            BufferRef.from_array("RED_X", reduce_x, device="cpu"),
            BufferRef.from_array("Y", y, device="cpu"),
        ]
    )
    inter_desc = GemvDesc(
        A=BufferSlice("INTER_A", offset=0, shape=(2, 3), leading_dim=3),
        x=BufferSlice("X", offset=0, shape=(3,)),
        y=BufferSlice("INTER", offset=0, shape=(2,)),
        trans_a="N",
        m=2,
        n=3,
        lda=3,
        dtype=np.dtype("float64"),
        tag=("pipeline", "inter"),
    )
    gemm_desc = MatmulDesc(
        A=BufferSlice("INTER", offset=0, shape=(2, 1), leading_dim=1),
        B=BufferSlice("B", offset=0, shape=(1, 2), leading_dim=2),
        C=BufferSlice("C", offset=0, shape=(2, 2), leading_dim=2),
        trans_a="N",
        trans_b="N",
        m=2,
        n=2,
        k=1,
        lda=1,
        ldb=2,
        ldc=2,
        dtype=np.dtype("float64"),
        tag=("pipeline", "gemm"),
    )
    reduce_desc = GemvDesc(
        A=BufferSlice("C", offset=0, shape=(2, 2), leading_dim=2),
        x=BufferSlice("RED_X", offset=0, shape=(2,)),
        y=BufferSlice("Y", offset=0, shape=(2,)),
        trans_a="N",
        m=2,
        n=2,
        lda=2,
        alpha=1.0,
        beta=0.25,
        dtype=np.dtype("float64"),
        tag=("pipeline", "reduce"),
    )
    block = HxBlock(
        block_id=0,
        input_slice=BufferSlice("X", offset=0, shape=(3,)),
        output_slice=BufferSlice("Y", offset=0, shape=(2,)),
        coefficient=1.0,
        inter_slice=BufferSlice("INTER", offset=0, shape=(2,)),
        matmul_stages=((gemm_desc,),),
        gemv_inter=(inter_desc,),
        gemv_reduce=(reduce_desc,),
        qn_key=("q0",),
        term_key=("pipeline",),
        cost=32.0,
    )
    task = build_hmm_task(
        HxBlockList.from_blocks(
            [block],
            center_kind="onedot",
            direct_intermediate=True,
            qn_adapted=True,
            output_shape=(2,),
        ),
        batch_size=1,
    )
    expected_inter = inter_a.reshape(2, 3) @ x
    expected_c = expected_inter.reshape(2, 1) @ gemm_b.reshape(1, 2)
    expected_y = expected_c.reshape(2, 2) @ reduce_x + 0.25 * y.copy()
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        profile = execute_hmm_task(
            backend,
            task,
            buffers=buffers,
            pack_threshold=1,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(inter, expected_inter)
    assert np.allclose(c.reshape(2, 2), expected_c)
    assert np.allclose(y, expected_y)
    assert profile["phase_counts"] == {
        "inter_gemv": 1,
        "gemm_stage": 1,
        "reduce_gemv": 1,
    }
    assert profile["phase_events"] == [
        "gemv_batch_execute",
        "grouped_gemm_execute",
        "gemv_batch_execute",
    ]

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    aggregate = next(
        payload
        for payload in events
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_task_buffers"
    )
    assert aggregate["hmm_center_kind"] == "onedot"
    assert aggregate["hmm_qn_adapted"] is True
    assert aggregate["num_gemv_desc"] == 2
    assert aggregate["num_gemm_desc"] == 1
    assert aggregate["phase_events"] == profile["phase_events"]
    assert aggregate["execution_primitives"] == ["gemv", "grouped_gemm"]
    assert aggregate["execution_policies"] == ["loop_gemv", "bucketed_loop_matmul"]
    assert aggregate["grouped_gemm_implementations"] == ["fallback_bucketed_loop_matmul"]
    assert aggregate["requires_grouped_gemm_fallback"] is True
    assert aggregate["fallback_from"] == "grouped_gemm"
    assert aggregate["fallback_to"] == "bucketed_loop_matmul"
    assert aggregate["fallback_policy"] == "record"
    assert aggregate["fallback_reasons"] == [
        "backend-owned grouped_gemm unavailable; used bucketed fallback"
    ]
    assert [item["phase"] for item in aggregate["hmm_execution_trace"]] == [
        "inter_gemv",
        "gemm_stage",
        "reduce_gemv",
    ]
    assert aggregate["hmm_execution_trace"][1]["task_block_ids"] == [0]
    assert aggregate["hmm_execution_trace"][1]["task_descriptors"][0]["tag"] == ["pipeline", "gemm"]
    gemm_event = next(payload for payload in events if payload["event"] == "grouped_gemm_execute")
    assert gemm_event["hmm_phase"] == "gemm_stage"
    assert gemm_event["hmm_batch"] == 0
    assert gemm_event["hmm_stage"] == 0
    assert gemm_event["hmm_center_kind"] == "onedot"
    assert gemm_event["hmm_direct_intermediate"] is True
    assert gemm_event["hmm_qn_adapted"] is True
    assert gemm_event["hmm_trace_item"]["phase"] == "gemm_stage"
    assert gemm_event["hmm_trace_item"]["task_block_ids"] == [0]
    assert gemm_event["hmm_trace_item"]["task_descriptors"][0]["tag"] == ["pipeline", "gemm"]


def test_build_hmm_task_batches_hx_blocks_into_gemm_and_gemv_batches():
    from renormalizer.backend.gemm import (
        BufferSlice,
        GemmTask,
        GemvDesc,
        HxBlock,
        HxBlockList,
        build_hmm_task,
    )

    def gemm(scale, shape=(16, 8, 4), tag=None):
        m, k, n = shape
        return GemmTask(
            np.full((m, k), scale, dtype=np.float64),
            np.ones((k, n), dtype=np.float64),
            tag=tag,
        )

    def gemv(block_id, tag):
        return GemvDesc(
            A=BufferSlice("A", offset=block_id * 64, shape=(8, 8), leading_dim=8),
            x=BufferSlice("X", offset=block_id * 8, shape=(8,)),
            y=BufferSlice("Y", offset=block_id * 8, shape=(8,)),
            trans_a="N",
            m=8,
            n=8,
            lda=8,
            dtype=np.dtype("float64"),
            tag=tag,
        )

    blocks = [
        HxBlock(
            block_id=0,
            input_slice=BufferSlice("X", offset=0, shape=(16, 8)),
            output_slice=BufferSlice("Y", offset=0, shape=(16, 4)),
            coefficient=1.0,
            inter_slice=BufferSlice("INTER", offset=0, shape=(16, 4)),
            matmul_stages=((gemm(1.0, tag="b0-s0"),), (gemm(2.0, tag="b0-s1"),)),
            gemv_inter=(gemv(0, "b0-inter"),),
            gemv_reduce=(gemv(0, "b0-reduce"),),
            qn_key=("q0",),
            term_key=("t0",),
            cost=10.0,
        ),
        HxBlock(
            block_id=1,
            input_slice=BufferSlice("X", offset=128, shape=(16, 8)),
            output_slice=BufferSlice("Y", offset=64, shape=(16, 4)),
            coefficient=2.0,
            inter_slice=BufferSlice("INTER", offset=64, shape=(16, 4)),
            matmul_stages=((gemm(3.0, tag="b1-s0"),), (gemm(4.0, tag="b1-s1"),)),
            gemv_inter=(gemv(1, "b1-inter"),),
            gemv_reduce=(gemv(1, "b1-reduce"),),
            qn_key=("q1",),
            term_key=("t1",),
            cost=20.0,
        ),
        HxBlock(
            block_id=2,
            input_slice=BufferSlice("X", offset=256, shape=(3, 4)),
            output_slice=BufferSlice("Y", offset=128, shape=(3, 5)),
            coefficient=3.0,
            inter_slice=None,
            matmul_stages=((gemm(5.0, shape=(3, 4, 5), tag="b2-s0"),), ()),
            gemv_inter=(),
            gemv_reduce=(gemv(2, "b2-reduce"),),
            qn_key=("q2",),
            term_key=("t2",),
            cost=5.0,
        ),
    ]
    hxlist = HxBlockList.from_blocks(
        blocks,
        center_kind="onedot",
        direct_intermediate=True,
        qn_adapted=True,
    )

    task = build_hmm_task(hxlist, batch_size=2, workspace_limit_bytes=4096)

    assert hxlist.total_cost == 35.0
    assert hxlist.max_block_size == 128
    assert task.center_kind == "onedot"
    assert task.direct_intermediate is True
    assert task.batch_size == 2
    assert task.num_batches == 2
    assert task.total_cost == 35.0
    assert len(task.batches) == 2
    assert len(task.batches[0]) == 2
    assert task.batches[0][0].group_sizes == (2,)
    assert task.batches[0][1].group_sizes == (2,)
    assert task.batches[1][0].group_sizes == (1,)
    assert task.batches[1][1].descs == ()
    assert task.inter_batches[0].group_sizes == (2,)
    assert task.inter_batches[1].descs == ()
    assert task.reduce_batches[0].group_sizes == (2,)
    assert task.reduce_batches[1].group_sizes == (1,)
    assert task.workspace_size == 128
    assert task.inter_size == 64


def test_hmm_batch_size_can_be_chosen_from_workspace_budget():
    from renormalizer.backend.gemm import (
        BufferSlice,
        GemmTask,
        HxBlock,
        HxBlockList,
        build_hmm_task,
        choose_batch_size,
    )

    assert choose_batch_size(
        max_blocks=10,
        block_workspace_size=128,
        static_workspace_size=64,
        memory_budget_bytes=576,
    ) == 4
    assert choose_batch_size(
        max_blocks=10,
        block_workspace_size=128,
        static_workspace_size=64,
        memory_budget_bytes=4096,
    ) == 10

    with pytest.raises(ValueError, match="memory_budget_bytes.*insufficient"):
        choose_batch_size(
            max_blocks=10,
            block_workspace_size=128,
            static_workspace_size=64,
            memory_budget_bytes=191,
        )

    def block(block_id):
        return HxBlock(
            block_id=block_id,
            input_slice=BufferSlice("X", offset=block_id * 64, shape=(8, 8)),
            output_slice=BufferSlice("Y", offset=block_id * 64, shape=(8, 8)),
            coefficient=1.0,
            inter_slice=None,
            matmul_stages=((
                GemmTask(
                    np.ones((8, 4), dtype=np.float64),
                    np.ones((4, 8), dtype=np.float64),
                    tag=("auto-batch", block_id),
                ),
            ),),
            gemv_inter=(),
            gemv_reduce=(),
            cost=1.0,
        )

    hxlist = HxBlockList.from_blocks(
        [block(0), block(1), block(2)],
        center_kind="onedot",
        direct_intermediate=False,
        qn_adapted=False,
    )

    task = build_hmm_task(
        hxlist,
        batch_size=None,
        workspace_limit_bytes=2 * 64 * np.dtype("float64").itemsize + 16,
        static_workspace_bytes=16,
    )

    assert task.batch_size == 2
    assert task.num_batches == 2
    assert [len(batch[0].descs) for batch in task.batches] == [2, 1]
    assert task.workspace_size == 64


def test_hmm_task_build_profile_records_workspace_budget_batch_selection(tmp_path):
    from renormalizer.backend.gemm import (
        BufferSlice,
        GemmTask,
        HxBlock,
        HxBlockList,
        build_hmm_task,
    )
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    def block(block_id):
        return HxBlock(
            block_id=block_id,
            input_slice=BufferSlice("X", offset=block_id * 64, shape=(8, 8)),
            output_slice=BufferSlice("Y", offset=block_id * 64, shape=(8, 8)),
            coefficient=1.0,
            inter_slice=None,
            matmul_stages=((
                GemmTask(
                    np.ones((8, 4), dtype=np.float64),
                    np.ones((4, 8), dtype=np.float64),
                    tag=("auto-batch-profile", block_id),
                ),
            ),),
            gemv_inter=(),
            gemv_reduce=(),
            cost=1.0,
        )

    hxlist = HxBlockList.from_blocks(
        [block(0), block(1), block(2)],
        center_kind="onedot",
        direct_intermediate=False,
        qn_adapted=False,
    )
    static_workspace_bytes = 16
    workspace_limit_bytes = 2 * 64 * np.dtype("float64").itemsize + static_workspace_bytes
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        task = build_hmm_task(
            hxlist,
            batch_size=None,
            workspace_limit_bytes=workspace_limit_bytes,
            static_workspace_bytes=static_workspace_bytes,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert task.batch_size == 2
    assert task.batch_size_auto is True
    assert task.workspace_limit_bytes == workspace_limit_bytes
    assert task.static_workspace_bytes == static_workspace_bytes
    assert task.block_workspace_size == 64
    assert task.block_workspace_bytes == 64 * np.dtype("float64").itemsize

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "hmm_task_build")
    assert event["batch_size_auto"] is True
    assert event["workspace_limit_bytes"] == workspace_limit_bytes
    assert event["static_workspace_bytes"] == static_workspace_bytes
    assert event["block_workspace_size"] == 64
    assert event["block_workspace_bytes"] == 64 * np.dtype("float64").itemsize
    assert event["hmm_memory_plan"] == {
        "max_blocks": 3,
        "batch_size": 2,
        "batch_size_auto": True,
        "workspace_limit_bytes": workspace_limit_bytes,
        "static_workspace_bytes": static_workspace_bytes,
        "block_workspace_size": 64,
        "block_workspace_bytes": 64 * np.dtype("float64").itemsize,
    }


def test_hmm_task_lowers_buffer_slice_matmul_descs_to_grouped_plan():
    from renormalizer.backend.gemm import (
        BufferSlice,
        HxBlock,
        HxBlockList,
        MatmulDesc,
        build_hmm_task,
        lower_hmm_task_to_contraction_plan,
    )

    def matmul_desc(block_id, tag):
        return MatmulDesc(
            A=BufferSlice("A", offset=block_id * 12, shape=(4, 3), leading_dim=3),
            B=BufferSlice("B", offset=block_id * 6, shape=(3, 2), leading_dim=2),
            C=BufferSlice("C", offset=block_id * 8, shape=(4, 2), leading_dim=2),
            trans_a="N",
            trans_b="N",
            m=4,
            n=2,
            k=3,
            lda=3,
            ldb=2,
            ldc=2,
            dtype=np.dtype("float64"),
            tag=tag,
        )

    blocks = [
        HxBlock(
            block_id=0,
            input_slice=BufferSlice("X", offset=0, shape=(4, 3)),
            output_slice=BufferSlice("Y", offset=0, shape=(4, 2)),
            coefficient=1.0,
            inter_slice=None,
            matmul_stages=((matmul_desc(0, "same-0"),),),
            gemv_inter=(),
            gemv_reduce=(),
        ),
        HxBlock(
            block_id=1,
            input_slice=BufferSlice("X", offset=12, shape=(4, 3)),
            output_slice=BufferSlice("Y", offset=8, shape=(4, 2)),
            coefficient=1.0,
            inter_slice=None,
            matmul_stages=((matmul_desc(1, "same-1"),),),
            gemv_inter=(),
            gemv_reduce=(),
        ),
    ]
    hxlist = HxBlockList.from_blocks(
        blocks,
        center_kind="onedot",
        direct_intermediate=False,
        qn_adapted=True,
        equation="ik,kj->ij",
        operand_shapes=((4, 3), (3, 2)),
        output_shape=(4, 2),
    )

    task = build_hmm_task(hxlist, batch_size=2)
    plan = lower_hmm_task_to_contraction_plan(task)

    assert task.batches[0][0].group_sizes == (2,)
    assert plan.steps[0].kind == "grouped_gemm"
    assert plan.steps[0].plan.descs[0].m == 4
    assert plan.steps[0].plan.descs[0].n == 2
    assert plan.steps[0].plan.descs[0].k == 3
    assert plan.steps[0].plan.estimated_flops == 2 * 2 * 4 * 2 * 3


def test_hmm_task_profiles_explicit_output_reduction_modes(tmp_path):
    from renormalizer.backend import OutputMode
    from renormalizer.backend.gemm import (
        BufferSlice,
        GemmTask,
        HxBlock,
        HxBlockList,
        build_hmm_task,
        record_hmm_contraction_plan,
    )
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    def block(block_id, output_mode, output_offset):
        return HxBlock(
            block_id=block_id,
            input_slice=BufferSlice("X", offset=block_id * 12, shape=(4, 3)),
            output_slice=BufferSlice("Y", offset=output_offset, shape=(4, 2)),
            coefficient=1.0,
            inter_slice=None,
            matmul_stages=((
                GemmTask(
                    np.ones((4, 3), dtype=np.float64),
                    np.ones((3, 2), dtype=np.float64),
                    tag=("mode", output_mode.value),
                ),
            ),),
            gemv_inter=(),
            gemv_reduce=(),
            output_mode=output_mode,
        )

    hxlist = HxBlockList.from_blocks(
        [
            block(0, OutputMode.OVERWRITE, 0),
            block(1, OutputMode.ADD, 0),
            block(2, OutputMode.REDUCE_BY_GEMV, 8),
        ],
        center_kind="onedot",
        direct_intermediate=False,
        qn_adapted=True,
        equation="ik,kj->ij",
        operand_shapes=((4, 3), (3, 2)),
        output_shape=(4, 2),
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        task = build_hmm_task(hxlist, batch_size=3)
        record_hmm_contraction_plan(task)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    build_event = next(payload for payload in payloads if payload["event"] == "hmm_task_build")
    plan_event = next(payload for payload in payloads if payload["event"] == "contraction_plan")

    assert task.output_modes == ("overwrite", "add", "reduce_by_gemv")
    assert task.output_mode_counts == {
        "overwrite": 1,
        "add": 1,
        "reduce_by_gemv": 1,
    }
    assert task.scatter_add_required is True
    assert task.reduce_by_gemv_required is True
    assert build_event["output_modes"] == ["overwrite", "add", "reduce_by_gemv"]
    assert build_event["output_mode_counts"] == {
        "overwrite": 1,
        "add": 1,
        "reduce_by_gemv": 1,
    }
    assert build_event["reduction_mode"] == "mixed"
    assert build_event["scatter_add_required"] is True
    assert build_event["reduce_by_gemv_required"] is True
    assert plan_event["output_contribution_modes"] == ["overwrite", "add", "reduce_by_gemv"]
    assert plan_event["reduction_mode"] == "mixed"


def test_hmm_task_builds_output_reduction_groups_for_overlapping_slices(tmp_path):
    from renormalizer.backend.gemm import (
        BufferSlice,
        GemmTask,
        HxBlock,
        HxBlockList,
        OutputMode,
        build_hmm_task,
    )
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    def block(block_id, output_offset, output_mode):
        return HxBlock(
            block_id=block_id,
            input_slice=BufferSlice("X", offset=block_id * 12, shape=(4, 3)),
            output_slice=BufferSlice("Y", offset=output_offset, shape=(4, 2)),
            coefficient=block_id + 1.0,
            inter_slice=None,
            matmul_stages=((
                GemmTask(
                    np.ones((4, 3), dtype=np.float64),
                    np.ones((3, 2), dtype=np.float64),
                    tag=("block", block_id),
                ),
            ),),
            gemv_inter=(),
            gemv_reduce=(),
            output_mode=output_mode,
        )

    hxlist = HxBlockList.from_blocks(
        [
            block(0, 0, OutputMode.OVERWRITE),
            block(1, 0, OutputMode.ADD),
            block(2, 8, OutputMode.REDUCE_BY_GEMV),
        ],
        center_kind="onedot",
        direct_intermediate=False,
        qn_adapted=True,
        equation="ik,kj->ij",
        operand_shapes=((4, 3), (3, 2)),
        output_shape=(4, 2),
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        task = build_hmm_task(hxlist, batch_size=3)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert [group.output_key for group in task.output_reduction_groups] == [
        ("Y", 0, (4, 2)),
        ("Y", 8, (4, 2)),
    ]
    assert task.output_reduction_groups[0].block_ids == (0, 1)
    assert task.output_reduction_groups[0].output_modes == ("overwrite", "add")
    assert task.output_reduction_groups[0].reduction_mode == "mixed"
    assert task.output_reduction_groups[0].scatter_add_required is True
    assert task.output_reduction_groups[1].block_ids == (2,)
    assert task.output_reduction_groups[1].output_modes == ("reduce_by_gemv",)
    assert task.output_reduction_groups[1].reduce_by_gemv_required is True

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "hmm_task_build")
    assert event["num_output_reduction_groups"] == 2
    assert event["output_reduction_groups"][0] == {
        "output_key": ["Y", 0, [4, 2]],
        "contribution_indices": [0, 1],
        "block_ids": [0, 1],
        "output_modes": ["overwrite", "add"],
        "reduction_mode": "mixed",
        "scatter_add_required": True,
        "reduce_by_gemv_required": False,
        "contributions": [
            {
                "contribution_index": 0,
                "block_id": 0,
                "input_slice": ["X", 0, [4, 3]],
                "output_slice": ["Y", 0, [4, 2]],
                "inter_slice": None,
                "output_mode": "overwrite",
                "coefficient": 1.0,
            },
            {
                "contribution_index": 1,
                "block_id": 1,
                "input_slice": ["X", 12, [4, 3]],
                "output_slice": ["Y", 0, [4, 2]],
                "inter_slice": None,
                "output_mode": "add",
                "coefficient": 2.0,
            },
        ],
    }


def test_hmm_task_build_profile_records_focus_batch_stage_boundaries(tmp_path, caplog):
    from renormalizer.backend.gemm import (
        BufferSlice,
        GemmTask,
        GemvDesc,
        HxBlock,
        HxBlockList,
        build_hmm_task,
        hmm_task_execution_metadata,
        lower_hmm_task_to_contraction_plan,
        record_hmm_contraction_plan,
    )
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    def block(block_id, *, shape=(4, 3, 2), tag=None):
        m, k, n = shape
        return HxBlock(
            block_id=block_id,
            input_slice=BufferSlice("X", offset=block_id * m * k, shape=(m, k)),
            output_slice=BufferSlice("Y", offset=block_id * m * n, shape=(m, n)),
            coefficient=1.0,
            inter_slice=None,
            matmul_stages=((
                GemmTask(
                    np.full((m, k), block_id + 1.0, dtype=np.float64),
                    np.ones((k, n), dtype=np.float64),
                    tag=tag,
                ),
            ),),
            gemv_inter=(
                GemvDesc(
                    A=BufferSlice("INTER_A", offset=block_id * m * k, shape=(m, k), leading_dim=k),
                    x=BufferSlice("X", offset=block_id * k, shape=(k,)),
                    y=BufferSlice("INTER_Y", offset=block_id * m, shape=(m,)),
                    trans_a="N",
                    m=m,
                    n=k,
                    lda=k,
                    dtype=np.dtype("float64"),
                    tag=(tag, "inter"),
                ),
            ),
            gemv_reduce=(
                GemvDesc(
                    A=BufferSlice("RED_A", offset=block_id * m * n, shape=(m, n), leading_dim=n),
                    x=BufferSlice("INTER_Y", offset=block_id * n, shape=(n,)),
                    y=BufferSlice("Y", offset=block_id * m, shape=(m,)),
                    trans_a="N",
                    m=m,
                    n=n,
                    lda=n,
                    dtype=np.dtype("float64"),
                    tag=(tag, "reduce"),
                ),
            ),
            cost=float(m * n * k),
        )

    hxlist = HxBlockList.from_blocks(
        [
            block(0, tag="same-0"),
            block(1, tag="same-1"),
            block(2, shape=(2, 3, 5), tag="ragged"),
        ],
        center_kind="onedot",
        direct_intermediate=False,
        qn_adapted=True,
        equation="ik,kj->ij",
        operand_shapes=((4, 3), (3, 2)),
        output_shape=(4, 2),
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    caplog.set_level(PROFILING, logger="renormalizer")
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        task = build_hmm_task(hxlist, batch_size=2)
        record_hmm_contraction_plan(task)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        payload
        for payload in (
            json.loads(line)
            for line in event_path.read_text().splitlines()
            if line.strip()
        )
    ]
    event = next(payload for payload in payloads if payload["event"] == "hmm_task_build")
    plan_event = next(payload for payload in payloads if payload["event"] == "contraction_plan")
    logged_payloads = [
        json.loads(record.getMessage()[len(profiling.LOG_PREFIX):])
        for record in caplog.records
        if record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    summary = next(
        payload
        for payload in logged_payloads
        if payload["event"] == "profile_compute_summary"
        and payload["source_events"] == ["hmm_task_build"]
    )

    assert event["stage_group_sizes"] == [[2, 1]]
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_subclass"] == "hmm_task_build"
    assert event["compute_role"] == "planning"
    assert event["compute_accounting"] == "primary"
    assert event["lowering"] == "hmm_task_build"
    assert event["flops"] == event["total_flops"]
    assert event["num_blocks"] == event["num_hx_blocks"]
    assert event["workspace_bytes"] == event["workspace_size"] * np.dtype("float64").itemsize
    assert event["compute_profile"]["measurement_scope"] == "hmm_task_build"
    assert event["compute_profile"]["work"]["unit"] == "hmm_task_build"
    assert event["compute_profile"]["work"]["num_batches"] == 2
    assert event["compute_profile"]["work"]["num_gemm_desc"] == 3
    assert event["compute_profile"]["work"]["num_gemv_desc"] == 6
    assert summary["compute_class"] == "contraction_plan"
    assert summary["compute_subclass"] == "hmm_task_build"
    assert summary["compute_role"] == "planning"
    assert summary["total_flops_estimate"] == event["total_flops"]
    assert summary["total_blocks"] == 3
    assert summary["total_shape_buckets"] == 2
    assert summary["total_hmm_gemv_shape_buckets"] == 4
    assert summary["total_hmm_batches"] == 2
    assert summary["total_hmm_gemm_desc"] == 3
    assert summary["total_hmm_gemv_desc"] == 6
    assert summary["compute_profile"]["measurement_scope"] == "contraction_plan_summary"
    assert summary["compute_profile"]["work"]["num_batches"] == 2
    assert summary["compute_profile"]["work"]["num_gemm_desc"] == 3
    assert summary["compute_profile"]["work"]["num_gemv_desc"] == 6
    assert event["hmm_hx_blocks"] == [
        {
            "block_id": 0,
            "input_slice": ["X", 0, [4, 3]],
            "output_slice": ["Y", 0, [4, 2]],
            "inter_slice": None,
            "coefficient": 1.0,
            "output_mode": "add",
            "qn_key": None,
            "term_key": None,
            "cost": 24.0,
            "matmul_stage_count": 1,
            "matmul_stage_task_counts": [1],
            "gemv_inter_count": 1,
            "gemv_reduce_count": 1,
        },
        {
            "block_id": 1,
            "input_slice": ["X", 12, [4, 3]],
            "output_slice": ["Y", 8, [4, 2]],
            "inter_slice": None,
            "coefficient": 1.0,
            "output_mode": "add",
            "qn_key": None,
            "term_key": None,
            "cost": 24.0,
            "matmul_stage_count": 1,
            "matmul_stage_task_counts": [1],
            "gemv_inter_count": 1,
            "gemv_reduce_count": 1,
        },
        {
            "block_id": 2,
            "input_slice": ["X", 12, [2, 3]],
            "output_slice": ["Y", 20, [2, 5]],
            "inter_slice": None,
            "coefficient": 1.0,
            "output_mode": "add",
            "qn_key": None,
            "term_key": None,
            "cost": 30.0,
            "matmul_stage_count": 1,
            "matmul_stage_task_counts": [1],
            "gemv_inter_count": 1,
            "gemv_reduce_count": 1,
        },
    ]
    assert plan_event["hmm_hx_blocks"] == event["hmm_hx_blocks"]
    assert event["hmm_batches"][0]["batch"] == 0
    assert event["hmm_batches"][0]["num_stages"] == 1
    assert event["hmm_batches"][0]["stages"][0]["stage"] == 0
    assert event["hmm_batches"][0]["stages"][0]["num_tasks"] == 2
    assert event["hmm_batches"][0]["stages"][0]["num_groups"] == 1
    assert event["hmm_batches"][0]["stages"][0]["group_sizes"] == [2]
    assert event["hmm_batches"][0]["stages"][0]["planned_lowerings"] == ["grouped_gemm"]
    assert event["hmm_batches"][0]["stages"][0]["task_block_ids"] == [0, 1]
    assert event["hmm_batches"][0]["stages"][0]["task_block_stage_task_indices"] == [0, 0]
    assert event["hmm_batches"][0]["stages"][0]["gsta"] == [0, 2]
    assert event["hmm_batches"][0]["stages"][0]["sorted_indices"] == [0, 1]
    assert event["hmm_batches"][0]["stages"][0]["group_keys"][0]["m"] == 4
    assert event["hmm_batches"][0]["stages"][0]["group_keys"][0]["n"] == 2
    assert event["hmm_batches"][0]["stages"][0]["group_keys"][0]["k"] == 3
    assert event["hmm_batches"][0]["stages"][0]["task_descriptors"][0] == {
        "task_index": 0,
        "original_index": 0,
        "block_id": 0,
        "stage_task_index": 0,
        "descriptor_kind": "gemm_task",
        "a_shape": [4, 3],
        "b_shape": [3, 2],
        "c_shape": None,
        "m": 4,
        "n": 2,
        "k": 3,
        "batch_shape": [],
        "trans_a": False,
        "trans_b": False,
        "conj_a": False,
        "conj_b": False,
        "alpha": 1.0,
        "beta": 0.0,
        "dtype_a": "float64",
        "dtype_b": "float64",
        "dtype_c": None,
        "tag": "same-0",
    }
    assert event["hmm_batches"][1]["batch"] == 1
    assert event["hmm_batches"][1]["stages"][0]["group_sizes"] == [1]
    assert event["hmm_batches"][1]["stages"][0]["planned_lowerings"] == ["gemm"]
    assert event["hmm_batches"][1]["stages"][0]["task_block_ids"] == [2]
    assert event["hmm_batches"][1]["stages"][0]["task_block_stage_task_indices"] == [0]
    assert event["hmm_batches"][1]["stages"][0]["gsta"] == [0, 1]
    assert event["hmm_batches"][1]["stages"][0]["sorted_indices"] == [0]
    assert event["hmm_batches"][1]["stages"][0]["group_keys"][0]["m"] == 2
    assert event["hmm_batches"][1]["stages"][0]["group_keys"][0]["n"] == 5
    assert event["hmm_batches"][1]["stages"][0]["group_keys"][0]["k"] == 3
    assert event["num_gemv_desc"] == 6
    assert event["num_shape_buckets"] == 2
    assert event["num_gemv_shape_buckets"] == 4
    assert event["num_total_shape_buckets"] == 6
    assert event["shape_buckets"] == [
        {
            "batch": 0,
            "stage": 0,
            "group": 0,
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "batch_count": 1,
            "m": 4,
            "n": 2,
            "k": 3,
            "lda": 3,
            "ldb": 2,
            "ldc": 2,
            "task_indices": [0, 1],
            "task_count": 2,
            "task_block_ids": [0, 1],
            "task_block_stage_task_indices": [0, 0],
            "planned_lowering": "grouped_gemm",
            "execution_primitives": ["grouped_gemm"],
            "execution_policies": ["backend_grouped_gemm"],
            "fallback_reasons": [],
            "planned_plan_count": 1,
        },
        {
            "batch": 1,
            "stage": 0,
            "group": 0,
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "batch_count": 1,
            "m": 2,
            "n": 5,
            "k": 3,
            "lda": 3,
            "ldb": 5,
            "ldc": 5,
            "task_indices": [0],
            "task_count": 1,
            "task_block_ids": [2],
            "task_block_stage_task_indices": [0],
            "planned_lowering": "gemm",
            "execution_primitives": ["matmul"],
            "execution_policies": ["backend_matmul"],
            "fallback_reasons": [],
            "planned_plan_count": 1,
        },
    ]
    assert event["hmm_gemv_shape_buckets"] == [
        {
            "phase": "inter",
            "batch": 0,
            "group": 0,
            "dtype": "float64",
            "trans_a": "N",
            "conj_a": False,
            "m": 4,
            "n": 3,
            "lda": 3,
            "incx": 1,
            "incy": 1,
            "task_indices": [0, 1],
            "task_count": 2,
            "task_block_ids": [0, 1],
            "task_block_gemv_indices": [0, 0],
            "execution_primitives": ["gemv"],
            "execution_policies": ["backend_gemv"],
            "fallback_reasons": [],
        },
        {
            "phase": "inter",
            "batch": 1,
            "group": 0,
            "dtype": "float64",
            "trans_a": "N",
            "conj_a": False,
            "m": 2,
            "n": 3,
            "lda": 3,
            "incx": 1,
            "incy": 1,
            "task_indices": [0],
            "task_count": 1,
            "task_block_ids": [2],
            "task_block_gemv_indices": [0],
            "execution_primitives": ["gemv"],
            "execution_policies": ["backend_gemv"],
            "fallback_reasons": [],
        },
        {
            "phase": "reduce",
            "batch": 0,
            "group": 0,
            "dtype": "float64",
            "trans_a": "N",
            "conj_a": False,
            "m": 4,
            "n": 2,
            "lda": 2,
            "incx": 1,
            "incy": 1,
            "task_indices": [0, 1],
            "task_count": 2,
            "task_block_ids": [0, 1],
            "task_block_gemv_indices": [0, 0],
            "execution_primitives": ["gemv"],
            "execution_policies": ["backend_gemv"],
            "fallback_reasons": [],
        },
        {
            "phase": "reduce",
            "batch": 1,
            "group": 0,
            "dtype": "float64",
            "trans_a": "N",
            "conj_a": False,
            "m": 2,
            "n": 5,
            "lda": 5,
            "incx": 1,
            "incy": 1,
            "task_indices": [0],
            "task_count": 1,
            "task_block_ids": [2],
            "task_block_gemv_indices": [0],
            "execution_primitives": ["gemv"],
            "execution_policies": ["backend_gemv"],
            "fallback_reasons": [],
        },
    ]
    assert event["hmm_gemv_batches"]["inter"][0]["batch"] == 0
    assert event["hmm_gemv_batches"]["inter"][0]["group_sizes"] == [2]
    assert event["hmm_gemv_batches"]["inter"][0]["gsta"] == [0, 2]
    assert event["hmm_gemv_batches"]["inter"][0]["sorted_indices"] == [0, 1]
    assert event["hmm_gemv_batches"]["inter"][0]["task_block_ids"] == [0, 1]
    assert event["hmm_gemv_batches"]["inter"][0]["task_block_gemv_indices"] == [0, 0]
    assert event["hmm_gemv_batches"]["inter"][0]["execution_primitives"] == ["gemv"]
    assert event["hmm_gemv_batches"]["inter"][0]["execution_policies"] == ["backend_gemv"]
    assert event["hmm_gemv_batches"]["inter"][0]["fallback_reasons"] == []
    assert event["hmm_gemv_batches"]["inter"][0]["group_keys"][0]["m"] == 4
    assert event["hmm_gemv_batches"]["inter"][0]["group_keys"][0]["n"] == 3
    assert event["hmm_gemv_batches"]["inter"][0]["task_descriptors"][0] == {
        "task_index": 0,
        "original_index": 0,
        "block_id": 0,
        "gemv_index": 0,
        "descriptor_kind": "gemv_desc",
        "A_slice": ["INTER_A", 0, [4, 3]],
        "x_slice": ["X", 0, [3]],
        "y_slice": ["INTER_Y", 0, [4]],
        "trans_a": "N",
        "conj_a": False,
        "m": 4,
        "n": 3,
        "lda": 3,
        "incx": 1,
        "incy": 1,
        "alpha": 1.0,
        "beta": 0.0,
        "dtype": "float64",
        "tag": ["same-0", "inter"],
    }
    assert event["hmm_gemv_batches"]["inter"][1]["batch"] == 1
    assert event["hmm_gemv_batches"]["inter"][1]["group_sizes"] == [1]
    assert event["hmm_gemv_batches"]["inter"][1]["gsta"] == [0, 1]
    assert event["hmm_gemv_batches"]["inter"][1]["task_block_ids"] == [2]
    assert event["hmm_gemv_batches"]["inter"][1]["task_block_gemv_indices"] == [0]
    assert event["hmm_gemv_batches"]["inter"][1]["group_keys"][0]["m"] == 2
    assert event["hmm_gemv_batches"]["inter"][1]["group_keys"][0]["n"] == 3
    assert event["hmm_gemv_batches"]["reduce"][0]["group_sizes"] == [2]
    assert event["hmm_gemv_batches"]["reduce"][0]["gsta"] == [0, 2]
    assert event["hmm_gemv_batches"]["reduce"][0]["task_block_ids"] == [0, 1]
    assert event["hmm_gemv_batches"]["reduce"][0]["task_block_gemv_indices"] == [0, 0]
    assert event["hmm_gemv_batches"]["reduce"][0]["execution_primitives"] == ["gemv"]
    assert event["hmm_gemv_batches"]["reduce"][0]["execution_policies"] == ["backend_gemv"]
    assert event["hmm_gemv_batches"]["reduce"][0]["fallback_reasons"] == []
    assert event["hmm_gemv_batches"]["reduce"][0]["group_keys"][0]["m"] == 4
    assert event["hmm_gemv_batches"]["reduce"][0]["group_keys"][0]["n"] == 2
    assert event["hmm_gemv_batches"]["reduce"][1]["group_sizes"] == [1]
    assert event["hmm_gemv_batches"]["reduce"][1]["task_block_ids"] == [2]
    assert event["hmm_gemv_batches"]["reduce"][1]["task_block_gemv_indices"] == [0]
    assert event["hmm_gemv_batches"]["reduce"][1]["group_keys"][0]["m"] == 2
    assert event["hmm_gemv_batches"]["reduce"][1]["group_keys"][0]["n"] == 5
    assert event["hmm_execution_trace"][0]["phase"] == "inter_gemv"
    assert event["hmm_execution_trace"][0]["batch"] == 0
    assert event["hmm_execution_trace"][0]["num_tasks"] == 2
    assert event["hmm_execution_trace"][0]["task_block_ids"] == [0, 1]
    assert event["hmm_execution_trace"][0]["task_block_gemv_indices"] == [0, 0]
    assert event["hmm_execution_trace"][0]["execution_primitives"] == ["gemv"]
    assert event["hmm_execution_trace"][0]["execution_policies"] == ["backend_gemv"]
    assert event["hmm_execution_trace"][0]["fallback_reasons"] == []
    assert event["hmm_execution_trace"][0]["task_descriptors"][0] == event["hmm_gemv_batches"]["inter"][0]["task_descriptors"][0]
    assert event["hmm_execution_trace"][1]["phase"] == "gemm_stage"
    assert event["hmm_execution_trace"][1]["batch"] == 0
    assert event["hmm_execution_trace"][1]["stage"] == 0
    assert event["hmm_execution_trace"][1]["group_sizes"] == [2]
    assert event["hmm_execution_trace"][1]["planned_lowerings"] == ["grouped_gemm"]
    assert event["hmm_execution_trace"][1]["task_block_ids"] == [0, 1]
    assert event["hmm_execution_trace"][1]["task_block_stage_task_indices"] == [0, 0]
    assert event["hmm_execution_trace"][1]["task_descriptors"] == event["hmm_batches"][0]["stages"][0]["task_descriptors"]
    assert event["hmm_execution_trace"][2]["phase"] == "reduce_gemv"
    assert event["hmm_execution_trace"][2]["batch"] == 0
    assert event["hmm_execution_trace"][2]["task_block_ids"] == [0, 1]
    assert event["hmm_execution_trace"][2]["task_block_gemv_indices"] == [0, 0]
    assert event["hmm_execution_trace"][2]["execution_primitives"] == ["gemv"]
    assert event["hmm_execution_trace"][2]["execution_policies"] == ["backend_gemv"]
    assert event["hmm_execution_trace"][2]["fallback_reasons"] == []
    assert event["hmm_execution_trace"][2]["output_reduction_group_indices"] == [0, 1]
    assert event["hmm_execution_trace"][2]["output_reduction_groups"] == event["output_reduction_groups"][:2]
    assert event["hmm_execution_trace"][3]["phase"] == "inter_gemv"
    assert event["hmm_execution_trace"][3]["batch"] == 1
    assert event["hmm_execution_trace"][3]["task_block_ids"] == [2]
    assert event["hmm_execution_trace"][3]["task_block_gemv_indices"] == [0]
    assert event["hmm_execution_trace"][4]["phase"] == "gemm_stage"
    assert event["hmm_execution_trace"][4]["batch"] == 1
    assert event["hmm_execution_trace"][4]["stage"] == 0
    assert event["hmm_execution_trace"][4]["planned_lowerings"] == ["gemm"]
    assert event["hmm_execution_trace"][4]["task_block_ids"] == [2]
    assert event["hmm_execution_trace"][4]["task_block_stage_task_indices"] == [0]
    assert event["hmm_execution_trace"][5]["phase"] == "reduce_gemv"
    assert event["hmm_execution_trace"][5]["batch"] == 1
    assert event["hmm_execution_trace"][5]["task_block_ids"] == [2]
    assert event["hmm_execution_trace"][5]["task_block_gemv_indices"] == [0]
    assert event["hmm_execution_trace"][5]["output_reduction_group_indices"] == [2]
    assert event["hmm_execution_trace"][5]["output_reduction_groups"] == event["output_reduction_groups"][2:]
    assert [item["phase"] for item in event["hmm_execution_trace"]] == [
        "inter_gemv",
        "gemm_stage",
        "reduce_gemv",
        "inter_gemv",
        "gemm_stage",
        "reduce_gemv",
    ]
    assert plan_event["hmm_execution_trace"][1]["plan_hashes"] == [
        plan_event["hmm_steps"][0]["plan_hash"]
    ]
    assert plan_event["hmm_execution_trace"][4]["plan_hashes"] == [
        plan_event["hmm_steps"][1]["plan_hash"]
    ]
    assert plan_event["hmm_gemv_shape_buckets"] == event["hmm_gemv_shape_buckets"]
    assert plan_event["hmm_gemv_batches"] == event["hmm_gemv_batches"]
    metadata = hmm_task_execution_metadata(lower_hmm_task_to_contraction_plan(task), task)
    assert metadata["hmm_gemv_shape_buckets"] == event["hmm_gemv_shape_buckets"]
    assert metadata["hmm_gemv_batches"] == event["hmm_gemv_batches"]
    assert metadata["hmm_execution_trace"] == plan_event["hmm_execution_trace"]


def test_single_site_hmm_scaffold_records_hvec_shape_groups(tmp_path):
    from renormalizer.backend.gemm import build_single_site_hmm_scaffold
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    ltensor = np.ones((2, 3, 4), dtype=np.float64)
    mpo = np.ones((3, 5, 6, 7), dtype=np.float64)
    rtensor = np.ones((8, 7, 9), dtype=np.float64)
    center = np.ones((4, 6, 9), dtype=np.float64)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        hxlist, task = build_single_site_hmm_scaffold(
            ltensor,
            mpo,
            rtensor,
            center,
            batch_size=1,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert hxlist.center_kind == "onedot"
    assert hxlist.equation == "abc,bdef,lfk,cek->adl"
    assert hxlist.operand_shapes == ((2, 3, 4), (3, 5, 6, 7), (8, 7, 9), (4, 6, 9))
    assert hxlist.output_shape == (2, 5, 8)
    assert task.center_kind == "onedot"
    assert task.equation == "abc,bdef,lfk,cek->adl"
    assert task.num_batches == 1
    assert len(task.batches[0]) == 3
    assert [batch.group_sizes for batch in task.batches[0]] == [(1,), (1,), (1,)]
    assert [(batch.max_m, batch.max_n, batch.max_k) for batch in task.batches[0]] == [
        (24, 56, 9),
        (15, 32, 42),
        (2, 40, 12),
    ]
    assert task.inter_size == 24 * 56
    assert task.workspace_size == max(center.size, np.prod(hxlist.output_shape), task.inter_size)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "hmm_task_build")
    assert event["equation"] == "abc,bdef,lfk,cek->adl"
    assert event["center_kind"] == "onedot"
    assert event["num_hx_blocks"] == 1
    assert event["num_batches"] == 1
    assert event["batch_size"] == 1
    assert event["num_gemm_desc"] == 3
    assert event["num_gemv_desc"] == 0
    assert event["stage_group_sizes"] == [[1], [1], [1]]
    assert event["operand_shapes"] == [[2, 3, 4], [3, 5, 6, 7], [8, 7, 9], [4, 6, 9]]
    assert event["output_shape"] == [2, 5, 8]


def test_two_site_hmm_scaffold_records_hvec_shape_groups(tmp_path):
    from renormalizer.backend.gemm import build_two_site_hmm_scaffold
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    ltensor = np.ones((2, 3, 4), dtype=np.float64)
    mpo0 = np.ones((3, 5, 6, 7), dtype=np.float64)
    mpo1 = np.ones((7, 8, 9, 10), dtype=np.float64)
    rtensor = np.ones((11, 10, 12), dtype=np.float64)
    center = np.ones((4, 6, 9, 12), dtype=np.float64)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        hxlist, task = build_two_site_hmm_scaffold(
            ltensor,
            mpo0,
            mpo1,
            rtensor,
            center,
            batch_size=1,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert hxlist.center_kind == "twodot"
    assert hxlist.equation == "abc,bdef,fghj,ljk,cehk->adgl"
    assert hxlist.operand_shapes == (
        (2, 3, 4),
        (3, 5, 6, 7),
        (7, 8, 9, 10),
        (11, 10, 12),
        (4, 6, 9, 12),
    )
    assert hxlist.output_shape == (2, 5, 8, 11)
    assert task.center_kind == "twodot"
    assert task.equation == "abc,bdef,fghj,ljk,cehk->adgl"
    assert task.num_batches == 1
    assert len(task.batches[0]) == 4
    assert [batch.group_sizes for batch in task.batches[0]] == [(1,), (1,), (1,), (1,)]
    assert [(batch.max_m, batch.max_n, batch.max_k) for batch in task.batches[0]] == [
        (216, 110, 12),
        (56, 264, 90),
        (15, 352, 42),
        (2, 440, 12),
    ]
    assert task.inter_size == 216 * 110
    assert task.workspace_size == max(center.size, np.prod(hxlist.output_shape), task.inter_size)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "hmm_task_build")
    assert event["equation"] == "abc,bdef,fghj,ljk,cehk->adgl"
    assert event["center_kind"] == "twodot"
    assert event["num_hx_blocks"] == 1
    assert event["num_batches"] == 1
    assert event["batch_size"] == 1
    assert event["num_gemm_desc"] == 4
    assert event["num_gemv_desc"] == 0
    assert event["stage_group_sizes"] == [[1], [1], [1], [1]]
    assert event["operand_shapes"] == [
        [2, 3, 4],
        [3, 5, 6, 7],
        [7, 8, 9, 10],
        [11, 10, 12],
        [4, 6, 9, 12],
    ]
    assert event["output_shape"] == [2, 5, 8, 11]


def test_hmm_task_lowers_scaffold_stages_to_matmul_plan_ir():
    from renormalizer.backend import MatmulPlan
    from renormalizer.backend.gemm import (
        build_single_site_hmm_scaffold_from_shapes,
        build_two_site_hmm_scaffold_from_shapes,
        lower_hmm_task_to_matmul_plans,
    )

    _single_hxlist, single_task = build_single_site_hmm_scaffold_from_shapes(
        (2, 3, 4),
        (3, 5, 6, 7),
        (8, 7, 9),
        (4, 6, 9),
        dtype=np.dtype("float64"),
    )
    _two_hxlist, two_task = build_two_site_hmm_scaffold_from_shapes(
        (2, 3, 4),
        (3, 5, 6, 7),
        (7, 8, 9, 10),
        (11, 10, 12),
        (4, 6, 9, 12),
        dtype=np.dtype("float64"),
    )

    single_plan_batches = lower_hmm_task_to_matmul_plans(single_task)
    two_plan_batches = lower_hmm_task_to_matmul_plans(two_task)

    assert len(single_plan_batches) == 1
    assert [len(stage_plans) for stage_plans in single_plan_batches[0]] == [1, 1, 1]
    single_plans = [stage_plans[0] for stage_plans in single_plan_batches[0]]
    assert all(isinstance(plan, MatmulPlan) for plan in single_plans)
    assert [plan.kind for plan in single_plans] == ["gemm", "gemm", "gemm"]
    assert [plan.output_shape for plan in single_plans] == [(24, 56), (15, 32), (2, 40)]
    assert [(plan.descs[0].m, plan.descs[0].n, plan.descs[0].k) for plan in single_plans] == [
        (24, 56, 9),
        (15, 32, 42),
        (2, 40, 12),
    ]
    assert [plan.estimated_flops for plan in single_plans] == [
        2 * 24 * 56 * 9,
        2 * 15 * 32 * 42,
        2 * 2 * 40 * 12,
    ]
    assert all(plan.reason.startswith("hmm_task_stage") for plan in single_plans)
    assert all(plan.descs[0].dtype_compute == np.dtype("float64") for plan in single_plans)

    assert len(two_plan_batches) == 1
    assert [len(stage_plans) for stage_plans in two_plan_batches[0]] == [1, 1, 1, 1]
    two_plans = [stage_plans[0] for stage_plans in two_plan_batches[0]]
    assert [plan.kind for plan in two_plans] == ["gemm", "gemm", "gemm", "gemm"]
    assert [plan.output_shape for plan in two_plans] == [(216, 110), (56, 264), (15, 352), (2, 440)]
    assert [(plan.descs[0].m, plan.descs[0].n, plan.descs[0].k) for plan in two_plans] == [
        (216, 110, 12),
        (56, 264, 90),
        (15, 352, 42),
        (2, 440, 12),
    ]
    assert sum(plan.estimated_flops for plan in two_plans) == int(two_task.total_cost)


def test_batched_rhs_hmm_scaffold_lowers_rhs_stages_to_batched_matmul_plan_ir():
    from renormalizer.backend.gemm import (
        build_batched_single_site_hmm_scaffold_from_shapes,
        build_batched_two_site_hmm_scaffold_from_shapes,
        lower_hmm_task_to_matmul_plans,
    )

    _single_hxlist, single_task = build_batched_single_site_hmm_scaffold_from_shapes(
        (2, 3, 4),
        (3, 5, 6, 7),
        (8, 7, 9),
        (4, 6, 9),
        nrhs=3,
        dtype=np.dtype("float64"),
    )
    _two_hxlist, two_task = build_batched_two_site_hmm_scaffold_from_shapes(
        (2, 3, 4),
        (3, 5, 6, 7),
        (7, 8, 9, 10),
        (11, 10, 12),
        (4, 6, 9, 12),
        nrhs=2,
        dtype=np.dtype("float64"),
    )

    single_plans = [stage_plans[0] for stage_plans in lower_hmm_task_to_matmul_plans(single_task)[0]]
    assert single_task.equation == "abc,bdef,lfk,cekr->adlr"
    assert single_task.output_shape == (2, 5, 8, 3)
    assert [plan.kind for plan in single_plans] == ["batched_gemm", "batched_gemm", "batched_gemm"]
    assert [plan.output_shape for plan in single_plans] == [(3, 24, 56), (3, 15, 32), (3, 2, 40)]
    assert [plan.descs[0].batch_shape for plan in single_plans] == [(3,), (3,), (3,)]
    assert [(plan.descs[0].m, plan.descs[0].n, plan.descs[0].k) for plan in single_plans] == [
        (24, 56, 9),
        (15, 32, 42),
        (2, 40, 12),
    ]
    assert sum(plan.estimated_flops for plan in single_plans) == int(single_task.total_cost)

    two_plans = [stage_plans[0] for stage_plans in lower_hmm_task_to_matmul_plans(two_task)[0]]
    assert two_task.equation == "abc,bdef,fghj,ljk,cehkr->adglr"
    assert two_task.output_shape == (2, 5, 8, 11, 2)
    assert [plan.kind for plan in two_plans] == ["batched_gemm", "batched_gemm", "batched_gemm", "batched_gemm"]
    assert [plan.output_shape for plan in two_plans] == [(2, 216, 110), (2, 56, 264), (2, 15, 352), (2, 2, 440)]
    assert [plan.descs[0].batch_shape for plan in two_plans] == [(2,), (2,), (2,), (2,)]
    assert [(plan.descs[0].m, plan.descs[0].n, plan.descs[0].k) for plan in two_plans] == [
        (216, 110, 12),
        (56, 264, 90),
        (15, 352, 42),
        (2, 440, 12),
    ]
    assert sum(plan.estimated_flops for plan in two_plans) == int(two_task.total_cost)


def test_execute_single_site_hmm_action_matches_einsum_and_uses_backend_gemm():
    from renormalizer.backend.gemm import execute_single_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CountingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.matmul_calls = 0
            self.batched_matmul_calls = 0
            self.execute_matmul_plan_calls = []

        def matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.matmul_calls += 1
            return super().matmul(*args, **kwargs)

        def batched_matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.batched_matmul_calls += 1
            return super().batched_matmul(*args, **kwargs)

        def execute_matmul_plan(self, plan, **kwargs):
            self.execute_matmul_plan_calls.append((plan, kwargs))
            return super().execute_matmul_plan(plan, **kwargs)

    rng = np.random.default_rng(20260703)
    backend = CountingBackend()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    center = rng.normal(size=(4, 6, 9))

    result = execute_single_site_hmm_action(
        backend,
        ltensor,
        mpo,
        rtensor,
        center,
    )

    expected = np.einsum("abc,bdef,lfk,cek->adl", ltensor, mpo, rtensor, center)
    assert result.shape == (2, 5, 8)
    assert np.allclose(result, expected)
    assert [call[0].kind for call in backend.execute_matmul_plan_calls] == ["gemm"] * 3
    assert [call[0].reason for call in backend.execute_matmul_plan_calls] == [
        "single_site_hmm_stage_0",
        "single_site_hmm_stage_1",
        "single_site_hmm_stage_2",
    ]
    assert backend.matmul_calls == 3
    assert backend.batched_matmul_calls == 0


def test_execute_single_site_hmm_action_records_aggregate_execution_profile(tmp_path):
    from renormalizer.backend.gemm import execute_single_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = np.random.default_rng(20260704)
    backend = NumpyBackend()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    center = rng.normal(size=(4, 6, 9))
    stream = object()
    workspace = backend.allocate_workspace(128)
    event_path = tmp_path / "events.jsonl"

    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = execute_single_site_hmm_action(
            backend,
            ltensor,
            mpo,
            rtensor,
            center,
            stream=stream,
            workspace=workspace,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    expected = np.einsum("abc,bdef,lfk,cek->adl", ltensor, mpo, rtensor, center)
    assert np.allclose(result, expected)
    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["backend"] == "numpy"
    assert profile["lowering"] == "hmm_task"
    assert profile["equation"] == "abc,bdef,lfk,cek->adl"
    assert profile["center_kind"] == "onedot"
    assert profile["output_shape"] == (2, 5, 8)
    assert profile["num_gemm"] == 3
    assert profile["num_batched_gemm"] == 0
    assert profile["num_grouped_tasks"] == 0
    assert profile["num_gemv_desc"] == 0
    assert profile["num_hx_blocks"] == 1
    assert profile["hmm_num_batches"] == 1
    assert [item["phase"] for item in profile["hmm_execution_trace"]] == [
        "inter_gemv",
        "gemm_stage",
        "gemm_stage",
        "gemm_stage",
        "reduce_gemv",
    ]
    assert profile["stream_provided"] is True
    assert profile["workspace_provided"] is True
    assert profile["workspace_nbytes"] == 128
    assert profile["fallback_reason"] is None
    assert profile["wall_s"] >= 0.0

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_task"
    )
    assert event["plan_hash"] == profile["plan_hash"]
    assert [step["plan_hash"] for step in event["hmm_steps"]] == [
        step["plan_hash"] for step in profile["hmm_steps"]
    ]


def test_execute_batched_single_site_hmm_action_matches_einsum_and_uses_batched_gemm():
    from renormalizer.backend.gemm import execute_single_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CountingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.matmul_calls = 0
            self.batched_matmul_calls = 0
            self.execute_matmul_plan_calls = []

        def matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.matmul_calls += 1
            return super().matmul(*args, **kwargs)

        def batched_matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.batched_matmul_calls += 1
            return super().batched_matmul(*args, **kwargs)

        def execute_matmul_plan(self, plan, **kwargs):
            self.execute_matmul_plan_calls.append((plan, kwargs))
            return super().execute_matmul_plan(plan, **kwargs)

    rng = np.random.default_rng(20260703)
    backend = CountingBackend()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    center = rng.normal(size=(4, 6, 9, 3))

    result = execute_single_site_hmm_action(
        backend,
        ltensor,
        mpo,
        rtensor,
        center,
    )

    expected = np.einsum("abc,bdef,lfk,cekr->adlr", ltensor, mpo, rtensor, center)
    assert result.shape == (2, 5, 8, 3)
    assert np.allclose(result, expected)
    assert [call[0].kind for call in backend.execute_matmul_plan_calls] == ["batched_gemm"] * 3
    assert [call[0].descs[0].batch_shape for call in backend.execute_matmul_plan_calls] == [(3,)] * 3
    assert [call[0].reason for call in backend.execute_matmul_plan_calls] == [
        "single_site_hmm_batched_rhs_stage_0",
        "single_site_hmm_batched_rhs_stage_1",
        "single_site_hmm_batched_rhs_stage_2",
    ]
    assert backend.matmul_calls == 0
    assert backend.batched_matmul_calls == 3


def test_execute_batched_single_site_hmm_action_profiles_rhs_batching_and_primitives():
    from renormalizer.backend.gemm import execute_single_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend

    rng = np.random.default_rng(20260704)
    backend = NumpyBackend()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    center = rng.normal(size=(4, 6, 9, 3))

    result = execute_single_site_hmm_action(
        backend,
        ltensor,
        mpo,
        rtensor,
        center,
    )

    expected = np.einsum("abc,bdef,lfk,cekr->adlr", ltensor, mpo, rtensor, center)
    assert np.allclose(result, expected)
    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "hmm_task"
    assert profile["equation"] == "abc,bdef,lfk,cekr->adlr"
    assert profile["rhs_batch_mode"] == "r"
    assert profile["num_rhs"] == 3
    assert profile["num_rhs_loop_calls"] == 0
    assert profile["num_gemm"] == 0
    assert profile["num_batched_gemm"] == 3
    assert profile["execution_primitives"] == ["batched_matmul"]
    assert profile["execution_policies"] == ["backend_batched_matmul"]
    assert profile["fallback_reasons"] == []
    assert profile["hmm_execution_trace"][1]["execution_primitives"] == ["batched_matmul"]
    assert profile["hmm_execution_trace"][1]["execution_policies"] == ["backend_batched_matmul"]


def test_execute_two_site_hmm_action_matches_einsum_and_uses_backend_gemm():
    from renormalizer.backend.gemm import execute_two_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CountingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.matmul_calls = 0
            self.batched_matmul_calls = 0
            self.execute_matmul_plan_calls = []

        def matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.matmul_calls += 1
            return super().matmul(*args, **kwargs)

        def batched_matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.batched_matmul_calls += 1
            return super().batched_matmul(*args, **kwargs)

        def execute_matmul_plan(self, plan, **kwargs):
            self.execute_matmul_plan_calls.append((plan, kwargs))
            return super().execute_matmul_plan(plan, **kwargs)

    rng = np.random.default_rng(20260703)
    backend = CountingBackend()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo0 = rng.normal(size=(3, 5, 6, 7))
    mpo1 = rng.normal(size=(7, 8, 9, 10))
    rtensor = rng.normal(size=(11, 10, 12))
    center = rng.normal(size=(4, 6, 9, 12))

    result = execute_two_site_hmm_action(
        backend,
        ltensor,
        mpo0,
        mpo1,
        rtensor,
        center,
    )

    expected = np.einsum("abc,bdef,fghj,ljk,cehk->adgl", ltensor, mpo0, mpo1, rtensor, center)
    assert result.shape == (2, 5, 8, 11)
    assert np.allclose(result, expected)
    assert [call[0].kind for call in backend.execute_matmul_plan_calls] == ["gemm"] * 4
    assert [call[0].reason for call in backend.execute_matmul_plan_calls] == [
        "two_site_hmm_stage_0",
        "two_site_hmm_stage_1",
        "two_site_hmm_stage_2",
        "two_site_hmm_stage_3",
    ]
    assert backend.matmul_calls == 4
    assert backend.batched_matmul_calls == 0


def test_execute_batched_two_site_hmm_action_matches_einsum_and_uses_batched_gemm():
    from renormalizer.backend.gemm import execute_two_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CountingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.matmul_calls = 0
            self.batched_matmul_calls = 0
            self.execute_matmul_plan_calls = []

        def matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.matmul_calls += 1
            return super().matmul(*args, **kwargs)

        def batched_matmul(self, *args, **kwargs):
            if not (len(args) == 1 and self._is_matmul_desc(args[0])):
                self.batched_matmul_calls += 1
            return super().batched_matmul(*args, **kwargs)

        def execute_matmul_plan(self, plan, **kwargs):
            self.execute_matmul_plan_calls.append((plan, kwargs))
            return super().execute_matmul_plan(plan, **kwargs)

    rng = np.random.default_rng(20260703)
    backend = CountingBackend()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo0 = rng.normal(size=(3, 5, 6, 7))
    mpo1 = rng.normal(size=(7, 8, 9, 10))
    rtensor = rng.normal(size=(11, 10, 12))
    center = rng.normal(size=(4, 6, 9, 12, 2))

    result = execute_two_site_hmm_action(
        backend,
        ltensor,
        mpo0,
        mpo1,
        rtensor,
        center,
    )

    expected = np.einsum("abc,bdef,fghj,ljk,cehkr->adglr", ltensor, mpo0, mpo1, rtensor, center)
    assert result.shape == (2, 5, 8, 11, 2)
    assert np.allclose(result, expected)
    assert [call[0].kind for call in backend.execute_matmul_plan_calls] == ["batched_gemm"] * 4
    assert [call[0].descs[0].batch_shape for call in backend.execute_matmul_plan_calls] == [(2,)] * 4
    assert [call[0].reason for call in backend.execute_matmul_plan_calls] == [
        "two_site_hmm_batched_rhs_stage_0",
        "two_site_hmm_batched_rhs_stage_1",
        "two_site_hmm_batched_rhs_stage_2",
        "two_site_hmm_batched_rhs_stage_3",
    ]
    assert backend.matmul_calls == 0
    assert backend.batched_matmul_calls == 4


def test_hmm_task_lowers_scaffold_to_contraction_plan_ir():
    from renormalizer.backend import ContractionPlan
    from renormalizer.backend.gemm import (
        build_single_site_hmm_scaffold_from_shapes,
        build_two_site_hmm_scaffold_from_shapes,
        lower_hmm_task_to_contraction_plan,
    )

    _single_hxlist, single_task = build_single_site_hmm_scaffold_from_shapes(
        (2, 3, 4),
        (3, 5, 6, 7),
        (8, 7, 9),
        (4, 6, 9),
        dtype=np.dtype("float64"),
    )
    _two_hxlist, two_task = build_two_site_hmm_scaffold_from_shapes(
        (2, 3, 4),
        (3, 5, 6, 7),
        (7, 8, 9, 10),
        (11, 10, 12),
        (4, 6, 9, 12),
        dtype=np.dtype("float64"),
    )

    single_plan = lower_hmm_task_to_contraction_plan(single_task)
    two_plan = lower_hmm_task_to_contraction_plan(two_task)

    assert isinstance(single_plan, ContractionPlan)
    assert single_plan.output_modes == ("a", "d", "l")
    assert [operand.modes for operand in single_plan.input_specs] == [
        ("a", "b", "c"),
        ("b", "d", "e", "f"),
        ("l", "f", "k"),
        ("c", "e", "k"),
    ]
    assert [operand.array.shape for operand in single_plan.input_specs] == [
        (2, 3, 4),
        (3, 5, 6, 7),
        (8, 7, 9),
        (4, 6, 9),
    ]
    assert len(single_plan.steps) == 3
    assert [step.kind for step in single_plan.steps] == ["gemm", "gemm", "gemm"]
    assert all(step.plan.kind == step.kind for step in single_plan.steps)
    assert [step.plan.output_shape for step in single_plan.steps] == [(24, 56), (15, 32), (2, 40)]
    assert single_plan.steps[-1].output_modes == ("a", "d", "l")
    assert single_plan.estimated_flops == int(single_task.total_cost)
    assert single_plan.estimated_read_bytes == sum(
        desc.estimated_read_bytes
        for step in single_plan.steps
        for desc in step.plan.descs
    )
    assert single_plan.estimated_write_bytes == sum(
        desc.estimated_write_bytes
        for step in single_plan.steps
        for desc in step.plan.descs
    )
    assert single_plan.plan_hash

    assert isinstance(two_plan, ContractionPlan)
    assert two_plan.output_modes == ("a", "d", "g", "l")
    assert [operand.modes for operand in two_plan.input_specs] == [
        ("a", "b", "c"),
        ("b", "d", "e", "f"),
        ("f", "g", "h", "j"),
        ("l", "j", "k"),
        ("c", "e", "h", "k"),
    ]
    assert len(two_plan.steps) == 4
    assert [step.kind for step in two_plan.steps] == ["gemm", "gemm", "gemm", "gemm"]
    assert [step.plan.output_shape for step in two_plan.steps] == [(216, 110), (56, 264), (15, 352), (2, 440)]
    assert two_plan.steps[-1].output_modes == ("a", "d", "g", "l")
    assert two_plan.estimated_flops == int(two_task.total_cost)


def test_grouped_gemm_applies_flags_alpha_beta_and_updates_c():
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(6, dtype=np.float64).reshape(3, 2)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    c = np.ones((2, 4), dtype=np.float64)
    task = GemmTask(a, b, C=c, trans_a=True, alpha=2.0, beta=3.0, tag="accumulate")

    results = backend.grouped_gemm([task], pack_threshold=2)

    expected = 2.0 * (a.T @ b) + 3.0 * np.ones((2, 4), dtype=np.float64)
    assert results == [c]
    assert np.allclose(c, expected)


def test_grouped_gemm_forbid_policy_rejects_bucketed_fallback():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    tasks = [
        GemmTask(
            np.arange(6, dtype=np.float64).reshape(2, 3),
            np.arange(12, dtype=np.float64).reshape(3, 4),
        )
    ]

    with pytest.raises(BackendFeatureError, match="backend-owned grouped_gemm unavailable"):
        backend.grouped_gemm(tasks)


def test_grouped_gemm_warn_policy_emits_warning_and_returns_result():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="warn"))
    task = GemmTask(
        np.arange(6, dtype=np.float64).reshape(2, 3),
        np.arange(12, dtype=np.float64).reshape(3, 4),
    )

    with pytest.warns(RuntimeWarning, match="backend-owned grouped_gemm unavailable"):
        results = backend.grouped_gemm([task])

    assert len(results) == 1
    assert np.allclose(results[0], task.A @ task.B)


def test_grouped_gemm_call_fallback_policy_overrides_backend_fallback_policy(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    task = GemmTask(
        np.arange(6, dtype=np.float64).reshape(2, 3),
        np.arange(12, dtype=np.float64).reshape(3, 4),
    )
    record_backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))

    with pytest.raises(BackendFeatureError, match="backend-owned grouped_gemm unavailable"):
        record_backend.grouped_gemm([task], fallback_policy="forbid")

    strict_backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = strict_backend.grouped_gemm([task], fallback_policy="record")
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], task.A @ task.B)
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in events if payload["event"] == "grouped_gemm_execute")
    assert event["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert event["fallback_policy"] == "record"


def test_grouped_gemm_execution_policy_can_force_loop_bucket():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.eye(128, dtype=np.float64), tag="large-0"),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.eye(128, dtype=np.float64), tag="large-1"),
    ]

    results = backend.grouped_gemm(tasks, pack_threshold=2, policy="bucketed_loop_matmul")

    assert np.allclose(results[0], tasks[0].A @ tasks[0].B)
    assert np.allclose(results[1], tasks[1].A @ tasks[1].B)
    profile = backend.last_execution_profile()
    assert profile["policy"] == "bucketed_loop_matmul"
    assert profile["execution_policies"] == ["bucketed_loop_matmul"]
    assert profile["num_batched_gemm"] == 0
    assert profile["num_gemm"] == 2
    assert profile["batched_kernel_calls"] == 0
    assert profile["loop_kernel_calls"] == 2
    assert profile["bucket_kernel_primitives"] == ["matmul"]
    assert profile["bucket_kernel_calls_by_primitive"] == {"matmul": 2}
    assert profile["bucket_execution_profiles"][0]["execution"] == "loop"


def test_native_grouped_gemm_batches_same_shape_bucket_despite_copy_heuristic(tmp_path):
    from renormalizer.backend.gemm import GemmTask, should_batch
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NativeGroupedNumpyBackend(NumpyBackend):
        supports_grouped_gemm = True

    backend = NativeGroupedNumpyBackend()
    tasks = [
        GemmTask(np.ones((64, 64), dtype=np.float64), np.ones((64, 64), dtype=np.float64))
        for _ in range(8)
    ]
    assert should_batch(tasks, xp=np, pack_threshold=4) is False

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.grouped_gemm(tasks, pack_threshold=4)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["fallback_reason"] is None
    assert event["num_batched_gemm"] == 1
    assert event["num_gemm"] == 0


def test_grouped_gemm_fallback_respects_missing_batched_matmul_capability(tmp_path):
    from renormalizer.backend.gemm import GemmTask, should_batch
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoBatchedNumpyBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedNumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64))
        for _ in range(4)
    ]
    assert should_batch(tasks, xp=np, pack_threshold=4, flop_copy_ratio=10) is True

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.grouped_gemm(tasks, pack_threshold=4)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert len(result) == len(tasks)
    assert all(np.allclose(item, np.full((128, 128), 128.0)) for item in result)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["fallback_from"] == "grouped_gemm"
    assert event["fallback_to"] == "bucketed_loop_matmul"
    assert event["num_batched_gemm"] == 0
    assert event["batched_bucket_count"] == 0
    assert event["batched_task_count"] == 0
    assert event["num_gemm"] == len(tasks)
    assert event["loop_bucket_count"] == 1
    assert event["loop_task_count"] == len(tasks)
    assert event["copy_bytes"] == 0


def test_grouped_gemm_bucketed_fallback_uses_backend_batched_matmul_when_available(tmp_path):
    from renormalizer.backend.gemm import GemmTask, should_batch
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class CountingBatchedNumpyBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.batched_matmul_calls = 0

        def batched_matmul(self, *args, **kwargs):
            self.batched_matmul_calls += 1
            return super().batched_matmul(*args, **kwargs)

    backend = CountingBatchedNumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64)),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64)),
        GemmTask(np.full((128, 128), 3.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64)),
        GemmTask(np.full((128, 128), 4.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64)),
    ]
    assert should_batch(tasks, xp=np, pack_threshold=4, flop_copy_ratio=10) is True

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = backend.grouped_gemm(tasks, pack_threshold=4)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert backend.batched_matmul_calls == 1
    assert [item.shape for item in results] == [(128, 128)] * len(tasks)
    assert np.allclose(results[0], np.full((128, 128), 128.0))
    assert np.allclose(results[1], np.full((128, 128), 256.0))
    assert np.allclose(results[2], np.full((128, 128), 384.0))
    assert np.allclose(results[3], np.full((128, 128), 512.0))

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["fallback_from"] == "grouped_gemm"
    assert event["fallback_to"] == "bucketed_batched_matmul"
    assert event["num_batched_gemm"] == 1
    assert event["num_gemm"] == 0


def test_grouped_gemm_profile_records_one_input_shape_pair_per_task(tmp_path):
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((2, 3), dtype=np.float64), np.ones((3, 4), dtype=np.float64)),
        GemmTask(np.ones((5, 6), dtype=np.float64), np.ones((6, 7), dtype=np.float64)),
    ]
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.grouped_gemm(tasks, pack_threshold=4)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")

    assert event["input_shapes"] == [
        [[2, 3], [3, 4]],
        [[5, 6], [6, 7]],
    ]
    assert event["input_dtypes"] == [
        ["float64", "float64"],
        ["float64", "float64"],
    ]
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_subclass"] == "backend_execute"
    assert event["compute_role"] == "kernel"
    assert event["task_operands"][0][0]["name"] == "task0.A"
    assert event["task_operands"][0][0]["modes"] == ["m", "k"]
    assert event["task_operands"][0][0]["shape"] == [2, 3]
    assert event["task_operands"][0][0]["itemsize"] == tasks[0].A.itemsize
    assert event["task_operands"][0][0]["is_host"] is True
    assert event["task_operands"][0][0]["is_distributed"] is False
    assert event["task_operands"][0][1]["name"] == "task0.B"
    assert event["task_operands"][0][1]["modes"] == ["k", "n"]
    assert event["task_operands"][0][1]["shape"] == [3, 4]
    assert event["task_specs"][0] == {
        "index": 0,
        "m": 2,
        "n": 4,
        "k": 3,
        "trans_a": False,
        "trans_b": False,
        "conj_a": False,
        "conj_b": False,
        "alpha": 1.0,
        "beta": 0.0,
        "tag": None,
    }


def test_torch_grouped_gemm_is_not_implemented_in_np2_cupy_phase():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available
    from renormalizer.backend.gemm import GemmTask

    if not is_backend_available("torch"):
        pytest.skip("torch unavailable")

    backend = create_backend("torch", config=BackendConfig(device="cpu", fallback_policy="record"))
    a_np = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    b_np = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    tasks = [
        GemmTask(backend.to_backend(a_np[index]), backend.to_backend(b_np[index]), tag=index)
        for index in range(2)
    ]

    assert backend.supports_grouped_gemm is False
    assert backend.capabilities.grouped_gemm is False
    with pytest.raises(NotImplementedError, match="Torch grouped_gemm is not implemented"):
        backend.grouped_gemm(tasks, pack_threshold=2)


def test_torch_contraction_execute_supports_full_axis_permutation_when_available():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available("torch"):
        pytest.skip("torch unavailable")

    backend = create_backend("torch", config=BackendConfig(device="cpu"))
    left_np = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    right_np = np.arange(5 * 6, dtype=np.float64).reshape(5, 6)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    spec = backend.parse_einsum("abfh,hc->ahbfc", left, right)
    plan = backend.plan_contraction(spec)

    result = backend.execute(plan)

    assert np.allclose(backend.to_numpy(result), np.einsum("abfh,hc->ahbfc", left_np, right_np))


def test_torch_array_info_and_layout_report_contiguous_strides_when_available():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available("torch"):
        pytest.skip("torch unavailable")

    backend = create_backend("torch", config=BackendConfig(device="cpu", precision=64))
    tensor = backend.to_backend(np.arange(6, dtype=np.float64).reshape(2, 3))

    info = backend.array_info(tensor)
    layout = backend.layout(tensor)

    assert info.shape == (2, 3)
    assert info.strides == (24, 8)
    assert info.order == "C"
    assert info.contiguous is True
    assert layout.logical_modes == (0, 1)
    assert layout.strides == (24, 8)
    assert layout.order == "C"
    assert layout.contiguous_groups == ((0, 1),)


def test_cupy_grouped_gemm_records_backend_owned_bucketed_execution():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available
    from renormalizer.backend.gemm import GemmTask

    if not is_backend_available("cupy"):
        pytest.skip("cupy unavailable")

    try:
        backend = create_backend("cupy", config=BackendConfig(device="gpu", fallback_policy="record"))
    except (ImportError, ValueError, RuntimeError) as exc:
        pytest.skip("cupy backend unavailable: {0}".format(exc))
    a_np = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    b_np = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    tasks = [
        GemmTask(backend.to_backend(a_np[index]), backend.to_backend(b_np[index]), tag=index)
        for index in range(2)
    ]

    results = backend.grouped_gemm(tasks, pack_threshold=2)

    assert backend.supports_grouped_gemm is True
    assert backend.capabilities.grouped_gemm is True
    assert [tuple(result.shape) for result in results] == [(4, 4), (4, 4)]
    assert np.allclose(backend.to_numpy(results[0]), a_np[0] @ b_np[0])
    assert np.allclose(backend.to_numpy(results[1]), a_np[1] @ b_np[1])
    profile = backend.last_execution_profile()
    assert profile["supports_grouped_gemm"] is True
    assert profile["requires_grouped_gemm_fallback"] is False
    assert profile["fallback_reason"] is None
    assert profile["grouped_gemm_implementation"] == "backend_bucketed_batched_matmul"
    assert profile["batched_matmul_provider"] == "cupy.matmul"
    assert profile["bucket_execution_profiles"][0]["batched_matmul_provider"] == "cupy.matmul"

    strict_backend = create_backend("cupy", config=BackendConfig(device="gpu", fallback_policy="forbid"))
    strict_tasks = [
        GemmTask(strict_backend.to_backend(a_np[index]), strict_backend.to_backend(b_np[index]), tag=index)
        for index in range(2)
    ]
    strict_results = strict_backend.grouped_gemm(strict_tasks, pack_threshold=2)
    assert np.allclose(strict_backend.to_numpy(strict_results[0]), a_np[0] @ b_np[0])
    assert strict_backend.last_execution_profile()["fallback_reason"] is None


@pytest.mark.parametrize(
    "backend_name,device,expected_provider",
    (
        ("cupy", "gpu", "cupy.matmul"),
    ),
)
def test_backend_owned_grouped_gemm_contraction_event_records_batched_provider(
    tmp_path,
    backend_name,
    device,
    expected_provider,
):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    if not is_backend_available(backend_name):
        pytest.skip("{0} unavailable".format(backend_name))

    try:
        backend = create_backend(
            backend_name,
            config=BackendConfig(device=device, fallback_policy="record"),
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        pytest.skip("{0} backend unavailable: {1}".format(backend_name, exc))

    a_np = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    b_np = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    tasks = [
        GemmTask(backend.to_backend(a_np[index]), backend.to_backend(b_np[index]), tag=index)
        for index in range(2)
    ]
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.grouped_gemm(tasks, pack_threshold=2)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["backend"] == backend_name
    assert event["grouped_gemm_implementation"] == "backend_bucketed_batched_matmul"
    assert event["batched_matmul_provider"] == expected_provider
    assert event["bucket_execution_profiles"][0]["batched_matmul_provider"] == expected_provider


def test_should_batch_uses_copy_to_flop_heuristic():
    from renormalizer.backend.gemm import GemmTask, should_batch

    tiny_tasks = [
        GemmTask(np.ones((1, 1), dtype=np.float64), np.ones((1, 1), dtype=np.float64))
        for _ in range(8)
    ]
    large_tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64))
        for _ in range(8)
    ]

    assert should_batch(tiny_tasks, xp=np, pack_threshold=4) is False
    assert should_batch(large_tasks, xp=np, pack_threshold=4) is True


def test_grouped_gemm_records_bucketed_fallback_profile(tmp_path):
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="batch-0"),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="batch-1"),
        GemmTask(np.ones((3, 4), dtype=np.float64), np.ones((4, 5), dtype=np.float64), tag="loop"),
    ]
    event_path = tmp_path / "events.jsonl"
    stream = object()
    workspace = backend.allocate_workspace(128)
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = backend.grouped_gemm(tasks, pack_threshold=2, stream=stream, workspace=workspace)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], tasks[0].A @ tasks[0].B)
    assert np.allclose(results[1], tasks[1].A @ tasks[1].B)
    assert np.allclose(results[2], tasks[2].A @ tasks[2].B)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    events = [payload for payload in payloads if payload["event"] == "contraction_execute"]

    assert len(events) == 1
    event = events[0]
    assert event["backend"] == "numpy"
    assert event["lowering"] == "grouped_gemm"
    assert event["num_grouped_tasks"] == 3
    assert event["num_shape_buckets"] == 2
    assert event["num_batched_gemm"] == 1
    assert event["num_gemm"] == 1
    assert event["num_blocks"] == 3
    assert event["operands"][0][0]["name"] == "task0.A"
    assert event["operands"][0][0]["modes"] == ["m", "k"]
    assert event["operands"][0][1]["name"] == "task0.B"
    assert event["task_operands"] == event["operands"]
    assert event["flops"] == 8388728
    assert event["read_bytes"] == 524544
    assert event["write_bytes"] == 262264
    assert event["copy_bytes"] == 524288
    assert event["peak_bytes"] == event["write_bytes"] + event["copy_bytes"] + event["workspace_bytes"]
    assert event["fallback_from"] == "grouped_gemm"
    assert event["grouped_gemm_implementation"] == "fallback_bucketed_mixed_matmul"
    assert event["requires_grouped_gemm_fallback"] is True
    assert event["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert event["fallback_policy"] == "record"
    assert event["bucket_task_counts"] == [1, 2]
    assert event["shape_buckets"] == [
        {"m": 3, "n": 5, "k": 4, "task_indices": [2], "task_count": 1},
        {"m": 128, "n": 128, "k": 128, "task_indices": [0, 1], "task_count": 2},
    ]
    assert event["group_sizes"] == [2, 1]
    assert event["gsta"] == [0, 2, 3]
    assert event["sorted_indices"] == [0, 1, 2]
    assert event["group_keys"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "lda": 128,
            "ldb": 128,
            "ldc": 128,
        },
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 3,
            "n": 5,
            "k": 4,
            "lda": 4,
            "ldb": 5,
            "ldc": 5,
        },
    ]
    assert event["group_key_buckets"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "lda": 128,
            "ldb": 128,
            "ldc": 128,
            "task_indices": [0, 1],
            "task_count": 2,
        },
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 3,
            "n": 5,
            "k": 4,
            "lda": 4,
            "ldb": 5,
            "ldc": 5,
            "task_indices": [2],
            "task_count": 1,
        },
    ]
    assert event["group_descriptors"] == [
        {
            "group_index": 0,
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "lda": 128,
            "ldb": 128,
            "ldc": 128,
            "task_indices": [0, 1],
            "task_count": 2,
            "execution": "batched",
            "flops": 8388608,
            "read_bytes": 524288,
            "write_bytes": 262144,
            "copy_bytes": 524288,
            "pack_strategy": "stack_per_call",
            "kernel_calls": 1,
            "batched_matmul_provider": "numpy.matmul",
        },
        {
            "group_index": 1,
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 3,
            "n": 5,
            "k": 4,
            "lda": 4,
            "ldb": 5,
            "ldc": 5,
            "task_indices": [2],
            "task_count": 1,
            "execution": "loop",
            "flops": 120,
            "read_bytes": 256,
            "write_bytes": 120,
            "copy_bytes": 0,
            "pack_strategy": "none",
            "kernel_calls": 1,
            "batched_matmul_provider": None,
        },
    ]
    assert event["batched_bucket_count"] == 1
    assert event["loop_bucket_count"] == 1
    assert event["stream_provided"] is True
    assert event["stream_type"] == "object"
    assert event["workspace_provided"] is True
    assert event["workspace_nbytes"] == 128
    assert event["workspace_device_kind"] == "cpu"
    assert event["workspace_device_index"] is None
    assert event["wall_s"] >= 0.0

    primitive_event = next(payload for payload in payloads if payload["event"] == "grouped_gemm_execute")
    assert primitive_event["device_kind"] == "cpu"
    assert primitive_event["device_index"] is None
    assert primitive_event["stream_provided"] is True
    assert primitive_event["stream_type"] == "object"
    assert primitive_event["workspace_provided"] is True
    assert primitive_event["workspace_nbytes"] == 128
    assert primitive_event["workspace_device_kind"] == "cpu"
    assert primitive_event["workspace_device_index"] is None
    assert primitive_event["group_key_buckets"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "lda": 128,
            "ldb": 128,
            "ldc": 128,
            "task_indices": [0, 1],
            "task_count": 2,
        },
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 3,
            "n": 5,
            "k": 4,
            "lda": 4,
            "ldb": 5,
            "ldc": 5,
            "task_indices": [2],
            "task_count": 1,
        },
    ]

    profile = backend.last_execution_profile()
    assert profile["device_kind"] == "cpu"
    assert profile["device_index"] is None
    assert profile["stream_provided"] is True
    assert profile["stream_type"] == "object"
    assert profile["workspace_provided"] is True
    assert profile["workspace_nbytes"] == 128
    assert profile["workspace_device_kind"] == "cpu"
    assert profile["workspace_device_index"] is None


def test_grouped_gemm_caches_execution_profile_without_profiling_log_level():
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="batch-0"),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="batch-1"),
        GemmTask(np.ones((3, 4), dtype=np.float64), np.ones((4, 5), dtype=np.float64), tag="loop"),
    ]
    old_level = package_logger.level
    try:
        init_log(DEBUG)
        profiling.flush_summaries()

        results = backend.grouped_gemm(tasks, pack_threshold=2)
    finally:
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], tasks[0].A @ tasks[0].B)
    assert np.allclose(results[1], tasks[1].A @ tasks[1].B)
    assert np.allclose(results[2], tasks[2].A @ tasks[2].B)

    profile = backend.last_execution_profile()
    assert profile["event"] == "grouped_gemm_execute"
    assert profile["backend"] == "numpy"
    assert profile["policy"] == "bucketed_mixed_matmul"
    assert profile["grouped_gemm_implementation"] == "fallback_bucketed_mixed_matmul"
    assert profile["requires_grouped_gemm_fallback"] is True
    assert profile["fallback_from"] == "grouped_gemm"
    assert profile["fallback_to"] == "bucketed_mixed_matmul"
    assert profile["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert profile["num_tasks"] == 3
    assert profile["num_groups"] == 2
    assert profile["group_sizes"] == [2, 1]
    assert profile["gsta"] == [0, 2, 3]
    assert profile["sorted_indices"] == [0, 1, 2]
    assert profile["group_key_buckets"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "lda": 128,
            "ldb": 128,
            "ldc": 128,
            "task_indices": [0, 1],
            "task_count": 2,
        },
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 3,
            "n": 5,
            "k": 4,
            "lda": 4,
            "ldb": 5,
            "ldc": 5,
            "task_indices": [2],
            "task_count": 1,
        },
    ]
    assert profile["num_batched_gemm"] == 1
    assert profile["num_gemm"] == 1
    assert profile["total_flops"] == 8388728
    assert profile["read_bytes"] == 524544
    assert profile["write_bytes"] == 262264
    assert profile["copy_bytes"] == 524288
    assert profile["pack_threshold"] == 2
    assert profile["pointer_setup_s"] >= 0.0
    assert profile["compute_s"] >= 0.0
    assert profile["wall_s"] >= profile["compute_s"]
    batched_bucket, loop_bucket = profile["bucket_execution_profiles"]
    assert batched_bucket["execution"] == "batched"
    assert batched_bucket["selection_reason"] == (
        "batched: task_count 2 >= pack_threshold 2 and flops 8388608 >= copy_threshold 5242880"
    )
    assert batched_bucket["fallback_reason"] is None
    assert loop_bucket["execution"] == "loop"
    assert loop_bucket["selection_reason"] == "loop: task_count 1 below pack_threshold 2"
    assert loop_bucket["fallback_reason"] == "task_count below pack_threshold"


def test_grouped_gemm_without_profiling_uses_lightweight_profile_cache(monkeypatch):
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64)),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64)),
    ]
    old_level = package_logger.level

    def group_descriptors_must_not_run(*args, **kwargs):
        raise AssertionError("group descriptors are full profiling metadata")

    monkeypatch.setattr(backend, "_grouped_gemm_group_descriptors", group_descriptors_must_not_run)
    try:
        init_log(DEBUG)
        profiling.flush_summaries()

        results = backend.grouped_gemm(tasks, pack_threshold=2)
    finally:
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], tasks[0].A @ tasks[0].B)
    profile = backend.last_execution_profile()
    assert profile["event"] == "grouped_gemm_execute"
    assert profile["num_tasks"] == 2
    assert profile["num_batched_gemm"] == 1
    assert profile["compute_profile"]["compute_class"] == "contraction_plan"


def test_grouped_gemm_rejects_workspace_on_wrong_device():
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec, Workspace
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    task = GemmTask(np.ones((2, 3), dtype=np.float64), np.ones((3, 4), dtype=np.float64))
    workspace = Workspace(device=DeviceSpec(kind="cuda"), nbytes=0, buffer=np.empty(0, dtype=np.uint8))

    with pytest.raises(BackendFeatureError, match="workspace device"):
        backend.grouped_gemm([task], workspace=workspace)


def test_dense_block_rejects_invalid_shape_metadata():
    from renormalizer.backend import BlockKey, DenseBlock

    key = BlockKey((0,), (1,))

    with pytest.raises(ValueError, match="DenseBlock shape dimensions must be non-negative"):
        DenseBlock(key, np.ones((2, 3)), ("i", "k"), (-2, 3))

    with pytest.raises(ValueError, match="DenseBlock modes must match shape rank"):
        DenseBlock(key, np.ones((2, 3)), ("i",), (2, 3))

    with pytest.raises(ValueError, match="DenseBlock shape must match array shape"):
        DenseBlock(key, np.ones((2, 3)), ("i", "k"), (2, 4))

    with pytest.raises(ValueError, match="DenseBlock offset must match shape rank"):
        DenseBlock(key, np.ones((2, 3)), ("i", "k"), (2, 3), offset=(0,))

    with pytest.raises(ValueError, match="DenseBlock offset entries must be non-negative"):
        DenseBlock(key, np.ones((2, 3)), ("i", "k"), (2, 3), offset=(-1, 0))


def test_block_tensor_rejects_inconsistent_block_metadata():
    from renormalizer.backend import BlockKey, BlockTensor, DenseBlock

    key = BlockKey((0,), (1,))
    block = DenseBlock(key, np.ones((2, 3)), ("i", "k"), (2, 3))

    with pytest.raises(ValueError, match="BlockTensor global_shape dimensions must be non-negative"):
        BlockTensor({key: block}, global_shape=(-2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy")

    with pytest.raises(ValueError, match="BlockTensor modes must match global_shape rank"):
        BlockTensor({key: block}, global_shape=(2, 3), modes=("i",), block_axis_meta=None, backend="numpy")

    with pytest.raises(ValueError, match="BlockTensor block dictionary key must match block.key"):
        BlockTensor(
            {BlockKey((2,), (3,)): block},
            global_shape=(2, 3),
            modes=("i", "k"),
            block_axis_meta=None,
            backend="numpy",
        )

    wrong_modes = DenseBlock(key, np.ones((2, 3)), ("row", "col"), (2, 3))
    with pytest.raises(ValueError, match="BlockTensor block modes must match tensor modes"):
        BlockTensor({key: wrong_modes}, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy")

    too_large = DenseBlock(key, np.ones((3, 3)), ("i", "k"), (3, 3))
    with pytest.raises(ValueError, match="BlockTensor block shape must fit within global_shape"):
        BlockTensor({key: too_large}, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy")

    out_of_bounds = DenseBlock(key, np.ones((2, 3)), ("i", "k"), (2, 3), offset=(1, 0))
    with pytest.raises(ValueError, match="BlockTensor block offset and shape must stay within global_shape"):
        BlockTensor({key: out_of_bounds}, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy")


def test_block_tensor_dense_conversion_requires_explicit_materialization():
    from renormalizer.backend import BackendFeatureError, BlockKey, BlockTensor, DenseBlock

    first = BlockKey((0,), (0,))
    second = BlockKey((1,), (0,))
    tensor = BlockTensor(
        {
            first: DenseBlock(first, np.full((2, 3), 2.0), ("i", "j"), (2, 3), offset=(0, 0)),
            second: DenseBlock(second, np.full((1, 3), 4.0), ("i", "j"), (1, 3), offset=(3, 0)),
        },
        global_shape=(5, 3),
        modes=("i", "j"),
        block_axis_meta=None,
        backend="numpy",
    )

    with pytest.raises(BackendFeatureError, match="explicit"):
        tensor.to_dense()

    dense = tensor.to_dense(explicit=True)

    expected = np.zeros((5, 3))
    expected[0:2, 0:3] = 2.0
    expected[3:4, 0:3] = 4.0
    np.testing.assert_allclose(dense, expected)


def test_block_tensor_dense_conversion_requires_offsets():
    from renormalizer.backend import BackendFeatureError, BlockKey, BlockTensor, DenseBlock

    key = BlockKey((0,), (0,))
    tensor = BlockTensor(
        {key: DenseBlock(key, np.ones((2, 3)), ("i", "j"), (2, 3))},
        global_shape=(2, 3),
        modes=("i", "j"),
        block_axis_meta=None,
        backend="numpy",
    )

    with pytest.raises(BackendFeatureError, match="offset"):
        tensor.to_dense(explicit=True)


def test_block_contraction_spec_rejects_invalid_metadata():
    from renormalizer.backend import BlockContractionSpec, BlockKey, BlockTensor, DenseBlock

    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    left = BlockTensor(
        {left_key: DenseBlock(left_key, np.ones((2, 3)), ("i", "k"), (2, 3))},
        global_shape=(2, 3),
        modes=("i", "k"),
        block_axis_meta=None,
        backend="numpy",
    )
    right = BlockTensor(
        {right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4))},
        global_shape=(3, 4),
        modes=("k", "j"),
        block_axis_meta=None,
        backend="numpy",
    )
    other_backend = BlockTensor(
        {right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4))},
        global_shape=(3, 4),
        modes=("k", "j"),
        block_axis_meta=None,
        backend="torch",
    )
    inconsistent_shared_mode = BlockTensor(
        {right_key: DenseBlock(right_key, np.ones((4, 4)), ("k", "j"), (4, 4))},
        global_shape=(4, 4),
        modes=("k", "j"),
        block_axis_meta=None,
        backend="numpy",
    )

    with pytest.raises(ValueError, match="BlockContractionSpec qn_rule must be callable"):
        BlockContractionSpec(left, right, output_modes=("i", "j"), qn_rule=None)

    with pytest.raises(ValueError, match="BlockContractionSpec output_modes must be unique"):
        BlockContractionSpec(left, right, output_modes=("i", "i"), qn_rule=lambda left, right: None)

    with pytest.raises(ValueError, match="BlockContractionSpec output_modes are not present in operands"):
        BlockContractionSpec(left, right, output_modes=("i", "missing"), qn_rule=lambda left, right: None)

    with pytest.raises(ValueError, match="BlockContractionSpec left and right backends must match"):
        BlockContractionSpec(left, other_backend, output_modes=("i", "j"), qn_rule=lambda left, right: None)

    with pytest.raises(ValueError, match="BlockContractionSpec shared mode 'k' has inconsistent global sizes"):
        BlockContractionSpec(
            left,
            inconsistent_shared_mode,
            output_modes=("i", "j"),
            qn_rule=lambda left, right: None,
        )


def test_lower_block_contraction_builds_deterministic_grouped_gemm_plan():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    out_key = BlockKey((0,), (1,))
    skipped_key = BlockKey((3,), (4,))
    left_blocks = {
        BlockKey((0,), (2,), extra=("second",)): DenseBlock(
            BlockKey((0,), (2,), extra=("second",)),
            np.full((2, 3), 2.0),
            ("i", "k"),
            (2, 3),
        ),
        BlockKey((0,), (0,), extra=("first",)): DenseBlock(
            BlockKey((0,), (0,), extra=("first",)),
            np.ones((2, 3)),
            ("i", "k"),
            (2, 3),
        ),
    }
    right_blocks = {
        BlockKey((2,), (1,)): DenseBlock(
            BlockKey((2,), (1,)),
            np.ones((3, 4)),
            ("k", "j"),
            (3, 4),
        ),
        BlockKey((3,), (4,)): DenseBlock(
            BlockKey((3,), (4,)),
            np.ones((0, 4)),
            ("k", "j"),
            (0, 4),
        ),
        BlockKey((0,), (1,)): DenseBlock(
            BlockKey((0,), (1,)),
            np.full((3, 4), 3.0),
            ("k", "j"),
            (3, 4),
        ),
    }
    left = BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy")
    right = BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy")

    def qn_rule(left_key, right_key):
        if left_key.qn_right != right_key.qn_left:
            return None
        if right_key == skipped_key:
            raise AssertionError("zero block should not be materialized")
        return BlockKey(left_key.qn_left, right_key.qn_right)

    plan = backend.lower_block_contraction(
        BlockContractionSpec(left, right, output_modes=("i", "j"), qn_rule=qn_rule)
    )

    assert plan.output_blocks == (out_key, out_key)
    assert [(desc.m, desc.n, desc.k) for desc in plan.tasks] == [(2, 4, 3), (2, 4, 3)]
    assert plan.bucketed_by_shape == {(2, 4, 3): (0, 1)}
    assert plan.scatter_add_required is True
    assert plan.estimated_flops == 96
    assert plan.estimated_read_bytes == 288
    assert plan.estimated_write_bytes == 128
    assert plan.output_modes == ("i", "j")


def test_block_grouped_gemm_profile_records_zero_sized_block_skips(tmp_path):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    left_zero = BlockKey((9,), (9,), extra=("left-zero",))
    right_zero = BlockKey((8,), (8,), extra=("right-zero",))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        left_zero: DenseBlock(left_zero, np.ones((0, 3)), ("i", "k"), (0, 3)),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(BlockKey((0,), (1,)), np.ones((3, 4)), ("k", "j"), (3, 4)),
        right_zero: DenseBlock(right_zero, np.ones((3, 0)), ("k", "j"), (3, 0)),
    }

    def qn_rule(left_key, right_key):
        if left_key in {left_zero, right_zero} or right_key in {left_zero, right_zero}:
            raise AssertionError("zero-sized block keys must not reach qn_rule")
        if left_key.qn_right != right_key.qn_left:
            return None
        return BlockKey(left_key.qn_left, right_key.qn_right)

    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=qn_rule,
    )
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.lower_block_contraction(spec)
        result = backend.execute_grouped_gemm_plan(plan)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert tuple(result.blocks) == (BlockKey((0,), (1,)),)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    events = [
        payload
        for payload in payloads
        if payload.get("event") in {"contraction_plan", "contraction_execute"}
        and payload.get("lowering") == "block_grouped_gemm"
    ]
    assert [event["num_zero_sized_left_blocks_skipped"] for event in events] == [1, 1]
    assert [event["num_zero_sized_right_blocks_skipped"] for event in events] == [1, 1]
    assert [event["zero_sized_left_block_keys"] for event in events] == [
        [{"extra": ["left-zero"], "qn_left": [9], "qn_right": [9]}],
        [{"extra": ["left-zero"], "qn_left": [9], "qn_right": [9]}],
    ]
    assert [event["zero_sized_right_block_keys"] for event in events] == [
        [{"extra": ["right-zero"], "qn_left": [8], "qn_right": [8]}],
        [{"extra": ["right-zero"], "qn_left": [8], "qn_right": [8]}],
    ]


def test_lower_block_contraction_delegates_pair_lowering_to_backend_method():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CountingLoweringBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.lower_pair_calls = []

        def lower_pair_contraction_to_matmul(self, spec, *, record_profile=True):
            self.lower_pair_calls.append((spec, record_profile))
            return super().lower_pair_contraction_to_matmul(
                spec,
                record_profile=record_profile,
            )

    backend = CountingLoweringBackend()
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    left_blocks = {
        left_key: DenseBlock(left_key, np.ones((2, 3)), ("i", "k"), (2, 3)),
    }
    right_blocks = {
        right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4)),
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=lambda left, right: BlockKey(left.qn_left, right.qn_right),
    )

    plan = backend.lower_block_contraction(spec)

    assert len(backend.lower_pair_calls) == 1
    pair_spec, record_profile = backend.lower_pair_calls[0]
    assert record_profile is False
    assert pair_spec.left.modes == ("i", "k")
    assert pair_spec.right.modes == ("k", "j")
    assert pair_spec.output_modes == ("i", "j")
    assert plan.tasks[0].m == 2
    assert plan.tasks[0].n == 4
    assert plan.tasks[0].k == 3


def test_lower_block_contraction_uses_operand_global_shape_for_output_metadata():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    out_key = BlockKey((0,), (1,))
    spec = BlockContractionSpec(
        BlockTensor(
            {left_key: DenseBlock(left_key, np.ones((2, 3)), ("i", "k"), (2, 3), offset=(1, 0))},
            global_shape=(5, 3),
            modes=("i", "k"),
            block_axis_meta=None,
            backend="numpy",
        ),
        BlockTensor(
            {right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4), offset=(0, 2))},
            global_shape=(3, 7),
            modes=("k", "j"),
            block_axis_meta=None,
            backend="numpy",
        ),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: out_key,
    )

    plan = backend.lower_block_contraction(spec)
    result = backend.execute_grouped_gemm_plan(plan)

    assert plan.global_shape == (5, 7)
    assert result.global_shape == (5, 7)
    assert result.blocks[out_key].shape == (2, 4)
    assert result.blocks[out_key].offset == (1, 2)

    dense = result.to_dense(explicit=True)
    expected = np.zeros((5, 7))
    expected[1:3, 2:6] = np.ones((2, 3)) @ np.ones((3, 4))
    assert np.allclose(dense, expected)


def test_block_grouped_gemm_profile_records_output_block_offsets(tmp_path):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    out_key = BlockKey((0,), (1,))
    spec = BlockContractionSpec(
        BlockTensor(
            {left_key: DenseBlock(left_key, np.ones((2, 3)), ("i", "k"), (2, 3), offset=(1, 0))},
            global_shape=(5, 3),
            modes=("i", "k"),
            block_axis_meta=None,
            backend="numpy",
        ),
        BlockTensor(
            {right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4), offset=(0, 2))},
            global_shape=(3, 7),
            modes=("k", "j"),
            block_axis_meta=None,
            backend="numpy",
        ),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: out_key,
    )
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.lower_block_contraction(spec)
        result = backend.execute_grouped_gemm_plan(plan)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.blocks[out_key].offset == (1, 2)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_event = next(
        payload
        for payload in payloads
        if payload.get("event") == "contraction_plan"
        and payload.get("lowering") == "block_grouped_gemm"
    )
    execute_event = next(
        payload
        for payload in payloads
        if payload.get("event") == "contraction_execute"
        and payload.get("lowering") == "block_grouped_gemm"
    )
    expected_offsets = [
        {"extra": [], "qn_left": [0], "qn_right": [1], "offset": [1, 2]},
    ]

    assert plan_event["output_block_offsets"] == expected_offsets
    assert plan_event["unique_output_block_offsets"] == expected_offsets
    assert execute_event["output_block_offsets"] == expected_offsets
    assert execute_event["unique_output_block_offsets"] == expected_offsets
    assert execute_event["result_block_shapes"] == [
        {"extra": [], "qn_left": [0], "qn_right": [1], "shape": [2, 4], "offset": [1, 2]},
    ]


def test_block_grouped_gemm_profile_records_output_contribution_offsets(tmp_path):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    out_key = BlockKey((0,), (1,))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(
            BlockKey((0,), (0,)),
            np.ones((2, 3)),
            ("i", "k"),
            (2, 3),
            offset=(1, 0),
        ),
        BlockKey((0,), (2,)): DenseBlock(
            BlockKey((0,), (2,)),
            np.full((2, 3), 2.0),
            ("i", "k"),
            (2, 3),
            offset=(1, 0),
        ),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(
            BlockKey((0,), (1,)),
            np.full((3, 4), 3.0),
            ("k", "j"),
            (3, 4),
            offset=(0, 2),
        ),
        BlockKey((2,), (1,)): DenseBlock(
            BlockKey((2,), (1,)),
            np.ones((3, 4)),
            ("k", "j"),
            (3, 4),
            offset=(0, 2),
        ),
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(5, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 7), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: (
            BlockKey(left_key.qn_left, right_key.qn_right)
            if left_key.qn_right == right_key.qn_left
            else None
        ),
    )
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.lower_block_contraction(spec)
        result = backend.execute_grouped_gemm_plan(plan)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert plan.output_blocks == (out_key, out_key)
    assert result.blocks[out_key].offset == (1, 2)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    contribution_counts = [
        payload["output_contribution_counts"]
        for payload in payloads
        if payload.get("event") in {"contraction_plan", "contraction_execute"}
        and payload.get("lowering") == "block_grouped_gemm"
    ]
    assert contribution_counts == [
        [
            {
                "extra": [],
                "qn_left": [0],
                "qn_right": [1],
                "offset": [1, 2],
                "contribution_count": 2,
                "task_indices": [0, 1],
            },
        ],
        [
            {
                "extra": [],
                "qn_left": [0],
                "qn_right": [1],
                "offset": [1, 2],
                "contribution_count": 2,
                "task_indices": [0, 1],
            },
        ],
    ]
    contribution_details = [
        payload["output_contributions"]
        for payload in payloads
        if payload.get("event") in {"contraction_plan", "contraction_execute"}
        and payload.get("lowering") == "block_grouped_gemm"
    ]
    assert contribution_details == [
        [
            {
                "extra": [],
                "qn_left": [0],
                "qn_right": [1],
                "offset": [1, 2],
                "contribution_count": 2,
                "task_indices": [0, 1],
                "total_write_bytes": 128,
                "contributions": [
                    {"task_index": 0, "shape": [2, 4], "nbytes": 64, "output_offset": [1, 2]},
                    {"task_index": 1, "shape": [2, 4], "nbytes": 64, "output_offset": [1, 2]},
                ],
            },
        ],
        [
            {
                "extra": [],
                "qn_left": [0],
                "qn_right": [1],
                "offset": [1, 2],
                "contribution_count": 2,
                "task_indices": [0, 1],
                "total_write_bytes": 128,
                "contributions": [
                    {"task_index": 0, "shape": [2, 4], "nbytes": 64, "output_offset": [1, 2]},
                    {"task_index": 1, "shape": [2, 4], "nbytes": 64, "output_offset": [1, 2]},
                ],
            },
        ],
    ]


def test_lower_block_contraction_rejects_duplicate_outputs_without_accumulation():
    from renormalizer.backend import (
        BackendFeatureError,
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        BlockKey((0,), (2,)): DenseBlock(BlockKey((0,), (2,)), np.full((2, 3), 2.0), ("i", "k"), (2, 3)),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(BlockKey((0,), (1,)), np.full((3, 4), 3.0), ("k", "j"), (3, 4)),
        BlockKey((2,), (1,)): DenseBlock(BlockKey((2,), (1,)), np.ones((3, 4)), ("k", "j"), (3, 4)),
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: (
            BlockKey(left_key.qn_left, right_key.qn_right)
            if left_key.qn_right == right_key.qn_left
            else None
        ),
        accumulate=False,
    )

    with pytest.raises(BackendFeatureError, match="duplicate output block.*accumulate=False"):
        backend.lower_block_contraction(spec)


def test_execute_grouped_gemm_plan_accumulates_sparse_output_blocks():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    out_key = BlockKey((0,), (1,))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        BlockKey((0,), (2,)): DenseBlock(BlockKey((0,), (2,)), np.full((2, 3), 2.0), ("i", "k"), (2, 3)),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(BlockKey((0,), (1,)), np.full((3, 4), 3.0), ("k", "j"), (3, 4)),
        BlockKey((2,), (1,)): DenseBlock(BlockKey((2,), (1,)), np.ones((3, 4)), ("k", "j"), (3, 4)),
    }
    left = BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta={"left": True}, backend="numpy")
    right = BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta={"right": True}, backend="numpy")
    spec = BlockContractionSpec(
        left,
        right,
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: (
            BlockKey(left_key.qn_left, right_key.qn_right)
            if left_key.qn_right == right_key.qn_left
            else None
        ),
    )

    plan = backend.lower_block_contraction(spec)
    result = backend.execute_grouped_gemm_plan(plan, pack_threshold=2)

    expected = left_blocks[BlockKey((0,), (0,))].array @ right_blocks[BlockKey((0,), (1,))].array
    expected += left_blocks[BlockKey((0,), (2,))].array @ right_blocks[BlockKey((2,), (1,))].array
    assert result.modes == ("i", "j")
    assert result.global_shape == (2, 4)
    assert result.backend == "numpy"
    assert tuple(result.blocks) == (out_key,)
    assert np.allclose(result.blocks[out_key].array, expected)
    assert result.blocks[out_key].shape == (2, 4)


def test_execute_grouped_gemm_plan_preserves_block_batch_shape():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    out_key = BlockKey((0,), (1,))
    left_array = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    right_array = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)
    spec = BlockContractionSpec(
        BlockTensor(
            {left_key: DenseBlock(left_key, left_array, ("b", "i", "k"), left_array.shape)},
            global_shape=(2, 3, 4),
            modes=("b", "i", "k"),
            block_axis_meta=None,
            backend="numpy",
        ),
        BlockTensor(
            {right_key: DenseBlock(right_key, right_array, ("b", "k", "j"), right_array.shape)},
            global_shape=(2, 4, 5),
            modes=("b", "k", "j"),
            block_axis_meta=None,
            backend="numpy",
        ),
        output_modes=("b", "i", "j"),
        qn_rule=lambda left_key, right_key: (
            out_key if left_key.qn_right == right_key.qn_left else None
        ),
    )

    plan = backend.lower_block_contraction(spec)
    result = backend.execute_grouped_gemm_plan(plan, pack_threshold=2)

    assert plan.tasks[0].batch_shape == (2,)
    assert plan.bucketed_by_shape == {((2,), 3, 5, 4): (0,)}
    assert result.global_shape == (2, 3, 5)
    assert result.blocks[out_key].shape == (2, 3, 5)
    assert np.allclose(result.blocks[out_key].array, np.matmul(left_array, right_array))


def test_execute_grouped_gemm_plan_reduces_duplicate_outputs_outside_grouped_gemm():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CapturingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.grouped_task_c_values = None

        def grouped_gemm(self, tasks, *, pack_threshold=4, stream=None, workspace=None):
            self.grouped_task_c_values = [task.C for task in tasks]
            return [task.A @ task.B for task in tasks]

    backend = CapturingBackend()
    out_key = BlockKey((0,), (1,))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        BlockKey((0,), (2,)): DenseBlock(BlockKey((0,), (2,)), np.full((2, 3), 2.0), ("i", "k"), (2, 3)),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(BlockKey((0,), (1,)), np.full((3, 4), 3.0), ("k", "j"), (3, 4)),
        BlockKey((2,), (1,)): DenseBlock(BlockKey((2,), (1,)), np.ones((3, 4)), ("k", "j"), (3, 4)),
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: (
            BlockKey(left_key.qn_left, right_key.qn_right)
            if left_key.qn_right == right_key.qn_left
            else None
        ),
    )

    plan = backend.lower_block_contraction(spec)
    result = backend.execute_grouped_gemm_plan(plan, pack_threshold=2)

    expected = left_blocks[BlockKey((0,), (0,))].array @ right_blocks[BlockKey((0,), (1,))].array
    expected += left_blocks[BlockKey((0,), (2,))].array @ right_blocks[BlockKey((2,), (1,))].array
    assert plan.scatter_add_required is True
    assert all(value is None for value in backend.grouped_task_c_values)
    assert np.allclose(result.blocks[out_key].array, expected)


def test_grouped_gemm_plan_cost_model_reports_work_and_peak_memory():
    from renormalizer.backend import (
        BackendFeatureError,
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
        HardwareModel,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    left = BlockTensor(
        {left_key: DenseBlock(left_key, np.ones((2, 3), dtype=np.float64), ("i", "k"), (2, 3))},
        global_shape=(2, 3),
        modes=("i", "k"),
        block_axis_meta=None,
        backend="numpy",
    )
    right = BlockTensor(
        {right_key: DenseBlock(right_key, np.ones((3, 4), dtype=np.float64), ("k", "j"), (3, 4))},
        global_shape=(3, 4),
        modes=("k", "j"),
        block_axis_meta=None,
        backend="numpy",
    )
    plan = backend.lower_block_contraction(
        BlockContractionSpec(
            left,
            right,
            output_modes=("i", "j"),
            qn_rule=lambda left_key, right_key: BlockKey(left_key.qn_left, right_key.qn_right),
        )
    )
    hw = HardwareModel(flop_per_s=24.0, memory_bandwidth_Bps=208.0, max_memory_bytes=64)

    estimate = backend.estimate_contraction(plan, hw)

    assert estimate.flops == 48
    assert estimate.read_bytes == 144
    assert estimate.write_bytes == 64
    assert estimate.workspace_bytes == 0
    assert estimate.peak_bytes == 64
    assert estimate.compute_s == pytest.approx(2.0)
    assert estimate.memory_s == pytest.approx(1.0)
    assert estimate.total_s == pytest.approx(3.0)
    with pytest.raises(BackendFeatureError, match="max_memory.*63.*peak.*64"):
        backend.estimate_contraction(plan, HardwareModel(max_memory_bytes=63))


def _valid_grouped_gemm_plan_kwargs():
    from renormalizer.backend import BlockKey, MatmulDesc

    desc = MatmulDesc(
        np.ones((2, 3), dtype=np.float64),
        np.ones((3, 4), dtype=np.float64),
        None,
        m=2,
        n=4,
        k=3,
    )
    return {
        "tasks": (desc,),
        "output_blocks": (BlockKey((0,), (1,)),),
        "bucketed_by_shape": {(2, 4, 3): (0,)},
        "scatter_add_required": False,
        "estimated_flops": 48,
        "estimated_read_bytes": 80,
        "estimated_write_bytes": 64,
        "estimated_workspace_bytes": 0,
        "output_modes": ("i", "j"),
        "global_shape": (2, 4),
        "backend": "numpy",
    }


def test_grouped_gemm_plan_hash_includes_task_semantics():
    from dataclasses import replace

    from renormalizer.backend import GroupedGemmPlan

    kwargs = _valid_grouped_gemm_plan_kwargs()
    desc = kwargs["tasks"][0]
    desc = replace(
        desc,
        C=np.zeros((2, 4), dtype=np.float64),
        dtype_compute=np.float64,
        dtype_output=np.float64,
    )
    kwargs["tasks"] = (desc,)
    base_hash = GroupedGemmPlan(**kwargs).plan_hash

    for changed_desc in (
        replace(desc, alpha=2.0),
        replace(desc, beta=1.0),
        replace(desc, dtype_compute=np.float32),
        replace(desc, dtype_output=np.float32),
    ):
        assert GroupedGemmPlan(**{**kwargs, "tasks": (changed_desc,)}).plan_hash != base_hash


def test_grouped_gemm_plan_rejects_inconsistent_task_metadata():
    from renormalizer.backend import GroupedGemmPlan

    kwargs = _valid_grouped_gemm_plan_kwargs()

    with pytest.raises(ValueError, match="GroupedGemmPlan tasks and output_blocks must have the same length"):
        GroupedGemmPlan(**{**kwargs, "output_blocks": ()})

    with pytest.raises(ValueError, match="GroupedGemmPlan bucket shapes must be rank-3"):
        GroupedGemmPlan(**{**kwargs, "bucketed_by_shape": {(2, 4): (0,)}})

    with pytest.raises(ValueError, match="GroupedGemmPlan bucket shapes must be non-negative"):
        GroupedGemmPlan(**{**kwargs, "bucketed_by_shape": {(-2, 4, 3): (0,)}})

    with pytest.raises(ValueError, match="GroupedGemmPlan bucket task indices out of range"):
        GroupedGemmPlan(**{**kwargs, "bucketed_by_shape": {(2, 4, 3): (1,)}})

    with pytest.raises(ValueError, match="GroupedGemmPlan bucketed_by_shape must cover each task exactly once"):
        GroupedGemmPlan(**{**kwargs, "bucketed_by_shape": {}})

    with pytest.raises(ValueError, match="GroupedGemmPlan bucket shape must match task descriptor"):
        GroupedGemmPlan(**{**kwargs, "bucketed_by_shape": {(2, 5, 3): (0,)}})

    with pytest.raises(ValueError, match="GroupedGemmPlan bucketed_by_shape must cover each task exactly once"):
        GroupedGemmPlan(**{**kwargs, "bucketed_by_shape": {(2, 4, 3): (0, 0)}})

    with pytest.raises(ValueError, match="GroupedGemmPlan output_modes must match global_shape rank"):
        GroupedGemmPlan(**{**kwargs, "output_modes": ("i",), "global_shape": (2, 4)})


def test_grouped_gemm_plan_buckets_batched_tasks_by_batch_shape():
    from renormalizer.backend import BlockKey, GroupedGemmPlan, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    desc0 = MatmulDesc(
        np.ones((2, 2, 3), dtype=np.float64),
        np.ones((2, 3, 4), dtype=np.float64),
        None,
        m=2,
        n=4,
        k=3,
        batch_shape=(2,),
    )
    desc1 = MatmulDesc(
        np.ones((3, 2, 3), dtype=np.float64),
        np.ones((3, 3, 4), dtype=np.float64),
        None,
        m=2,
        n=4,
        k=3,
        batch_shape=(3,),
    )
    descs = (desc0, desc1)

    assert NumpyBackend._bucketed_by_desc_shape(descs) == {
        ((2,), 2, 4, 3): (0,),
        ((3,), 2, 4, 3): (1,),
    }

    plan = GroupedGemmPlan(
        tasks=descs,
        output_blocks=(BlockKey((0,), (0,)), BlockKey((1,), (1,))),
        bucketed_by_shape={
            ((2,), 2, 4, 3): (0,),
            ((3,), 2, 4, 3): (1,),
        },
        scatter_add_required=False,
        estimated_flops=2 * 2 * 2 * 4 * 3 + 2 * 3 * 2 * 4 * 3,
        estimated_read_bytes=desc0.estimated_read_bytes + desc1.estimated_read_bytes,
        estimated_write_bytes=desc0.estimated_write_bytes + desc1.estimated_write_bytes,
        estimated_workspace_bytes=0,
        output_modes=("i", "j"),
        global_shape=(2, 4),
        backend="numpy",
    )

    assert plan.bucketed_by_shape == {
        ((2,), 2, 4, 3): (0,),
        ((3,), 2, 4, 3): (1,),
    }

    with pytest.raises(ValueError, match="GroupedGemmPlan bucket batch_shape must match task descriptor"):
        GroupedGemmPlan(
            tasks=descs,
            output_blocks=(BlockKey((0,), (0,)), BlockKey((1,), (1,))),
            bucketed_by_shape={(2, 4, 3): (0, 1)},
            scatter_add_required=False,
            estimated_flops=plan.estimated_flops,
            estimated_read_bytes=plan.estimated_read_bytes,
            estimated_write_bytes=plan.estimated_write_bytes,
            estimated_workspace_bytes=0,
            output_modes=("i", "j"),
            global_shape=(2, 4),
            backend="numpy",
        )


@pytest.mark.parametrize(
    "field",
    [
        "estimated_flops",
        "estimated_read_bytes",
        "estimated_write_bytes",
        "estimated_copy_bytes",
        "estimated_workspace_bytes",
    ],
)
def test_grouped_gemm_plan_rejects_negative_estimates(field):
    from renormalizer.backend import GroupedGemmPlan

    kwargs = _valid_grouped_gemm_plan_kwargs()
    kwargs[field] = -1

    with pytest.raises(ValueError, match="GroupedGemmPlan {0} must be non-negative".format(field)):
        GroupedGemmPlan(**kwargs)


def test_lower_block_contraction_records_grouped_plan_profile(tmp_path):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    out_key = BlockKey((0,), (1,))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        BlockKey((0,), (2,)): DenseBlock(BlockKey((0,), (2,)), np.full((2, 3), 2.0), ("i", "k"), (2, 3)),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(BlockKey((0,), (1,)), np.full((3, 4), 3.0), ("k", "j"), (3, 4)),
        BlockKey((2,), (1,)): DenseBlock(BlockKey((2,), (1,)), np.ones((3, 4)), ("k", "j"), (3, 4)),
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: (
            BlockKey(left_key.qn_left, right_key.qn_right)
            if left_key.qn_right == right_key.qn_left
            else None
        ),
    )
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.lower_block_contraction(spec)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload.get("event") == "contraction_plan"
        and payload.get("lowering") == "block_grouped_gemm"
    )
    assert plan.output_blocks == (out_key, out_key)
    assert event["output_modes"] == ["i", "j"]
    assert event["output_shape"] == [2, 4]
    assert event["flops"] == plan.estimated_flops
    assert event["read_bytes"] == plan.estimated_read_bytes
    assert event["write_bytes"] == plan.estimated_write_bytes
    assert event["num_grouped_tasks"] == 2
    assert event["num_blocks"] == 1
    assert event["num_shape_buckets"] == 1
    assert event["grouped_gemm_policy"] == "bucketed_grouped_gemm"
    assert event["grouped_gemm_implementation"] == "fallback_bucketed_grouped_gemm"
    assert event["requires_grouped_gemm_fallback"] is True
    assert event["scatter_add_required"] is True
    assert event["reduction_mode"] == "scatter_add"
    assert event["num_output_reduction_groups"] == 1
    assert event["num_scatter_add_tasks"] == 2
    assert event["max_output_contributions"] == 2
    assert event["output_contribution_counts"] == [
        {
            "extra": [],
            "qn_left": [0],
            "qn_right": [1],
            "contribution_count": 2,
            "task_indices": [0, 1],
        },
    ]
    assert event["output_contributions"] == [
        {
            "extra": [],
            "qn_left": [0],
            "qn_right": [1],
            "contribution_count": 2,
            "task_indices": [0, 1],
            "total_write_bytes": 128,
            "contributions": [
                {"task_index": 0, "shape": [2, 4], "nbytes": 64, "output_offset": None},
                {"task_index": 1, "shape": [2, 4], "nbytes": 64, "output_offset": None},
            ],
        },
    ]
    assert event["dense_materialized"] is False
    assert event["materialized_dense_bytes"] == 0
    assert event["shape_buckets"] == [
        {"m": 2, "n": 4, "k": 3, "task_indices": [0, 1], "task_count": 2},
    ]
    assert event["group_sizes"] == [2]
    assert event["gsta"] == [0, 2]
    assert event["sorted_indices"] == [0, 1]
    assert event["group_keys"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
        },
    ]
    assert event["group_key_buckets"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
            "task_indices": [0, 1],
            "task_count": 2,
        },
    ]
    assert event["input_dtypes"] == [
        ["float64", "float64"],
        ["float64", "float64"],
    ]
    assert event["task_operands"][0][0]["modes"] == ["i", "k"]
    assert event["task_operands"][0][0]["shape"] == [2, 3]
    assert event["task_operands"][0][0]["itemsize"] == 8
    assert event["task_operands"][0][0]["contiguous"] is True
    assert event["task_operands"][0][0]["is_distributed"] is False
    assert event["operands"] == event["task_operands"]
    assert event["task_specs"][0]["m"] == 2
    assert event["task_specs"][0]["n"] == 4
    assert event["task_specs"][0]["k"] == 3
    assert event["task_specs"][0]["dtype_compute"] == "None"
    assert event["task_specs"][0]["estimated_flops"] == 48
    assert event["task_specs"][0]["estimated_read_bytes"] == 144
    assert event["task_specs"][0]["estimated_write_bytes"] == 64
    assert event["task_specs"][0]["layout_a"]["logical_modes"] == ["i", "k"]
    assert event["task_specs"][0]["layout_b"]["logical_modes"] == ["k", "j"]
    assert event["task_specs"][0]["layout_c"]["logical_modes"] == ["i", "j"]


def test_block_grouped_gemm_profile_correlates_plan_and_execute_events(tmp_path):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    spec = BlockContractionSpec(
        BlockTensor(
            {left_key: DenseBlock(left_key, np.ones((2, 3)), ("i", "k"), (2, 3))},
            global_shape=(2, 3),
            modes=("i", "k"),
            block_axis_meta=None,
            backend="numpy",
        ),
        BlockTensor(
            {right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4))},
            global_shape=(3, 4),
            modes=("k", "j"),
            block_axis_meta=None,
            backend="numpy",
        ),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: BlockKey(left_key.qn_left, right_key.qn_right),
    )
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.lower_block_contraction(spec)
        backend.execute(plan)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_event = next(
        event
        for event in events
        if event["event"] == "contraction_plan"
        and event["lowering"] == "block_grouped_gemm"
    )
    execute_event = next(
        event
        for event in events
        if event["event"] == "contraction_execute"
        and event["lowering"] == "block_grouped_gemm"
    )

    assert isinstance(plan.plan_hash, str)
    assert plan.plan_hash
    assert plan_event["plan_hash"] == plan.plan_hash
    assert execute_event["plan_hash"] == plan.plan_hash
    assert plan_event["grouped_gemm_implementation"] == "fallback_bucketed_grouped_gemm"
    assert plan_event["requires_grouped_gemm_fallback"] is True
    assert plan_event["execution_primitives"] == ["grouped_gemm"]
    assert plan_event["execution_policies"] == ["bucketed_grouped_gemm"]
    assert plan_event["fallback_reasons"] == [plan_event["fallback_reason"]]
    assert execute_event["grouped_gemm_implementation"] == "fallback_bucketed_loop_matmul"
    assert execute_event["requires_grouped_gemm_fallback"] is True
    assert execute_event["execution_primitives"] == ["grouped_gemm"]
    assert execute_event["execution_policies"] == ["bucketed_loop_matmul"]
    assert execute_event["fallback_reasons"] == [execute_event["fallback_reason"]]


def test_execute_grouped_gemm_plan_records_block_profile(tmp_path):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    out_key = BlockKey((0,), (1,))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        BlockKey((0,), (2,)): DenseBlock(BlockKey((0,), (2,)), np.full((2, 3), 2.0), ("i", "k"), (2, 3)),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(BlockKey((0,), (1,)), np.full((3, 4), 3.0), ("k", "j"), (3, 4)),
        BlockKey((2,), (1,)): DenseBlock(BlockKey((2,), (1,)), np.ones((3, 4)), ("k", "j"), (3, 4)),
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: (
            BlockKey(left_key.qn_left, right_key.qn_right)
            if left_key.qn_right == right_key.qn_left
            else None
        ),
    )
    plan = backend.lower_block_contraction(spec)
    stream = object()
    workspace = backend.allocate_workspace(128)
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_grouped_gemm_plan(
            plan,
            pack_threshold=2,
            stream=stream,
            workspace=workspace,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert tuple(result.blocks) == (out_key,)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload.get("event") == "contraction_execute"
        and payload.get("lowering") == "block_grouped_gemm"
    )
    assert event["backend"] == "numpy"
    assert event["output_modes"] == ["i", "j"]
    assert event["global_shape"] == [2, 4]
    assert event["num_grouped_tasks"] == 2
    assert event["num_blocks"] == 1
    assert event["num_shape_buckets"] == 1
    assert event["num_batched_gemm"] == 0
    assert event["num_gemm"] == 2
    assert event["batched_bucket_count"] == 0
    assert event["loop_bucket_count"] == 1
    assert event["batched_task_count"] == 0
    assert event["loop_task_count"] == 2
    assert event["copy_bytes"] == 0
    assert event["scatter_add_required"] is True
    assert event["reduction_mode"] == "scatter_add"
    assert event["output_reduction_executor"] == "post_grouped_gemm"
    assert event["grouped_gemm_output_write_mode"] == "workspace_then_reduce"
    assert event["num_output_reduction_groups"] == 1
    assert event["num_scatter_add_tasks"] == 2
    assert event["max_output_contributions"] == 2
    assert event["output_contribution_counts"] == [
        {
            "extra": [],
            "qn_left": [0],
            "qn_right": [1],
            "contribution_count": 2,
            "task_indices": [0, 1],
        },
    ]
    assert event["dense_materialized"] is False
    assert event["materialized_dense_bytes"] == 0
    assert event["shape_buckets"] == [
        {"m": 2, "n": 4, "k": 3, "task_indices": [0, 1], "task_count": 2},
    ]
    assert event["group_sizes"] == [2]
    assert event["gsta"] == [0, 2]
    assert event["sorted_indices"] == [0, 1]
    assert event["group_keys"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
        },
    ]
    assert event["group_key_buckets"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
            "task_indices": [0, 1],
            "task_count": 2,
        },
    ]
    assert event["output_block_keys"] == [
        {"extra": [], "qn_left": [0], "qn_right": [1]},
        {"extra": [], "qn_left": [0], "qn_right": [1]},
    ]
    assert event["unique_output_block_keys"] == [
        {"extra": [], "qn_left": [0], "qn_right": [1]},
    ]
    assert event["result_block_shapes"] == [{"extra": [], "qn_left": [0], "qn_right": [1], "shape": [2, 4]}]
    assert event["result_block_layouts"] == [
        {
            "extra": [],
            "qn_left": [0],
            "qn_right": [1],
            "shape": [2, 4],
            "strides": [32, 8],
            "order": "C",
            "contiguous": True,
            "dtype": "float64",
            "itemsize": 8,
            "nbytes": 64,
            "backend": "numpy",
            "device": "DeviceSpec(kind='cpu', index=None, local_rank=None, global_rank=None, visible_id=None)",
            "device_kind": "cpu",
            "device_index": None,
            "location": "host",
            "is_host": True,
            "is_device": False,
            "is_distributed": False,
        },
    ]
    assert event["input_dtypes"] == [
        ["float64", "float64"],
        ["float64", "float64"],
    ]
    assert event["task_operands"][0][0]["name"] == "task0.A"
    assert event["task_operands"][0][0]["modes"] == ["i", "k"]
    assert event["task_operands"][0][0]["shape"] == [2, 3]
    assert event["task_operands"][0][1]["name"] == "task0.B"
    assert event["task_operands"][0][1]["modes"] == ["k", "j"]
    assert event["task_operands"][0][1]["shape"] == [3, 4]
    assert event["operands"] == event["task_operands"]
    assert event["task_specs"][0]["m"] == 2
    assert event["task_specs"][0]["n"] == 4
    assert event["task_specs"][0]["k"] == 3
    assert event["task_specs"][0]["dtype_output"] == "None"
    assert event["task_specs"][0]["estimated_flops"] == 48
    assert event["task_specs"][0]["layout_a"]["logical_modes"] == ["i", "k"]
    assert event["task_specs"][0]["layout_b"]["logical_modes"] == ["k", "j"]
    assert event["task_specs"][0]["layout_c"]["logical_modes"] == ["i", "j"]
    assert event["peak_bytes"] == event["write_bytes"] + event["workspace_bytes"]
    assert event["stream_provided"] is True
    assert event["stream_type"] == "object"
    assert event["workspace_provided"] is True
    assert event["workspace_nbytes"] == 128
    assert event["workspace_device_kind"] == "cpu"
    assert event["workspace_device_index"] is None
    assert event["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert event["fallback_policy"] == "record"
    assert event["wall_s"] >= 0.0

    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "block_grouped_gemm"
    assert profile["plan_hash"] == plan.plan_hash
    assert profile["num_grouped_tasks"] == 2
    assert profile["num_blocks"] == 1
    assert profile["num_shape_buckets"] == 1
    assert profile["scatter_add_required"] is True
    assert profile["dense_materialized"] is False
    assert profile["materialized_dense_bytes"] == 0


def test_grouped_gemm_records_focus_style_execute_profile(tmp_path, caplog):
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="large-0"),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="large-1"),
        GemmTask(np.ones((3, 4), dtype=np.float64), np.ones((4, 5), dtype=np.float64), tag="small"),
    ]
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    caplog.set_level(PROFILING, logger="renormalizer")
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.grouped_gemm(tasks, pack_threshold=2)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "grouped_gemm_execute")

    assert event["backend"] == "numpy"
    assert event["policy"] == "bucketed_mixed_matmul"
    assert event["num_tasks"] == 3
    assert event["num_groups"] == 2
    assert event["group_sizes"] == [2, 1]
    assert event["gsta"] == [0, 2, 3]
    assert event["sorted_indices"] == [0, 1, 2]
    assert event["group_keys"][0]["m"] == 128
    assert event["group_keys"][0]["n"] == 128
    assert event["group_keys"][0]["k"] == 128
    assert event["group_keys"][1]["m"] == 3
    assert event["group_keys"][1]["n"] == 5
    assert event["group_keys"][1]["k"] == 4
    assert event["shape_buckets"] == [
        {
            "dtype_a": "float64",
            "dtype_b": "float64",
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "trans_a": False,
            "trans_b": False,
            "conj_a": False,
            "conj_b": False,
            "execution": "batched",
            "task_count": 2,
        },
        {
            "dtype_a": "float64",
            "dtype_b": "float64",
            "batch_shape": [],
            "m": 3,
            "n": 5,
            "k": 4,
            "trans_a": False,
            "trans_b": False,
            "conj_a": False,
            "conj_b": False,
            "execution": "loop",
            "task_count": 1,
        },
    ]
    assert event["total_flops"] == 2 * 128 * 128 * 128 * 2 + 2 * 3 * 5 * 4
    assert event["num_batched_gemm"] == 1
    assert event["num_gemm"] == 1
    assert event["grouped_gemm_implementation"] == "fallback_bucketed_mixed_matmul"
    assert event["requires_grouped_gemm_fallback"] is True
    assert event["fallback_from"] == "grouped_gemm"
    assert event["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert event["execution_primitives"] == ["grouped_gemm"]
    assert event["execution_policies"] == ["bucketed_mixed_matmul"]
    assert event["fallback_reasons"] == [event["fallback_reason"]]
    assert event["bucket_fallback_reasons"] == ["task_count below pack_threshold"]
    assert event["pointer_setup_s"] >= 0.0
    assert event["compute_s"] >= 0.0
    assert event["wall_s"] >= event["compute_s"]
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_subclass"] == "grouped_gemm_execute"
    assert event["compute_accounting"] == "inclusive"
    assert event["lowering"] == "grouped_gemm"
    assert event["num_grouped_tasks"] == 3
    assert event["num_shape_buckets"] == 2
    assert event["compute_profile"]["estimated_cost"]["flops"] == event["total_flops"]
    assert event["compute_profile"]["workload_signature"]["shape_signature"] == (
        "lowering=grouped_gemm,gemm=1,batched=1,grouped=3,blocks=3,buckets=2"
    )
    assert event["output_shape"] == [[128, 128], [128, 128], [3, 5]]
    assert event["output_strides"] == [[1024, 8], [1024, 8], [40, 8]]
    assert event["output_order"] == ["C", "C", "C"]
    assert event["output_contiguous"] == [True, True, True]
    assert event["output_backend"] == ["numpy", "numpy", "numpy"]
    assert event["output_device_kind"] == ["cpu", "cpu", "cpu"]
    assert event["output_location"] == ["host", "host", "host"]
    assert event["output_is_host"] == [True, True, True]
    assert event["output_is_device"] == [False, False, False]
    assert event["output_is_distributed"] == [False, False, False]
    assert event["layout_profile"]["output_layout"] == {
        "shape": [[128, 128], [128, 128], [3, 5]],
        "strides": [[1024, 8], [1024, 8], [40, 8]],
        "order": ["C", "C", "C"],
        "contiguous": [True, True, True],
    }

    logged_payloads = [
        json.loads(record.getMessage()[len(profiling.LOG_PREFIX):])
        for record in caplog.records
        if record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    summary = next(
        payload
        for payload in logged_payloads
        if payload["event"] == "profile_compute_summary"
        and payload["source_events"] == ["grouped_gemm_execute"]
    )
    assert summary["compute_class"] == "contraction_plan"
    assert summary["compute_subclass"] == "grouped_gemm_execute"
    assert summary["compute_accounting"] == "inclusive"
    assert summary["total_flops_estimate"] == event["total_flops"]
    assert summary["total_grouped_tasks"] == 3
    assert summary["total_shape_buckets"] == 2
    assert summary["cost_profile"]["compute_s"] == pytest.approx(event["compute_s"])
    assert summary["total_compute_s"] == pytest.approx(event["compute_s"])
    assert summary["total_pointer_setup_s"] == pytest.approx(event["pointer_setup_s"])
    assert summary["phase_timing"] == {
        "compute_s": pytest.approx(event["compute_s"]),
        "pointer_setup_s": pytest.approx(event["pointer_setup_s"]),
        "pack_s": pytest.approx(event["pack_s"]),
        "kernel_s": pytest.approx(event["kernel_s"]),
        "scatter_s": pytest.approx(event["scatter_s"]),
        "loop_s": pytest.approx(event["loop_s"]),
    }


def test_grouped_gemm_execution_profile_separates_pack_kernel_and_scatter_costs():
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((128, 128), dtype=np.float64)
    right = np.full((128, 128), 2.0, dtype=np.float64)
    out0 = np.zeros((128, 128), dtype=np.float64)
    out1 = np.zeros((128, 128), dtype=np.float64)
    tasks = [
        GemmTask(left, right, C=out0),
        GemmTask(left + 1.0, right + 1.0, C=out1),
    ]

    backend.grouped_gemm(tasks, pack_threshold=2)

    profile = backend.last_execution_profile()
    expected_pack_bytes = sum(task.A.nbytes + task.B.nbytes for task in tasks)
    assert profile["event"] == "grouped_gemm_execute"
    assert profile["num_tasks"] == 2
    assert profile["num_batched_gemm"] == 1
    assert profile["compute_profile"]["compute_class"] == "contraction_plan"
    assert profile["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert profile["cost_profile"]["flops"] == profile["total_flops"]
    assert profile["copy_profile"]["copy_required"] is True
    assert profile["copy_profile"]["copy_bytes"] == profile["copy_bytes"]
    assert profile["workspace_profile"]["workspace_required"] is False
    assert profile["lowering_profile"]["lowering"] == "grouped_gemm"
    assert profile["pack_strategy"] == "stack_per_call"
    assert profile["prepacked"] is False
    assert profile["pack_bytes"] == expected_pack_bytes
    assert profile["pack_s"] >= 0.0
    assert profile["kernel_s"] >= 0.0
    assert profile["scatter_s"] >= 0.0
    assert profile["loop_s"] == 0.0
    assert profile["kernel_calls"] == 1
    assert profile["batched_kernel_calls"] == 1
    assert profile["loop_kernel_calls"] == 0
    assert profile["bucket_fallback_reasons"] == []
    assert len(profile["bucket_execution_profiles"]) == 1
    bucket = profile["bucket_execution_profiles"][0]
    assert bucket["execution"] == "batched"
    assert bucket["task_count"] == 2
    assert bucket["pack_strategy"] == "stack_per_call"
    assert bucket["pack_bytes"] == expected_pack_bytes
    assert bucket["kernel_calls"] == 1


def test_prepacked_grouped_gemm_reuses_packed_buckets_and_profiles_no_repack():
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left0 = np.arange(128 * 128, dtype=np.float64).reshape(128, 128)
    right0 = np.eye(128, dtype=np.float64)
    left1 = left0 + 10.0
    right1 = np.full((128, 128), 2.0, dtype=np.float64)
    tasks = [
        GemmTask(left0, right0, tag="bucket-0"),
        GemmTask(left1, right1, tag="bucket-1"),
    ]

    prepacked = backend.prepack_grouped_gemm(tasks, pack_threshold=2)
    first = backend.execute_prepacked_grouped_gemm(prepacked)
    second = backend.execute_prepacked_grouped_gemm(prepacked)

    assert np.allclose(first[0], left0 @ right0)
    assert np.allclose(first[1], left1 @ right1)
    assert np.allclose(second[0], first[0])
    assert np.allclose(second[1], first[1])

    plan_profile = prepacked.profile
    expected_pack_bytes = sum(task.A.nbytes + task.B.nbytes for task in tasks)
    assert plan_profile["event"] == "grouped_gemm_prepack"
    assert plan_profile["pack_strategy"] == "prepack_once"
    assert plan_profile["prepacked"] is True
    assert plan_profile["pack_bytes"] == expected_pack_bytes
    assert plan_profile["num_batched_gemm"] == 1
    assert plan_profile["num_gemm"] == 0
    assert plan_profile["compute_profile"]["compute_class"] == "contraction_plan"
    assert plan_profile["copy_profile"]["copy_required"] is True
    assert plan_profile["copy_profile"]["copy_bytes"] == expected_pack_bytes
    assert plan_profile["lowering_profile"]["lowering"] == "grouped_gemm"

    profile = backend.last_execution_profile()
    assert profile["event"] == "grouped_gemm_execute"
    assert profile["compute_profile"]["compute_class"] == "contraction_plan"
    assert profile["cost_profile"]["flops"] == profile["total_flops"]
    assert profile["copy_profile"]["copy_required"] is True
    assert profile["copy_profile"]["copy_bytes"] == profile["copy_bytes"]
    assert profile["lowering_profile"]["lowering"] == "grouped_gemm"
    assert profile["execution_primitives"] == ["grouped_gemm"]
    assert profile["execution_policies"] == ["prepacked_bucketed_matmul"]
    assert profile["fallback_reasons"] == [profile["fallback_reason"]]
    assert profile["pointer_setup_s"] >= 0.0
    assert profile["wall_s"] >= profile["pointer_setup_s"] + profile["compute_s"]
    assert profile["pack_strategy"] == "prepacked_reuse"
    assert profile["prepacked"] is True
    assert profile["pack_s"] == 0.0
    assert profile["pack_bytes"] == 0
    assert profile["reused_pack_bytes"] == expected_pack_bytes
    assert profile["kernel_calls"] == 1
    assert profile["batched_kernel_calls"] == 1
    assert profile["loop_kernel_calls"] == 0
    assert profile["bucket_execution_profiles"][0]["pack_strategy"] == "prepacked_reuse"
    assert profile["bucket_execution_profiles"][0]["pack_bytes"] == 0
    assert profile["output_shape"] == [(128, 128), (128, 128)]
    assert profile["output_strides"] == [(1024, 8), (1024, 8)]
    assert profile["output_order"] == ["C", "C"]
    assert profile["output_contiguous"] == [True, True]
    assert profile["output_backend"] == ["numpy", "numpy"]
    assert profile["output_device_kind"] == ["cpu", "cpu"]
    assert profile["output_location"] == ["host", "host"]
    assert profile["output_is_host"] == [True, True]
    assert profile["output_is_device"] == [False, False]
    assert profile["output_is_distributed"] == [False, False]
    layout_profile = json.loads(json.dumps(profile["layout_profile"]))
    assert layout_profile["output_layout"] == {
        "shape": [[128, 128], [128, 128]],
        "strides": [[1024, 8], [1024, 8]],
        "order": ["C", "C"],
        "contiguous": [True, True],
    }


def test_backend_owned_prepack_grouped_gemm_reports_native_implementation():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NativePrepackNumpyBackend(NumpyBackend):
        supports_grouped_gemm = True

    backend = NativePrepackNumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    tasks = [
        GemmTask(np.ones((64, 64), dtype=np.float64), np.eye(64, dtype=np.float64)),
        GemmTask(np.full((64, 64), 2.0, dtype=np.float64), np.eye(64, dtype=np.float64)),
    ]

    prepacked = backend.prepack_grouped_gemm(tasks, pack_threshold=2)

    assert prepacked.profile["supports_grouped_gemm"] is True
    assert prepacked.profile["requires_grouped_gemm_fallback"] is False
    assert prepacked.profile["fallback_reason"] is None
    assert prepacked.profile["fallback_reasons"] == []
    assert prepacked.profile["grouped_gemm_implementation"] == "backend_prepacked_bucketed_matmul"
    assert prepacked.profile["num_batched_gemm"] == 1


def test_prepack_grouped_gemm_writes_pack_cost_profile_event(tmp_path):
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="bucket-0"),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.eye(128, dtype=np.float64), tag="bucket-1"),
    ]
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        prepacked = backend.prepack_grouped_gemm(tasks, pack_threshold=2)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in events if payload["event"] == "grouped_gemm_prepack")
    expected_pack_bytes = sum(task.A.nbytes + task.B.nbytes for task in tasks)

    assert event["backend"] == "numpy"
    assert event["descriptor_source"] == "gemm_task"
    assert event["pack_strategy"] == "prepack_once"
    assert event["prepacked"] is True
    assert event["pack_bytes"] == expected_pack_bytes
    assert event["pack_s"] >= 0.0
    assert event["lowering"] == "grouped_gemm"
    assert event["flops"] == 2 * 2 * 128 * 128 * 128
    assert event["total_flops"] == event["flops"]
    assert event["read_bytes"] == expected_pack_bytes
    assert event["write_bytes"] == 2 * 128 * 128 * 8
    assert event["copy_bytes"] == expected_pack_bytes
    assert event["workspace_bytes"] == 0
    assert event["num_grouped_tasks"] == 2
    assert event["num_shape_buckets"] == 1
    assert event["num_tasks"] == 2
    assert event["num_groups"] == 1
    assert event["num_batched_gemm"] == 1
    assert event["num_gemm"] == 0
    assert event["kernel_calls"] == 1
    assert event["batched_kernel_calls"] == 1
    assert event["loop_kernel_calls"] == 0
    assert event["pack_threshold"] == 2
    assert event["fallback_from"] == "grouped_gemm"
    assert event["fallback_to"] == "prepacked_bucketed_matmul"
    assert event["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert event["fallback_policy"] == "record"
    assert event["execution_primitives"] == ["grouped_gemm"]
    assert event["execution_policies"] == ["prepacked_bucketed_matmul"]
    assert event["fallback_reasons"] == [event["fallback_reason"]]
    assert event["input_shapes"] == [
        [[128, 128], [128, 128]],
        [[128, 128], [128, 128]],
    ]
    assert event["input_dtypes"] == [
        ["float64", "float64"],
        ["float64", "float64"],
    ]
    assert event["output_shape"] == [[128, 128], [128, 128]]
    assert event["dtype"] == "float64"
    assert event["task_specs"][0]["m"] == 128
    assert event["task_specs"][0]["n"] == 128
    assert event["task_specs"][0]["k"] == 128
    assert event["task_specs"][0]["tag"] == "bucket-0"
    assert event["task_operands"][0][0]["name"] == "task0.A"
    assert event["task_operands"][0][0]["shape"] == [128, 128]
    assert event["task_operands"][0][0]["dtype"] == "float64"
    assert event["task_operands"][0][0]["is_host"] is True
    assert event["device_profile"]["backend"] == "numpy"
    assert event["device_profile"]["array_location"] == "host"
    assert event["dtype_profile"]["input_dtypes"] == ["float64", "float64", "float64", "float64"]
    assert event["dtype_profile"]["output_dtype"] == "float64"
    assert event["layout_profile"]["input_layouts"][0]["shape"] == [128, 128]
    assert event["layout_profile"]["input_layouts"][0]["order"] == "C"
    assert event["layout_profile"]["input_layouts"][0]["contiguous"] is True
    assert len(event["layout_profile"]["input_layouts"]) == 4
    assert event["copy_profile"] == {
        "copy_required": True,
        "copy_bytes": expected_pack_bytes,
        "copy_kind": "packing_or_bucketed_execution",
        "copy_source": "copy_bytes",
    }
    assert event["workspace_profile"]["workspace_required"] is False
    assert event["lowering_profile"]["lowering"] == "grouped_gemm"
    assert event["cost_profile"]["flops"] == event["flops"]
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["shape_buckets"] == [
        {
            "dtype_a": "float64",
            "dtype_b": "float64",
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "trans_a": False,
            "trans_b": False,
            "conj_a": False,
            "conj_b": False,
            "execution": "batched",
            "task_count": 2,
        },
    ]
    assert event["bucket_task_counts"] == [2]
    assert event["bucket_execution_profiles"][0]["execution"] == "batched"
    assert event["bucket_execution_profiles"][0]["pack_strategy"] == "prepack_once"
    assert event["bucket_execution_profiles"][0]["pack_bytes"] == expected_pack_bytes
    assert event["group_descriptors"] == [
        {
            "group_index": 0,
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "lda": 128,
            "ldb": 128,
            "ldc": 128,
            "task_indices": [0, 1],
            "task_count": 2,
            "execution": "batched",
            "flops": 8388608,
            "read_bytes": expected_pack_bytes,
            "write_bytes": 262144,
            "copy_bytes": expected_pack_bytes,
            "pack_strategy": "prepack_once",
            "kernel_calls": 1,
            "batched_matmul_provider": "numpy.matmul",
        },
    ]
    assert prepacked.profile["pack_bytes"] == event["pack_bytes"]
    assert prepacked.profile["lowering"] == event["lowering"]
    assert prepacked.profile["num_grouped_tasks"] == event["num_grouped_tasks"]
    assert prepacked.profile["num_shape_buckets"] == event["num_shape_buckets"]
    assert prepacked.profile["flops"] == event["flops"]
    assert prepacked.profile["execution_primitives"] == event["execution_primitives"]
    assert prepacked.profile["execution_policies"] == event["execution_policies"]
    assert prepacked.profile["fallback_reasons"] == event["fallback_reasons"]
    assert prepacked.profile["group_descriptors"] == event["group_descriptors"]


def test_prepack_grouped_gemm_execution_policy_can_force_loop_buckets():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.eye(128, dtype=np.float64)),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.eye(128, dtype=np.float64)),
    ]

    prepacked = backend.prepack_grouped_gemm(
        tasks,
        pack_threshold=2,
        policy="bucketed_loop_matmul",
    )

    assert [bucket.execution for bucket in prepacked.buckets] == ["loop"]
    assert prepacked.profile["requested_policy"] == "bucketed_loop_matmul"
    assert prepacked.profile["policy"] == "bucketed_loop_matmul"
    assert prepacked.profile["effective_allow_batched"] is False
    assert prepacked.profile["num_batched_gemm"] == 0
    assert prepacked.profile["num_gemm"] == 2
    assert prepacked.profile["batched_kernel_calls"] == 0
    assert prepacked.profile["loop_kernel_calls"] == 2
    assert prepacked.profile["bucket_execution_profiles"][0]["execution"] == "loop"
    assert prepacked.profile["bucket_execution_profiles"][0]["selection_reason"] == "loop: batched execution disabled"


def test_prepack_grouped_gemm_call_fallback_policy_overrides_backend_config():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    tasks = [
        GemmTask(np.ones((2, 3), dtype=np.float64), np.eye(3, 4, dtype=np.float64)),
        GemmTask(np.full((2, 3), 2.0, dtype=np.float64), np.eye(3, 4, dtype=np.float64)),
    ]
    strict_backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    prepacked = strict_backend.prepack_grouped_gemm(tasks, fallback_policy="record")

    assert prepacked.profile["fallback_policy"] == "record"
    assert prepacked.profile["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"

    record_backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))
    with pytest.raises(BackendFeatureError, match="backend-owned grouped_gemm unavailable"):
        record_backend.prepack_grouped_gemm(tasks, fallback_policy="forbid")


def test_prepack_grouped_gemm_profile_records_focus_buffer_table(tmp_path):
    from renormalizer.backend.gemm import BufferRef, BufferSlice, BufferTable, GemmBatch, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    a_buffer = np.arange(12, dtype=np.float64)
    b_buffer = np.arange(24, dtype=np.float64)
    c_buffer = np.zeros(16, dtype=np.float64)
    buffers = BufferTable.from_refs(
        [
            BufferRef.from_array("A", a_buffer, device="cpu"),
            BufferRef.from_array("B", b_buffer, device="cpu"),
            BufferRef.from_array("C", c_buffer, device="cpu"),
        ]
    )
    batch = GemmBatch.from_descs(
        [
            MatmulDesc(
                A=BufferSlice("A", offset=0, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=0, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=0, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
            ),
            MatmulDesc(
                A=BufferSlice("A", offset=6, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=12, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=8, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
            ),
        ]
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.prepack_grouped_gemm(batch, buffers=buffers, pack_threshold=2)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in events if payload["event"] == "grouped_gemm_prepack")

    assert event["descriptor_source"] == "focus_buffer_slice"
    assert event["buffer_names"] == ["A", "B", "C"]
    assert event["buffer_table"] == [
        {
            "name": "A",
            "shape": [12],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": a_buffer.nbytes,
        },
        {
            "name": "B",
            "shape": [24],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": b_buffer.nbytes,
        },
        {
            "name": "C",
            "shape": [16],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": c_buffer.nbytes,
        },
    ]


def test_execute_prepacked_grouped_gemm_writes_output_layout_profile_event(tmp_path):
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.eye(128, dtype=np.float64), tag="bucket-0"),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.eye(128, dtype=np.float64), tag="bucket-1"),
    ]
    prepacked = backend.prepack_grouped_gemm(tasks, pack_threshold=2)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.execute_prepacked_grouped_gemm(prepacked)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    assert [payload["event"] for payload in events] == ["contraction_execute", "grouped_gemm_execute"]
    contraction_event = events[0]
    event = events[1]

    expected_pack_bytes = sum(task.A.nbytes + task.B.nbytes for task in tasks)
    assert contraction_event["lowering"] == "grouped_gemm"
    assert contraction_event["pack_strategy"] == "prepacked_reuse"
    assert contraction_event["prepacked"] is True
    assert contraction_event["pack_bytes"] == 0
    assert contraction_event["reused_pack_bytes"] == expected_pack_bytes
    assert contraction_event["batched_matmul_provider"] == "numpy.matmul"
    assert contraction_event["bucket_execution_profiles"][0]["pack_strategy"] == "prepacked_reuse"
    assert contraction_event["bucket_execution_profiles"][0]["pack_bytes"] == 0
    assert event["pack_strategy"] == "prepacked_reuse"
    assert event["prepacked"] is True
    assert event["pack_bytes"] == 0
    assert event["group_descriptors"] == [
        {
            "group_index": 0,
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 128,
            "n": 128,
            "k": 128,
            "lda": 128,
            "ldb": 128,
            "ldc": 128,
            "task_indices": [0, 1],
            "task_count": 2,
            "execution": "batched",
            "flops": 8388608,
            "read_bytes": expected_pack_bytes,
            "write_bytes": 262144,
            "copy_bytes": 0,
            "pack_strategy": "prepacked_reuse",
            "kernel_calls": 1,
            "batched_matmul_provider": "numpy.matmul",
        },
    ]
    assert contraction_event["group_descriptors"] == event["group_descriptors"]
    assert event["output_shape"] == [[128, 128], [128, 128]]
    assert event["output_strides"] == [[1024, 8], [1024, 8]]
    assert event["output_order"] == ["C", "C"]
    assert event["output_contiguous"] == [True, True]
    assert event["output_backend"] == ["numpy", "numpy"]
    assert event["output_device_kind"] == ["cpu", "cpu"]
    assert event["output_location"] == ["host", "host"]
    assert event["layout_profile"]["output_layout"] == {
        "shape": [[128, 128], [128, 128]],
        "strides": [[1024, 8], [1024, 8]],
        "order": ["C", "C"],
        "contiguous": [True, True],
    }


def test_execute_prepacked_grouped_gemm_profile_preserves_focus_buffer_table(tmp_path):
    from renormalizer.backend.gemm import BufferRef, BufferSlice, BufferTable, GemmBatch, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    a0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    a1 = a0 + 10.0
    b0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    b1 = b0 + 20.0
    a_buffer = np.concatenate([a0.ravel(), a1.ravel()])
    b_buffer = np.concatenate([b0.ravel(), b1.ravel()])
    c_buffer = np.zeros(16, dtype=np.float64)
    buffers = BufferTable.from_refs(
        [
            BufferRef.from_array("A", a_buffer, device="cpu"),
            BufferRef.from_array("B", b_buffer, device="cpu"),
            BufferRef.from_array("C", c_buffer, device="cpu"),
        ]
    )
    batch = GemmBatch.from_descs(
        [
            MatmulDesc(
                A=BufferSlice("A", offset=0, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=0, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=0, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
            ),
            MatmulDesc(
                A=BufferSlice("A", offset=6, shape=(2, 3), leading_dim=3),
                B=BufferSlice("B", offset=12, shape=(3, 4), leading_dim=4),
                C=BufferSlice("C", offset=8, shape=(2, 4), leading_dim=4),
                trans_a="N",
                trans_b="N",
                m=2,
                n=4,
                k=3,
                lda=3,
                ldb=4,
                ldc=4,
                dtype=np.dtype("float64"),
            ),
        ]
    )
    prepacked = backend.prepack_grouped_gemm(batch, buffers=buffers, pack_threshold=2)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_prepacked_grouped_gemm(prepacked)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result[0], a0 @ b0)
    assert np.allclose(result[1], a1 @ b1)
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    assert [payload["event"] for payload in events] == ["contraction_execute", "grouped_gemm_execute"]
    contraction_event = events[0]
    event = events[1]

    assert contraction_event["descriptor_source"] == "focus_buffer_slice"
    assert contraction_event["buffer_names"] == ["A", "B", "C"]
    assert contraction_event["buffer_slice_descriptors"][1]["C_slice"] == {
        "buffer": "C",
        "offset": 8,
        "shape": [2, 4],
        "leading_dim": 4,
    }
    assert contraction_event["lowering"] == "grouped_gemm"
    assert contraction_event["pack_strategy"] == event["pack_strategy"]
    assert contraction_event["prepacked"] is True
    assert event["descriptor_source"] == "focus_buffer_slice"
    assert event["buffer_names"] == ["A", "B", "C"]
    assert event["buffer_slice_descriptors"] == contraction_event["buffer_slice_descriptors"]
    assert event["buffer_table"] == [
        {
            "name": "A",
            "shape": [12],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": a_buffer.nbytes,
        },
        {
            "name": "B",
            "shape": [24],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": b_buffer.nbytes,
        },
        {
            "name": "C",
            "shape": [16],
            "dtype": "float64",
            "device": "cpu",
            "nbytes": c_buffer.nbytes,
        },
    ]


def test_execute_grouped_gemm_plan_forbid_policy_rejects_bucketed_fallback():
    from renormalizer.backend import (
        BackendConfig,
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    spec = BlockContractionSpec(
        BlockTensor(
            {left_key: DenseBlock(left_key, np.ones((2, 3)), ("i", "k"), (2, 3))},
            global_shape=(2, 3),
            modes=("i", "k"),
            block_axis_meta=None,
            backend="numpy",
        ),
        BlockTensor(
            {right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4))},
            global_shape=(3, 4),
            modes=("k", "j"),
            block_axis_meta=None,
            backend="numpy",
        ),
        output_modes=("i", "j"),
        qn_rule=lambda left_block_key, right_block_key: BlockKey(
            left_block_key.qn_left,
            right_block_key.qn_right,
        ),
    )
    plan = backend.lower_block_contraction(spec)

    with pytest.raises(BackendFeatureError, match="backend-owned grouped_gemm unavailable"):
        backend.execute_grouped_gemm_plan(plan)


def test_execute_grouped_gemm_plan_call_policy_overrides_backend_config(tmp_path):
    from renormalizer.backend import (
        BackendConfig,
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        BlockKey((1,), (0,)): DenseBlock(BlockKey((1,), (0,)), np.full((2, 3), 2.0), ("i", "k"), (2, 3)),
    }
    right_key = BlockKey((0,), (1,))
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(4, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(
            {right_key: DenseBlock(right_key, np.eye(3, 4), ("k", "j"), (3, 4))},
            global_shape=(3, 4),
            modes=("k", "j"),
            block_axis_meta=None,
            backend="numpy",
        ),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: BlockKey(left_key.qn_left, right_key.qn_right),
    )
    plan = backend.lower_block_contraction(spec)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_grouped_gemm_plan(
            plan,
            pack_threshold=2,
            policy="bucketed_batched_matmul",
            fallback_policy="record",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert len(result.blocks) == 2
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload.get("event") == "contraction_execute"
        and payload.get("lowering") == "block_grouped_gemm"
    )
    assert event["grouped_gemm_policy"] == "bucketed_batched_matmul"
    assert event["execution_policies"] == ["bucketed_batched_matmul"]
    assert event["fallback_policy"] == "record"
    assert event["num_batched_gemm"] == 1
    assert event["num_gemm"] == 0
    assert event["batched_bucket_count"] == 1
    assert event["loop_bucket_count"] == 0
    assert event["effective_pack_threshold"] == 0
    assert event["effective_allow_batched"] is True


def test_execute_prepacked_grouped_gemm_forbid_policy_rejects_bucketed_fallback():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    record_backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))
    forbid_backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    tasks = [
        GemmTask(np.ones((2, 3)), np.ones((3, 4))),
        GemmTask(np.ones((2, 3)), np.ones((3, 4))),
    ]
    prepacked = record_backend.prepack_grouped_gemm(tasks, pack_threshold=2)

    with pytest.raises(BackendFeatureError, match="backend-owned grouped_gemm unavailable"):
        forbid_backend.execute_prepacked_grouped_gemm(prepacked)


def test_execute_prepacked_grouped_gemm_call_fallback_policy_overrides_backend_config(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    record_backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))
    strict_backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    tasks = [
        GemmTask(np.ones((2, 3), dtype=np.float64), np.eye(3, 4, dtype=np.float64)),
        GemmTask(np.full((2, 3), 2.0, dtype=np.float64), np.eye(3, 4, dtype=np.float64)),
    ]
    prepacked = record_backend.prepack_grouped_gemm(tasks, pack_threshold=2)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = strict_backend.execute_prepacked_grouped_gemm(
            prepacked,
            fallback_policy="record",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result[0], tasks[0].A @ tasks[0].B)
    assert np.allclose(result[1], tasks[1].A @ tasks[1].B)
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in events if payload["event"] == "grouped_gemm_execute")

    assert event["fallback_policy"] == "record"
    assert event["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert strict_backend.last_execution_profile()["fallback_policy"] == "record"


def test_empty_block_contraction_does_not_materialize_blocks_or_report_fallback(tmp_path, monkeypatch):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    spec = BlockContractionSpec(
        BlockTensor(
            {left_key: DenseBlock(left_key, np.ones((2, 3)), ("i", "k"), (2, 3))},
            global_shape=(2, 3),
            modes=("i", "k"),
            block_axis_meta={"left": True},
            backend="numpy",
        ),
        BlockTensor(
            {right_key: DenseBlock(right_key, np.ones((3, 4)), ("k", "j"), (3, 4))},
            global_shape=(3, 4),
            modes=("k", "j"),
            block_axis_meta={"right": True},
            backend="numpy",
        ),
        output_modes=("i", "j"),
        qn_rule=lambda _left_key, _right_key: None,
    )
    event_path = tmp_path / "events.jsonl"

    def grouped_gemm_must_not_run(*args, **kwargs):
        raise AssertionError("empty block contraction must not execute grouped_gemm")

    monkeypatch.setattr(backend, "grouped_gemm", grouped_gemm_must_not_run)
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.lower_block_contraction(spec)
        result = backend.execute_grouped_gemm_plan(plan, pack_threshold=2)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert plan.tasks == ()
    assert plan.output_blocks == ()
    assert plan.bucketed_by_shape == {}
    assert plan.scatter_add_required is False
    assert plan.estimated_flops == 0
    assert plan.estimated_read_bytes == 0
    assert plan.estimated_write_bytes == 0
    assert plan.estimated_workspace_bytes == 0
    assert plan.global_shape == (2, 4)
    assert result.blocks == {}
    assert result.global_shape == (2, 4)
    assert result.modes == ("i", "j")
    assert result.block_axis_meta == {"left": {"left": True}, "right": {"right": True}}

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_event = next(payload for payload in payloads if payload["event"] == "contraction_plan")
    execute_event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    for event in (plan_event, execute_event):
        assert event["lowering"] == "block_grouped_gemm"
        assert event["num_grouped_tasks"] == 0
        assert event["num_blocks"] == 0
        assert event["num_shape_buckets"] == 0
        assert event["flops"] == 0
        assert event["read_bytes"] == 0
        assert event["write_bytes"] == 0
        assert event["shape_buckets"] == []
        assert event["output_block_keys"] == []
        assert event["unique_output_block_keys"] == []
        assert event.get("fallback_reason") is None
    assert execute_event["result_block_shapes"] == []


def test_block_grouped_gemm_profile_overview_summarizes_execution_plan(caplog):
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    out_key = BlockKey((0,), (1,))
    left_blocks = {
        BlockKey((0,), (0,)): DenseBlock(BlockKey((0,), (0,)), np.ones((2, 3)), ("i", "k"), (2, 3)),
        BlockKey((0,), (2,)): DenseBlock(BlockKey((0,), (2,)), np.full((2, 3), 2.0), ("i", "k"), (2, 3)),
    }
    right_blocks = {
        BlockKey((0,), (1,)): DenseBlock(BlockKey((0,), (1,)), np.full((3, 4), 3.0), ("k", "j"), (3, 4)),
        BlockKey((2,), (1,)): DenseBlock(BlockKey((2,), (1,)), np.ones((3, 4)), ("k", "j"), (3, 4)),
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="numpy"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="numpy"),
        output_modes=("i", "j"),
        qn_rule=lambda left_key, right_key: (
            out_key
            if left_key.qn_right == right_key.qn_left
            else None
        ),
    )
    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")
    old_level = package_logger.level
    try:
        init_log(PROFILING)

        plan = backend.lower_block_contraction(spec)
        backend.execute_grouped_gemm_plan(plan, pack_threshold=2)
        profiling.flush_summaries()
    finally:
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(record.getMessage()[len(profiling.LOG_PREFIX):])
        for record in caplog.records
        if record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    plan_row = next(row for row in overview["classes"] if row["compute_class"] == "contraction_plan")
    assert plan_row["kernel_kinds"] == ["block_grouped_gemm"]
    assert plan_row["lowerings"] == ["block_grouped_gemm"]
    assert plan_row["total_grouped_tasks"] == 2
    assert plan_row["total_shape_buckets"] == 1
    assert plan_row["fallback_calls"] == 1
    assert plan_row["max_m"] == 2
    assert plan_row["max_n"] == 4
    assert plan_row["max_k"] == 3
    assert plan_row["compute_profile"]["estimated_cost"]["flops"] == 96
    assert plan_row["compute_profile"]["estimated_cost"]["read_bytes"] == 288
    assert plan_row["compute_profile"]["estimated_cost"]["write_bytes"] == 128
    assert plan_row["compute_profile"]["execution_breakdown"]["gemm_like_work_units"] == 2
    assert plan_row["compute_profile"]["execution_breakdown"]["batchable_work_units"] == 2
    assert plan_row["compute_profile"]["execution_breakdown"]["batchable_group_count"] == 1
    assert plan_row["compute_profile"]["execution_breakdown"]["independent_work_units"] == 1
    assert plan_row["compute_profile"]["workload_signature"]["shape_signature"] == (
        "lowering=block_grouped_gemm,gemm=2,batched=0,grouped=2,blocks=1,buckets=1"
    )
    assert plan_row["compute_profile"]["cost_factor_breakdown"]["factors"][0] == {
        "name": "backend_plan_fallback",
        "value": 1,
        "unit": "calls",
        "source": "fallback_calls",
        "resource": "missing_backend_kernel",
        "accounting": "primary",
    }
    kernel_mix = plan_row["compute_profile"]["core_compute_profile"]["kernel_mix"]
    grouped_item = next(item for item in kernel_mix if item["kind"] == "grouped_gemm_task")
    assert {
        "count": grouped_item["count"],
        "flops_estimate": grouped_item["flops_estimate"],
        "resource": grouped_item["resource"],
        "source": grouped_item["source"],
        "accounting": grouped_item["accounting"],
    } == {
        "count": 2,
        "flops_estimate": 96,
        "resource": "backend_grouped_kernel",
        "source": "num_grouped_tasks",
        "accounting": "primary",
    }
    bucket_item = next(item for item in kernel_mix if item["kind"] == "shape_bucket")
    assert {
        "count": bucket_item["count"],
        "resource": bucket_item["resource"],
        "source": bucket_item["source"],
        "accounting": bucket_item["accounting"],
    } == {
        "count": 1,
        "resource": "shape_grouping",
        "source": "num_shape_buckets",
        "accounting": "subset",
    }


def test_torch_block_contraction_lowering_accepts_tensor_size_method_when_available():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.config import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available("torch"):
        pytest.skip("torch unavailable")

    backend = create_backend("torch", config=BackendConfig(device="cpu"))
    left_key = BlockKey((0,), (0,))
    right_key = BlockKey((0,), (1,))
    left_array = backend.to_backend(np.ones((2, 3), dtype=np.float64))
    right_array = backend.to_backend(np.ones((3, 4), dtype=np.float64))
    spec = BlockContractionSpec(
        BlockTensor(
            {left_key: DenseBlock(left_key, left_array, ("i", "k"), (2, 3))},
            global_shape=(2, 3),
            modes=("i", "k"),
            block_axis_meta=None,
            backend="torch",
        ),
        BlockTensor(
            {right_key: DenseBlock(right_key, right_array, ("k", "j"), (3, 4))},
            global_shape=(3, 4),
            modes=("k", "j"),
            block_axis_meta=None,
            backend="torch",
        ),
        output_modes=("i", "j"),
        qn_rule=lambda left, right: BlockKey(left.qn_left, right.qn_right),
    )

    plan = backend.lower_block_contraction(spec)

    assert [(desc.m, desc.n, desc.k) for desc in plan.tasks] == [(2, 4, 3)]


def test_jax_grouped_gemm_plan_execution_is_not_implemented_in_np2_cupy_phase():
    from renormalizer.backend import (
        BlockContractionSpec,
        BlockKey,
        BlockTensor,
        DenseBlock,
    )
    from renormalizer.backend.config import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available("jax"):
        pytest.skip("jax unavailable")

    try:
        backend = create_backend("jax", config=BackendConfig(device="cpu"))
    except (ImportError, ValueError, RuntimeError) as exc:
        pytest.skip("jax backend unavailable: {0}".format(exc))

    out_key = BlockKey((0,), (1,))
    left_np = {
        BlockKey((0,), (0,)): np.ones((2, 3), dtype=np.float64),
        BlockKey((0,), (2,)): np.full((2, 3), 2.0, dtype=np.float64),
    }
    right_np = {
        BlockKey((0,), (1,)): np.full((3, 4), 3.0, dtype=np.float64),
        BlockKey((2,), (1,)): np.ones((3, 4), dtype=np.float64),
    }
    left_blocks = {
        key: DenseBlock(key, backend.to_backend(value), ("i", "k"), value.shape)
        for key, value in left_np.items()
    }
    right_blocks = {
        key: DenseBlock(key, backend.to_backend(value), ("k", "j"), value.shape)
        for key, value in right_np.items()
    }
    spec = BlockContractionSpec(
        BlockTensor(left_blocks, global_shape=(2, 3), modes=("i", "k"), block_axis_meta=None, backend="jax"),
        BlockTensor(right_blocks, global_shape=(3, 4), modes=("k", "j"), block_axis_meta=None, backend="jax"),
        output_modes=("i", "j"),
        qn_rule=lambda left, right: (
            BlockKey(left.qn_left, right.qn_right)
            if left.qn_right == right.qn_left
            else None
        ),
    )

    assert backend.supports_grouped_gemm is False
    with pytest.raises(NotImplementedError, match="JAX grouped_gemm is not implemented"):
        backend.execute_grouped_gemm_plan(backend.lower_block_contraction(spec), pack_threshold=2)


def test_execute_matmul_plan_runs_gemm_and_records_execute_event(tmp_path):
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("i", "k"), name="left"),
        TensorOperand(right, ("k", "j"), name="right"),
        output_modes=("i", "j"),
    )
    plan = backend.lower_pair_contraction_to_matmul(spec)
    assert isinstance(plan.plan_hash, str)
    assert plan.plan_hash
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_matmul_plan(plan)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    executes = [payload for payload in payloads if payload["event"] == "contraction_execute"]

    assert len(executes) == 1
    event = executes[0]
    assert event["backend"] == "numpy"
    assert event["lowering"] == "gemm"
    assert event["plan_hash"] == plan.plan_hash
    assert event["input_shapes"] == [[2, 3], [3, 4]]
    assert event["output_shape"] == [2, 4]
    assert event["dtype"] == "float64"
    assert event["flops"] == 48
    assert event["read_bytes"] == 144
    assert event["write_bytes"] == 64
    assert event["num_gemm"] == 1
    assert event["num_batched_gemm"] == 0
    assert event["fallback_reason"] is None
    assert event["wall_s"] >= 0.0


def test_execute_matmul_plan_exposes_latest_execution_profile():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("i", "k"), name="left"),
        TensorOperand(right, ("k", "j"), name="right"),
        output_modes=("i", "j"),
    )
    plan = backend.lower_pair_contraction_to_matmul(spec)

    old_level = package_logger.level
    try:
        init_log(PROFILING)
        result = backend.execute_matmul_plan(
            plan,
            equation="ik,kj->ij",
            input_modes=(("i", "k"), ("k", "j")),
            output_modes=("i", "j"),
        )
    finally:
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)
    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["backend"] == "numpy"
    assert profile["lowering"] == "gemm"
    assert profile["equation"] == "ik,kj->ij"
    assert profile["plan_hash"] == plan.plan_hash
    assert profile["input_shapes"] == [(2, 3), (3, 4)]
    assert profile["output_shape"] == (2, 4)
    assert profile["flops"] == 48
    assert profile["read_bytes"] == 144
    assert profile["write_bytes"] == 64
    assert profile["copy_bytes"] == 0
    assert profile["workspace_bytes"] == 0
    assert profile["fallback_reason"] is None
    assert profile["fallback_from"] is None
    assert profile["fallback_to"] is None
    assert profile["wall_s"] >= 0.0


def test_execute_matmul_plan_caches_profile_without_profiling_log_level():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils.log import DEBUG, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("i", "k"), name="left"),
        TensorOperand(right, ("k", "j"), name="right"),
        output_modes=("i", "j"),
    )
    plan = backend.lower_pair_contraction_to_matmul(spec)

    old_level = package_logger.level
    try:
        init_log(DEBUG)
        result = backend.execute_matmul_plan(plan, equation="ik,kj->ij")
    finally:
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)
    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "gemm"
    assert profile["equation"] == "ik,kj->ij"
    assert profile["plan_hash"] == plan.plan_hash
    assert profile["fallback_reason"] is None
    assert profile["wall_s"] >= 0.0
    assert profile["compute_profile"]["compute_class"] == "contraction_plan"
    assert profile["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    layout_profile = json.loads(json.dumps(profile["layout_profile"]))
    assert layout_profile["output_layout"] == {
        "shape": [2, 4],
        "strides": [32, 8],
        "order": "C",
        "contiguous": True,
    }
    assert profile["dtype_profile"]["output_dtype"] == "float64"
    assert profile["copy_profile"]["copy_required"] is False
    assert profile["workspace_profile"]["workspace_required"] is False
    assert profile["lowering_profile"]["lowering"] == "gemm"


def test_execute_fallback_batched_plan_records_fallback_source(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    right = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4)
    equation = "bik,bkj->bij"
    spec = backend.parse_einsum(equation, left, right)
    plan = backend.plan_contraction(spec)
    matmul_plan = plan.steps[0].plan
    assert matmul_plan.kind == "fallback_tensordot"

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_matmul_plan(
            matmul_plan,
            plan_hash=matmul_plan.plan_hash,
            equation=equation,
            input_modes=tuple(operand.modes for operand in spec.operands),
            output_modes=spec.output_modes,
        )
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, np.matmul(left, right))
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["lowering"] == "fallback_tensordot"
    assert event["fallback_from"] == "batched_gemm"
    assert event["fallback_to"] == "loop_matmul"
    assert event["fallback_reason"] == "backend lacks batched_matmul for batch shape (5,)"
    assert event["num_batched_gemm"] == 0


def test_execute_fallback_matmul_plan_uses_tensor_contraction_not_matmul(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoMatmulNamespace:
        def __init__(self, base):
            self._base = base
            self.matmul_calls = 0

        def __getattr__(self, name):
            return getattr(self._base, name)

        def matmul(self, *args, **kwargs):
            self.matmul_calls += 1
            raise AssertionError("fallback_tensordot must not call matmul")

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False
        supports_batched_matmul = False
        supports_strided_batched_gemm = False

        def __init__(self, config=None):
            super().__init__(config=config)
            self.array_namespace = NoMatmulNamespace(np)

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(2 * 3, dtype=np.float64).reshape(2, 3)
    right = np.arange(3 * 4, dtype=np.float64).reshape(3, 4)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("i", "k"), name="left"),
        TensorOperand(right, ("k", "j"), name="right"),
        output_modes=("i", "j"),
    )
    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert plan.kind == "fallback_tensordot"
    assert plan.fallback_reason == "backend lacks matmul"

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_matmul_plan(
            plan,
            equation="ik,kj->ij",
            input_modes=(("i", "k"), ("k", "j")),
            output_modes=("i", "j"),
        )
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert backend.array_namespace.matmul_calls == 0
    assert np.allclose(result, left @ right)
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["lowering"] == "fallback_tensordot"
    assert event["fallback_from"] == "gemm"
    assert event["fallback_to"] == "tensordot"
    assert event["fallback_reason"] == "backend lacks matmul"
    assert event["num_gemm"] == 0


@pytest.mark.parametrize("has_matmul", [True, False])
def test_execute_fallback_matmul_plan_preserves_alpha_beta_c_semantics(has_matmul):
    from renormalizer.backend import BackendConfig, LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    class MaybeMatmulBackend(NumpyBackend):
        supports_matmul = has_matmul

    backend = MaybeMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    c = np.full((2, 4), 2.0, dtype=np.float64)
    c_initial = c.copy()

    def layout(shape, modes, strides):
        return LayoutSpec(
            logical_shape=shape,
            physical_shape=shape,
            logical_modes=modes,
            strides=strides,
            order="C",
            contiguous_groups=(tuple(range(len(shape))),),
        )

    desc = MatmulDesc(
        left,
        right,
        c,
        2,
        4,
        3,
        alpha=0.5,
        beta=3.0,
        layout_a=layout(left.shape, ("i", "k"), left.strides),
        layout_b=layout(right.shape, ("k", "j"), right.strides),
        layout_c=layout(c.shape, ("i", "j"), c.strides),
        estimated_flops=48,
        estimated_read_bytes=left.nbytes + right.nbytes,
        estimated_write_bytes=c.nbytes,
    )
    plan = MatmulPlan(
        kind="fallback_tensordot",
        descs=(desc,),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=48,
        estimated_time_s=None,
        reason="forced fallback for alpha beta C test",
        fallback_reason="forced fallback for alpha beta C test",
    )

    result = backend.execute_matmul_plan(plan)

    expected = 0.5 * (left @ right) + 3.0 * c_initial
    assert result is c
    assert np.allclose(c, expected)


def test_batched_fallback_without_matmul_lowers_and_profiles_as_einsum(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoMatmulNamespace:
        def __init__(self, base):
            self._base = base
            self.matmul_calls = 0

        def __getattr__(self, name):
            return getattr(self._base, name)

        def matmul(self, *args, **kwargs):
            self.matmul_calls += 1
            raise AssertionError("fallback_einsum must not call matmul")

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False
        supports_batched_matmul = False
        supports_strided_batched_gemm = False

        def __init__(self, config=None):
            super().__init__(config=config)
            self.array_namespace = NoMatmulNamespace(np)

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    right = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4)
    equation = "bik,bkj->bij"
    spec = backend.parse_einsum(equation, left, right)
    plan = backend.plan_contraction(spec)
    matmul_plan = plan.steps[0].plan

    assert matmul_plan.kind == "fallback_einsum"
    assert matmul_plan.fallback_reason == "backend lacks batched_matmul and matmul for batch shape (5,)"

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute(plan)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert backend.array_namespace.matmul_calls == 0
    assert np.allclose(result, np.matmul(left, right))
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["lowering"] == "fallback_einsum"
    assert event["fallback_from"] == "batched_gemm"
    assert event["fallback_to"] == "einsum"
    assert event["fallback_reason"] == matmul_plan.fallback_reason
    assert event["num_batched_gemm"] == 0


def test_execute_grouped_matmul_plan_records_all_descriptor_shapes(tmp_path):
    from renormalizer.backend import LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    right0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    left1 = left0 + 10.0
    right1 = right0 - 2.0

    def layout(shape, modes, strides):
        return LayoutSpec(
            logical_shape=shape,
            physical_shape=shape,
            logical_modes=modes,
            strides=strides,
            order="C",
            contiguous_groups=(tuple(range(len(shape))),),
        )

    layout_a = layout((2, 3), ("i", "k"), left0.strides)
    layout_b = layout((3, 4), ("k", "j"), right0.strides)
    layout_c = layout((2, 4), ("i", "j"), None)
    descs = (
        MatmulDesc(
            left0,
            right0,
            None,
            2,
            4,
            3,
            dtype_compute=np.float64,
            dtype_output=np.float64,
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
            estimated_flops=48,
            estimated_read_bytes=left0.nbytes + right0.nbytes,
            estimated_write_bytes=2 * 4 * left0.itemsize,
        ),
        MatmulDesc(
            left1,
            right1,
            None,
            2,
            4,
            3,
            dtype_compute=np.float64,
            dtype_output=np.float64,
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
            estimated_flops=48,
            estimated_read_bytes=left1.nbytes + right1.nbytes,
            estimated_write_bytes=2 * 4 * left1.itemsize,
        ),
    )
    plan = MatmulPlan(
        kind="grouped_gemm",
        descs=descs,
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=96,
        estimated_time_s=None,
        reason="test grouped matmul plan",
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_matmul_plan(plan, plan_hash="grouped-plan")
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert len(result) == 2
    assert np.allclose(result[0], left0 @ right0)
    assert np.allclose(result[1], left1 @ right1)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload.get("plan_hash") == "grouped-plan")

    assert event["lowering"] == "grouped_gemm"
    assert event["input_shapes"] == [[[2, 3], [3, 4]], [[2, 3], [3, 4]]]
    assert event["input_dtypes"] == [["float64", "float64"], ["float64", "float64"]]
    assert event["output_shape"] == [[2, 4], [2, 4]]
    assert event["num_grouped_tasks"] == 2
    assert event["num_shape_buckets"] == 1
    assert event["fallback_from"] == "grouped_gemm"
    assert event["fallback_to"] == "bucketed_loop_matmul"
    assert event["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
    assert event["bucket_task_counts"] == [2]
    assert event["shape_buckets"] == [
        {"m": 2, "n": 4, "k": 3, "task_indices": [0, 1], "task_count": 2}
    ]
    assert event["group_sizes"] == [2]
    assert event["gsta"] == [0, 2]
    assert event["sorted_indices"] == [0, 1]
    assert event["group_keys"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
        }
    ]
    assert event["group_key_buckets"] == [
        {
            "dtype": "float64",
            "trans_a": "N",
            "trans_b": "N",
            "conj_a": False,
            "conj_b": False,
            "batch_shape": [],
            "m": 2,
            "n": 4,
            "k": 3,
            "lda": 3,
            "ldb": 4,
            "ldc": 4,
            "task_indices": [0, 1],
            "task_count": 2,
        }
    ]
    assert event["task_specs"] == [
        {
            "index": 0,
            "m": 2,
            "n": 4,
            "k": 3,
            "batch_shape": [],
            "trans_a": False,
            "trans_b": False,
            "conj_a": False,
            "conj_b": False,
            "alpha": 1.0,
            "beta": 0.0,
            "dtype_compute": "<class 'numpy.float64'>",
            "dtype_output": "<class 'numpy.float64'>",
            "estimated_flops": 48,
            "estimated_read_bytes": 144,
            "estimated_write_bytes": 64,
            "estimated_workspace_bytes": 0,
            "layout_a": {
                "logical_shape": [2, 3],
                "physical_shape": [2, 3],
                "logical_modes": ["i", "k"],
                "strides": [24, 8],
                "order": "C",
                "contiguous_groups": [[0, 1]],
                "requires_transpose": False,
                "transpose_perm": None,
                "estimated_copy_bytes": 0,
            },
            "layout_b": {
                "logical_shape": [3, 4],
                "physical_shape": [3, 4],
                "logical_modes": ["k", "j"],
                "strides": [32, 8],
                "order": "C",
                "contiguous_groups": [[0, 1]],
                "requires_transpose": False,
                "transpose_perm": None,
                "estimated_copy_bytes": 0,
            },
            "layout_c": {
                "logical_shape": [2, 4],
                "physical_shape": [2, 4],
                "logical_modes": ["i", "j"],
                "strides": None,
                "order": "C",
                "contiguous_groups": [[0, 1]],
                "requires_transpose": False,
                "transpose_perm": None,
                "estimated_copy_bytes": 0,
            },
        },
        {
            "index": 1,
            "m": 2,
            "n": 4,
            "k": 3,
            "batch_shape": [],
            "trans_a": False,
            "trans_b": False,
            "conj_a": False,
            "conj_b": False,
            "alpha": 1.0,
            "beta": 0.0,
            "dtype_compute": "<class 'numpy.float64'>",
            "dtype_output": "<class 'numpy.float64'>",
            "estimated_flops": 48,
            "estimated_read_bytes": 144,
            "estimated_write_bytes": 64,
            "estimated_workspace_bytes": 0,
            "layout_a": {
                "logical_shape": [2, 3],
                "physical_shape": [2, 3],
                "logical_modes": ["i", "k"],
                "strides": [24, 8],
                "order": "C",
                "contiguous_groups": [[0, 1]],
                "requires_transpose": False,
                "transpose_perm": None,
                "estimated_copy_bytes": 0,
            },
            "layout_b": {
                "logical_shape": [3, 4],
                "physical_shape": [3, 4],
                "logical_modes": ["k", "j"],
                "strides": [32, 8],
                "order": "C",
                "contiguous_groups": [[0, 1]],
                "requires_transpose": False,
                "transpose_perm": None,
                "estimated_copy_bytes": 0,
            },
            "layout_c": {
                "logical_shape": [2, 4],
                "physical_shape": [2, 4],
                "logical_modes": ["i", "j"],
                "strides": None,
                "order": "C",
                "contiguous_groups": [[0, 1]],
                "requires_transpose": False,
                "transpose_perm": None,
                "estimated_copy_bytes": 0,
            },
        },
    ]


def test_execute_grouped_matmul_plan_call_fallback_policy_forbids_bucketed_fallback():
    from renormalizer.backend import (
        BackendConfig,
        BackendFeatureError,
        LayoutSpec,
        MatmulDesc,
        MatmulPlan,
    )
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    layout_a = LayoutSpec(
        logical_shape=(2, 3),
        physical_shape=(2, 3),
        logical_modes=("i", "k"),
        strides=left.strides,
        order="C",
        contiguous_groups=((0, 1),),
    )
    layout_b = LayoutSpec(
        logical_shape=(3, 4),
        physical_shape=(3, 4),
        logical_modes=("k", "j"),
        strides=right.strides,
        order="C",
        contiguous_groups=((0, 1),),
    )
    layout_c = LayoutSpec(
        logical_shape=(2, 4),
        physical_shape=(2, 4),
        logical_modes=("i", "j"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1),),
    )
    plan = MatmulPlan(
        kind="grouped_gemm",
        descs=(
            MatmulDesc(
                left,
                right,
                None,
                2,
                4,
                3,
                dtype_compute=np.float64,
                dtype_output=np.float64,
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c,
                estimated_flops=48,
                estimated_read_bytes=left.nbytes + right.nbytes,
                estimated_write_bytes=2 * 4 * left.itemsize,
            ),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=48,
        estimated_time_s=None,
        reason="test grouped matmul plan fallback policy",
    )

    with pytest.raises(BackendFeatureError, match="backend-owned grouped_gemm unavailable"):
        backend.execute_matmul_plan(plan, fallback_policy="forbid")


def test_execute_gemm_matmul_plan_call_fallback_policy_forbids_primitive_fallback():
    from renormalizer.backend import BackendConfig, BackendFeatureError, LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    layout_a = LayoutSpec(
        logical_shape=(2, 3),
        physical_shape=(2, 3),
        logical_modes=("i", "k"),
        strides=left.strides,
        order="C",
        contiguous_groups=((0, 1),),
    )
    layout_b = LayoutSpec(
        logical_shape=(3, 4),
        physical_shape=(3, 4),
        logical_modes=("k", "j"),
        strides=right.strides,
        order="C",
        contiguous_groups=((0, 1),),
    )
    layout_c = LayoutSpec(
        logical_shape=(2, 4),
        physical_shape=(2, 4),
        logical_modes=("i", "j"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1),),
    )
    plan = MatmulPlan(
        kind="gemm",
        descs=(
            MatmulDesc(
                left,
                right,
                None,
                2,
                4,
                3,
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c,
                estimated_flops=48,
                estimated_read_bytes=left.nbytes + right.nbytes,
                estimated_write_bytes=2 * 4 * left.itemsize,
            ),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=48,
        estimated_time_s=None,
        reason="test gemm matmul plan fallback policy",
    )

    with pytest.raises(BackendFeatureError, match="backend lacks matmul"):
        backend.execute_matmul_plan(plan, fallback_policy="forbid")


def test_execute_batched_matmul_plan_call_fallback_policy_forbids_primitive_fallback():
    from renormalizer.backend import BackendConfig, BackendFeatureError, LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    right = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4)
    layout_a = LayoutSpec(
        logical_shape=(5, 2, 3),
        physical_shape=(5, 2, 3),
        logical_modes=("batch", "i", "k"),
        strides=left.strides,
        order="C",
        contiguous_groups=((0, 1, 2),),
    )
    layout_b = LayoutSpec(
        logical_shape=(5, 3, 4),
        physical_shape=(5, 3, 4),
        logical_modes=("batch", "k", "j"),
        strides=right.strides,
        order="C",
        contiguous_groups=((0, 1, 2),),
    )
    layout_c = LayoutSpec(
        logical_shape=(5, 2, 4),
        physical_shape=(5, 2, 4),
        logical_modes=("batch", "i", "j"),
        strides=None,
        order="C",
        contiguous_groups=((0, 1, 2),),
    )
    plan = MatmulPlan(
        kind="batched_gemm",
        descs=(
            MatmulDesc(
                left,
                right,
                None,
                2,
                4,
                3,
                batch_shape=(5,),
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c,
                estimated_flops=5 * 48,
                estimated_read_bytes=left.nbytes + right.nbytes,
                estimated_write_bytes=5 * 2 * 4 * left.itemsize,
            ),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(5, 2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=5 * 48,
        estimated_time_s=None,
        reason="test batched matmul plan fallback policy",
    )

    with pytest.raises(BackendFeatureError, match="backend lacks batched_matmul"):
        backend.execute_matmul_plan(plan, fallback_policy="forbid")


def test_execute_gemm_matmul_plan_records_primitive_fallback_in_aggregate_profile():
    from renormalizer.backend import BackendConfig, LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = MatmulPlan(
        kind="gemm",
        descs=(
            MatmulDesc(
                left,
                right,
                None,
                2,
                4,
                3,
                layout_a=LayoutSpec(
                    logical_shape=(2, 3),
                    physical_shape=(2, 3),
                    logical_modes=("i", "k"),
                    strides=left.strides,
                    order="C",
                    contiguous_groups=((0, 1),),
                ),
                layout_b=LayoutSpec(
                    logical_shape=(3, 4),
                    physical_shape=(3, 4),
                    logical_modes=("k", "j"),
                    strides=right.strides,
                    order="C",
                    contiguous_groups=((0, 1),),
                ),
                layout_c=LayoutSpec(
                    logical_shape=(2, 4),
                    physical_shape=(2, 4),
                    logical_modes=("i", "j"),
                    strides=None,
                    order="C",
                    contiguous_groups=((0, 1),),
                ),
                estimated_flops=48,
                estimated_read_bytes=left.nbytes + right.nbytes,
                estimated_write_bytes=2 * 4 * left.itemsize,
            ),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=48,
        estimated_time_s=None,
        reason="test gemm aggregate primitive fallback profile",
    )

    result = backend.execute_matmul_plan(plan, fallback_policy="record")

    assert np.allclose(result, left @ right)
    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "fallback_tensordot"
    assert profile["fallback_reason"] == "backend lacks matmul"
    assert profile["fallback_from"] == "gemm"
    assert profile["fallback_to"] == "tensordot"
    assert profile["fallback_policy"] == "record"
    assert profile["fallback_reasons"] == ["backend lacks matmul"]
    assert profile["execution_primitives"] == ["tensordot"]
    assert profile["execution_policies"] == ["fallback_tensordot"]
    assert profile["num_gemm"] == 0
    assert profile["num_batched_gemm"] == 0
    assert profile["lowering_profile"]["silent_fallback"] is False


def test_execute_batched_matmul_plan_records_primitive_fallback_in_aggregate_profile():
    from renormalizer.backend import BackendConfig, LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    right = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4)
    plan = MatmulPlan(
        kind="batched_gemm",
        descs=(
            MatmulDesc(
                left,
                right,
                None,
                2,
                4,
                3,
                batch_shape=(5,),
                layout_a=LayoutSpec(
                    logical_shape=(5, 2, 3),
                    physical_shape=(5, 2, 3),
                    logical_modes=("batch", "i", "k"),
                    strides=left.strides,
                    order="C",
                    contiguous_groups=((0, 1, 2),),
                ),
                layout_b=LayoutSpec(
                    logical_shape=(5, 3, 4),
                    physical_shape=(5, 3, 4),
                    logical_modes=("batch", "k", "j"),
                    strides=right.strides,
                    order="C",
                    contiguous_groups=((0, 1, 2),),
                ),
                layout_c=LayoutSpec(
                    logical_shape=(5, 2, 4),
                    physical_shape=(5, 2, 4),
                    logical_modes=("batch", "i", "j"),
                    strides=None,
                    order="C",
                    contiguous_groups=((0, 1, 2),),
                ),
                estimated_flops=5 * 48,
                estimated_read_bytes=left.nbytes + right.nbytes,
                estimated_write_bytes=5 * 2 * 4 * left.itemsize,
            ),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(5, 2, 4),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=5 * 48,
        estimated_time_s=None,
        reason="test batched aggregate primitive fallback profile",
    )

    result = backend.execute_matmul_plan(plan, fallback_policy="record")

    assert np.allclose(result, np.matmul(left, right))
    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "fallback_tensordot"
    assert profile["fallback_reason"] == "backend lacks batched_matmul for batch shape (5,)"
    assert profile["fallback_from"] == "batched_gemm"
    assert profile["fallback_to"] == "loop_matmul"
    assert profile["fallback_policy"] == "record"
    assert profile["fallback_reasons"] == ["backend lacks batched_matmul for batch shape (5,)"]
    assert profile["execution_primitives"] == ["loop_matmul"]
    assert profile["execution_policies"] == ["fallback_loop_matmul"]
    assert profile["num_gemm"] == 0
    assert profile["num_batched_gemm"] == 0
    assert profile["lowering_profile"]["silent_fallback"] is False


def test_estimate_grouped_matmul_plan_peak_counts_all_outputs():
    from renormalizer.backend import MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    right0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    left1 = left0 + 10.0
    right1 = right0 - 2.0
    descs = (
        MatmulDesc(
            left0,
            right0,
            None,
            2,
            4,
            3,
            estimated_flops=48,
            estimated_read_bytes=left0.nbytes + right0.nbytes,
            estimated_write_bytes=2 * 4 * left0.itemsize,
        ),
        MatmulDesc(
            left1,
            right1,
            None,
            2,
            4,
            3,
            estimated_flops=48,
            estimated_read_bytes=left1.nbytes + right1.nbytes,
            estimated_write_bytes=2 * 4 * left1.itemsize,
        ),
    )
    plan = MatmulPlan(
        kind="grouped_gemm",
        descs=descs,
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=32,
        estimated_flops=96,
        estimated_time_s=None,
        reason="test grouped matmul plan",
    )

    estimate = backend.estimate_matmul(plan)

    assert estimate.write_bytes == 2 * 2 * 4 * left0.itemsize
    assert estimate.workspace_bytes == 32
    assert estimate.peak_bytes == estimate.write_bytes + estimate.workspace_bytes


def test_estimate_grouped_gemm_contraction_includes_plan_copy_bytes():
    from renormalizer.backend import GroupedGemmPlan, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    kwargs = _valid_grouped_gemm_plan_kwargs()
    kwargs["estimated_copy_bytes"] = 48
    kwargs["estimated_workspace_bytes"] = 32
    plan = GroupedGemmPlan(**kwargs)

    estimate = backend.estimate_contraction(plan, HardwareModel(p2p_bandwidth_Bps=24.0))

    assert estimate.copy_bytes == 48
    assert estimate.copy_s == pytest.approx(2.0)
    assert estimate.workspace_bytes == 32
    assert estimate.peak_bytes == plan.estimated_write_bytes + 48 + 32


def test_estimate_matmul_peak_includes_layout_copy_workspace_and_outputs():
    from renormalizer.backend import BackendFeatureError, HardwareModel, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    right0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    left1 = left0 + 10.0
    right1 = right0 - 2.0
    descs = (
        MatmulDesc(
            left0,
            right0,
            None,
            2,
            4,
            3,
            estimated_flops=48,
            estimated_read_bytes=left0.nbytes + right0.nbytes,
            estimated_write_bytes=2 * 4 * left0.itemsize,
        ),
        MatmulDesc(
            left1,
            right1,
            None,
            2,
            4,
            3,
            estimated_flops=48,
            estimated_read_bytes=left1.nbytes + right1.nbytes,
            estimated_write_bytes=2 * 4 * left1.itemsize,
        ),
    )
    plan = MatmulPlan(
        kind="grouped_gemm",
        descs=descs,
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=64,
        workspace_bytes=32,
        estimated_flops=96,
        estimated_time_s=None,
        reason="test grouped matmul plan with layout copies",
    )
    expected_write = 2 * 2 * 4 * left0.itemsize
    expected_peak = expected_write + plan.copy_bytes + plan.workspace_bytes

    estimate = backend.estimate_matmul(plan, HardwareModel(p2p_bandwidth_Bps=32.0))

    assert estimate.write_bytes == expected_write
    assert estimate.copy_bytes == 64
    assert estimate.workspace_bytes == 32
    assert estimate.peak_bytes == expected_peak
    assert estimate.copy_s == pytest.approx(2.0)
    with pytest.raises(BackendFeatureError, match="max_memory.*223.*peak.*224"):
        backend.estimate_matmul(plan, HardwareModel(max_memory_bytes=expected_peak - 1))


def test_estimate_matmul_desc_accounts_for_layout_copy_bytes():
    from renormalizer.backend import BackendFeatureError, HardwareModel, LayoutSpec, MatmulDesc
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    def layout(shape, modes, strides, copy_bytes):
        return LayoutSpec(
            logical_shape=shape,
            physical_shape=shape,
            logical_modes=modes,
            strides=strides,
            order="C",
            contiguous_groups=(tuple(range(len(shape))),),
            estimated_copy_bytes=copy_bytes,
        )

    desc = MatmulDesc(
        left,
        right,
        None,
        2,
        4,
        3,
        layout_a=layout(left.shape, ("i", "k"), left.strides, 48),
        layout_b=layout(right.shape, ("k", "j"), right.strides, 96),
        layout_c=layout((2, 4), ("i", "j"), None, 0),
        estimated_flops=48,
        estimated_read_bytes=left.nbytes + right.nbytes,
        estimated_write_bytes=2 * 4 * left.itemsize,
        estimated_workspace_bytes=32,
    )
    expected_copy = 48 + 96
    expected_peak = desc.estimated_write_bytes + expected_copy + desc.estimated_workspace_bytes

    estimate = backend.estimate_matmul(desc, HardwareModel(p2p_bandwidth_Bps=48.0))

    assert estimate.copy_bytes == expected_copy
    assert estimate.workspace_bytes == 32
    assert estimate.peak_bytes == expected_peak
    assert estimate.copy_s == pytest.approx(3.0)
    with pytest.raises(BackendFeatureError, match="max_memory.*239.*peak.*240"):
        backend.estimate_matmul(desc, HardwareModel(max_memory_bytes=expected_peak - 1))


def test_contraction_step_from_matmul_plan_peak_counts_all_outputs():
    from renormalizer.backend import MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    left0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    right0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    left1 = left0 + 10.0
    right1 = right0 - 2.0
    descs = (
        MatmulDesc(left0, right0, None, 2, 4, 3, estimated_write_bytes=2 * 4 * left0.itemsize),
        MatmulDesc(left1, right1, None, 2, 4, 3, estimated_write_bytes=2 * 4 * left1.itemsize),
    )
    plan = MatmulPlan(
        kind="grouped_gemm",
        descs=descs,
        pre_ops=(),
        post_ops=(),
        output_shape=(2, 4),
        copy_bytes=0,
        workspace_bytes=32,
        estimated_flops=96,
        estimated_time_s=None,
        reason="test grouped matmul plan",
    )

    step = NumpyBackend._contraction_step_from_matmul_plan(
        plan,
        (("i", "k"), ("k", "j")),
        ("i", "j"),
    )

    assert step.estimated_write_bytes == 2 * 2 * 4 * left0.itemsize
    assert step.required_workspace_bytes == 32
    assert step.estimated_peak_bytes == step.estimated_write_bytes + step.required_workspace_bytes


def test_execute_batched_matmul_plan_preserves_generic_output_mode_order():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    right = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("batch", "i", "k"), name="left"),
        TensorOperand(right, ("batch", "k", "j"), name="right"),
        output_modes=("j", "batch", "i"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)
    result = backend.execute_matmul_plan(plan)
    expected = np.einsum("bik,bkj->bij", left, right).transpose(2, 0, 1)

    assert plan.kind == "strided_batched_gemm"
    assert result.shape == (5, 2, 3)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "kind,left_shape,right_shape,output_shape,left_modes,right_modes,output_modes,expected_fn",
    [
        (
            "gemm",
            (2, 3),
            (3, 4),
            (4, 2),
            ("i", "k"),
            ("k", "j"),
            ("j", "i"),
            lambda left, right: (left @ right).T,
        ),
        (
            "batched_gemm",
            (2, 3, 4),
            (2, 4, 5),
            (5, 2, 3),
            ("batch", "i", "k"),
            ("batch", "k", "j"),
            ("j", "batch", "i"),
            lambda left, right: np.matmul(left, right).transpose(2, 0, 1),
        ),
    ],
)
def test_execute_matmul_plan_accumulates_c_after_output_layout_finalize(
    kind,
    left_shape,
    right_shape,
    output_shape,
    left_modes,
    right_modes,
    output_modes,
    expected_fn,
):
    from renormalizer.backend import LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(np.prod(left_shape), dtype=np.float64).reshape(left_shape)
    right = np.arange(np.prod(right_shape), dtype=np.float64).reshape(right_shape)
    c = np.full(output_shape, 2.0, dtype=np.float64)
    c_initial = c.copy()

    def layout(shape, modes, strides):
        return LayoutSpec(
            logical_shape=shape,
            physical_shape=shape,
            logical_modes=modes,
            strides=strides,
            order="C",
            contiguous_groups=(tuple(range(len(shape))),),
        )

    desc = MatmulDesc(
        left,
        right,
        c,
        m=left_shape[-2],
        n=right_shape[-1],
        k=left_shape[-1],
        batch_shape=left_shape[:-2],
        alpha=0.5,
        beta=3.0,
        layout_a=layout(left.shape, left_modes, left.strides),
        layout_b=layout(right.shape, right_modes, right.strides),
        layout_c=layout(c.shape, output_modes, c.strides),
    )
    plan = MatmulPlan(
        kind=kind,
        descs=(desc,),
        pre_ops=(),
        post_ops=(),
        output_shape=output_shape,
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=1,
        estimated_time_s=None,
        reason="test layout-aware C accumulation",
    )

    result = backend.execute_matmul_plan(plan)

    expected = 0.5 * expected_fn(left, right) + 3.0 * c_initial
    assert result is c
    assert np.allclose(c, expected)


def test_execute_grouped_matmul_plan_accumulates_c_after_output_layout_finalize():
    from renormalizer.backend import LayoutSpec, MatmulDesc, MatmulPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    right0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    left1 = left0 + 10.0
    right1 = right0 - 2.0
    c0 = np.full((4, 2), 2.0, dtype=np.float64)
    c1 = np.full((4, 2), -1.0, dtype=np.float64)
    c0_initial = c0.copy()
    c1_initial = c1.copy()

    def layout(shape, modes, strides):
        return LayoutSpec(
            logical_shape=shape,
            physical_shape=shape,
            logical_modes=modes,
            strides=strides,
            order="C",
            contiguous_groups=(tuple(range(len(shape))),),
        )

    def desc(left, right, c, alpha, beta):
        return MatmulDesc(
            left,
            right,
            c,
            m=2,
            n=4,
            k=3,
            alpha=alpha,
            beta=beta,
            layout_a=layout(left.shape, ("i", "k"), left.strides),
            layout_b=layout(right.shape, ("k", "j"), right.strides),
            layout_c=layout(c.shape, ("j", "i"), c.strides),
        )

    plan = MatmulPlan(
        kind="grouped_gemm",
        descs=(
            desc(left0, right0, c0, 0.5, 3.0),
            desc(left1, right1, c1, 2.0, -4.0),
        ),
        pre_ops=(),
        post_ops=(),
        output_shape=(4, 2),
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=96,
        estimated_time_s=None,
        reason="test grouped layout-aware C accumulation",
    )

    result = backend.execute_matmul_plan(plan, pack_threshold=2)

    assert result == [c0, c1]
    assert np.allclose(c0, 0.5 * (left0 @ right0).T + 3.0 * c0_initial)
    assert np.allclose(c1, 2.0 * (left1 @ right1).T - 4.0 * c1_initial)


def test_gemm_primitives_execute_inside_backend_stream_context():
    from contextlib import contextmanager

    from renormalizer.backend import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    class StreamCheckingNamespace:
        def __init__(self, base, backend):
            self._base = base
            self._backend = backend

        def __getattr__(self, name):
            return getattr(self._base, name)

        def matmul(self, left, right):
            assert self._backend._inside_stream_context
            return self._base.matmul(left, right)

    class StreamAwareNumpyBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self._inside_stream_context = False
            self.stream_contexts = []
            self.array_namespace = StreamCheckingNamespace(np, self)

        @contextmanager
        def _stream_context(self, stream):
            self.stream_contexts.append(stream)
            old = self._inside_stream_context
            self._inside_stream_context = True
            try:
                yield
            finally:
                self._inside_stream_context = old

    backend = StreamAwareNumpyBackend()
    stream = object()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    batch_left = np.stack((left, left + 1.0), axis=0)
    batch_right = np.stack((right, right - 1.0), axis=0)

    assert np.allclose(backend.matmul(left, right, stream=stream), left @ right)
    assert np.allclose(backend.batched_matmul(batch_left, batch_right, stream=stream), np.matmul(batch_left, batch_right))
    grouped = backend.grouped_gemm(
        [GemmTask(left, right), GemmTask(left + 2.0, right + 3.0)],
        pack_threshold=1,
        stream=stream,
    )
    assert np.allclose(grouped[0], left @ right)
    assert np.allclose(grouped[1], (left + 2.0) @ (right + 3.0))
    assert backend.stream_contexts == [stream, stream, stream]


def test_backend_contraction_plan_event_records_generic_operands(tmp_path):
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    left = np.ones((2, 3, 4), dtype=np.float32)
    right = np.ones((2, 4, 5), dtype=np.float32)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("batch", ("left", "site"), "bond"), name="left_tensor"),
        TensorOperand(right, ("batch", "bond", ("right", "site")), name="right_tensor"),
        output_modes=("batch", ("left", "site"), ("right", "site")),
    )
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.lower_pair_contraction_to_matmul(spec)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan = next(payload for payload in payloads if payload["event"] == "contraction_plan")

    assert plan["equation"] == "batch('left', 'site')bond,batchbond('right', 'site')->batch('left', 'site')('right', 'site')"
    assert isinstance(plan["plan_hash"], str)
    assert plan["plan_hash"]
    assert plan["lowering"] == "strided_batched_gemm"
    assert plan["compute_profile"]["schema"] == "renormalizer.compute_profile.v1"
    assert plan["compute_profile"]["event_scope"] == "plan"
    assert plan["compute_profile"]["compute_class"] == "contraction_plan"
    assert plan["compute_profile"]["measurement_scope"] == "contraction_plan"
    assert plan["compute_profile"]["primary_kernel"] == "strided_batched_gemm"
    assert plan["compute_profile"]["estimated_cost"]["flops"] == plan["flops"]
    assert plan["compute_profile"]["estimated_cost"]["working_set_bytes"] == (
        plan["read_bytes"] + plan["write_bytes"] + plan["copy_bytes"]
    )
    assert plan["compute_profile"]["work"] == {
        "unit": "contraction_plan",
        "lowering": "strided_batched_gemm",
        "num_gemm": 0,
        "num_batched_gemm": 1,
        "num_grouped_tasks": 0,
        "num_blocks": 0,
        "num_shape_buckets": 0,
    }
    assert plan["compute_profile"]["parallelism"] == {
        "unit": "planned_backend_lowering",
        "backend_kernel": "strided_batched_gemm",
        "batched_kernel": True,
        "grouped_kernel": False,
        "distributed": False,
    }
    assert plan["batch_modes"] == ["batch"]
    assert plan["output_modes"] == ["batch", "('left', 'site')", "('right', 'site')"]
    assert plan["operands"] == [
        {
            "name": "left_tensor",
            "modes": ["batch", "('left', 'site')", "bond"],
            "shape": [2, 3, 4],
            "dtype": "float32",
            "itemsize": 4,
            "size": 24,
            "nbytes": 96,
            "ndim": 3,
            "strides": [48, 16, 4],
            "order": "C",
            "contiguous": True,
            "writeable": True,
            "owns_data": True,
            "backend": "numpy",
            "device": "DeviceSpec(kind='cpu', index=None, local_rank=None, global_rank=None, visible_id=None)",
            "device_kind": "cpu",
            "device_index": None,
            "is_host": True,
            "is_device": False,
            "is_distributed": False,
        },
        {
            "name": "right_tensor",
            "modes": ["batch", "bond", "('right', 'site')"],
            "shape": [2, 4, 5],
            "dtype": "float32",
            "itemsize": 4,
            "size": 40,
            "nbytes": 160,
            "ndim": 3,
            "strides": [80, 20, 4],
            "order": "C",
            "contiguous": True,
            "writeable": True,
            "owns_data": True,
            "backend": "numpy",
            "device": "DeviceSpec(kind='cpu', index=None, local_rank=None, global_rank=None, visible_id=None)",
            "device_kind": "cpu",
            "device_index": None,
            "is_host": True,
            "is_device": False,
            "is_distributed": False,
        },
    ]


def test_pair_tensor_contract_records_generic_contraction_plan(tmp_path):
    from renormalizer.mps.matrix import pair_tensor_contract
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    left = np.ones((2, 3))
    right = np.ones((3, 4))
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = pair_tensor_contract(left, "ik", right, "kj", {"k"})
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, np.tensordot(left, right, axes=((1,), (0,))))

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plans = [payload for payload in payloads if payload["event"] == "contraction_plan"]
    executes = [payload for payload in payloads if payload["event"] == "contraction_execute"]

    assert len(plans) == 1
    assert len(executes) == 1
    plan = plans[0]
    assert plan["backend"] == "numpy"
    assert plan["lowering"] == "gemm"
    assert plan["left_modes"] == ["i", "k"]
    assert plan["right_modes"] == ["k", "j"]
    assert plan["output_modes"] == ["i", "j"]
    assert plan["contracted_modes"] == ["k"]
    assert plan["input_shapes"] == [[2, 3], [3, 4]]
    assert plan["output_shape"] == [2, 4]
    assert plan["dtype"] == "float64"
    assert plan["flops"] == 48
    assert plan["read_bytes"] == 144
    assert plan["write_bytes"] == 64
    assert plan["copy_bytes"] == 0
    assert plan["workspace_bytes"] == 0
    assert plan["peak_bytes"] == 64
    assert plan["largest_intermediate"] == 64
    assert plan["largest_intermediate_elements"] == 8
    assert plan["largest_intermediate_bytes"] == 64
    assert plan["num_gemm"] == 1
    assert plan["execution_primitives"] == ["matmul"]
    assert plan["execution_policies"] == ["backend_matmul"]
    assert plan["fallback_reason"] is None
    execute = executes[0]
    assert execute["backend"] == "numpy"
    assert execute["lowering"] == "gemm"
    assert execute["input_shapes"] == [[2, 3], [3, 4]]
    assert execute["output_shape"] == [2, 4]
    assert execute["flops"] == 48
    assert execute["read_bytes"] == 144
    assert execute["write_bytes"] == 64
    assert execute["copy_bytes"] == 0
    assert execute["workspace_bytes"] == 0
    assert execute["peak_bytes"] == 64
    assert execute["largest_intermediate"] == 64
    assert execute["largest_intermediate_elements"] == 8
    assert execute["largest_intermediate_bytes"] == 64
    assert execute["num_gemm"] == 1
    assert execute["execution_primitives"] == ["matmul"]
    assert execute["execution_policies"] == ["backend_matmul"]
    assert execute["fallback_reason"] is None
    assert execute["wall_s"] >= 0.0


def test_pair_tensor_contract_uses_backend_execute_without_profiling(monkeypatch):
    from renormalizer.mps import matrix

    calls = []
    current_backend = matrix.backend.current
    original_execute = current_backend.execute

    def counting_execute(plan, **kwargs):
        calls.append(plan)
        return original_execute(plan, **kwargs)

    monkeypatch.setattr(current_backend, "execute", counting_execute)
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    result = matrix.pair_tensor_contract(left, "ik", right, "kj", {"k"})

    assert len(calls) >= 1
    assert calls[0].steps[0].kind == "gemm"
    assert np.allclose(matrix.asnumpy(result), left @ right)


def test_pair_tensor_contract_forwards_stream_and_workspace_to_backend_execute(monkeypatch):
    from renormalizer.mps import matrix

    calls = []
    current_backend = matrix.backend.current
    original_execute = current_backend.execute
    stream = object()
    workspace = current_backend.allocate_workspace(64)

    def counting_execute(plan, **kwargs):
        calls.append((plan, kwargs))
        return original_execute(plan, **kwargs)

    monkeypatch.setattr(current_backend, "execute", counting_execute)
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    result = matrix.pair_tensor_contract(
        left,
        "ik",
        right,
        "kj",
        {"k"},
        stream=stream,
        workspace=workspace,
    )

    assert len(calls) == 1
    assert calls[0][1]["stream"] is stream
    assert calls[0][1]["workspace"] is workspace
    assert np.allclose(matrix.asnumpy(result), left @ right)


def test_pair_tensor_contract_reuses_plan_without_reusing_old_arrays(monkeypatch):
    from renormalizer.mps import matrix

    cache = getattr(matrix, "_PAIR_CONTRACTION_PLAN_CACHE", None)
    if cache is not None:
        cache.clear()

    calls = []
    current_backend = matrix.backend.current
    original_plan_contraction = current_backend.plan_contraction

    def counting_plan_contraction(spec, **kwargs):
        calls.append(spec)
        return original_plan_contraction(spec, **kwargs)

    monkeypatch.setattr(current_backend, "plan_contraction", counting_plan_contraction)
    left0 = np.arange(6, dtype=np.float64).reshape(2, 3)
    right0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    left1 = left0 + 10.0
    right1 = right0 - 2.0

    result0 = matrix.pair_tensor_contract(left0, "ik", right0, "kj", {"k"})
    result1 = matrix.pair_tensor_contract(left1, "ik", right1, "kj", {"k"})

    assert len(calls) == 1
    assert np.allclose(matrix.asnumpy(result0), left0 @ right0)
    assert np.allclose(matrix.asnumpy(result1), left1 @ right1)


def test_pair_tensor_contract_cache_key_tracks_backend_capabilities(monkeypatch):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.mps import matrix

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

    cache = getattr(matrix, "_PAIR_CONTRACTION_PLAN_CACHE", None)
    if cache is not None:
        cache.clear()
    original_backend = matrix.backend.current
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    normal_backend = NumpyBackend(config=BackendConfig(fallback_policy="record"))
    fallback_backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    fallback_plan_calls = []
    original_fallback_plan_contraction = fallback_backend.plan_contraction

    def counting_fallback_plan_contraction(spec, **kwargs):
        plan = original_fallback_plan_contraction(spec, **kwargs)
        fallback_plan_calls.append(plan)
        return plan

    monkeypatch.setattr(fallback_backend, "plan_contraction", counting_fallback_plan_contraction)
    try:
        matrix.backend._manager.current = normal_backend
        first = matrix.pair_tensor_contract(left, "ik", right, "kj", {"k"})
        assert normal_backend.last_execution_profile()["lowering"] == "gemm"

        matrix.backend._manager.current = fallback_backend
        second = matrix.pair_tensor_contract(left, "ik", right, "kj", {"k"})
    finally:
        matrix.backend._manager.current = original_backend
        if cache is not None:
            cache.clear()

    assert np.allclose(matrix.asnumpy(first), left @ right)
    assert np.allclose(matrix.asnumpy(second), left @ right)
    assert len(fallback_plan_calls) == 1
    assert fallback_plan_calls[0].steps[0].kind == "fallback_tensordot"


def test_plan_contraction_returns_dense_gemm_step_for_pair_einsum():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    plan = backend.plan_contraction(spec, allow_distribution=False)

    assert plan.output_modes == ("i", "j")
    assert plan.estimated_flops == 48
    assert plan.estimated_read_bytes == left.nbytes + right.nbytes
    assert plan.estimated_write_bytes == 2 * 4 * left.itemsize
    assert plan.required_workspace_bytes == 0
    assert plan.sliced_modes == ()
    assert plan.distributed_modes == ()
    assert isinstance(plan.plan_hash, str)
    assert plan.plan_hash
    assert len(plan.steps) == 1

    step = plan.steps[0]
    assert step.kind == "gemm"
    assert step.inputs == (0, 1)
    assert step.output == 2
    assert step.input_modes == (("i", "k"), ("k", "j"))
    assert step.output_modes == ("i", "j")
    assert step.plan.kind == "gemm"
    assert step.estimated_flops == 48
    assert step.estimated_read_bytes == left.nbytes + right.nbytes
    assert step.estimated_write_bytes == 2 * 4 * left.itemsize
    assert step.estimated_comm_bytes == 0


def test_plan_contraction_returns_multi_step_gemm_path_for_three_operand_einsum():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    middle = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)
    spec = backend.parse_einsum("ab,bc,cd->ad", left, middle, right, optimize="greedy")

    plan = backend.plan_contraction(spec, allow_distribution=False)
    result = backend.execute(plan)

    assert len(plan.steps) == 2
    assert [step.kind for step in plan.steps] == ["gemm", "gemm"]
    assert plan.steps[0].inputs == (2, 1)
    assert plan.steps[0].input_modes == (("c", "d"), ("b", "c"))
    assert plan.steps[0].output_modes == ("d", "b")
    assert plan.steps[1].inputs == (1, 0)
    assert plan.steps[1].input_modes == (("d", "b"), ("a", "b"))
    assert plan.steps[1].output_modes == ("a", "d")
    assert plan.estimated_flops == 180
    assert plan.output_modes == ("a", "d")
    assert np.allclose(result, np.einsum("ab,bc,cd->ad", left, middle, right))


def test_multi_step_contraction_peak_tracks_live_intermediate_and_step_output():
    from renormalizer.backend import BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    middle = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)
    spec = backend.parse_einsum("ab,bc,cd->ad", left, middle, right, optimize="greedy")

    plan = backend.plan_contraction(spec, allow_distribution=False)

    first_intermediate_bytes = plan.steps[0].estimated_write_bytes
    final_output_bytes = plan.steps[1].estimated_write_bytes
    expected_peak = first_intermediate_bytes + final_output_bytes
    assert plan.estimated_peak_bytes == expected_peak

    with pytest.raises(BackendFeatureError, match="memory_limit.*199.*peak.*200"):
        backend.plan_contraction(
            spec,
            memory_limit=expected_peak - 1,
            allow_slicing=False,
        )


def test_explicit_multi_operand_contract_uses_planner_and_executor():
    from renormalizer.backend.numpy_backend import NumpyBackend

    class RecordingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.plan_calls = []
            self.execute_calls = []

        def plan_contraction(self, spec, **kwargs):
            self.plan_calls.append(spec)
            return super().plan_contraction(spec, **kwargs)

        def execute(self, plan, **kwargs):
            self.execute_calls.append(plan)
            return super().execute(plan, **kwargs)

    backend = RecordingBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    middle = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)

    result = backend.contract("ab,bc,cd->ad", left, middle, right, optimize="greedy")

    assert len(backend.plan_calls) == 1
    assert len(backend.execute_calls) == 1
    assert len(backend.execute_calls[0].steps) == 2
    assert np.allclose(result, np.einsum("ab,bc,cd->ad", left, middle, right))


def test_multi_constant_contract_expression_uses_planner_and_executor():
    from renormalizer.backend.numpy_backend import NumpyBackend

    class RecordingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.plan_calls = []
            self.execute_calls = []

        def plan_contraction(self, spec, **kwargs):
            self.plan_calls.append(spec)
            return super().plan_contraction(spec, **kwargs)

        def execute(self, plan, **kwargs):
            self.execute_calls.append(plan)
            return super().execute(plan, **kwargs)

    backend = RecordingBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    middle = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)

    expr = backend.contract_expression(
        "ab,bc,cd->ad",
        left,
        middle,
        right.shape,
        constants=[0, 1],
        optimize="greedy",
    )
    result = expr(right)

    assert len(backend.plan_calls) == 1
    assert len(backend.execute_calls) == 1
    assert len(backend.execute_calls[0].steps) == 2
    assert np.allclose(result, np.einsum("ab,bc,cd->ad", left, middle, right))


def test_contract_expression_forwards_plan_policy_to_unified_executor():
    from renormalizer.backend.numpy_backend import NumpyBackend

    class RecordingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.plan_kwargs = []
            self.execute_calls = []

        def plan_contraction(self, spec, **kwargs):
            self.plan_kwargs.append(dict(kwargs))
            return super().plan_contraction(spec, **kwargs)

        def execute(self, plan, **kwargs):
            self.execute_calls.append(plan)
            return super().execute(plan, **kwargs)

    backend = RecordingBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    expr = backend.contract_expression(
        "ik,kj->ij",
        left.shape,
        right.shape,
        optimize="greedy",
        memory_limit=63,
        allow_slicing=True,
    )
    result = expr(left, right)

    assert backend.plan_kwargs == [
        {
            "memory_limit": 63,
            "allow_slicing": True,
        }
    ]
    assert len(backend.execute_calls) == 1
    assert backend.execute_calls[0].steps[0].kind == "slice"
    assert np.allclose(result, left @ right)


def test_execute_contraction_plan_profile_records_plan_hash(tmp_path):
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
        result = backend.execute(plan)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_event = next(event for event in events if event["event"] == "contraction_plan")
    execute = next(event for event in events if event["event"] == "contraction_execute")

    assert plan_event["equation"] == "ik,kj->ij"
    assert plan_event["plan_hash"] == plan.plan_hash
    assert plan_event["peak_bytes"] == plan.estimated_peak_bytes
    assert execute["plan_hash"] == plan.plan_hash
    assert execute["equation"] == "ik,kj->ij"
    assert execute["input_modes"] == [["i", "k"], ["k", "j"]]
    assert execute["output_modes"] == ["i", "j"]
    assert execute["input_dtypes"] == ["float64", "float64"]
    assert execute["operands"][0]["name"] == "operand0"
    assert execute["operands"][1]["name"] == "operand1"
    assert execute["operands"][0]["modes"] == ["i", "k"]
    assert execute["operands"][0]["shape"] == [2, 3]
    assert execute["operands"][0]["itemsize"] == left.itemsize
    assert execute["operands"][0]["is_host"] is True
    assert execute["operands"][0]["is_distributed"] is False
    assert execute["peak_bytes"] == execute["write_bytes"] + execute["workspace_bytes"]
    assert execute["device_kind"] == "cpu"
    assert execute["device_index"] is None
    assert execute["device_info"] == {
        "kind": "cpu",
        "index": None,
        "local_rank": None,
        "global_rank": None,
        "visible_id": None,
    }


def test_multi_step_contraction_execute_profile_records_aggregate_event(tmp_path):
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    middle = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)
    stream = object()
    workspace = backend.allocate_workspace(32)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        spec = backend.parse_einsum("ab,bc,cd->ad", left, middle, right, optimize="greedy")
        plan = backend.plan_contraction(spec)
        result = backend.execute(plan, stream=stream, workspace=workspace)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, np.einsum("ab,bc,cd->ad", left, middle, right))
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    execute_events = [
        event
        for event in events
        if event["event"] == "contraction_execute" and event["plan_hash"] == plan.plan_hash
    ]

    assert len(execute_events) == 1
    execute = execute_events[0]
    assert execute["lowering"] == "multi_step"
    assert execute["compute_class"] == "contraction_plan"
    assert execute["compute_profile"]["compute_class"] == "contraction_plan"
    assert execute["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert execute["compute_subclass"] == "backend_execute"
    assert execute["compute_role"] == "kernel"
    assert execute["step_lowerings"] == ["gemm", "gemm"]
    assert execute["step_count"] == 2
    assert execute["contraction_steps"] == [
        {
            "index": 0,
            "kind": "gemm",
            "lowering": "gemm",
            "execution_primitives": ["matmul"],
            "execution_policies": ["backend_matmul"],
            "fallback_reasons": [],
            "inputs": [2, 1],
            "output": 3,
            "input_modes": [["c", "d"], ["b", "c"]],
            "output_modes": ["d", "b"],
            "plan_hash": plan.steps[0].plan.plan_hash,
            "flops": 120,
            "read_bytes": plan.steps[0].estimated_read_bytes,
            "write_bytes": plan.steps[0].estimated_write_bytes,
            "copy_bytes": plan.steps[0].estimated_copy_bytes,
            "peak_bytes": plan.steps[0].estimated_peak_bytes,
            "comm_bytes": plan.steps[0].estimated_comm_bytes,
            "workspace_bytes": plan.steps[0].required_workspace_bytes,
            "fallback_reason": None,
            "reason": "pair contraction lowered to GEMM",
        },
        {
            "index": 1,
            "kind": "gemm",
            "lowering": "gemm",
            "execution_primitives": ["matmul"],
            "execution_policies": ["backend_matmul"],
            "fallback_reasons": [],
            "inputs": [1, 0],
            "output": 4,
            "input_modes": [["d", "b"], ["a", "b"]],
            "output_modes": ["a", "d"],
            "plan_hash": plan.steps[1].plan.plan_hash,
            "flops": 60,
            "read_bytes": plan.steps[1].estimated_read_bytes,
            "write_bytes": plan.steps[1].estimated_write_bytes,
            "copy_bytes": plan.steps[1].estimated_copy_bytes,
            "peak_bytes": plan.steps[1].estimated_peak_bytes,
            "comm_bytes": plan.steps[1].estimated_comm_bytes,
            "workspace_bytes": plan.steps[1].required_workspace_bytes,
            "fallback_reason": None,
            "reason": "pair contraction lowered to GEMM",
        },
    ]
    assert execute["equation"] == "ab,bc,cd->ad"
    assert execute["input_shapes"] == [[2, 3], [3, 4], [4, 5]]
    assert execute["input_dtypes"] == ["float64", "float64", "float64"]
    assert execute["operands"][0]["modes"] == ["a", "b"]
    assert execute["operands"][0]["shape"] == [2, 3]
    assert execute["operands"][0]["itemsize"] == left.itemsize
    assert execute["operands"][0]["is_host"] is True
    assert execute["operands"][0]["is_distributed"] is False
    assert execute["output_shape"] == [2, 5]
    assert execute["flops"] == 180
    assert execute["read_bytes"] == plan.estimated_read_bytes
    assert execute["write_bytes"] == plan.estimated_write_bytes
    assert execute["num_gemm"] == 2
    assert execute["num_batched_gemm"] == 0
    assert execute["wall_s"] >= 0.0
    assert execute["stream_provided"] is True
    assert execute["stream_type"] == "object"
    assert execute["workspace_provided"] is True
    assert execute["workspace_nbytes"] == 32
    assert execute["workspace_device_kind"] == "cpu"
    assert execute["workspace_device_index"] is None


def test_multi_step_contraction_plan_profile_records_aggregate_event(tmp_path):
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    middle = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        spec = backend.parse_einsum("ab,bc,cd->ad", left, middle, right, optimize="greedy")
        plan = backend.plan_contraction(spec)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_events = [
        event
        for event in events
        if event["event"] == "contraction_plan" and event["plan_hash"] == plan.plan_hash
    ]

    assert len(plan_events) == 1
    event = plan_events[0]
    assert event["lowering"] == "multi_step"
    assert event["step_lowerings"] == ["gemm", "gemm"]
    assert event["step_count"] == 2
    assert event["contraction_steps"] == [
        {
            "index": 0,
            "kind": "gemm",
            "lowering": "gemm",
            "execution_primitives": ["matmul"],
            "execution_policies": ["backend_matmul"],
            "fallback_reasons": [],
            "inputs": [2, 1],
            "output": 3,
            "input_modes": [["c", "d"], ["b", "c"]],
            "output_modes": ["d", "b"],
            "plan_hash": plan.steps[0].plan.plan_hash,
            "flops": 120,
            "read_bytes": plan.steps[0].estimated_read_bytes,
            "write_bytes": plan.steps[0].estimated_write_bytes,
            "copy_bytes": plan.steps[0].estimated_copy_bytes,
            "peak_bytes": plan.steps[0].estimated_peak_bytes,
            "comm_bytes": plan.steps[0].estimated_comm_bytes,
            "workspace_bytes": plan.steps[0].required_workspace_bytes,
            "fallback_reason": None,
            "reason": "pair contraction lowered to GEMM",
        },
        {
            "index": 1,
            "kind": "gemm",
            "lowering": "gemm",
            "execution_primitives": ["matmul"],
            "execution_policies": ["backend_matmul"],
            "fallback_reasons": [],
            "inputs": [1, 0],
            "output": 4,
            "input_modes": [["d", "b"], ["a", "b"]],
            "output_modes": ["a", "d"],
            "plan_hash": plan.steps[1].plan.plan_hash,
            "flops": 60,
            "read_bytes": plan.steps[1].estimated_read_bytes,
            "write_bytes": plan.steps[1].estimated_write_bytes,
            "copy_bytes": plan.steps[1].estimated_copy_bytes,
            "peak_bytes": plan.steps[1].estimated_peak_bytes,
            "comm_bytes": plan.steps[1].estimated_comm_bytes,
            "workspace_bytes": plan.steps[1].required_workspace_bytes,
            "fallback_reason": None,
            "reason": "pair contraction lowered to GEMM",
        },
    ]
    assert event["equation"] == "ab,bc,cd->ad"
    assert event["input_shapes"] == [[2, 3], [3, 4], [4, 5]]
    assert event["output_shape"] == [2, 5]
    assert event["dtype"] == "float64"
    assert event["operands"][0]["itemsize"] == left.itemsize
    assert event["operands"][0]["size"] == left.size
    assert event["operands"][0]["writeable"] is True
    assert event["operands"][0]["owns_data"] is False
    assert event["operands"][0]["is_distributed"] is False
    assert event["flops"] == 180
    assert event["read_bytes"] == plan.estimated_read_bytes
    assert event["write_bytes"] == plan.estimated_write_bytes
    assert event["largest_intermediate"] == 120
    assert event["largest_intermediate_elements"] == 15
    assert event["largest_intermediate_bytes"] == 120
    assert event["num_gemm"] == 2


def test_distributed_contraction_spec_rejects_output_sharding_mode_mismatch():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    output_spec = ShardingSpec(
        global_shape=(2, 4),
        modes=("i", "x"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )

    with pytest.raises(ValueError, match="output_sharding modes must match equation output"):
        DistributedContractionSpec(
            equation="ik,kj->ij",
            operands=(
                np.ones((2, 3), dtype=np.float64),
                np.ones((3, 4), dtype=np.float64),
            ),
            output_sharding=output_spec,
        )


def test_distributed_contraction_spec_rejects_output_sharding_shape_mismatch():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    output_spec = ShardingSpec(
        global_shape=(2, 5),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )

    with pytest.raises(ValueError, match="output_sharding global_shape must match inferred output shape"):
        DistributedContractionSpec(
            equation="ik,kj->ij",
            operands=(
                np.ones((2, 3), dtype=np.float64),
                np.ones((3, 4), dtype=np.float64),
            ),
            output_sharding=output_spec,
        )


def test_distributed_contract_matches_dense_for_row_sharded_matmul():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    sharded_left = backend.shard_tensor(left, left_spec)
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(sharded_left, right),
    )

    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    result = backend.distributed_contract(contract_spec)

    assert plan.distributed_modes == ("i",)
    assert len(plan.steps) == 1
    assert plan.steps[0].kind == "distributed_contract"
    assert plan.steps[0].estimated_comm_bytes == 5 * 4 * left.itemsize
    assert backend.is_distributed_array(result) is True
    assert result.global_shape == (5, 4)
    assert result.modes == ("i", "j")
    assert result.sharding.sharded_modes == ("i",)
    assert result.local_shape == (3, 4)
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_distributed_contract_executes_local_matmul_plan_instead_of_direct_einsum():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    class PlanOnlyNumpyBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.local_matmul_plan_calls = 0

        def execute_matmul_plan(self, plan, **kwargs):
            self.local_matmul_plan_calls += 1
            return super().execute_matmul_plan(plan, **kwargs)

        def _execute_einsum(self, equation, operands):
            raise AssertionError("distributed local contraction bypassed backend execution plan")

    backend = PlanOnlyNumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), right),
    )

    result = backend.distributed_contract(contract_spec)

    assert backend.local_matmul_plan_calls == 2
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_local_contraction_plan_fallback_respects_forbid_policy():
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NoLocalPlanBackend(NumpyBackend):
        def _plan_einsum_contraction(self, *args, **kwargs):
            raise BackendFeatureError("local planner unavailable")

    backend = NoLocalPlanBackend(config=BackendConfig(fallback_policy="forbid"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    with pytest.raises(BackendFeatureError, match="local planner unavailable"):
        backend._execute_local_contraction_plan("ik,kj->ij", (left, right))


def test_local_contraction_plan_record_policy_profiles_einsum_fallback(tmp_path):
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoLocalPlanBackend(NumpyBackend):
        def _plan_einsum_contraction(self, *args, **kwargs):
            raise BackendFeatureError("local planner unavailable")

    backend = NoLocalPlanBackend(config=BackendConfig(fallback_policy="record"))
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend._execute_local_contraction_plan("ik,kj->ij", (left, right))
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in events if payload["event"] == "contraction_execute")

    assert event["equation"] == "ik,kj->ij"
    assert event["lowering"] == "fallback_einsum"
    assert event["fallback_from"] == "local_contraction_plan"
    assert event["fallback_to"] == "einsum"
    assert event["fallback_policy"] == "record"
    assert "local planner unavailable" in event["fallback_reason"]
    assert event["input_shapes"] == [[2, 3], [3, 4]]
    assert event["output_shape"] == [2, 4]
    assert event["num_gemm"] == 0
    assert event["num_batched_gemm"] == 0
    assert event["num_grouped_tasks"] == 0


def test_distributed_contract_allreduces_when_contracted_mode_is_sharded():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(24, dtype=np.float64).reshape(4, 6)
    right = np.arange(30, dtype=np.float64).reshape(6, 5)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    sharded_left = backend.shard_tensor(left, left_spec)
    sharded_right = backend.shard_tensor(right, right_spec)
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(sharded_left, sharded_right),
    )

    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    result = backend.distributed_contract(contract_spec, plan=plan)
    communication = plan.steps[0].plan.steps[0].communication

    assert plan.distributed_modes == ("k",)
    assert communication[0].kind == "allreduce"
    assert communication[0].bytes == left.shape[0] * right.shape[1] * left.itemsize
    assert result.sharding.sharded_modes == ()
    assert result.sharding.replicated_modes == ("i", "j")
    assert np.allclose(result.local_array, left @ right)
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_distributed_contract_reduce_scatters_when_reducing_to_requested_output_sharding():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(24, dtype=np.float64).reshape(4, 6)
    right = np.arange(30, dtype=np.float64).reshape(6, 5)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    output_spec = ShardingSpec(
        global_shape=(4, 5),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
        output_sharding=output_spec,
    )

    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    result = backend.distributed_contract(contract_spec, plan=plan)
    distributed_plan = plan.steps[0].plan
    communication = distributed_plan.steps[0].communication

    assert communication[0].kind == "reduce_scatter"
    assert communication[0].modes == ("k", "j")
    assert distributed_plan.total_allreduce_bytes == left.shape[0] * right.shape[1] * left.itemsize
    assert distributed_plan.total_redistribute_bytes == 0
    assert result.sharding == output_spec
    assert result.local_shape == (4, 3)
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_real_distributed_contract_uses_reduce_scatter_for_requested_output_shard():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    class CollectiveNumpyBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.reduce_scatter_calls = []

        @property
        def rank(self):
            return 0

        @property
        def size(self):
            return 2

        @property
        def is_distributed(self):
            return True

        def allreduce(self, x, op="sum"):
            raise AssertionError("requested output shard should use reduce_scatter, not allreduce")

        def reduce_scatter(self, x, op="sum", axis=0):
            self.reduce_scatter_calls.append((tuple(x.shape), op, axis))
            assert axis == 1
            return self._expected[:, :3]

        def allgather(self, x):
            return [self._expected[:, :3], self._expected[:, 3:]]

    backend = CollectiveNumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(24, dtype=np.float64).reshape(4, 6)
    right = np.arange(30, dtype=np.float64).reshape(6, 5)
    backend._expected = left @ right
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    output_spec = ShardingSpec(
        global_shape=(4, 5),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
        output_sharding=output_spec,
    )

    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    result = backend.distributed_contract(contract_spec, plan=plan)

    assert backend.reduce_scatter_calls == [((4, 5), "sum", 1)]
    assert result.sharding == output_spec
    assert result.rank_local_arrays is None
    assert result.local_shape == (4, 3)
    assert np.allclose(result.local_array, backend._expected[:, :3])
    assert np.allclose(backend.gather_tensor(result), backend._expected)


def test_distributed_contract_records_communication_profile(tmp_path):
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
    )
    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    stream = object()
    workspace = backend.allocate_workspace(32)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.distributed_contract(contract_spec, plan=plan, stream=stream, workspace=workspace)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(backend.gather_tensor(result), left @ right)
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    events = [payload for payload in payloads if payload["event"] == "contraction_execute"]

    assert len(events) == 1
    event = events[0]
    assert event["backend"] == "numpy"
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert event["equation"] == "ik,kj->ij"
    assert event["lowering"] == "distributed"
    assert event["is_distributed_runtime"] is False
    assert event["plan_hash"] == plan.plan_hash
    assert event["local_plan_hash"] == plan.steps[0].plan.path.plan_hash
    assert event["input_modes"] == [["i", "k"], ["k", "j"]]
    assert event["output_modes"] == ["i", "j"]
    assert event["input_dtypes"] == ["float64", "float64"]
    assert event["operands"][0]["modes"] == ["i", "k"]
    assert event["operands"][0]["shape"] == [5, 3]
    assert event["operands"][0]["nbytes"] == 72
    assert event["operands"][0]["is_host"] is False
    assert event["operands"][0]["is_device"] is False
    assert event["operands"][0]["is_distributed"] is True
    assert event["operands"][0]["device_kind"] == "distributed"
    assert event["operands"][1]["modes"] == ["k", "j"]
    assert event["operands"][1]["shape"] == [3, 4]
    assert event["operands"][1]["nbytes"] == 48
    assert event["local_lowering"] == "gemm"
    assert event["local_execution_primitives"] == ["matmul"]
    assert event["local_execution_policies"] == ["backend_matmul"]
    assert event["local_fallback_reasons"] == []
    assert event["execution_primitives"] == ["distributed_contract", "matmul"]
    assert event["execution_policies"] == ["backend_distributed_contract", "backend_matmul"]
    assert event["fallback_reasons"] == []
    assert event["num_gemm"] == 1
    assert event["num_batched_gemm"] == 0
    assert event["num_grouped_tasks"] == 0
    assert event["num_blocks"] == 0
    assert event["num_shape_buckets"] == 0
    assert event["fallback_reason"] is None
    assert event["distributed_modes"] == ["i"]
    assert event["rank"] == 0
    assert event["world_size"] == 2
    assert event["global_shape"] == [5, 4]
    assert event["stream_provided"] is True
    assert event["stream_type"] == "object"
    assert event["workspace_provided"] is True
    assert event["workspace_nbytes"] == 32
    assert event["workspace_device_kind"] == "cpu"
    assert event["workspace_device_index"] is None
    assert event["local_shape"] == [3, 4]
    assert event["input_states"] == [
        {
            "operand_index": 0,
            "tensor_id": 0,
            "modes": ["i", "k"],
            "shape": [5, 3],
            "distributed_modes": ["i"],
            "replicated_modes": ["k"],
            "local_shape": [3, 3],
            "local_nbytes": 72,
            "sharding": {
                "global_shape": [5, 3],
                "modes": ["i", "k"],
                "sharded_modes": ["i"],
                "replicated_modes": ["k"],
                "ranks_per_mode": {"i": 2},
                "mode_to_mesh_axis": {"i": "rank"},
                "mesh_shape": [2],
                "mesh_axis_names": ["rank"],
                "mesh_backend": "numpy",
                "mesh_world_size": 2,
                "mesh_local_rank": 0,
                "mesh_global_rank": 0,
                "local_slice": [[0, 3, None], [None, None, None]],
            },
        },
        {
            "operand_index": 1,
            "tensor_id": 1,
            "modes": ["k", "j"],
            "shape": [3, 4],
            "distributed_modes": ["j"],
            "replicated_modes": ["k"],
            "local_shape": [3, 2],
            "local_nbytes": 48,
            "sharding": {
                "global_shape": [3, 4],
                "modes": ["k", "j"],
                "sharded_modes": ["j"],
                "replicated_modes": ["k"],
                "ranks_per_mode": {"j": 2},
                "mode_to_mesh_axis": {"j": "rank"},
                "mesh_shape": [2],
                "mesh_axis_names": ["rank"],
                "mesh_backend": "numpy",
                "mesh_world_size": 2,
                "mesh_local_rank": 0,
                "mesh_global_rank": 0,
                "local_slice": [[None, None, None], [0, 2, None]],
            },
        },
    ]
    assert event["output_state"] == {
        "operand_index": 2,
        "tensor_id": 2,
        "modes": ["i", "j"],
        "shape": [5, 4],
        "distributed_modes": ["i"],
        "replicated_modes": ["j"],
        "local_shape": [3, 4],
        "local_nbytes": 96,
        "sharding": {
            "global_shape": [5, 4],
            "modes": ["i", "j"],
            "sharded_modes": ["i"],
            "replicated_modes": ["j"],
            "ranks_per_mode": {"i": 2},
            "mode_to_mesh_axis": {"i": "rank"},
            "mesh_shape": [2],
            "mesh_axis_names": ["rank"],
            "mesh_backend": "numpy",
            "mesh_world_size": 2,
            "mesh_local_rank": 0,
            "mesh_global_rank": 0,
            "local_slice": [[0, 3, None], [None, None, None]],
        },
    }
    assert event["communication"] == [
        {
            "kind": "redistribute",
            "primitive": "redistribute",
            "collective": "redistribute",
            "is_collective": False,
            "is_point_to_point": False,
            "bytes": 96,
            "local_bytes": 48,
            "modes": ["j"],
            "num_messages": 1,
            "block_size": 48,
            "wall_s": 0.0,
        },
        {
            "kind": "gather",
            "primitive": "gather",
            "collective": "gather",
            "is_collective": True,
            "is_point_to_point": False,
            "bytes": 160,
            "local_bytes": 96,
            "modes": ["i"],
            "num_messages": 1,
            "block_size": 96,
            "wall_s": 0.0,
        },
    ]
    assert event["distributed_step_count"] == 1
    assert event["distributed_steps"][0]["index"] == 0
    assert event["distributed_steps"][0]["kind"] == "distributed_contract"
    assert event["distributed_steps"][0]["local_step"]["kind"] == "distributed_contract"
    assert event["distributed_steps"][0]["local_step"]["lowering"] == "gemm"
    assert event["distributed_steps"][0]["local_step"]["inputs"] == [0, 1]
    assert event["distributed_steps"][0]["local_step"]["output"] == 2
    assert event["distributed_steps"][0]["local_step"]["input_modes"] == [["i", "k"], ["k", "j"]]
    assert event["distributed_steps"][0]["local_step"]["output_modes"] == ["i", "j"]
    assert event["distributed_steps"][0]["local_step"]["plan_hash"] == plan.steps[0].plan.steps[0].local_step.plan.plan_hash
    assert event["distributed_steps"][0]["local_step"]["comm_bytes"] == 160
    assert event["distributed_steps"][0]["input_states"] == event["input_states"]
    assert event["distributed_steps"][0]["output_state"] == event["output_state"]

    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "distributed"
    assert profile["plan_hash"] == plan.plan_hash
    assert profile["rank"] == event["rank"]
    assert profile["world_size"] == event["world_size"]
    assert profile["communication"] == event["communication"]
    assert profile["distributed_steps"] == event["distributed_steps"]
    assert json.loads(json.dumps(profile["compute_profile"])) == event["compute_profile"]
    assert json.loads(json.dumps(profile["layout_profile"])) == event["layout_profile"]
    assert json.loads(json.dumps(profile["communication_profile"])) == event["communication_profile"]
    assert event["distributed_steps"][0]["communication"] == event["communication"]
    assert event["comm_bytes"] == 256
    assert event["redistribute_bytes"] == 96
    assert event["allreduce_bytes"] == 0
    assert event["gather_bytes"] == 160
    assert event["point_to_point_bytes"] == 0
    assert event["broadcast_bytes"] == 0
    assert event["reduce_scatter_bytes"] == 0
    assert event["alltoall_bytes"] == 0
    assert event["allgather_bytes"] == 0
    assert event["wall_s"] >= 0.0
    communications = plan.steps[0].plan.steps[0].communication
    assert [(item.num_messages, item.block_size) for item in communications] == [(1, 48), (1, 96)]


def test_distributed_contraction_plan_records_planner_profile_event(tmp_path):
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    events = [
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan" and payload["lowering"] == "distributed"
    ]

    assert len(events) == 1
    event = events[0]
    assert event["backend"] == "numpy"
    assert event["equation"] == "ik,kj->ij"
    assert event["is_distributed_runtime"] is False
    assert event["plan_hash"] == plan.plan_hash
    assert event["local_plan_hash"] == plan.steps[0].plan.path.plan_hash
    assert event["input_modes"] == [["i", "k"], ["k", "j"]]
    assert event["output_modes"] == ["i", "j"]
    assert event["input_shapes"] == [[5, 3], [3, 4]]
    assert event["output_shape"] == [5, 4]
    assert event["rank"] == 0
    assert event["world_size"] == 2
    assert event["local_shape"] == [3, 4]
    assert event["global_shape"] == [5, 4]
    assert event["input_dtypes"] == ["float64", "float64"]
    assert event["dtype"] == "float64"
    assert event["operands"][0]["is_distributed"] is True
    assert event["operands"][0]["device_kind"] == "distributed"
    assert event["distributed_modes"] == ["i"]
    assert event["local_lowering"] == "gemm"
    assert event["local_execution_primitives"] == ["matmul"]
    assert event["local_execution_policies"] == ["backend_matmul"]
    assert event["local_fallback_reasons"] == []
    assert event["execution_primitives"] == ["distributed_contract", "matmul"]
    assert event["execution_policies"] == ["backend_distributed_contract", "backend_matmul"]
    assert event["fallback_reasons"] == []
    assert event["largest_intermediate"] == 96
    assert event["largest_intermediate_elements"] == 12
    assert event["largest_intermediate_bytes"] == 96
    assert event["comm_bytes"] == 256
    assert event["communication"][0]["collective"] == "redistribute"
    assert event["communication"][0]["bytes"] == 96
    assert event["communication"][1]["collective"] == "gather"
    assert event["communication"][1]["bytes"] == 160
    assert event["input_states"][0]["distributed_modes"] == ["i"]
    assert event["input_states"][1]["distributed_modes"] == ["j"]
    assert event["output_state"]["distributed_modes"] == ["i"]
    assert event["distributed_step_count"] == 1
    assert event["distributed_steps"][0]["index"] == 0
    assert event["distributed_steps"][0]["kind"] == "distributed_contract"
    assert event["distributed_steps"][0]["local_step"]["kind"] == "distributed_contract"
    assert event["distributed_steps"][0]["local_step"]["lowering"] == "gemm"
    assert event["distributed_steps"][0]["local_step"]["inputs"] == [0, 1]
    assert event["distributed_steps"][0]["local_step"]["output"] == 2
    assert event["distributed_steps"][0]["local_step"]["input_modes"] == [["i", "k"], ["k", "j"]]
    assert event["distributed_steps"][0]["local_step"]["output_modes"] == ["i", "j"]
    assert event["distributed_steps"][0]["local_step"]["plan_hash"] == plan.steps[0].plan.steps[0].local_step.plan.plan_hash
    assert event["distributed_steps"][0]["local_step"]["comm_bytes"] == 160
    assert event["distributed_steps"][0]["input_states"] == event["input_states"]
    assert event["distributed_steps"][0]["output_state"] == event["output_state"]
    assert event["distributed_steps"][0]["communication"] == event["communication"]


def test_distributed_contract_profile_measures_reduce_scatter_wall_time(tmp_path):
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class TimedCollectiveNumpyBackend(NumpyBackend):
        @property
        def rank(self):
            return 0

        @property
        def size(self):
            return 2

        @property
        def is_distributed(self):
            return True

        def allreduce(self, x, op="sum"):
            raise AssertionError("requested output shard should use reduce_scatter, not allreduce")

        def reduce_scatter(self, x, op="sum", axis=0):
            import time

            assert axis == 1
            time.sleep(0.002)
            return self._expected[:, :3]

        def allgather(self, x):
            return [self._expected[:, :3], self._expected[:, 3:]]

    backend = TimedCollectiveNumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(24, dtype=np.float64).reshape(4, 6)
    right = np.arange(30, dtype=np.float64).reshape(6, 5)
    backend._expected = left @ right
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    output_spec = ShardingSpec(
        global_shape=(4, 5),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
        output_sharding=output_spec,
    )
    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.distributed_contract(contract_spec, plan=plan)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result.local_array, backend._expected[:, :3])
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")

    assert event["communication"][0]["collective"] == "reduce_scatter"
    assert event["communication"][0]["wall_s"] > 0.0
    assert event["reduce_scatter_bytes"] == left.shape[0] * right.shape[1] * left.itemsize
    assert event["alltoall_bytes"] == 0
    assert event["broadcast_bytes"] == 0
    assert event["allgather_bytes"] == 0


def test_distributed_contract_redistributes_when_output_sharding_changes():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    output_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    sharded_left = backend.shard_tensor(left, left_spec)
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(sharded_left, right),
        output_sharding=output_spec,
    )

    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    result = backend.distributed_contract(contract_spec, plan=plan)
    distributed_plan = plan.steps[0].plan
    communication = distributed_plan.steps[0].communication

    assert communication[0].kind == "alltoall"
    assert communication[0].bytes == left.shape[0] * right.shape[1] * left.itemsize
    assert distributed_plan.total_alltoall_bytes == left.shape[0] * right.shape[1] * left.itemsize
    assert distributed_plan.total_reduce_scatter_bytes == 0
    assert distributed_plan.total_broadcast_bytes == 0
    assert distributed_plan.total_allgather_bytes == 0
    assert result.sharding == output_spec
    assert result.local_shape == (5, 2)
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_distributed_contract_redistributes_incompatible_input_sharding():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
    )

    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    result = backend.distributed_contract(contract_spec, plan=plan)
    communication = plan.steps[0].plan.steps[0].communication

    assert communication[0].kind == "redistribute"
    assert communication[0].modes == ("j",)
    assert communication[1].kind == "gather"
    assert communication[1].modes == ("i",)
    assert result.sharding.sharded_modes == ("i",)
    assert result.local_shape == (3, 4)
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_execute_runs_distributed_contraction_plan():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionPlan, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    contract_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), right),
    )
    plan = backend.plan_contraction(contract_spec, allow_distribution=True)
    distributed_plan = plan.steps[0].plan
    workspace = backend.allocate_workspace(32)
    stream = backend.default_stream()

    assert isinstance(distributed_plan, DistributedContractionPlan)
    assert distributed_plan.equation == "ik,kj->ij"

    direct_result = backend.execute(distributed_plan, stream=stream, workspace=workspace)
    wrapped_result = backend.execute(plan, stream=stream, workspace=workspace)

    assert direct_result.sharding.sharded_modes == ("i",)
    assert wrapped_result.sharding.sharded_modes == ("i",)
    assert np.allclose(backend.gather_tensor(direct_result), left @ right)
    assert np.allclose(backend.gather_tensor(wrapped_result), left @ right)


def test_plan_distributed_contraction_path_reports_communication_totals():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionPlan, DistributedContractionSpec, HardwareModel, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(15, dtype=np.float64).reshape(5, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    redistribute_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
    )
    hw = HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25)

    path = backend.plan_contraction(redistribute_spec, allow_distribution=True)
    distributed_path = backend.plan_distributed_contraction_path(
        path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=hw,
    )

    assert isinstance(distributed_path, DistributedContractionPlan)
    assert distributed_path.total_flops == path.estimated_flops
    assert distributed_path.total_comm_bytes == 256
    assert distributed_path.total_redistribute_bytes == 96
    assert distributed_path.total_allreduce_bytes == 0
    assert distributed_path.total_gather_bytes == 160
    assert distributed_path.total_alltoall_bytes == 0
    assert distributed_path.total_reduce_scatter_bytes == 0
    assert distributed_path.total_broadcast_bytes == 0
    assert distributed_path.total_allgather_bytes == 0
    assert distributed_path.peak_local_bytes == 96
    assert distributed_path.estimated_comm_bytes == 256
    assert distributed_path.steps[0].local_contraction_plan is distributed_path.steps[0].local_step.plan
    assert distributed_path.steps[0].communication_plan == distributed_path.steps[0].communication[0]
    assert [item.kind for item in distributed_path.steps[0].communication] == ["redistribute", "gather"]
    assert [(item.bytes, item.local_bytes) for item in distributed_path.steps[0].communication] == [(96, 48), (160, 96)]
    assert [(item.num_messages, item.block_size) for item in distributed_path.steps[0].communication] == [(1, 48), (1, 96)]
    assert distributed_path.steps[0].estimated_comm_s == pytest.approx(2.3)
    assert distributed_path.steps[0].estimated_total_s == float("inf")
    left_state, right_state = distributed_path.steps[0].input_states
    assert left_state.tensor_id == 0
    assert left_state.shape == left.shape
    assert left_state.local_shape == (3, 3)
    assert left_state.local_nbytes == 3 * 3 * left.itemsize
    assert right_state.tensor_id == 1
    assert right_state.shape == right.shape
    assert right_state.local_shape == (3, 2)
    assert right_state.local_nbytes == 3 * 2 * right.itemsize
    output_state = distributed_path.steps[0].output_state
    assert output_state.tensor_id == distributed_path.steps[0].local_step.output
    assert output_state.modes == ("i", "j")
    assert output_state.shape == (5, 4)
    assert output_state.sharding == distributed_path.output_sharding
    assert output_state.distributed_modes == ("i",)
    assert output_state.replicated_modes == ("j",)
    assert output_state.local_shape == (3, 4)
    assert output_state.local_nbytes == 3 * 4 * left.itemsize

    left_reduced = np.arange(24, dtype=np.float64).reshape(4, 6)
    right_reduced = np.arange(30, dtype=np.float64).reshape(6, 5)
    reduced_left_spec = ShardingSpec(
        global_shape=left_reduced.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    reduced_right_spec = ShardingSpec(
        global_shape=right_reduced.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"k": 2},
        mode_to_mesh_axis={"k": "rank"},
    )
    allreduce_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(
            backend.shard_tensor(left_reduced, reduced_left_spec),
            backend.shard_tensor(right_reduced, reduced_right_spec),
        ),
    )

    allreduce_path = backend.plan_distributed_contraction_path(
        backend.plan_contraction(allreduce_spec, allow_distribution=True),
        mesh,
        memory_limit_per_device=1024,
        cost_model=hw,
    )

    assert allreduce_path.total_comm_bytes == 160
    assert allreduce_path.total_redistribute_bytes == 0
    assert allreduce_path.total_allreduce_bytes == 160
    assert allreduce_path.total_gather_bytes == 0
    assert allreduce_path.total_alltoall_bytes == 0
    assert allreduce_path.total_reduce_scatter_bytes == 0
    assert allreduce_path.total_broadcast_bytes == 0
    assert allreduce_path.total_allgather_bytes == 0
    assert [item.kind for item in allreduce_path.steps[0].communication] == ["allreduce"]

    reduce_scatter_output_spec = ShardingSpec(
        global_shape=(4, 5),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )
    reduce_scatter_spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(
            backend.shard_tensor(left_reduced, reduced_left_spec),
            backend.shard_tensor(right_reduced, reduced_right_spec),
        ),
        output_sharding=reduce_scatter_output_spec,
    )

    reduce_scatter_path = backend.plan_distributed_contraction_path(
        backend.plan_contraction(reduce_scatter_spec, allow_distribution=True),
        mesh,
        memory_limit_per_device=1024,
        cost_model=hw,
    )

    assert reduce_scatter_path.total_comm_bytes == 160
    assert reduce_scatter_path.total_reduce_scatter_bytes == 160
    assert reduce_scatter_path.total_alltoall_bytes == 0
    assert reduce_scatter_path.total_broadcast_bytes == 0
    assert reduce_scatter_path.total_allgather_bytes == 0
    assert [item.kind for item in reduce_scatter_path.steps[0].communication] == ["reduce_scatter"]


def test_plan_distributed_contraction_path_uses_cost_model_memory_limit():
    from renormalizer.backend import BackendFeatureError, DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.ones((4, 3), dtype=np.float64)
    right = np.ones((3, 2), dtype=np.float64)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    with pytest.raises(BackendFeatureError, match="peak local bytes 96 exceeds memory limit 95"):
        backend.plan_distributed_contraction_path(
            dense_path,
            mesh,
            cost_model=HardwareModel(max_memory_bytes=95),
        )


def _valid_distribution_state_kwargs():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    sharding = ShardingSpec(
        global_shape=(4, 3),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    return {
        "operand_index": 0,
        "tensor_id": 0,
        "modes": ("i", "j"),
        "sharding": sharding,
        "shape": (4, 3),
        "distributed_modes": ("i",),
        "replicated_modes": ("j",),
        "local_shape": (2, 3),
        "local_nbytes": 48,
    }


def test_distribution_state_rejects_invalid_identity_and_shape_metadata():
    from renormalizer.backend import DistributionState

    kwargs = _valid_distribution_state_kwargs()

    with pytest.raises(ValueError, match="DistributionState operand_index must be non-negative"):
        DistributionState(**{**kwargs, "operand_index": -1})

    with pytest.raises(ValueError, match="DistributionState tensor_id must be non-negative"):
        DistributionState(**{**kwargs, "tensor_id": -1})

    with pytest.raises(ValueError, match="DistributionState shape dimensions must be non-negative"):
        DistributionState(**{**kwargs, "shape": (-1, 3)})

    with pytest.raises(ValueError, match="DistributionState modes must match shape rank"):
        DistributionState(**{**kwargs, "modes": ("i",)})

    with pytest.raises(ValueError, match="DistributionState local_shape dimensions must be non-negative"):
        DistributionState(**{**kwargs, "local_shape": (-1, 3)})

    with pytest.raises(ValueError, match="DistributionState local_shape must match shape rank"):
        DistributionState(**{**kwargs, "local_shape": (2,)})

    with pytest.raises(ValueError, match="DistributionState local_nbytes must be non-negative"):
        DistributionState(**{**kwargs, "local_nbytes": -1})


def test_distribution_state_rejects_inconsistent_mode_and_sharding_metadata():
    from renormalizer.backend import DistributionState, ShardingSpec

    kwargs = _valid_distribution_state_kwargs()

    with pytest.raises(ValueError, match="DistributionState distributed_modes contain modes not present in modes"):
        DistributionState(**{**kwargs, "distributed_modes": ("missing",)})

    with pytest.raises(ValueError, match="DistributionState distributed_modes and replicated_modes must not overlap"):
        DistributionState(**{**kwargs, "replicated_modes": ("i", "j")})

    with pytest.raises(ValueError, match="DistributionState distributed_modes and replicated_modes must cover all modes"):
        DistributionState(**{**kwargs, "replicated_modes": ()})

    mesh = kwargs["sharding"].mesh
    shape_mismatch = ShardingSpec(
        global_shape=(5, 3),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    with pytest.raises(ValueError, match="DistributionState sharding global_shape must match shape"):
        DistributionState(**{**kwargs, "sharding": shape_mismatch})

    mode_mismatch = ShardingSpec(
        global_shape=(4, 3),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    with pytest.raises(ValueError, match="DistributionState sharding modes must match modes"):
        DistributionState(**{**kwargs, "sharding": mode_mismatch})

    with pytest.raises(ValueError, match="DistributionState distributed_modes must match sharding sharded_modes"):
        DistributionState(**{**kwargs, "distributed_modes": (), "replicated_modes": ("i", "j")})


def _valid_distributed_step_plan_kwargs():
    from renormalizer.backend import ContractionStep, DistributionState

    left = DistributionState(
        operand_index=0,
        tensor_id=0,
        modes=("i", "k"),
        sharding=None,
        shape=(2, 3),
        distributed_modes=(),
        replicated_modes=("i", "k"),
        local_shape=(2, 3),
        local_nbytes=48,
    )
    right = DistributionState(
        operand_index=1,
        tensor_id=1,
        modes=("k", "j"),
        sharding=None,
        shape=(3, 4),
        distributed_modes=(),
        replicated_modes=("k", "j"),
        local_shape=(3, 4),
        local_nbytes=96,
    )
    output = DistributionState(
        operand_index=2,
        tensor_id=2,
        modes=("i", "j"),
        sharding=None,
        shape=(2, 4),
        distributed_modes=(),
        replicated_modes=("i", "j"),
        local_shape=(2, 4),
        local_nbytes=64,
    )
    return {
        "local_step": ContractionStep(**_valid_contraction_step_kwargs()),
        "input_states": (left, right),
        "output_sharding": None,
        "output_state": output,
    }


def test_distributed_step_plan_rejects_unknown_kind():
    from renormalizer.backend import DistributedStepPlan

    kwargs = _valid_distributed_step_plan_kwargs()

    with pytest.raises(ValueError, match="Unknown DistributedStepPlan kind"):
        DistributedStepPlan(**{**kwargs, "kind": "mystery"})


@pytest.mark.parametrize("field", ["estimated_compute_s", "estimated_comm_s", "estimated_total_s"])
def test_distributed_step_plan_rejects_negative_estimates(field):
    from renormalizer.backend import DistributedStepPlan

    kwargs = _valid_distributed_step_plan_kwargs()
    kwargs[field] = -1

    with pytest.raises(ValueError, match="DistributedStepPlan {0} must be non-negative".format(field)):
        DistributedStepPlan(**kwargs)


def test_distributed_step_plan_total_time_covers_compute_and_comm():
    from renormalizer.backend import DistributedStepPlan

    kwargs = _valid_distributed_step_plan_kwargs()
    step = DistributedStepPlan(
        **{
            **kwargs,
            "estimated_compute_s": 1.25,
            "estimated_comm_s": 2.75,
        }
    )

    assert step.estimated_total_s == pytest.approx(4.0)

    with pytest.raises(ValueError, match="DistributedStepPlan estimated_total_s must cover compute and communication"):
        DistributedStepPlan(
            **{
                **kwargs,
                "estimated_compute_s": 1.25,
                "estimated_comm_s": 2.75,
                "estimated_total_s": 3.99,
            }
        )


def _valid_distributed_contraction_plan_kwargs():
    from renormalizer.backend import ContractionPlan, DistributedStepPlan

    return {
        "path": ContractionPlan(**_valid_contraction_plan_kwargs()),
        "steps": (DistributedStepPlan(**_valid_distributed_step_plan_kwargs()),),
        "output_sharding": None,
    }


def test_distributed_contraction_plan_rejects_empty_steps():
    from renormalizer.backend import DistributedContractionPlan

    kwargs = _valid_distributed_contraction_plan_kwargs()
    kwargs["steps"] = ()

    with pytest.raises(ValueError, match="DistributedContractionPlan steps must be non-empty"):
        DistributedContractionPlan(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "estimated_comm_bytes",
        "peak_local_bytes",
        "total_flops",
        "total_comm_bytes",
        "total_redistribute_bytes",
        "total_allreduce_bytes",
        "total_gather_bytes",
        "total_point_to_point_bytes",
        "total_broadcast_bytes",
        "total_reduce_scatter_bytes",
        "total_alltoall_bytes",
        "total_allgather_bytes",
    ],
)
def test_distributed_contraction_plan_rejects_negative_estimates(field):
    from renormalizer.backend import DistributedContractionPlan

    kwargs = _valid_distributed_contraction_plan_kwargs()
    kwargs[field] = -1

    with pytest.raises(ValueError, match="DistributedContractionPlan {0} must be non-negative".format(field)):
        DistributedContractionPlan(**kwargs)


def test_distributed_contraction_plan_derives_and_validates_step_totals():
    from renormalizer.backend import (
        CommunicationPlan,
        ContractionPlan,
        ContractionStep,
        DistributedContractionPlan,
        DistributedStepPlan,
        TensorOperand,
    )

    local_step = ContractionStep(
        kind="gemm",
        inputs=(0, 1),
        output=2,
        input_modes=(("i", "k"), ("k", "j")),
        output_modes=("i", "j"),
        estimated_flops=25,
        estimated_read_bytes=48,
        estimated_write_bytes=64,
        estimated_peak_bytes=120,
        required_workspace_bytes=80,
    )
    path = ContractionPlan(
        steps=(local_step,),
        input_specs=(
            TensorOperand(np.ones((2, 3)), ("i", "k")),
            TensorOperand(np.ones((3, 4)), ("k", "j")),
        ),
        output_modes=("i", "j"),
        estimated_flops=25,
        estimated_read_bytes=48,
        estimated_write_bytes=64,
        estimated_peak_bytes=120,
        required_workspace_bytes=80,
    )
    distributed_step = DistributedStepPlan(
        **{
            **_valid_distributed_step_plan_kwargs(),
            "local_step": local_step,
            "communication": (
                CommunicationPlan(kind="redistribute", bytes=11, local_bytes=64),
                CommunicationPlan(kind="allreduce", bytes=13, local_bytes=72),
                CommunicationPlan(kind="gather", bytes=17, local_bytes=88),
                CommunicationPlan(kind="point_to_point", bytes=19, local_bytes=32),
            ),
        }
    )

    plan = DistributedContractionPlan(path=path, steps=(distributed_step,), output_sharding=None)

    assert plan.estimated_comm_bytes == 60
    assert plan.total_comm_bytes == 60
    assert plan.total_redistribute_bytes == 11
    assert plan.total_allreduce_bytes == 13
    assert plan.total_gather_bytes == 17
    assert plan.total_point_to_point_bytes == 19
    assert plan.total_flops == 25
    assert plan.peak_local_bytes == 120

    valid_kwargs = {
        "path": path,
        "steps": (distributed_step,),
        "output_sharding": None,
        "estimated_comm_bytes": 60,
        "total_comm_bytes": 60,
        "total_redistribute_bytes": 11,
        "total_allreduce_bytes": 13,
        "total_gather_bytes": 17,
        "total_point_to_point_bytes": 19,
        "total_flops": 25,
        "peak_local_bytes": 120,
    }
    for field, too_small in (
        ("estimated_comm_bytes", 40),
        ("total_comm_bytes", 40),
        ("total_redistribute_bytes", 10),
        ("total_allreduce_bytes", 12),
        ("total_gather_bytes", 16),
        ("total_point_to_point_bytes", 18),
        ("total_flops", 24),
        ("peak_local_bytes", 119),
    ):
        with pytest.raises(ValueError, match="DistributedContractionPlan {0} must cover step".format(field)):
            DistributedContractionPlan(**{**valid_kwargs, field: too_small})


def test_plan_distributed_contraction_path_activates_distribution_for_dense_plan():
    from renormalizer.backend import DeviceMesh, DeviceSpec, DistributedContractionPlan, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(64, dtype=np.float64).reshape(16, 4)
    right = np.arange(12, dtype=np.float64).reshape(4, 3)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    hw = HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25)

    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=hw,
    )

    assert isinstance(distributed_path, DistributedContractionPlan)
    assert distributed_path.output_sharding is not None
    assert distributed_path.output_sharding.modes == ("i", "j")
    assert distributed_path.output_sharding.sharded_modes == ("i",)
    assert distributed_path.steps[0].kind == "activate_distribution"
    assert distributed_path.steps[0].local_contraction_plan is distributed_path.steps[0].local_step.plan
    assert distributed_path.steps[0].communication_plan == distributed_path.steps[0].communication[0]
    assert [item.kind for item in distributed_path.steps[0].communication] == ["activate_distribution"]
    assert distributed_path.total_comm_bytes == left.nbytes + right.nbytes
    assert distributed_path.total_redistribute_bytes == left.nbytes + right.nbytes
    assert [
        (item.bytes, item.local_bytes)
        for item in distributed_path.steps[0].communication
    ] == [(left.nbytes + right.nbytes, 352)]
    assert distributed_path.steps[0].estimated_comm_s == pytest.approx(352 / 80.0 + 0.25)
    left_state, right_state = distributed_path.steps[0].input_states
    assert left_state.tensor_id == 0
    assert left_state.shape == left.shape
    assert left_state.local_shape == left.shape
    assert left_state.local_nbytes == left.nbytes
    assert right_state.tensor_id == 1
    assert right_state.shape == right.shape
    assert right_state.local_shape == right.shape
    assert right_state.local_nbytes == right.nbytes
    output_state = distributed_path.steps[0].output_state
    assert output_state.tensor_id == distributed_path.steps[0].local_step.output
    assert output_state.modes == ("i", "j")
    assert output_state.shape == (16, 3)
    assert output_state.sharding == distributed_path.output_sharding
    assert output_state.distributed_modes == ("i",)
    assert output_state.replicated_modes == ("j",)
    assert output_state.local_shape == (8, 3)
    assert output_state.local_nbytes == 8 * 3 * left.itemsize


def test_execute_auto_distributed_dense_plan_shards_output():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(64, dtype=np.float64).reshape(16, 4)
    right = np.arange(12, dtype=np.float64).reshape(4, 3)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
    )

    result = backend.execute(distributed_path)

    assert backend.is_distributed_array(result)
    assert result.sharding == distributed_path.output_sharding
    assert result.local_shape == (8, 3)
    assert np.allclose(result.local_array, (left @ right)[:8, :])
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_execute_auto_distributed_profile_preserves_plan_identity_and_estimates(tmp_path):
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(64, dtype=np.float64).reshape(16, 4)
    right = np.arange(12, dtype=np.float64).reshape(4, 3)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute(distributed_path)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(backend.gather_tensor(result), left @ right)
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    step = distributed_path.steps[0]

    assert event["plan_hash"] == dense_path.plan_hash
    assert event["distributed_step_kind"] == "activate_distribution"
    assert event["distributed_modes"] == ["i"]
    assert event["read_bytes"] == dense_path.estimated_read_bytes
    assert event["write_bytes"] == dense_path.estimated_write_bytes
    assert event["copy_bytes"] == dense_path.estimated_copy_bytes
    assert event["workspace_bytes"] == dense_path.required_workspace_bytes
    assert event["peak_bytes"] == dense_path.estimated_peak_bytes
    assert event["estimated_compute_s"] == pytest.approx(step.estimated_compute_s)
    assert event["estimated_comm_s"] == pytest.approx(step.estimated_comm_s)
    assert event["estimated_total_s"] == pytest.approx(step.estimated_total_s)
    assert event["communication"][0]["collective"] == "activate_distribution"
    assert event["communication"][0]["bytes"] == left.nbytes + right.nbytes
    assert event["communication"][0]["local_bytes"] == 352
    assert event["communication"][0]["num_messages"] == 1
    assert event["communication"][0]["block_size"] == 352
    assert event["communication"][0]["wall_s"] == pytest.approx(0.0)
    assert event["communication_profile"]["bytes"] == left.nbytes + right.nbytes
    assert event["communication_profile"]["bytes_by_collective"] == {
        "activate_distribution": left.nbytes + right.nbytes
    }
    assert event["communication_profile"]["messages_by_collective"] == {
        "activate_distribution": 1
    }
    assert event["communication_profile"]["num_messages"] == 1
    assert event["communication_profile"]["wall_s"] == pytest.approx(0.0)


def test_execute_auto_distributed_dense_plan_places_inputs_before_contracting():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    class LocalOnlyBackend(NumpyBackend):
        def matmul(self, A, B=None, **kwargs):
            if B is not None and getattr(A, "shape", None) == (16, 4):
                raise AssertionError("auto-distributed execution used full dense matmul")
            return super().matmul(A, B, **kwargs)

    backend = LocalOnlyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(64, dtype=np.float64).reshape(16, 4)
    right = np.arange(12, dtype=np.float64).reshape(4, 3)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
    )

    result = backend.execute(distributed_path)

    assert result.local_shape == (8, 3)
    assert np.allclose(result.local_array, (left @ right)[:8, :])
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_plan_contraction_rejects_plan_exceeding_memory_limit_without_slicing():
    from renormalizer.backend import BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    with pytest.raises(BackendFeatureError, match="memory_limit.*63.*peak.*64"):
        backend.plan_contraction(spec, memory_limit=63, allow_slicing=False)


def test_plan_contraction_slices_output_to_satisfy_memory_limit():
    from renormalizer.backend import SlicedContractionPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    plan = backend.plan_contraction(spec, memory_limit=63, allow_slicing=True)

    assert plan.sliced_modes
    assert plan.estimated_peak_bytes <= 63
    assert plan.steps[0].kind == "slice"
    assert isinstance(plan.steps[0].plan, SlicedContractionPlan)
    assert plan.steps[0].plan.base_plan.estimated_peak_bytes == 64
    assert np.allclose(backend.execute(plan), left @ right)


def test_execute_sliced_contraction_plan_records_profile_event(tmp_path):
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(
        backend.parse_einsum("ik,kj->ij", left, right),
        memory_limit=63,
        allow_slicing=True,
    )
    stream = object()
    workspace = backend.allocate_workspace(128)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute(plan, stream=stream, workspace=workspace)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")

    assert event["plan_hash"] == plan.plan_hash
    assert event["lowering"] == "slice"
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert event["equation"] == "ik,kj->ij"
    assert event["input_modes"] == [["i", "k"], ["k", "j"]]
    assert event["output_modes"] == ["i", "j"]
    assert event["input_dtypes"] == ["float64", "float64"]
    assert event["operands"][0]["modes"] == ["i", "k"]
    assert event["operands"][0]["shape"] == [2, 3]
    assert event["operands"][0]["itemsize"] == left.itemsize
    assert event["operands"][0]["is_host"] is True
    assert event["operands"][0]["is_device"] is False
    assert event["operands"][0]["is_distributed"] is False
    assert event["operands"][1]["modes"] == ["k", "j"]
    assert event["operands"][1]["shape"] == [3, 4]
    assert event["operands"][1]["backend"] == backend.name
    assert event["sliced_modes"] == [str(mode) for mode in plan.sliced_modes]
    assert event["num_slices"] == len(plan.steps[0].plan.output_slices)
    assert event["base_lowering"] == "gemm"
    assert event["peak_bytes"] == plan.estimated_peak_bytes
    assert event["stream_provided"] is True
    assert event["stream_type"] == "object"
    assert event["workspace_provided"] is True
    assert event["workspace_nbytes"] == 128
    assert event["workspace_device_kind"] == "cpu"
    assert event["workspace_device_index"] is None


def test_execute_sliced_contraction_plan_does_not_profile_internal_slice_plans(tmp_path):
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(
        backend.parse_einsum("ik,kj->ij", left, right),
        memory_limit=63,
        allow_slicing=True,
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.execute(plan)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    assert [payload["event"] for payload in payloads] == ["contraction_execute"]


def test_plan_contraction_records_sliced_plan_profile_event(tmp_path):
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        plan = backend.plan_contraction(
            backend.parse_einsum("ik,kj->ij", left, right),
            memory_limit=63,
            allow_slicing=True,
        )
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload.get("plan_hash") == plan.plan_hash
    )

    assert event["lowering"] == "slice"
    assert event["equation"] == "ik,kj->ij"
    assert event["sliced_modes"] == [str(mode) for mode in plan.sliced_modes]
    assert event["num_slices"] == len(plan.steps[0].plan.output_slices)
    assert event["base_lowering"] == "gemm"
    assert event["input_dtypes"] == ["float64", "float64"]
    assert event["dtype"] == "float64"
    assert event["operands"][0]["modes"] == ["i", "k"]
    assert event["operands"][0]["itemsize"] == left.itemsize
    assert event["operands"][0]["size"] == left.size
    assert event["operands"][0]["writeable"] is True
    assert event["operands"][0]["owns_data"] is False
    assert event["operands"][0]["is_distributed"] is False
    assert event["peak_bytes"] == plan.estimated_peak_bytes
    assert event["read_bytes"] == plan.estimated_read_bytes
    assert event["largest_intermediate"] == 48
    assert event["largest_intermediate_elements"] == 6
    assert event["largest_intermediate_bytes"] == 48


def test_plan_contraction_rejects_target_devices_without_distribution():
    from renormalizer.backend import BackendFeatureError, DeviceSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    with pytest.raises(BackendFeatureError, match="target_devices.*allow_distribution=True"):
        backend.plan_contraction(
            spec,
            allow_distribution=False,
            target_devices=(DeviceSpec(kind="cuda", index=0), DeviceSpec(kind="cuda", index=1)),
        )


def test_plan_contraction_target_devices_builds_distributed_plan():
    from renormalizer.backend import DeviceSpec, DistributedContractionPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(24, dtype=np.float64).reshape(6, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    plan = backend.plan_contraction(
        spec,
        allow_distribution=True,
        target_devices=(DeviceSpec(kind="cuda", index=0), DeviceSpec(kind="cuda", index=1)),
    )
    distributed_plan = plan.steps[0].plan
    result = backend.execute(plan)

    assert isinstance(distributed_plan, DistributedContractionPlan)
    assert plan.distributed_modes
    assert distributed_plan.output_sharding.mesh.backend == "numpy"
    assert distributed_plan.output_sharding.mesh.shape == (2,)
    assert tuple(device.kind for device in distributed_plan.output_sharding.mesh.devices) == ("cuda", "cuda")
    assert tuple(device.index for device in distributed_plan.output_sharding.mesh.devices) == (0, 1)
    assert result.mesh.world_size == 2
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_plan_contraction_allow_distribution_auto_uses_visible_device_mesh():
    from renormalizer.backend import DistributedContractionPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    class TwoRankNumpyBackend(NumpyBackend):
        def device_count(self):
            return 2

    backend = TwoRankNumpyBackend()
    left = np.arange(24, dtype=np.float64).reshape(6, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    plan = backend.plan_contraction(spec, allow_distribution=True)
    distributed_plan = plan.steps[0].plan
    result = backend.execute(plan)

    assert isinstance(distributed_plan, DistributedContractionPlan)
    assert plan.distributed_modes == ("i",)
    assert tuple(device.kind for device in distributed_plan.output_sharding.mesh.devices) == ("cpu", "cpu")
    assert tuple(device.global_rank for device in distributed_plan.output_sharding.mesh.devices) == (0, 1)
    assert result.mesh.world_size == 2
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_plan_contraction_target_devices_memory_preference_keeps_distributed_metadata():
    from renormalizer.backend import DeviceSpec, DistributedContractionPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    plan = backend.plan_contraction(
        spec,
        prefer="memory",
        allow_distribution=True,
        target_devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
    )

    assert isinstance(plan.steps[0].plan, DistributedContractionPlan)
    assert plan.steps[0].kind == "gemm"
    assert plan.distributed_modes
    assert plan.sliced_modes == ()


def test_plan_contraction_rejects_target_devices_with_explicit_distributed_spec():
    from renormalizer.backend import BackendFeatureError, DeviceMesh, DeviceSpec, DistributedContractionSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
    )
    left = np.ones((4, 3), dtype=np.float64)
    right = np.ones((3, 2), dtype=np.float64)
    left_spec = ShardingSpec(
        global_shape=left.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "rank"},
    )
    spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), right),
    )

    with pytest.raises(BackendFeatureError, match="target_devices cannot be combined"):
        backend.plan_contraction(
            spec,
            allow_distribution=True,
            target_devices=(DeviceSpec(kind="cuda", index=0), DeviceSpec(kind="cuda", index=1)),
        )


def test_plan_contraction_rejects_unknown_preference_strategy():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    with pytest.raises(ValueError, match="prefer.*time.*memory.*balanced"):
        backend.plan_contraction(spec, prefer="fastest")


def test_plan_contraction_memory_preference_uses_slicing_to_reduce_peak():
    from renormalizer.backend import SlicedContractionPlan
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    time_plan = backend.plan_contraction(spec, prefer="time")
    memory_plan = backend.plan_contraction(spec, prefer="memory")

    assert time_plan.steps[0].kind == "gemm"
    assert memory_plan.steps[0].kind == "slice"
    assert isinstance(memory_plan.steps[0].plan, SlicedContractionPlan)
    assert memory_plan.estimated_peak_bytes < time_plan.estimated_peak_bytes
    assert memory_plan.estimated_read_bytes > time_plan.estimated_read_bytes
    assert np.allclose(backend.execute(memory_plan), left @ right)


@pytest.mark.parametrize(
    "field",
    [
        "flop_per_s",
        "memory_bandwidth_Bps",
        "h2d_bandwidth_Bps",
        "d2h_bandwidth_Bps",
        "p2p_bandwidth_Bps",
        "network_bandwidth_Bps",
        "latency_s",
        "max_memory_bytes",
        "workspace_limit_bytes",
        "device_flop_s",
        "host_bandwidth_bytes_s",
        "device_bandwidth_bytes_s",
        "interconnect_bandwidth_bytes_s",
    ],
)
def test_hardware_model_rejects_negative_physical_parameters(field):
    from renormalizer.backend import HardwareModel

    with pytest.raises(ValueError, match="{0} must be non-negative".format(field)):
        HardwareModel(**{field: -1})


@pytest.mark.parametrize(
    "field",
    [
        "flops",
        "read_bytes",
        "write_bytes",
        "copy_bytes",
        "comm_bytes",
        "workspace_bytes",
        "peak_bytes",
        "compute_s",
        "memory_s",
        "copy_s",
        "comm_s",
        "total_s",
        "estimated_time_s",
    ],
)
def test_cost_estimate_rejects_negative_values(field):
    from renormalizer.backend import CostEstimate

    with pytest.raises(ValueError, match="{0} must be non-negative".format(field)):
        CostEstimate(**{field: -1})


def test_cost_estimate_total_time_covers_component_times():
    from renormalizer.backend import CostEstimate

    estimate = CostEstimate(
        compute_s=1.0,
        memory_s=2.0,
        copy_s=3.0,
        comm_s=4.0,
    )

    assert estimate.total_s == pytest.approx(10.0)
    assert estimate.estimated_time_s == pytest.approx(10.0)

    with pytest.raises(ValueError, match="total_s must cover component time estimates"):
        CostEstimate(compute_s=1.0, memory_s=2.0, total_s=2.5)

    with pytest.raises(ValueError, match="estimated_time_s must cover total_s"):
        CostEstimate(compute_s=1.0, memory_s=2.0, total_s=3.0, estimated_time_s=2.5)


def test_contraction_cost_model_reports_peak_and_timing_estimates():
    from renormalizer.backend import HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    spec = backend.parse_einsum("ik,kj->ij", left, right)

    plan = backend.plan_contraction(spec)
    hw = HardwareModel(
        flop_per_s=24.0,
        memory_bandwidth_Bps=208.0,
        p2p_bandwidth_Bps=100.0,
        network_bandwidth_Bps=10.0,
        latency_s=0.5,
        max_memory_bytes=1024,
        workspace_limit_bytes=256,
    )
    estimate = backend.estimate_contraction(plan, hw)

    assert plan.estimated_peak_bytes == 64
    assert estimate.flops == 48
    assert estimate.read_bytes == left.nbytes + right.nbytes
    assert estimate.write_bytes == 64
    assert estimate.peak_bytes == 64
    assert estimate.compute_s == pytest.approx(2.0)
    assert estimate.memory_s == pytest.approx(1.0)
    assert estimate.copy_s == 0.0
    assert estimate.comm_s == 0.0
    assert estimate.total_s == pytest.approx(3.0)
    assert estimate.estimated_time_s == pytest.approx(3.0)


def test_cost_model_marks_positive_work_with_unknown_rate_as_infinite_time():
    from renormalizer.backend import HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    assert backend._rate_seconds(48, None) == float("inf")
    assert backend._rate_seconds(48, 0.0) == float("inf")
    assert backend._rate_seconds(0, None) == 0.0

    estimate = backend.estimate_contraction(plan, HardwareModel())

    assert estimate.compute_s == float("inf")
    assert estimate.memory_s == float("inf")
    assert estimate.total_s == float("inf")
    assert estimate.estimated_time_s == float("inf")


def test_contraction_cost_model_rejects_peak_memory_limit_violation():
    from renormalizer.backend import BackendFeatureError, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    with pytest.raises(BackendFeatureError, match="max_memory.*63.*peak.*64"):
        backend.estimate_contraction(plan, HardwareModel(max_memory_bytes=63))


def test_contraction_cost_model_rejects_workspace_limit_violation():
    from dataclasses import replace

    from renormalizer.backend import BackendFeatureError, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    matmul_plan = replace(plan.steps[0].plan, workspace_bytes=64)
    step = replace(plan.steps[0], plan=matmul_plan, required_workspace_bytes=64)
    plan = replace(plan, steps=(step,), required_workspace_bytes=64)

    with pytest.raises(BackendFeatureError, match="workspace_limit.*32.*requires.*64"):
        backend.estimate_contraction(plan, HardwareModel(workspace_limit_bytes=32))


def test_distributed_cost_model_rejects_peak_memory_limit_violation():
    from renormalizer.backend import BackendFeatureError, DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.ones((4, 3), dtype=np.float64)
    right = np.ones((3, 2), dtype=np.float64)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    distributed_path = backend.plan_distributed_contraction_path(dense_path, mesh)

    with pytest.raises(BackendFeatureError, match="max_memory.*95.*peak.*96"):
        backend.estimate_contraction(distributed_path, HardwareModel(max_memory_bytes=95))


def test_distributed_cost_model_rejects_workspace_limit_violation():
    from dataclasses import replace

    from renormalizer.backend import BackendFeatureError, DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.ones((4, 3), dtype=np.float64)
    right = np.ones((3, 2), dtype=np.float64)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    distributed_path = backend.plan_distributed_contraction_path(dense_path, mesh)
    local_step = replace(distributed_path.steps[0].local_step, required_workspace_bytes=64)
    distributed_step = replace(distributed_path.steps[0], local_step=local_step)
    distributed_path = replace(distributed_path, steps=(distributed_step,))

    with pytest.raises(BackendFeatureError, match="workspace_limit.*32.*requires.*64"):
        backend.estimate_contraction(distributed_path, HardwareModel(workspace_limit_bytes=32))


def test_estimate_distributed_contraction_reports_local_work_and_communication_cost():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.ones((4, 3), dtype=np.float64)
    right = np.ones((3, 2), dtype=np.float64)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    hw = HardwareModel(flop_per_s=10.0, memory_bandwidth_Bps=64.0, network_bandwidth_Bps=100.0, latency_s=0.2)
    distributed_path = backend.plan_distributed_contraction_path(dense_path, mesh, cost_model=hw)

    estimate = backend.estimate_contraction(distributed_path, hw)

    assert estimate.flops == 48
    assert estimate.read_bytes == 96
    assert estimate.write_bytes == 32
    assert estimate.comm_bytes == 144
    assert estimate.peak_bytes == 96
    assert estimate.compute_s == pytest.approx(2.4)
    assert estimate.memory_s == pytest.approx(2.0)
    assert estimate.comm_s == pytest.approx(1.16)
    assert estimate.total_s == pytest.approx(5.56)


def test_estimate_target_device_contraction_uses_distributed_local_cost():
    from renormalizer.backend import DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((4, 3), dtype=np.float64)
    right = np.ones((3, 2), dtype=np.float64)
    spec = backend.parse_einsum("ik,kj->ij", left, right)
    hw = HardwareModel(flop_per_s=10.0, memory_bandwidth_Bps=64.0, network_bandwidth_Bps=100.0, latency_s=0.2)
    plan = backend.plan_contraction(
        spec,
        allow_distribution=True,
        target_devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
    )

    estimate = backend.estimate_contraction(plan, hw)

    assert plan.distributed_modes == ("i",)
    assert estimate.flops == 48
    assert estimate.read_bytes == 96
    assert estimate.write_bytes == 32
    assert estimate.comm_bytes == 144
    assert estimate.peak_bytes == 96
    assert estimate.compute_s == pytest.approx(2.4)
    assert estimate.memory_s == pytest.approx(2.0)
    assert estimate.comm_s == pytest.approx(1.16)
    assert estimate.total_s == pytest.approx(5.56)


def test_estimate_redistribute_records_communication_cost():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    row_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    col_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"col": 2},
        mode_to_mesh_axis={"col": "rank"},
    )
    hw = HardwareModel(p2p_bandwidth_Bps=96.0, network_bandwidth_Bps=80.0, latency_s=0.25)

    estimate = backend.estimate_redistribute(row_spec, col_spec, (5, 4), hw, itemsize=8)

    assert estimate.comm_bytes == 160
    assert estimate.copy_bytes == 160
    assert estimate.peak_bytes == 96
    assert estimate.copy_s == pytest.approx(1.0)
    assert estimate.comm_s == pytest.approx(1.45)
    assert estimate.total_s == pytest.approx(2.45)


def test_estimate_redistribute_rejects_peak_memory_limit_violation():
    from renormalizer.backend import BackendFeatureError, DeviceMesh, DeviceSpec, HardwareModel, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    row_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    col_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"col": 2},
        mode_to_mesh_axis={"col": "rank"},
    )

    with pytest.raises(BackendFeatureError, match="max_memory.*95.*peak.*96"):
        backend.estimate_redistribute(
            row_spec,
            col_spec,
            (5, 4),
            HardwareModel(max_memory_bytes=95),
            itemsize=8,
        )


def test_estimate_redistribute_rejects_shape_mismatch_with_sharding_specs():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=(DeviceSpec("cpu", global_rank=0), DeviceSpec("cpu", global_rank=1)),
        shape=(2,),
        axis_names=("rank",),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    row_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"row": 2},
        mode_to_mesh_axis={"row": "rank"},
    )
    col_spec = ShardingSpec(
        global_shape=(5, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"col": 2},
        mode_to_mesh_axis={"col": "rank"},
    )
    shape_mismatch = ShardingSpec(
        global_shape=(6, 4),
        modes=("row", "col"),
        mesh=mesh,
        ranks_per_mode={"col": 2},
        mode_to_mesh_axis={"col": "rank"},
    )
    mode_mismatch = ShardingSpec(
        global_shape=(5, 4),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"j": 2},
        mode_to_mesh_axis={"j": "rank"},
    )

    with pytest.raises(ValueError, match="estimate_redistribute tensor_shape must match source global_shape"):
        backend.estimate_redistribute(row_spec, col_spec, (6, 4), itemsize=8)

    with pytest.raises(ValueError, match="estimate_redistribute tensor_shape must match destination global_shape"):
        backend.estimate_redistribute(row_spec, shape_mismatch, (5, 4), itemsize=8)

    with pytest.raises(ValueError, match="estimate_redistribute source and destination modes must match"):
        backend.estimate_redistribute(row_spec, mode_mismatch, (5, 4), itemsize=8)


def test_estimate_redistribute_rejects_invalid_tensor_shape_and_itemsize():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(ValueError, match="estimate_redistribute tensor_shape dimensions must be non-negative"):
        backend.estimate_redistribute(None, None, (5, -4), itemsize=8)

    with pytest.raises(ValueError, match="estimate_redistribute itemsize must be positive"):
        backend.estimate_redistribute(None, None, (5, 4), itemsize=0)

    with pytest.raises(ValueError, match="estimate_redistribute itemsize must be positive"):
        backend.estimate_redistribute(None, None, (5, 4), itemsize=-8)


def test_communication_plan_message_count_contributes_latency():
    from renormalizer.backend import CommunicationPlan, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    communication = CommunicationPlan(
        kind="allreduce",
        bytes=96,
        modes=("k",),
        num_messages=3,
        block_size=32,
    )

    assert communication.num_messages == 3
    assert communication.block_size == 32
    assert backend._estimate_communication_sequence_s(
        (communication,),
        HardwareModel(network_bandwidth_Bps=24.0, latency_s=0.5),
    ) == pytest.approx(5.5)


def test_communication_plan_local_bytes_drive_per_rank_timing():
    from renormalizer.backend import CommunicationPlan, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    communication = CommunicationPlan(
        kind="alltoall",
        bytes=96,
        local_bytes=48,
        modes=("row", "col"),
        num_messages=2,
    )

    assert communication.bytes == 96
    assert communication.local_bytes == 48
    assert communication.block_size == 24
    assert backend._estimate_communication_sequence_s(
        (communication,),
        HardwareModel(network_bandwidth_Bps=24.0, latency_s=0.5),
    ) == pytest.approx(3.0)


def test_point_to_point_communication_plan_uses_p2p_bandwidth():
    from renormalizer.backend import CommunicationPlan, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    communication = CommunicationPlan(
        kind="point_to_point",
        bytes=96,
        local_bytes=48,
        modes=("row",),
        num_messages=2,
    )
    collective = CommunicationPlan(
        kind="allreduce",
        bytes=96,
        local_bytes=48,
        modes=("row",),
        num_messages=2,
    )
    hw = HardwareModel(
        network_bandwidth_Bps=12.0,
        p2p_bandwidth_Bps=48.0,
        latency_s=0.5,
    )

    assert communication.block_size == 24
    assert backend._estimate_communication_sequence_s((communication,), hw) == pytest.approx(2.0)
    assert backend._estimate_communication_sequence_s((collective,), hw) == pytest.approx(5.0)


def test_point_to_point_communication_profile_exposes_primitive_metrics():
    from renormalizer.backend import CommunicationPlan
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling

    backend = NumpyBackend()
    communication = backend._profile_communication_plan(
        CommunicationPlan(
            kind="point_to_point",
            bytes=96,
            local_bytes=48,
            modes=("row",),
            num_messages=2,
        ),
        wall_s=0.25,
    )

    event = profiling.standardize_event_payload({
        "event": "contraction_execute",
        "lowering": "distributed",
        "distributed_modes": ["row"],
        "communication": [communication],
        "read_bytes": 0,
        "write_bytes": 0,
        "copy_bytes": 0,
    })

    assert event["communication"][0]["kind"] == "point_to_point"
    assert event["communication"][0]["primitive"] == "point_to_point"
    assert event["communication"][0]["is_point_to_point"] is True
    assert event["communication"][0]["is_collective"] is False
    assert event["communication_profile"]["bytes_by_primitive"] == {"point_to_point": 96}
    assert event["communication_profile"]["messages_by_primitive"] == {"point_to_point": 2}
    assert event["communication_profile"]["wall_s_by_primitive"] == {"point_to_point": pytest.approx(0.25)}
    assert event["communication_profile"]["dominant_primitive"] == "point_to_point"
    assert event["communication_profile"]["point_to_point_bytes"] == 96
    assert event["communication_profile"]["num_point_to_point_messages"] == 2


def test_communication_plan_requires_positive_message_count():
    from renormalizer.backend import CommunicationPlan

    with pytest.raises(ValueError, match="num_messages must be positive"):
        CommunicationPlan(kind="broadcast", bytes=64, num_messages=0)


def test_communication_plan_rejects_negative_block_size():
    from renormalizer.backend import CommunicationPlan

    with pytest.raises(ValueError, match="block_size must be non-negative"):
        CommunicationPlan(kind="alltoall", bytes=64, block_size=-1)


def test_communication_plan_rejects_block_size_larger_than_local_payload():
    from renormalizer.backend import CommunicationPlan

    with pytest.raises(ValueError, match="block_size must not exceed local_bytes"):
        CommunicationPlan(kind="alltoall", bytes=96, local_bytes=48, block_size=49)


def test_communication_plan_rejects_unknown_kind():
    from renormalizer.backend import CommunicationPlan

    with pytest.raises(ValueError, match="Unknown CommunicationPlan kind"):
        CommunicationPlan(kind="mystery", bytes=64)


def test_sharding_spec_can_replicate_over_unused_mesh_axes():
    from renormalizer.backend import DeviceMesh, DeviceSpec, ShardingSpec

    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(4)),
        shape=(2, 2),
        axis_names=("row", "col"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )

    spec = ShardingSpec(
        global_shape=(8, 4),
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": 2},
        mode_to_mesh_axis={"i": "row"},
    )

    assert spec.sharded_modes == ("i",)
    assert spec.replicated_modes == ("k",)
    assert spec.local_slices[0] == (slice(0, 4), slice(None))
    assert spec.local_slices[1] == (slice(0, 4), slice(None))
    assert spec.local_slices[2] == (slice(4, 8), slice(None))
    assert spec.local_slices[3] == (slice(4, 8), slice(None))


def test_auto_distributed_dense_plan_uses_multi_axis_output_sharding():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(4)),
        shape=(2, 2),
        axis_names=("row", "col"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(32, dtype=np.float64).reshape(8, 4)
    right = np.arange(24, dtype=np.float64).reshape(4, 6)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
    )
    result = backend.execute(distributed_path)

    assert distributed_path.output_sharding.sharded_modes == ("i", "j")
    assert distributed_path.output_sharding.ranks_per_mode == {"i": 2, "j": 2}
    assert distributed_path.output_sharding.mode_to_mesh_axis == {"i": "row", "j": "col"}
    assert distributed_path.output_sharding.local_slices[0] == (slice(0, 4), slice(0, 3))
    assert distributed_path.output_sharding.local_slices[3] == (slice(4, 8), slice(3, 6))
    assert result.local_shape == (4, 3)
    assert np.allclose(result.local_array, (left @ right)[:4, :3])
    assert np.allclose(backend.gather_tensor(result), left @ right)


def test_auto_distributed_dense_plan_matches_largest_output_mode_to_largest_mesh_axis():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(8)),
        shape=(2, 4),
        axis_names=("small", "large"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(16 * 4, dtype=np.float64).reshape(16, 4)
    right = np.arange(4 * 32, dtype=np.float64).reshape(4, 32)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=4096,
        cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
    )

    assert distributed_path.output_sharding.ranks_per_mode == {"i": 2, "j": 4}
    assert distributed_path.output_sharding.mode_to_mesh_axis == {"i": "small", "j": "large"}


def test_auto_distributed_dense_plan_does_not_shard_singleton_output_modes():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(8)),
        shape=(2, 4),
        axis_names=("small", "large"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(1 * 4, dtype=np.float64).reshape(1, 4)
    right = np.arange(4 * 8, dtype=np.float64).reshape(4, 8)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=4096,
        cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
    )

    assert distributed_path.output_sharding.ranks_per_mode == {"j": 4}
    assert distributed_path.output_sharding.mode_to_mesh_axis == {"j": "large"}
    assert distributed_path.output_sharding.sharded_modes == ("j",)


def test_auto_distributed_dense_plan_rejects_all_singleton_output_modes():
    from renormalizer.backend import BackendFeatureError, DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(8)),
        shape=(2, 4),
        axis_names=("small", "large"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(1 * 4, dtype=np.float64).reshape(1, 4)
    right = np.arange(4 * 1, dtype=np.float64).reshape(4, 1)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    with pytest.raises(BackendFeatureError, match="no shardable output modes"):
        backend.plan_distributed_contraction_path(
            dense_path,
            mesh,
            memory_limit_per_device=4096,
            cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
        )


def test_auto_distributed_dense_plan_avoids_oversharding_output_modes():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(8)),
        shape=(4, 2),
        axis_names=("large", "small"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(2 * 4, dtype=np.float64).reshape(2, 4)
    right = np.arange(4 * 1, dtype=np.float64).reshape(4, 1)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=4096,
        cost_model=HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25),
    )

    assert distributed_path.output_sharding.ranks_per_mode == {"i": 2}
    assert distributed_path.output_sharding.mode_to_mesh_axis == {"i": "small"}
    assert distributed_path.output_sharding.sharded_modes == ("i",)


def test_auto_distributed_dense_plan_reports_local_step_costs():
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(4)),
        shape=(2, 2),
        axis_names=("row", "col"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(32, dtype=np.float64).reshape(8, 4)
    right = np.arange(24, dtype=np.float64).reshape(4, 6)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=HardwareModel(flop_per_s=48.0, network_bandwidth_Bps=80.0, latency_s=0.25),
    )
    step = distributed_path.steps[0]

    assert dense_path.estimated_flops == 384
    assert step.local_step.estimated_flops == 96
    assert step.local_step.estimated_read_bytes == (4 * 4 + 4 * 3) * left.itemsize
    assert step.local_step.estimated_write_bytes == 4 * 3 * left.itemsize
    assert step.estimated_compute_s == pytest.approx(2.0)
    assert step.estimated_total_s == pytest.approx(step.estimated_compute_s + step.estimated_comm_s)


def test_auto_distributed_execution_profile_reports_local_work_metrics(tmp_path):
    from renormalizer.backend import DeviceMesh, DeviceSpec, HardwareModel
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    mesh = DeviceMesh(
        devices=tuple(DeviceSpec("cpu", global_rank=rank) for rank in range(4)),
        shape=(2, 2),
        axis_names=("row", "col"),
        backend="numpy",
        local_rank=0,
        global_rank=0,
    )
    left = np.arange(32, dtype=np.float64).reshape(8, 4)
    right = np.arange(24, dtype=np.float64).reshape(4, 6)
    dense_path = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    distributed_path = backend.plan_distributed_contraction_path(
        dense_path,
        mesh,
        memory_limit_per_device=1024,
        cost_model=HardwareModel(flop_per_s=48.0, network_bandwidth_Bps=80.0, latency_s=0.25),
    )
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        result = backend.execute(distributed_path)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(backend.gather_tensor(result), left @ right)
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    execute = next(event for event in events if event["event"] == "contraction_execute")

    assert execute["execution_primitives"] == ["distributed_contract", "matmul"]
    assert execute["execution_policies"] == ["backend_distributed_contract", "backend_matmul"]
    assert execute["fallback_reasons"] == []
    assert execute["flops"] == 384
    assert execute["read_bytes"] == left.nbytes + right.nbytes
    assert execute["write_bytes"] == 8 * 6 * left.itemsize
    assert execute["local_flops"] == 96
    assert execute["local_read_bytes"] == (4 * 4 + 4 * 3) * left.itemsize
    assert execute["local_write_bytes"] == 4 * 3 * left.itemsize
    assert execute["local_copy_bytes"] == 0
    assert execute["local_workspace_bytes"] == 0
    assert execute["local_peak_bytes"] == 4 * 3 * left.itemsize


def test_workspace_stream_and_unified_execute_api_are_explicit():
    from renormalizer.backend import StreamEvent, Workspace
    from renormalizer.backend.execution import DeviceSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    workspace = backend.allocate_workspace(32)
    stream = backend.default_stream()
    event = backend.record_event(stream=stream)
    result = backend.execute(plan, stream=stream, workspace=workspace)

    assert isinstance(workspace, Workspace)
    assert workspace.device == DeviceSpec(kind="cpu")
    assert workspace.nbytes == 32
    assert workspace.buffer.nbytes >= 32
    assert backend.new_stream() is None
    assert isinstance(event, StreamEvent)
    assert event.device == DeviceSpec(kind="cpu")
    assert backend.wait_event(event, stream=stream) is None
    assert backend.release_workspace(workspace) is None
    assert np.allclose(result, left @ right)


def test_allocate_workspace_forwards_requested_device_to_buffer_allocation():
    from renormalizer.backend.abstract import AbstractBackend
    from renormalizer.backend.execution import DeviceSpec

    class RecordingBackend(AbstractBackend):
        name = "recording"

        def __init__(self):
            super().__init__()
            self.seen_device = None

        def _setup(self):
            self.device = "cpu"
            self.xp = np

        def set_device(self, device):
            self._device_spec = DeviceSpec(kind="cpu") if device is None else DeviceSpec(kind="cuda", index=3)

        def current_device(self):
            return getattr(self, "_device_spec", DeviceSpec(kind="cpu"))

        def to_backend(self, x, *, device=None, dtype=None, copy=None):
            self.seen_device = device
            return np.asarray(x, dtype=dtype)

    backend = RecordingBackend()
    requested = DeviceSpec(kind="cuda", index=3)

    workspace = backend.allocate_workspace(16, device=requested)

    assert workspace.device == requested
    assert backend.seen_device == requested


def test_stream_event_and_workspace_normalize_device_metadata():
    from renormalizer.backend import StreamEvent, Workspace
    from renormalizer.backend.execution import DeviceSpec

    event = StreamEvent(device="gpu", stream="s", token="t")
    workspace = Workspace(device="cpu", nbytes="32", buffer=np.empty(32, dtype=np.uint8))

    assert event.device == DeviceSpec(kind="cuda")
    assert workspace.device == DeviceSpec(kind="cpu")
    assert workspace.nbytes == 32


def test_stream_event_and_workspace_reject_invalid_metadata():
    from renormalizer.backend import StreamEvent, Workspace
    from renormalizer.backend.execution import DeviceSpec

    with pytest.raises(ValueError, match="Unknown backend device"):
        StreamEvent(device="quantum")

    with pytest.raises(ValueError, match="Workspace nbytes must be non-negative"):
        Workspace(device=DeviceSpec(kind="cpu"), nbytes=-1, buffer=np.empty(0, dtype=np.uint8))

    with pytest.raises(ValueError, match="Unknown backend device"):
        Workspace(device="quantum", nbytes=0, buffer=np.empty(0, dtype=np.uint8))


def test_release_workspace_validates_workspace_metadata():
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec, Workspace
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(TypeError, match="workspace must be a Workspace instance"):
        backend.release_workspace(object())

    wrong_device = Workspace(
        device=DeviceSpec(kind="cuda"),
        nbytes=0,
        buffer=np.empty(0, dtype=np.uint8),
    )
    with pytest.raises(BackendFeatureError, match="workspace device"):
        backend.release_workspace(wrong_device)


def test_released_workspace_is_marked_and_rejected_by_execution():
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    workspace = backend.allocate_workspace(32)

    backend.release_workspace(workspace)

    assert workspace.released is True
    assert workspace.buffer is None
    with pytest.raises(BackendFeatureError, match="released workspace"):
        backend.execute(plan, workspace=workspace)


def test_wait_event_rejects_invalid_event_metadata():
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec, StreamEvent
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(TypeError, match="event must be a StreamEvent instance"):
        backend.wait_event(object())

    wrong_device = StreamEvent(device=DeviceSpec(kind="cuda"))
    with pytest.raises(BackendFeatureError, match="stream event device"):
        backend.wait_event(wrong_device)


def test_explicit_contract_forwards_workspace_to_unified_execute():
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec, Workspace
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    workspace = Workspace(device=DeviceSpec(kind="cuda"), nbytes=0, buffer=np.empty(0, dtype=np.uint8))

    with pytest.raises(BackendFeatureError, match="workspace device"):
        backend.contract("ik,kj->ij", left, right, workspace=workspace)


def test_explicit_contract_forwards_stream_and_workspace_to_unified_execute():
    from renormalizer.backend.numpy_backend import NumpyBackend

    class RecordingBackend(NumpyBackend):
        def execute(self, plan, *, stream=None, workspace=None):
            self.seen_stream = stream
            self.seen_workspace = workspace
            return super().execute(plan, stream=stream, workspace=workspace)

    backend = RecordingBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    stream = object()
    workspace = backend.allocate_workspace(32)

    result = backend.contract("ik,kj->ij", left, right, stream=stream, workspace=workspace)

    assert np.allclose(result, left @ right)
    assert backend.seen_stream is stream
    assert backend.seen_workspace is workspace


def test_contraction_execute_profile_records_stream_and_workspace_resources(tmp_path):
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    stream = object()
    workspace = backend.allocate_workspace(32)
    event_path = tmp_path / "events.jsonl"

    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute(plan, stream=stream, workspace=workspace)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)
    event = next(
        payload
        for payload in (
            json.loads(line)
            for line in event_path.read_text().splitlines()
            if line.strip()
        )
        if payload["event"] == "contraction_execute"
    )
    assert event["stream_provided"] is True
    assert event["stream_type"] == "object"
    assert event["workspace_provided"] is True
    assert event["workspace_nbytes"] == 32
    assert event["workspace_device_kind"] == "cpu"
    assert event["workspace_device_index"] is None


def test_execute_rejects_workspace_smaller_than_plan_requirement():
    from dataclasses import replace

    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    matmul_plan = replace(plan.steps[0].plan, workspace_bytes=64)
    step = replace(plan.steps[0], plan=matmul_plan, required_workspace_bytes=64)
    plan = replace(plan, steps=(step,), required_workspace_bytes=64)
    workspace = backend.allocate_workspace(32)

    with pytest.raises(BackendFeatureError, match="workspace.*64.*32"):
        backend.execute(plan, workspace=workspace)


def test_execute_rejects_workspace_on_wrong_device():
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec, Workspace
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    workspace = Workspace(device=DeviceSpec(kind="cuda"), nbytes=0, buffer=np.empty(0, dtype=np.uint8))

    with pytest.raises(BackendFeatureError, match="workspace device"):
        backend.execute(plan, workspace=workspace)
