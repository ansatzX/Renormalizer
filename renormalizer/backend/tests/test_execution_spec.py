# -*- coding: utf-8 -*-

import json

import numpy as np
import pytest


def test_device_spec_parses_cpu_and_indexed_cuda_aliases():
    from renormalizer.backend.execution import DeviceSpec, parse_device_spec

    assert parse_device_spec("cpu") == DeviceSpec(kind="cpu")
    assert parse_device_spec("gpu") == DeviceSpec(kind="cuda")
    assert parse_device_spec("cuda:1") == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert parse_device_spec("gpu:2") == DeviceSpec(kind="cuda", index=2, visible_id="2")

    with pytest.raises(ValueError, match="Unknown backend device"):
        parse_device_spec("quantum:0")


def test_backend_config_keeps_legacy_device_kind_and_exposes_device_spec():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec, FallbackPolicy

    config = BackendConfig(device="cuda:1", fallback_policy="record")

    assert config.device == "gpu"
    assert config.device_spec == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert config.fallback_policy is FallbackPolicy.RECORD


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
    assert backend.capabilities.distributed_array is True
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


def test_pair_contraction_lowering_reports_batched_gemm_for_same_shape_batch():
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

    assert plan.kind == "batched_gemm"
    assert plan.descs[0].batch_shape == (5,)
    assert plan.fallback_reason is None


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


def test_backend_parse_einsum_rejects_operand_rank_mismatch():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(ValueError, match="rank"):
        backend.parse_einsum("ab,bc->ac", np.ones((2, 3, 1)), np.ones((3, 4)))


def test_backend_raw_matmul_and_stacked_batched_matmul():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    batch_a = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    batch_b = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)

    assert np.allclose(backend.matmul(a, b), a @ b)
    assert np.allclose(backend.batched_matmul(batch_a, batch_b), np.matmul(batch_a, batch_b))


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

    with pytest.raises(BackendFeatureError, match="native grouped_gemm unavailable"):
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

    with pytest.warns(RuntimeWarning, match="native grouped_gemm unavailable"):
        results = backend.grouped_gemm([task])

    assert len(results) == 1
    assert np.allclose(results[0], task.A @ task.B)


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


def test_torch_grouped_gemm_is_backend_primitive_when_available():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available
    from renormalizer.backend.gemm import GemmTask

    if not is_backend_available("torch"):
        pytest.skip("torch unavailable")

    backend = create_backend("torch", config=BackendConfig(device="cpu", fallback_policy="forbid"))
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


def test_cupy_grouped_gemm_is_backend_primitive_when_available():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available
    from renormalizer.backend.gemm import GemmTask

    if not is_backend_available("cupy"):
        pytest.skip("cupy unavailable")

    try:
        backend = create_backend("cupy", config=BackendConfig(device="gpu", fallback_policy="forbid"))
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
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = backend.grouped_gemm(tasks, pack_threshold=2)
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
    assert event["flops"] == 8388728
    assert event["read_bytes"] == 524544
    assert event["write_bytes"] == 262264
    assert event["copy_bytes"] == 524288
    assert event["fallback_reason"] == "native grouped_gemm unavailable; used bucketed fallback"
    assert event["bucket_task_counts"] == [1, 2]
    assert event["shape_buckets"] == [
        {"m": 3, "n": 5, "k": 4, "task_indices": [2], "task_count": 1},
        {"m": 128, "n": 128, "k": 128, "task_indices": [0, 1], "task_count": 2},
    ]
    assert event["batched_bucket_count"] == 1
    assert event["loop_bucket_count"] == 1
    assert event["wall_s"] >= 0.0


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
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_grouped_gemm_plan(plan, pack_threshold=2)
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
    event = next(payload for payload in payloads if payload.get("lowering") == "block_grouped_gemm")
    assert event["backend"] == "numpy"
    assert event["output_modes"] == ["i", "j"]
    assert event["global_shape"] == [2, 4]
    assert event["num_grouped_tasks"] == 2
    assert event["num_blocks"] == 1
    assert event["num_shape_buckets"] == 1
    assert event["scatter_add_required"] is True
    assert event["shape_buckets"] == [
        {"m": 2, "n": 4, "k": 3, "task_indices": [0, 1], "task_count": 2},
    ]
    assert event["output_block_keys"] == [
        {"extra": [], "qn_left": [0], "qn_right": [1]},
        {"extra": [], "qn_left": [0], "qn_right": [1]},
    ]
    assert event["unique_output_block_keys"] == [
        {"extra": [], "qn_left": [0], "qn_right": [1]},
    ]
    assert event["result_block_shapes"] == [{"extra": [], "qn_left": [0], "qn_right": [1], "shape": [2, 4]}]
    assert event["fallback_reason"] is None
    assert event["wall_s"] >= 0.0


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


def test_jax_grouped_gemm_plan_accumulates_sparse_blocks_when_available():
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

    result = backend.execute_grouped_gemm_plan(backend.lower_block_contraction(spec), pack_threshold=2)

    expected = left_np[BlockKey((0,), (0,))] @ right_np[BlockKey((0,), (1,))]
    expected += left_np[BlockKey((0,), (2,))] @ right_np[BlockKey((2,), (1,))]
    assert np.allclose(backend.to_numpy(result.blocks[out_key].array), expected)


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

    assert plan.kind == "batched_gemm"
    assert result.shape == (5, 2, 3)
    assert np.allclose(result, expected)


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
    assert plan["lowering"] == "batched_gemm"
    assert plan["batch_modes"] == ["batch"]
    assert plan["output_modes"] == ["batch", "('left', 'site')", "('right', 'site')"]
    assert plan["operands"] == [
        {
            "name": "left_tensor",
            "modes": ["batch", "('left', 'site')", "bond"],
            "shape": [2, 3, 4],
            "dtype": "float32",
            "nbytes": 96,
            "ndim": 3,
            "strides": [48, 16, 4],
            "order": "C",
            "contiguous": True,
            "backend": "numpy",
            "device": "DeviceSpec(kind='cpu', index=None, local_rank=None, global_rank=None, visible_id=None)",
            "device_kind": "cpu",
            "device_index": None,
            "is_host": True,
            "is_device": False,
        },
        {
            "name": "right_tensor",
            "modes": ["batch", "bond", "('right', 'site')"],
            "shape": [2, 4, 5],
            "dtype": "float32",
            "nbytes": 160,
            "ndim": 3,
            "strides": [80, 20, 4],
            "order": "C",
            "contiguous": True,
            "backend": "numpy",
            "device": "DeviceSpec(kind='cpu', index=None, local_rank=None, global_rank=None, visible_id=None)",
            "device_kind": "cpu",
            "device_index": None,
            "is_host": True,
            "is_device": False,
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
    assert plan["flops"] == 48
    assert plan["num_gemm"] == 1
    assert plan["fallback_reason"] is None
    execute = executes[0]
    assert execute["backend"] == "numpy"
    assert execute["lowering"] == "gemm"
    assert execute["input_shapes"] == [[2, 3], [3, 4]]
    assert execute["output_shape"] == [2, 4]
    assert execute["flops"] == 48
    assert execute["num_gemm"] == 1
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
    assert execute["plan_hash"] == plan.plan_hash
    assert execute["equation"] == "ik,kj->ij"
    assert execute["input_modes"] == [["i", "k"], ["k", "j"]]
    assert execute["output_modes"] == ["i", "j"]


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
    assert event["equation"] == "ik,kj->ij"
    assert event["lowering"] == "distributed"
    assert event["local_lowering"] == "gemm"
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
            "collective": "redistribute",
            "bytes": 96,
            "local_bytes": 48,
            "modes": ["j"],
            "num_messages": 1,
            "block_size": 48,
            "wall_s": 0.0,
        },
        {
            "collective": "gather",
            "bytes": 160,
            "local_bytes": 96,
            "modes": ["i"],
            "num_messages": 1,
            "block_size": 96,
            "wall_s": 0.0,
        },
    ]
    assert event["comm_bytes"] == 256
    assert event["wall_s"] >= 0.0
    communications = plan.steps[0].plan.steps[0].communication
    assert [(item.num_messages, item.block_size) for item in communications] == [(1, 48), (1, 96)]


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
    communication = plan.steps[0].plan.steps[0].communication

    assert communication[0].kind == "alltoall"
    assert communication[0].bytes == left.shape[0] * right.shape[1] * left.itemsize
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
    assert distributed_path.peak_local_bytes == 96
    assert distributed_path.estimated_comm_bytes == 256
    assert distributed_path.steps[0].local_contraction_plan is distributed_path.steps[0].local_step.plan
    assert distributed_path.steps[0].communication_plan == distributed_path.steps[0].communication[0]
    assert [item.kind for item in distributed_path.steps[0].communication] == ["redistribute", "gather"]
    assert [(item.bytes, item.local_bytes) for item in distributed_path.steps[0].communication] == [(96, 48), (160, 96)]
    assert [(item.num_messages, item.block_size) for item in distributed_path.steps[0].communication] == [(1, 48), (1, 96)]
    assert distributed_path.steps[0].estimated_comm_s == pytest.approx(2.3)
    assert distributed_path.steps[0].estimated_total_s == pytest.approx(2.3)
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
    assert [item.kind for item in allreduce_path.steps[0].communication] == ["allreduce"]


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
    assert event["estimated_compute_s"] == pytest.approx(step.estimated_compute_s)
    assert event["estimated_comm_s"] == pytest.approx(step.estimated_comm_s)
    assert event["estimated_total_s"] == pytest.approx(step.estimated_total_s)
    assert event["communication"][0]["collective"] == "activate_distribution"
    assert event["communication"][0]["bytes"] == left.nbytes + right.nbytes
    assert event["communication"][0]["local_bytes"] == 352


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

    assert np.allclose(result, left @ right)
    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")

    assert event["plan_hash"] == plan.plan_hash
    assert event["lowering"] == "slice"
    assert event["equation"] == "ik,kj->ij"
    assert event["input_modes"] == [["i", "k"], ["k", "j"]]
    assert event["output_modes"] == ["i", "j"]
    assert event["sliced_modes"] == [str(mode) for mode in plan.sliced_modes]
    assert event["num_slices"] == len(plan.steps[0].plan.output_slices)
    assert event["base_lowering"] == "gemm"
    assert event["peak_bytes"] == plan.estimated_peak_bytes


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
    assert event["peak_bytes"] == plan.estimated_peak_bytes
    assert event["read_bytes"] == plan.estimated_read_bytes


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
    hw = HardwareModel(network_bandwidth_Bps=80.0, latency_s=0.25)

    estimate = backend.estimate_redistribute(row_spec, col_spec, (5, 4), hw, itemsize=8)

    assert estimate.comm_bytes == 160
    assert estimate.copy_bytes == 160
    assert estimate.peak_bytes == 96
    assert estimate.comm_s == pytest.approx(1.45)
    assert estimate.total_s == pytest.approx(1.45)


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
    assert communication.block_size == 48
    assert backend._estimate_communication_sequence_s(
        (communication,),
        HardwareModel(network_bandwidth_Bps=24.0, latency_s=0.5),
    ) == pytest.approx(3.0)


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
    from renormalizer.backend.execution import BackendFeatureError, DeviceSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    workspace = backend.allocate_workspace(0, device=DeviceSpec(kind="cuda"))

    with pytest.raises(BackendFeatureError, match="workspace device"):
        backend.execute(plan, workspace=workspace)
