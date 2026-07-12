from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.center import (
    CenterVectorMap,
    MeasuredCollective,
    QnMaskIdentity,
    MappedDistributedLocalOperator,
    _adapter_decision_digest,
    center_materialization_memory_profile,
    coordinate_adapter_decision,
    run_synchronized_state_update,
    validate_distributed_backend,
)
from renormalizer.backend._distributed.collectives import SingleProcessCollective
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.local_operator import DistributedLocalOperator
from renormalizer.backend._distributed.planner import plan_distributed_execution
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend._distributed.sharding import DistributedTensor
from renormalizer.backend._distributed.solvers import run_sharded_krylov
from renormalizer.backend._execution.model import ExecutionBindings
from renormalizer.backend._execution.planner import lower_einsum_path
from renormalizer.backend._distributed.sharding import shard_axis
from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import create_backend


class NoGatherBroadcast:
    def __init__(self, remote_shards):
        self.rank = 0
        self.size = len(remote_shards)
        self.remote_shards = tuple(remote_shards)
        self.broadcast_roots = []

    def broadcast(self, value, *, root):
        self.broadcast_roots.append(root)
        if root != self.rank:
            value[...] = self.remote_shards[root]
        return value

    def allreduce(self, value, *, op="sum"):
        return np.array(value, copy=True)

    def allgather(self, *_args, **_kwargs):
        raise AssertionError("center materialization must not allgather")


def _numpy_backend():
    return create_backend(
        "numpy",
        config=BackendConfig(device="cpu", execution_policy="execution_ir"),
    )


class _TraceCollective:
    rank = 0
    size = 2

    def __init__(self, *, remote_failure=False, digest_mismatch=False):
        self.trace = []
        self.remote_failure = remote_failure
        self.digest_mismatch = digest_mismatch

    def allreduce(self, value, *, op="sum"):
        self.trace.append(("allreduce", op, int(np.asarray(value).size)))
        result = np.array(value, copy=True)
        if self.remote_failure and op == "max" and result.dtype == np.int32:
            result[...] = 1
        if self.digest_mismatch and op == "max" and result.dtype == np.uint64:
            result[0] += 1
        return result

    def broadcast(self, value, *, root):
        self.trace.append(("broadcast", root, int(np.asarray(value).size)))
        return value

    def allgather(self, *_args, **_kwargs):
        raise AssertionError("complete center materialization must not allgather")


def _execution(collective):
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend.config import DistributedExecutionConfig

    return DistributedExecutionConfig(
        context=DistributedContext(0, 0, 2, 2),
        mesh=DeviceMesh((2,), ("rank",), 0),
        collective=collective,
        provider=DeviceResidentProvider(),
    )


def test_adapter_backend_digest_normalizes_valid_rank_local_devices():
    from types import SimpleNamespace

    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend.config import DistributedExecutionConfig

    def digest_for_rank(rank):
        class Collective:
            size = 2

            def __init__(self):
                self.rank = rank
                self.digest = None

            def allreduce(self, value, *, op="sum"):
                value = np.asarray(value)
                if op == "min" and value.dtype == np.uint64:
                    self.digest = np.array(value, copy=True)
                return np.array(value, copy=True)

        collective = Collective()
        context = DistributedContext(rank, rank, 2, 2)
        execution = DistributedExecutionConfig(
            context=context,
            mesh=DeviceMesh((2,), ("rank",), rank),
            collective=collective,
            provider=DeviceResidentProvider(),
            backend_name="cupy",
            backend_device="cuda:{}".format(rank),
            backend_precision=64,
        )
        selected = SimpleNamespace(
            name="cupy",
            device="cuda:{}".format(rank),
            config=SimpleNamespace(precision=64),
            supports_execution_ir=True,
        )
        validate_distributed_backend(execution, selected, network="mps")
        return collective.digest

    np.testing.assert_array_equal(digest_for_rank(0), digest_for_rank(1))


def test_wave9_oversized_qn_shape_product_fails_without_array_allocation():
    oversized_shape = (1 << 62, 4)
    sharding = shard_axis(oversized_shape, 0, 1)

    with pytest.raises(OverflowError, match="int64|range"):
        QnMaskIdentity(oversized_shape, (1,), "0" * 64)
    with pytest.raises(OverflowError, match="int64|range"):
        CenterVectorMap(sharding, sharding)


def _mapped_matrix_operator(matrix, mask=None):
    backend = _numpy_backend()
    source = lower_einsum_path(
        "ab,b->a", (matrix.shape, (matrix.shape[1],)), dtype=matrix.dtype
    )
    distributed = plan_distributed_execution(
        source, variable_key="input_1", world_size=1
    )
    context = DistributedContext(0, 0, 1, 1)
    collective = SingleProcessCollective()
    local = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=backend,
        context=context,
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )
    vector_map = CenterVectorMap(
        distributed.input_sharding,
        distributed.output_sharding,
        mask,
    )
    return MappedDistributedLocalOperator(local, vector_map)


def test_dense_center_map_round_trips_uneven_shards_without_allgather():
    dense = shard_axis((5, 3), axis=0, parts=2)
    vector_map = CenterVectorMap(dense, dense)
    center = np.arange(15, dtype=np.float64).reshape(5, 3)

    local = tuple(
        vector_map.extract_local(center, rank, _numpy_backend()) for rank in range(2)
    )

    assert vector_map.rank_counts == (9, 6)
    assert vector_map.solver_sharding.global_shape == (15,)
    assert vector_map.solver_sharding.local_slices == (
        (slice(0, 9),),
        (slice(9, 15),),
    )
    collective = NoGatherBroadcast(local)
    actual = vector_map.materialize(local[0], collective, _numpy_backend())
    np.testing.assert_array_equal(actual, center)
    assert collective.broadcast_roots == [0, 1]


def test_qn_center_map_uses_rank_major_dense_slab_order_and_baseline_pack():
    dense = shard_axis((3, 4), axis=1, parts=2)
    mask = np.array(
        [
            [True, False, True, False],
            [False, True, True, False],
            [True, True, False, True],
        ]
    )
    vector_map = CenterVectorMap(dense, dense, mask)
    center = np.arange(12, dtype=np.float64).reshape(3, 4)

    local = tuple(
        vector_map.extract_local(center, rank, _numpy_backend()) for rank in range(2)
    )

    np.testing.assert_array_equal(local[0], [0, 5, 8, 9])
    np.testing.assert_array_equal(local[1], [2, 6, 11])
    collective = NoGatherBroadcast(local)
    materialized = vector_map.materialize(local[0], collective, _numpy_backend())
    np.testing.assert_array_equal(materialized[mask], center[mask])
    np.testing.assert_array_equal(materialized[~mask], 0)
    np.testing.assert_array_equal(vector_map.pack_baseline(materialized), center[mask])


def test_center_map_rejects_mismatched_ownership_and_zero_count_qn_rank():
    rows = shard_axis((4, 4), axis=0, parts=2)
    columns = shard_axis((4, 4), axis=1, parts=2)

    with pytest.raises(ValueError, match="identical dense ownership"):
        CenterVectorMap(rows, columns)

    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, 0] = True
    with pytest.raises(ValueError, match="at least one allowed entry"):
        CenterVectorMap(rows, rows, mask)


def test_center_map_copies_mask_as_immutable_host_metadata():
    dense = shard_axis((4, 2), axis=0, parts=2)
    mask = np.ones((4, 2), dtype=bool)
    vector_map = CenterVectorMap(dense, dense, mask)
    mask[...] = False

    assert vector_map.qn_mask.flags.writeable is False
    assert vector_map.rank_counts == (4, 4)
    with pytest.raises(FrozenInstanceError):
        vector_map.rank_counts = (8,)


def test_mapped_operator_expands_qn_input_and_packs_dense_hv_output():
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4) / 7
    mask = np.array([True, False, True, False])
    operator = _mapped_matrix_operator(matrix, mask)
    local = np.array([2.0, -1.0])

    actual = operator(local).copy()

    dense = np.array([2.0, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(actual, (matrix @ dense)[mask])
    assert operator.solver_input_sharding == operator.vector_map.solver_sharding
    assert operator.solver_output_sharding == operator.vector_map.solver_sharding
    assert operator.solver_dtype == np.dtype("float64")
    assert operator.counters["allgather_calls"] == 0


def test_sharded_krylov_accepts_mapped_solver_operator_protocol():
    matrix = np.diag(np.arange(1, 5, dtype=np.float64))
    operator = _mapped_matrix_operator(matrix)
    initial = np.arange(1, 5, dtype=np.float64)
    vector = DistributedTensor(
        operator.solver_input_sharding,
        rank=0,
        local_array=initial.copy(),
    )

    result, iterations = run_sharded_krylov(
        operator,
        vector,
        -0.05,
        collective=operator.collective,
        config={"block_size": 4},
    )

    np.testing.assert_allclose(
        result.local_array,
        np.exp(-0.05 * np.diag(matrix)) * initial,
        rtol=1e-13,
        atol=1e-13,
    )
    assert iterations == 4
    assert operator.counters["allgather_calls"] == 0


def test_adapter_decision_digest_covers_complete_workflow_metadata():
    base = {
        "error": None,
        "artifact": None,
        "plan": None,
        "vector_map": None,
        "network": "mps",
        "operation": "krylov",
        "center_kind": "one_site",
        "center_shape": (4, 3),
        "topology": {"site_indices": (2,)},
        "qn_mask": None,
        "solver_controls": {"block_size": 12, "coefficient": -0.5j},
        "selectors": {"ivp_solver": "krylov"},
        "fallback_policy": "error",
        "fallback_reason_code": "none",
        "world_size": 2,
        "local_world_size": 2,
        "mesh_shape": (2,),
        "mesh_axis_names": ("rank",),
        "device_budget": 1024,
        "host_budget": 2048,
        "prefetch_depth": 1,
    }
    expected = _adapter_decision_digest(**base)
    variants = {
        "network": "ttns",
        "operation": "davidson",
        "center_kind": "two_site",
        "center_shape": (2, 6),
        "topology": {"node_index": 2, "parent_index": 1},
        "solver_controls": {"block_size": 13, "coefficient": -0.5j},
        "selectors": {"ivp_solver": "RK45"},
        "fallback_policy": "legacy_oe",
        "fallback_reason_code": "unsupported_solver",
        "world_size": 4,
        "local_world_size": 4,
        "mesh_shape": (1, 2),
        "mesh_axis_names": ("data", "rank"),
        "device_budget": 4096,
        "host_budget": 8192,
        "prefetch_depth": 2,
    }

    for key, value in variants.items():
        changed = dict(base)
        changed[key] = value
        assert _adapter_decision_digest(**changed) != expected, key


def test_adapter_decision_synchronizes_unsupported_fallback_and_error():
    fallback_collective = _TraceCollective()
    fallback_backend = create_backend(
        "numpy",
        config=BackendConfig(
            device="cpu",
            execution_policy="execution_ir",
            fallback_policy="legacy_oe",
        ),
    )
    route = coordinate_adapter_decision(
        _execution(fallback_collective),
        fallback_backend,
        network="mps",
        operation="ground_state",
        center_kind="two_site",
        center_shape=(4, 4),
        topology={"site_indices": (1, 2)},
        solver_controls={"algo": "primme", "nroots": 1},
        selectors={"omega": None, "stacked_mpo": False},
        supported=False,
        fallback_reason_code="unsupported_solver",
    )

    assert route == "fallback"
    assert fallback_collective.trace == [
        ("allreduce", "max", 1),
        ("allreduce", "min", 4),
        ("allreduce", "max", 4),
    ]

    error_collective = _TraceCollective()
    with pytest.raises(
        NotImplementedError, match="distributed adapter decision failed"
    ):
        coordinate_adapter_decision(
            _execution(error_collective),
            _numpy_backend(),
            network="mps",
            operation="ground_state",
            center_kind="two_site",
            center_shape=(4, 4),
            topology={"site_indices": (1, 2)},
            solver_controls={"algo": "primme", "nroots": 1},
            selectors={"omega": None, "stacked_mpo": False},
            supported=False,
            fallback_reason_code="unsupported_solver",
        )
    assert error_collective.trace == fallback_collective.trace


def test_adapter_decision_rejects_rank_mismatch_after_fixed_coordination_schedule():
    collective = _TraceCollective()
    execution = _execution(collective)
    collective.rank = 1

    with pytest.raises(ValueError, match="adapter coordination preflight failed"):
        coordinate_adapter_decision(
            execution,
            _numpy_backend(),
            network="mps",
            operation="krylov",
            center_kind="one_site",
            center_shape=(4,),
            topology={"site_indices": (1,)},
            solver_controls={"block_size": 4},
            supported=True,
        )

    assert collective.trace == [
        ("allreduce", "max", 1),
        ("allreduce", "min", 4),
        ("allreduce", "max", 4),
    ]


def test_center_materialization_synchronizes_allocation_failure_before_broadcast():
    dense = shard_axis((4, 2), axis=0, parts=2)
    vector_map = CenterVectorMap(dense, dense)
    collective = _TraceCollective()

    class FailingNamespace:
        @staticmethod
        def zeros(*_args, **_kwargs):
            raise MemoryError("injected allocation failure")

        empty = staticmethod(np.empty)

    class FailingBackend:
        array_namespace = FailingNamespace()

    with pytest.raises(RuntimeError, match="center materialization allocation failed"):
        vector_map.materialize(
            np.ones(4, dtype=np.float64), collective, FailingBackend()
        )

    assert collective.trace == [("allreduce", "max", 1)]


def test_center_materialization_synchronizes_root_source_staging_before_broadcast():
    dense = shard_axis((4, 2), axis=0, parts=2)
    vector_map = CenterVectorMap(dense, dense)
    collective = _TraceCollective()

    class InjectedSourceFailure(BaseException):
        pass

    class FailingReceive:
        def __getitem__(self, _key):
            return self

        def __setitem__(self, _key, _value):
            raise InjectedSourceFailure("injected source staging failure")

    class FailingNamespace:
        zeros = staticmethod(np.zeros)

        @staticmethod
        def empty(*_args, **_kwargs):
            return FailingReceive()

    class FailingBackend:
        array_namespace = FailingNamespace()

    with pytest.raises(
        RuntimeError, match="center materialization source staging failed"
    ):
        vector_map.materialize(
            np.ones(4, dtype=np.float64), collective, FailingBackend()
        )

    assert collective.trace == [
        ("allreduce", "max", 1),
        ("allreduce", "max", 1),
    ]


def test_center_materialization_synchronizes_each_unpack_before_next_source(
    monkeypatch,
):
    dense = shard_axis((4, 2), axis=0, parts=2)
    vector_map = CenterVectorMap(dense, dense)
    collective = _TraceCollective()
    original = CenterVectorMap.expand_local

    def fail_first_unpack(self, local_1d, rank, dense_buffer):
        if rank == 0:
            raise ValueError("injected unpack failure")
        return original(self, local_1d, rank, dense_buffer)

    monkeypatch.setattr(CenterVectorMap, "expand_local", fail_first_unpack)
    with pytest.raises(RuntimeError, match="center materialization unpack failed"):
        vector_map.materialize(
            np.ones(4, dtype=np.float64), collective, _numpy_backend()
        )

    assert collective.trace == [
        ("allreduce", "max", 1),
        ("allreduce", "max", 1),
        ("broadcast", 0, 4),
        ("allreduce", "max", 1),
    ]


def test_center_materialization_profile_bounds_retained_status_result(monkeypatch):
    import renormalizer.backend._distributed.center as center

    class TrackedArray(np.ndarray):
        pass

    refs = []
    peak_bytes = 0

    def track(value, dtype=None):
        nonlocal peak_bytes
        array = np.array(value, dtype=dtype, copy=True).view(TrackedArray)
        refs.append(weakref.ref(array))
        peak_bytes = max(
            peak_bytes,
            sum(live.nbytes for ref in refs for live in (ref(),) if live is not None),
        )
        return array

    class Collective(_TraceCollective):
        def allreduce(self, value, *, op="sum"):
            self.trace.append(("allreduce", op, int(np.asarray(value).size)))
            return track(value)

    dense = shard_axis((8,), axis=0, parts=2)
    vector_map = CenterVectorMap(dense, dense)
    collective = Collective()
    monkeypatch.setattr(
        center,
        "_control_array",
        lambda _collective, values, dtype: track(values, dtype),
    )

    vector_map.materialize(np.ones(4, dtype=np.float64), collective, _numpy_backend())
    profile = center_materialization_memory_profile(vector_map, np.dtype("float64"))

    assert peak_bytes == 3 * np.dtype(np.int32).itemsize
    assert profile.control_device_bytes == (peak_bytes, peak_bytes)


def test_state_update_failure_stops_before_digest_collectives():
    collective = _TraceCollective()
    execution = _execution(collective)

    def fail_update():
        raise ValueError("injected state update failure")

    with pytest.raises(RuntimeError, match="distributed state update failed"):
        run_synchronized_state_update(
            execution,
            fail_update,
            lambda: ([np.ones(2)], ()),
            metadata=("mps", "one_site"),
        )

    assert collective.trace == [("allreduce", "max", 1)]


def test_state_update_success_has_status_then_digest_min_max():
    collective = _TraceCollective()
    execution = _execution(collective)
    state = [np.zeros(2)]

    result = run_synchronized_state_update(
        execution,
        lambda: state[0].fill(3.0) or "updated",
        lambda: (state, ()),
        metadata=("ttns", "two_site"),
    )

    assert result == "updated"
    assert collective.trace == [
        ("allreduce", "max", 1),
        ("allreduce", "min", 4),
        ("allreduce", "max", 4),
    ]


def test_state_update_emits_truthful_measured_phase_summary(monkeypatch):
    import renormalizer.backend._distributed.center as center_boundary
    from renormalizer.utils import profiling

    collective = _TraceCollective()
    execution = _execution(collective)
    state = [np.zeros(2)]
    events = []
    ticks = iter([7.0, 7.25])
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )
    monkeypatch.setattr(center_boundary, "perf_counter", lambda: next(ticks))

    run_synchronized_state_update(
        execution,
        lambda: state[0].fill(2.0),
        lambda: (state, ()),
        metadata=("mps", "one_site", 3),
    )

    assert events == [
        {
            "event": "phase_summary",
            "phase": "state_update",
            "network": "mps",
            "operation": "synchronized_state_transition",
            "operation_count": 1,
            "wall_s": pytest.approx(0.25),
        }
    ]


def test_measured_collective_accounts_exact_scheduled_buffers_and_time(monkeypatch):
    import renormalizer.backend._distributed.center as center_boundary

    class Base(SingleProcessCollective):
        def barrier(self):
            return None

    ticks = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3, 3.0, 3.4])
    monkeypatch.setattr(center_boundary, "perf_counter", lambda: next(ticks))
    measured = MeasuredCollective(Base())

    measured.broadcast(np.zeros(3, dtype=np.float64), root=0)
    measured.allreduce(np.zeros(2, dtype=np.int32), op="max")
    measured.allreduce_inplace(np.zeros(1, dtype=np.complex128), op="sum")
    measured.reduce_scatter(np.zeros(8, dtype=np.float32), axis=0, op="sum")

    assert measured.metrics == {
        "broadcast_calls": 1,
        "allreduce_calls": 2,
        "reduce_scatter_calls": 1,
        "allgather_calls": 0,
        "broadcast_bytes": 24,
        "allreduce_bytes": 24,
        "reduce_scatter_bytes": 32,
        "allgather_bytes": 0,
        "collective_bytes": 80,
        "collective_s": pytest.approx(1.0),
    }
