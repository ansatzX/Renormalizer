from collections.abc import Mapping
import importlib
import os
from types import MappingProxyType

import numpy as np
import pytest
from scipy.linalg import eigh

from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.local_operator import DistributedLocalOperator
from renormalizer.backend._distributed.planner import plan_distributed_execution
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend._distributed.sharding import DistributedTensor
from renormalizer.backend._execution.model import ExecutionBindings
from renormalizer.backend._execution.planner import lower_einsum_path
from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import create_backend

from renormalizer.backend._distributed.solvers import (
    distributed_norm,
    distributed_vdot,
    run_sharded_krylov,
)


class RecordingCollective:
    def __init__(self, reduced):
        self.reduced = reduced
        self.allreduce_calls = []

    def allreduce(self, value, *, op="sum"):
        self.allreduce_calls.append((np.array(value, copy=True), op))
        return np.asarray(self.reduced, dtype=value.dtype)

    def allgather(self, *args, **kwargs):
        raise AssertionError("solver scalar helpers must not allgather")


class NoGatherSingleProcessCollective:
    rank = 0
    size = 1

    def __init__(self):
        self.allgather_calls = 0

    def barrier(self):
        return None

    def broadcast(self, array, *, root):
        assert root == 0
        return array

    def allreduce(self, array, *, op="sum"):
        return array.copy()

    def allreduce_inplace(self, array, *, op="sum"):
        return array

    def allgather(self, *args, **kwargs):
        self.allgather_calls += 1
        raise AssertionError("sharded solvers must not allgather")


class ScriptedAllreduceCollective(NoGatherSingleProcessCollective):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.allreduce_calls = []

    def allreduce(self, array, *, op="sum"):
        copied = np.array(array, copy=True)
        self.allreduce_calls.append((copied, op))
        if not self.responses:
            return copied
        response = self.responses.pop(0)
        if response is None:
            return copied
        if callable(response):
            response = response(copied, op)
        return np.asarray(response, dtype=copied.dtype)


class UserControlAbort(BaseException):
    pass


class InterruptingMapping(Mapping):
    def __getitem__(self, key):
        raise AssertionError("mapping item access should not be reached")

    def __iter__(self):
        raise UserControlAbort("mapping iteration interrupted")

    def __len__(self):
        return 1


def _hermitian_problem(dtype=np.float64):
    diagonal = np.arange(1.0, 8.0)
    matrix = np.diag(diagonal)
    matrix += np.diag(np.full(6, -0.2), 1)
    matrix += np.diag(np.full(6, -0.2), -1)
    if np.dtype(dtype).kind == "c":
        phase = np.exp(0.3j)
        matrix = np.diag(diagonal.astype(dtype))
        matrix += np.diag(np.full(6, -0.2 * phase, dtype=dtype), 1)
        matrix += np.diag(np.full(6, -0.2 * phase.conjugate(), dtype=dtype), -1)
    return np.asarray(matrix, dtype=dtype)


def _numpy_operator(matrix, collective=None):
    backend = create_backend(
        "numpy", config=BackendConfig(device="cpu", execution_policy="execution_ir")
    )
    context = DistributedContext(0, 0, 1, 1)
    if collective is None:
        collective = NoGatherSingleProcessCollective()
    source = lower_einsum_path(
        "ab,b->a", (matrix.shape, (matrix.shape[1],)), dtype=matrix.dtype.name
    )
    plan = plan_distributed_execution(source, variable_key="input_1", world_size=1)
    operator = DistributedLocalOperator(
        plan=plan,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=backend,
        context=context,
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )
    return operator, collective


def test_distributed_norm_allreduces_local_squared_norm_once():
    collective = RecordingCollective(25.0)

    result = distributed_norm(np.array([3.0, 4.0]), collective)

    assert result == 5.0
    assert isinstance(result, float)
    assert len(collective.allreduce_calls) == 1
    local_squared_norm, op = collective.allreduce_calls[0]
    assert local_squared_norm.shape == (1,)
    assert local_squared_norm[0] == 25.0
    assert op == "sum"


def test_distributed_vdot_conjugates_and_allreduces_one_scalar_once():
    x = np.array([1 + 2j, 3 - 1j])
    y = np.array([2 - 1j, -4 + 2j])
    expected = np.vdot(x, y)
    collective = RecordingCollective(expected)

    result = distributed_vdot(x, y, collective)

    assert result == expected
    assert isinstance(result, complex)
    assert len(collective.allreduce_calls) == 1
    local_dot, op = collective.allreduce_calls[0]
    assert local_dot.shape == (1,)
    assert local_dot[0] == expected
    assert op == "sum"


def test_expm_krylov_delegates_to_shared_lanczos_recurrence(monkeypatch):
    krylov = importlib.import_module("renormalizer.lib.krylov.krylov")
    vector = np.arange(7.0)
    expected = np.full(7, 3.0)
    calls = []

    def shared(aop, coefficient, start, *, block_size, vector_ops, global_size):
        calls.append((aop, coefficient, start, block_size, vector_ops, global_size))
        return expected, 4

    monkeypatch.setattr(krylov, "_lanczos_expm", shared)
    operator = lambda value: value

    result, iterations = krylov.expm_krylov(operator, 0.25, vector, 9)

    assert result is expected
    assert iterations == 4
    assert calls[0][0] is operator
    assert calls[0][1] == 0.25
    assert calls[0][2] is vector
    assert calls[0][3] == 9
    assert calls[0][5] == 7


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sharded_krylov_uses_task14_operator_and_owns_result(dtype):
    matrix = _hermitian_problem(dtype)
    operator, collective = _numpy_operator(matrix)
    start = np.asarray(np.arange(1.0, 8.0), dtype=dtype)
    if np.dtype(dtype).kind == "c":
        start += 0.1j * start[::-1]
    original = start.copy()
    vector = DistributedTensor(operator.plan.input_sharding, 0, start)

    result, iterations = run_sharded_krylov(
        operator,
        vector,
        -0.2j,
        collective=collective,
        config=MappingProxyType({"block_size": 3}),
    )

    eigenvalues, eigenvectors = eigh(matrix)
    expected = eigenvectors @ (
        np.exp(-0.2j * eigenvalues) * (eigenvectors.conj().T @ original)
    )
    np.testing.assert_allclose(result.local_array, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(start, original)
    assert result.spec == vector.spec
    assert result.rank == vector.rank
    assert 1 <= iterations <= matrix.shape[0]
    saved = result.local_array.copy()
    operator(original)
    np.testing.assert_array_equal(result.local_array, saved)
    assert collective.allgather_calls == 0


def test_sharded_krylov_strictly_validates_config_and_dtype():
    matrix = _hermitian_problem()
    operator, collective = _numpy_operator(matrix)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(TypeError, match="config.*Mapping"):
        run_sharded_krylov(operator, vector, 1.0, collective=collective, config=[])
    with pytest.raises(ValueError, match="unknown Krylov config"):
        run_sharded_krylov(
            operator,
            vector,
            1.0,
            collective=collective,
            config={"callback": None},
        )
    bad = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1, 8, dtype=np.int64)
    )
    with pytest.raises(ValueError, match="distributed solver preflight failed") as caught:
        run_sharded_krylov(operator, bad, 1.0, collective=collective, config={})
    assert "dtype" in str(caught.value.__cause__)


def test_sharded_krylov_synchronizes_rank_local_control_validation():
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, np.array([1])])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(ValueError, match="Krylov control validation failed"):
        run_sharded_krylov(
            operator,
            vector,
            0.1,
            collective=collective,
            config={"block_size": 3},
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]
    assert collective.allgather_calls == 0


def test_sharded_krylov_locally_invalid_control_raises_after_synchronization():
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, None])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(ValueError, match="block_size"):
        run_sharded_krylov(
            operator,
            vector,
            0.1,
            collective=collective,
            config={"block_size": 0},
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]
    assert collective.allgather_calls == 0


def test_structural_preflight_uses_operator_collective_until_caller_is_proven():
    matrix = _hermitian_problem()
    operator_collective = ScriptedAllreduceCollective([None])
    caller_collective = ScriptedAllreduceCollective([None])
    operator, _ = _numpy_operator(matrix, operator_collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(ValueError, match="distributed solver preflight failed") as caught:
        run_sharded_krylov(
            operator,
            vector,
            0.1,
            collective=caller_collective,
            config={},
        )

    assert "solver collective" in str(caught.value.__cause__)
    assert [op for _, op in operator_collective.allreduce_calls] == ["max"]
    assert caller_collective.allreduce_calls == []
    assert operator_collective.allgather_calls == 0
    assert caller_collective.allgather_calls == 0


@pytest.mark.parametrize("local_setup_error", [TypeError("bad setup"), None])
def test_sharded_solver_runs_deferred_setup_preflight_before_plan_access(
    local_setup_error,
):
    class PlanAccessForbidden:
        def __getattr__(self, name):
            raise AssertionError("solver accessed plan before setup preflight")

    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([np.array([1])])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )
    operator._bootstrap_world_size = 2
    operator._setup_error = local_setup_error
    operator._setup_preflight_complete = False
    operator.plan = PlanAccessForbidden()

    with pytest.raises(ValueError, match="distributed setup preflight failed") as caught:
        run_sharded_krylov(
            operator, vector, 0.1, collective=collective, config={}
        )

    assert caught.value.__cause__ is local_setup_error
    assert [op for _, op in collective.allreduce_calls] == ["max"]


def test_sharded_krylov_synchronizes_mapping_baseexception_before_reraising():
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, None])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(UserControlAbort, match="mapping iteration interrupted"):
        run_sharded_krylov(
            operator,
            vector,
            0.1,
            collective=collective,
            config=InterruptingMapping(),
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]


def test_sharded_krylov_synchronizes_coefficient_baseexception_before_reraising():
    class InterruptingCoefficient:
        def __complex__(self):
            raise UserControlAbort("coefficient conversion interrupted")

    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, None])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(UserControlAbort, match="coefficient conversion interrupted"):
        run_sharded_krylov(
            operator,
            vector,
            InterruptingCoefficient(),
            collective=collective,
            config={},
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]


@pytest.mark.parametrize("coefficient", [2**53 + 1, -(2**53 + 1)])
def test_sharded_krylov_rejects_integer_coefficient_outside_binary64_exact_range(
    coefficient,
):
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, None])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(ValueError, match="binary64 exact integer range"):
        run_sharded_krylov(
            operator, vector, coefficient, collective=collective, config={}
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]


@pytest.mark.parametrize(
    "coefficient, config, mismatch_index",
    [
        (0.1 + 0.2j, {"block_size": 3}, 1),
        (0.1, {"block_size": 3}, 2),
        (0.1, {"block_size": 2**53}, 2),
    ],
    ids=["complex-coefficient", "block-size", "large-integer-block-size"],
)
def test_sharded_krylov_rejects_cross_rank_control_mismatch(
    coefficient, config, mismatch_index
):
    def mismatched_max(values, op):
        assert op == "max"
        values[mismatch_index] += 1
        return values

    collective = ScriptedAllreduceCollective([None, None, None, mismatched_max])
    matrix = _hermitian_problem()
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )

    with pytest.raises(ValueError, match="Krylov controls must agree across ranks"):
        run_sharded_krylov(
            operator,
            vector,
            coefficient,
            collective=collective,
            config=config,
        )

    assert [op for _, op in collective.allreduce_calls] == [
        "max",
        "max",
        "min",
        "max",
    ]
    assert collective.allgather_calls == 0


def test_sharded_solver_preserves_synchronized_dense_layout_not_implemented(
    monkeypatch,
):
    solvers = importlib.import_module("renormalizer.backend._distributed.solvers")
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )
    monkeypatch.setattr(
        solvers,
        "_vector_contract_error",
        lambda *args: NotImplementedError("unsupported dense-axis layout"),
    )

    with pytest.raises(NotImplementedError, match="unsupported dense-axis layout"):
        run_sharded_krylov(
            operator, vector, 0.1, collective=collective, config={}
        )

    assert [op for _, op in collective.allreduce_calls] == ["max"]


def test_public_distributed_solver_facade_exports_sharded_solvers():
    from renormalizer.backend.distributed_solver import (
        distributed_norm as facade_norm,
        distributed_vdot as facade_vdot,
        run_sharded_davidson as facade_davidson,
        run_sharded_krylov as facade_krylov,
    )
    from renormalizer.backend._distributed.solvers import run_sharded_davidson

    assert facade_norm is distributed_norm
    assert facade_vdot is distributed_vdot
    assert facade_krylov is run_sharded_krylov
    assert facade_davidson is run_sharded_davidson


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_real_two_rank_cupy_sharded_krylov_matches_dense_without_allgather(dtype):
    cupy = pytest.importorskip("cupy")
    from renormalizer.backend.distributed_runtime import create_cupy_distributed_runtime

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        matrix_host = _hermitian_problem(dtype)
        source = lower_einsum_path(
            "ab,b->a", (matrix_host.shape, (7,)), dtype=matrix_host.dtype.name
        )
        plan = plan_distributed_execution(source, variable_key="input_1", world_size=2)
        rank = runtime.rank
        local_slice = plan.input_sharding.local_slices[rank]
        start_host = np.asarray(np.arange(1.0, 8.0), dtype=dtype)
        if np.dtype(dtype).kind == "c":
            start_host += 0.1j * start_host[::-1]
        local_start = cupy.asarray(start_host[local_slice])
        counters = {}
        operator = DistributedLocalOperator(
            plan=plan,
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            counters=counters,
            backend=runtime.backend,
            context=runtime.context,
            source_bindings=ExecutionBindings({"input_0": cupy.asarray(matrix_host)}),
        )
        original_allgather = runtime.collective.allgather
        runtime.collective.allgather = lambda *a, **k: pytest.fail("solver allgathered")
        try:
            result, iterations = run_sharded_krylov(
                operator,
                DistributedTensor(plan.input_sharding, rank, local_start),
                -0.2j,
                collective=runtime.collective,
                config={"block_size": 3},
            )
        finally:
            runtime.collective.allgather = original_allgather

        eigenvalues, eigenvectors = eigh(matrix_host)
        expected = eigenvectors @ (
            np.exp(-0.2j * eigenvalues) * (eigenvectors.conj().T @ start_host)
        )
        np.testing.assert_allclose(
            cupy.asnumpy(result.local_array),
            expected[local_slice],
            atol=2e-11,
            rtol=2e-11,
        )
        trace = cupy.asarray([iterations], dtype=cupy.int64)
        minimum = runtime.collective.allreduce(trace, op="min")
        maximum = runtime.collective.allreduce(trace, op="max")
        assert int(cupy.asnumpy(minimum)[0]) == iterations
        assert int(cupy.asnumpy(maximum)[0]) == iterations
        assert counters["broadcast_calls"] == 2 * iterations
        assert counters["execution_calls"] == 2 * iterations
        assert counters["allreduce_calls"] == 4 * iterations + 4
        assert counters["allgather_calls"] == 0
    finally:
        runtime.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_cupy_rank_local_dtype_failure_is_synchronized():
    cupy = pytest.importorskip("cupy")
    from renormalizer.backend.distributed_runtime import create_cupy_distributed_runtime

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        matrix_host = _hermitian_problem()
        source = lower_einsum_path(
            "ab,b->a", (matrix_host.shape, (7,)), dtype="float64"
        )
        plan = plan_distributed_execution(source, variable_key="input_1", world_size=2)
        rank = runtime.rank
        local_slice = plan.input_sharding.local_slices[rank]
        operator = DistributedLocalOperator(
            plan=plan,
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            counters={},
            backend=runtime.backend,
            context=runtime.context,
            source_bindings=ExecutionBindings({"input_0": cupy.asarray(matrix_host)}),
        )
        dtype = cupy.float32 if rank == 0 else cupy.float64
        bad = DistributedTensor(
            plan.input_sharding,
            rank,
            cupy.asarray(np.arange(1.0, 8.0)[local_slice], dtype=dtype),
        )
        with pytest.raises(ValueError, match="distributed solver preflight failed") as caught:
            run_sharded_krylov(
                operator, bad, 0.1, collective=runtime.collective, config={}
            )
        if rank == 0:
            assert "dtype" in str(caught.value.__cause__)
        else:
            assert caught.value.__cause__ is None
        runtime.barrier()

        operator._setup_error = TypeError("rank-local setup failure") if rank == 0 else None
        operator._setup_preflight_complete = False
        with pytest.raises(ValueError, match="distributed setup preflight failed") as caught:
            run_sharded_krylov(
                operator, bad, 0.1, collective=runtime.collective, config={}
            )
        if rank == 0:
            assert caught.value.__cause__ is operator._setup_error
        else:
            assert caught.value.__cause__ is None
        runtime.barrier()

        operator._setup_error = None
        good = DistributedTensor(
            plan.input_sharding,
            rank,
            cupy.asarray(np.arange(1.0, 8.0)[local_slice], dtype=cupy.float64),
        )

        class InterruptingCoefficient:
            def __complex__(self):
                raise UserControlAbort("rank-local coefficient interruption")

        coefficient = InterruptingCoefficient() if rank == 0 else 0.1
        if rank == 0:
            with pytest.raises(
                UserControlAbort, match="rank-local coefficient interruption"
            ):
                run_sharded_krylov(
                    operator,
                    good,
                    coefficient,
                    collective=runtime.collective,
                    config={},
                )
        else:
            with pytest.raises(ValueError, match="Krylov control validation failed"):
                run_sharded_krylov(
                    operator,
                    good,
                    coefficient,
                    collective=runtime.collective,
                    config={},
                )
        runtime.barrier()

        class RecordingCallerCollective:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.rank = wrapped.rank
                self.size = wrapped.size
                self.allreduce_calls = 0
                self.allgather_calls = 0

            def allreduce(self, array, *, op="sum"):
                self.allreduce_calls += 1
                return self.wrapped.allreduce(array, op=op)

            def allgather(self, *args, **kwargs):
                self.allgather_calls += 1
                raise AssertionError("mismatched caller must not allgather")

        caller = (
            RecordingCallerCollective(runtime.collective)
            if rank == 0
            else runtime.collective
        )
        with pytest.raises(ValueError, match="distributed solver preflight failed") as caught:
            run_sharded_krylov(
                operator,
                good,
                0.1,
                collective=caller,
                config={},
            )
        if rank == 0:
            assert "solver collective" in str(caught.value.__cause__)
        else:
            assert caught.value.__cause__ is None
        runtime.barrier()
        if rank == 0:
            assert caller.allreduce_calls == 0
            assert caller.allgather_calls == 0
    finally:
        runtime.close()
