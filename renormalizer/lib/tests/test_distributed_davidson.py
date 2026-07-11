from dataclasses import FrozenInstanceError
import importlib
from types import MappingProxyType

import numpy as np
import pytest

from renormalizer.backend._distributed.sharding import DistributedTensor
from renormalizer.backend._distributed.solvers import (
    distributed_norm,
    distributed_vdot,
    run_sharded_davidson,
)

from renormalizer.lib.tests.test_distributed_krylov import (
    _hermitian_problem,
    _numpy_operator,
    ScriptedAllreduceCollective,
)

from renormalizer.lib.davidson.backend import davidson_backend


def test_local_backend_davidson_finds_lowest_single_root():
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    diagonal = np.diag(matrix).copy()

    energy, vector, info = davidson_backend(
        matrix.dot,
        np.array([1.0, 0.1]),
        diagonal,
        tol=1e-12,
        max_cycle=20,
        max_space=4,
    )

    assert info.converged is True
    np.testing.assert_allclose(energy, 1.0, atol=1e-12)
    np.testing.assert_allclose(matrix @ vector, energy * vector, atol=1e-12)


def test_local_backend_davidson_uses_backend_vector_namespace(monkeypatch):
    backend_module = importlib.import_module("renormalizer.lib.davidson.backend")
    calls = {"asarray": 0, "vdot": 0, "norm": 0, "zeros_like": 0}

    class Linalg:
        @staticmethod
        def norm(value):
            calls["norm"] += 1
            return np.linalg.norm(value)

    class Namespace:
        name = "numpy"
        linalg = Linalg()

        @staticmethod
        def asarray(value):
            calls["asarray"] += 1
            return np.asarray(value)

        @staticmethod
        def vdot(left, right):
            calls["vdot"] += 1
            return np.vdot(left, right)

        @staticmethod
        def zeros_like(value, dtype=None):
            calls["zeros_like"] += 1
            return np.zeros_like(value, dtype=dtype)

    monkeypatch.setattr(backend_module, "xp", Namespace())
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])

    backend_module.davidson_backend(
        matrix.dot, np.array([1.0, 0.1]), np.diag(matrix), max_space=2
    )

    assert calls["asarray"] == 2
    assert calls["vdot"] > 0
    assert calls["norm"] > 0
    assert calls["zeros_like"] > 0


def test_local_backend_davidson_complex_characterization_and_nonconvergence():
    matrix = _hermitian_problem(np.complex128)
    diagonal = np.diag(matrix).copy()
    start = np.arange(1.0, 8.0).astype(np.complex128)
    start += 0.2j * start[::-1]
    original = start.copy()

    energy, vector, info = davidson_backend(
        matrix.dot, start, diagonal, tol=1e-11, max_cycle=20, max_space=7
    )

    expected = np.linalg.eigvalsh(matrix)[0]
    np.testing.assert_allclose(energy, expected, atol=1e-10)
    np.testing.assert_allclose(matrix @ vector, energy * vector, atol=1e-10)
    np.testing.assert_array_equal(start, original)
    assert info.converged is True
    assert info.h_v_count == info.iterations
    with pytest.raises(FrozenInstanceError):
        info.iterations = 0

    _, _, short = davidson_backend(
        matrix.dot, start, diagonal, tol=1e-15, max_cycle=3, max_space=2
    )
    assert short.converged is False
    assert short.iterations == short.h_v_count == 3
    assert short.restarts == 1


def test_local_backend_davidson_promotes_projected_dtype_from_applied_vectors():
    matrix = _hermitian_problem(np.complex128)
    start = np.arange(1.0, 8.0)
    diagonal = np.diag(matrix).real.copy()

    energy, vector, info = davidson_backend(
        matrix.dot, start, diagonal, tol=1e-11, max_cycle=20, max_space=7
    )

    np.testing.assert_allclose(energy, np.linalg.eigvalsh(matrix)[0], atol=1e-10)
    np.testing.assert_allclose(matrix @ vector, energy * vector, atol=1e-10)
    assert np.iscomplexobj(vector)
    assert info.converged is True


@pytest.mark.parametrize("backend_name", ["jax", "torch"])
def test_local_backend_davidson_explicitly_rejects_unsupported_backend(
    monkeypatch, backend_name
):
    backend_module = importlib.import_module("renormalizer.lib.davidson.backend")

    class UnsupportedNamespace:
        name = backend_name

        def __getattr__(self, attribute):
            raise AssertionError("unsupported backend entered Davidson recurrence")

    monkeypatch.setattr(backend_module, "xp", UnsupportedNamespace())

    with pytest.raises(
        NotImplementedError, match="Davidson supports only NumPy and CuPy backends"
    ):
        backend_module.davidson_backend(None, None, None)


def test_davidson_info_residual_norm_matches_returned_normalized_eigenpair():
    backend_module = importlib.import_module("renormalizer.lib.davidson.backend")
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])

    class ScaledCombinationVectorOps:
        @staticmethod
        def copy(vector):
            return vector.copy()

        @staticmethod
        def vdot(left, right):
            return np.vdot(left, right)

        @staticmethod
        def norm(vector):
            return float(np.linalg.norm(vector))

        @staticmethod
        def linear_combination(coefficients, vectors):
            return 3 * sum(
                coefficient * vector
                for coefficient, vector in zip(coefficients, vectors)
            )

    energy, vector, info = backend_module._davidson_single_root(
        matrix.dot,
        np.array([1.0, 0.1]),
        np.diag(matrix).copy(),
        vector_ops=ScaledCombinationVectorOps(),
        global_size=2,
        tol=10.0,
        max_cycle=1,
        max_space=2,
        lindep=1e-14,
    )

    expected_residual = np.linalg.norm(matrix @ vector - energy * vector)
    np.testing.assert_allclose(info.residual_norm, expected_residual, atol=1e-15)


def test_davidson_nextafter_lindep_runs_first_hv_then_applies_threshold():
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    start = np.array([0.1257302210933933, -0.1321048632913019])
    diagonal_host = np.diag(matrix).copy()
    lindep = np.nextafter(1.0, 0.0)

    local_energy, local_vector, local_info = davidson_backend(
        matrix.dot,
        start,
        diagonal_host,
        tol=1e-15,
        max_cycle=5,
        max_space=2,
        lindep=lindep,
    )

    operator, collective = _numpy_operator(matrix)
    vector = DistributedTensor(operator.plan.input_sharding, 0, start.copy())
    diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, diagonal_host.copy()
    )
    energy, result, info = run_sharded_davidson(
        operator,
        vector,
        collective=collective,
        config={
            "diagonal": diagonal,
            "tol": 1e-15,
            "max_cycle": 5,
            "max_space": 2,
            "lindep": lindep,
        },
    )

    assert local_info.h_v_count == info.h_v_count == 1
    assert local_info.iterations == info.iterations == 2
    assert np.isfinite(local_info.residual_norm)
    assert np.isfinite(info.residual_norm)
    local_residual = np.linalg.norm(matrix @ local_vector - local_energy * local_vector)
    sharded_residual = np.linalg.norm(
        matrix @ result.local_array - energy * result.local_array
    )
    np.testing.assert_allclose(local_info.residual_norm, local_residual, atol=1e-15)
    np.testing.assert_allclose(info.residual_norm, sharded_residual, atol=1e-15)
    np.testing.assert_allclose(energy, local_energy, atol=1e-15)
    np.testing.assert_allclose(
        abs(np.vdot(local_vector, result.local_array)), 1.0, atol=1e-15
    )
    assert collective.allgather_calls == 0


@pytest.mark.parametrize("lindep", [1.0, 2.0])
def test_local_backend_davidson_rejects_lindep_that_can_skip_first_hv(lindep):
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    calls = []

    def aop(vector):
        calls.append(vector)
        return matrix @ vector

    with pytest.raises(ValueError, match="lindep.*less than 1"):
        davidson_backend(
            aop,
            np.array([1.0, 0.1]),
            np.diag(matrix).copy(),
            lindep=lindep,
        )

    assert calls == []


def test_sharded_davidson_uses_shared_recurrence_and_copies_reused_hv_storage(
    monkeypatch,
):
    matrix = _hermitian_problem()
    operator, collective = _numpy_operator(matrix)
    start = np.arange(1.0, 8.0)
    vector = DistributedTensor(operator.plan.input_sharding, 0, start)
    diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, np.diag(matrix).copy()
    )

    first = operator(start)
    first_snapshot = first.copy()
    second = operator(start[::-1].copy())
    assert second is first
    assert not np.array_equal(first, first_snapshot)

    backend_module = importlib.import_module("renormalizer.lib.davidson.backend")

    calls = []
    original = backend_module._davidson_single_root

    def recording(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(backend_module, "_davidson_single_root", recording)
    energy, result, info = run_sharded_davidson(
        operator,
        vector,
        collective=collective,
        config=MappingProxyType(
            {"diagonal": diagonal, "tol": 1e-11, "max_cycle": 30, "max_space": 7}
        ),
    )

    assert calls == [True]
    local_energy, local_vector, local_info = davidson_backend(
        matrix.dot,
        start,
        np.diag(matrix).copy(),
        tol=1e-11,
        max_cycle=30,
        max_space=7,
    )
    assert calls == [True, True]
    assert info.converged == local_info.converged
    assert info.iterations == local_info.iterations
    assert info.h_v_count == local_info.h_v_count
    assert info.subspace_size == local_info.subspace_size
    assert info.restarts == local_info.restarts
    np.testing.assert_allclose(
        info.residual_norm, local_info.residual_norm, atol=2e-15
    )
    np.testing.assert_allclose(energy, local_energy, atol=1e-13)
    np.testing.assert_allclose(
        abs(np.vdot(local_vector, result.local_array)), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        matrix @ result.local_array,
        energy * result.local_array,
        atol=1e-10,
    )
    assert info.converged is True
    saved = result.local_array.copy()
    operator(start)
    np.testing.assert_array_equal(result.local_array, saved)
    assert collective.allgather_calls == 0


@pytest.mark.parametrize(
    "config, error, message",
    [
        ({}, ValueError, "requires diagonal"),
        ({"diagonal": None, "nroots": 2}, ValueError, "unknown Davidson config"),
        (
            {"diagonal": None, "callback": lambda: None},
            ValueError,
            "unknown Davidson config",
        ),
        ({"diagonal": None, "pick": object()}, ValueError, "unknown Davidson config"),
        ({"diagonal": None, "lessio": True}, ValueError, "unknown Davidson config"),
        (
            {"diagonal": None, "preconditioner": object()},
            ValueError,
            "unknown Davidson config",
        ),
    ],
)
def test_sharded_davidson_rejects_missing_unknown_and_unsupported_modes(
    config, error, message
):
    matrix = _hermitian_problem()
    operator, collective = _numpy_operator(matrix)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )
    with pytest.raises(error, match=message):
        run_sharded_davidson(operator, vector, collective=collective, config=config)


def test_sharded_davidson_require_convergence_and_diagonal_contract():
    matrix = _hermitian_problem()
    operator, collective = _numpy_operator(matrix)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )
    bad_diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(7, dtype=np.int64)
    )
    with pytest.raises(ValueError, match="diagonal preflight failed") as caught:
        run_sharded_davidson(
            operator, vector, collective=collective, config={"diagonal": bad_diagonal}
        )
    assert "diagonal dtype" in str(caught.value.__cause__)

    diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, np.diag(matrix).copy()
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        run_sharded_davidson(
            operator,
            vector,
            collective=collective,
            config={
                "diagonal": diagonal,
                "tol": 1e-15,
                "max_cycle": 1,
                "require_convergence": True,
            },
        )


def test_sharded_davidson_synchronizes_rank_local_control_validation():
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, np.array([1])])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )
    diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, np.diag(matrix).copy()
    )

    with pytest.raises(ValueError, match="Davidson control validation failed"):
        run_sharded_davidson(
            operator,
            vector,
            collective=collective,
            config={"diagonal": diagonal},
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]
    assert collective.allgather_calls == 0


def test_sharded_davidson_locally_invalid_control_raises_after_synchronization():
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, None])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )
    diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, np.diag(matrix).copy()
    )

    with pytest.raises(ValueError, match="max_cycle"):
        run_sharded_davidson(
            operator,
            vector,
            collective=collective,
            config={"diagonal": diagonal, "max_cycle": 0},
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]
    assert collective.allgather_calls == 0


@pytest.mark.parametrize("lindep", [1.0, 2.0])
def test_sharded_davidson_synchronizes_invalid_lindep_upper_bound(lindep):
    matrix = _hermitian_problem()
    collective = ScriptedAllreduceCollective([None, None])
    operator, _ = _numpy_operator(matrix, collective)
    vector = DistributedTensor(
        operator.plan.input_sharding, 0, np.arange(1.0, 8.0)
    )
    diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, np.diag(matrix).copy()
    )

    with pytest.raises(ValueError, match="lindep.*less than 1"):
        run_sharded_davidson(
            operator,
            vector,
            collective=collective,
            config={"diagonal": diagonal, "lindep": lindep},
        )

    assert [op for _, op in collective.allreduce_calls] == ["max", "max"]
    assert collective.allgather_calls == 0


@pytest.mark.parametrize(
    "field, mismatch_index",
    [
        ("tol", 0),
        ("max_cycle", 1),
        ("max_space", 2),
        ("lindep", 3),
        ("require_convergence", 4),
    ],
)
def test_sharded_davidson_rejects_cross_rank_recurrence_control_mismatch(
    field, mismatch_index
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
    diagonal = DistributedTensor(
        operator.plan.input_sharding, 0, np.diag(matrix).copy()
    )
    config = {
        "diagonal": diagonal,
        "tol": 1e-12,
        "max_cycle": 20,
        "max_space": 7,
        "lindep": 1e-14,
        "require_convergence": False,
    }

    with pytest.raises(ValueError, match="Davidson controls must agree across ranks"):
        run_sharded_davidson(
            operator,
            vector,
            collective=collective,
            config=config,
        )

    assert field in config
    assert [op for _, op in collective.allreduce_calls] == [
        "max",
        "max",
        "min",
        "max",
    ]
    assert collective.allgather_calls == 0


@pytest.mark.skipif(
    __import__("os").environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_cupy_sharded_davidson_energy_residual_and_aliasing():
    cupy = pytest.importorskip("cupy")
    from renormalizer.backend.distributed_runtime import create_cupy_distributed_runtime
    from renormalizer.backend._distributed.local_operator import DistributedLocalOperator
    from renormalizer.backend._distributed.planner import plan_distributed_execution
    from renormalizer.backend._distributed.providers import DeviceResidentProvider
    from renormalizer.backend._execution.model import ExecutionBindings
    from renormalizer.backend._execution.planner import lower_einsum_path

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        matrix_host = _hermitian_problem(np.complex128)
        source = lower_einsum_path(
            "ab,b->a", (matrix_host.shape, (7,)), dtype="complex128"
        )
        plan = plan_distributed_execution(source, variable_key="input_1", world_size=2)
        rank = runtime.rank
        local_slice = plan.input_sharding.local_slices[rank]
        start_host = np.arange(1.0, 8.0).astype(np.complex128)
        start_host += 0.2j * start_host[::-1]
        local_start = cupy.asarray(start_host[local_slice])
        counters = {}
        operator = DistributedLocalOperator(
            plan=plan,
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            counters=counters,
            backend=runtime.backend,
            context=runtime.context,
            source_bindings=ExecutionBindings(
                {"input_0": cupy.asarray(matrix_host)}
            ),
        )
        alias = operator(local_start)
        alias_snapshot = alias.copy()
        assert operator(local_start * 2) is alias
        assert not bool(cupy.all(alias == alias_snapshot))
        diagonal = DistributedTensor(
            plan.input_sharding, rank, cupy.asarray(np.diag(matrix_host)[local_slice])
        )
        original_allgather = runtime.collective.allgather
        runtime.collective.allgather = lambda *a, **k: pytest.fail(
            "solver allgathered"
        )
        try:
            energy, result, info = run_sharded_davidson(
                operator,
                DistributedTensor(plan.input_sharding, rank, local_start),
                collective=runtime.collective,
                config={
                    "diagonal": diagonal,
                    "tol": 2e-10,
                    "max_cycle": 30,
                    "max_space": 7,
                },
            )
        finally:
            runtime.collective.allgather = original_allgather

        expected_energy = np.linalg.eigvalsh(matrix_host)[0]
        np.testing.assert_allclose(energy, expected_energy, atol=2e-9)
        _, expected_vectors = np.linalg.eigh(matrix_host)
        expected_local = cupy.asarray(expected_vectors[:, 0][local_slice])
        overlap = distributed_vdot(
            expected_local, result.local_array, runtime.collective
        )
        phase = overlap / abs(overlap)
        np.testing.assert_allclose(
            cupy.asnumpy(result.local_array / phase),
            cupy.asnumpy(expected_local),
            atol=2e-9,
            rtol=2e-9,
        )
        local_residual = (
            operator(result.local_array).copy() - energy * result.local_array
        )
        global_residual = distributed_norm(local_residual, runtime.collective)
        global_norm = distributed_norm(result.local_array, runtime.collective)
        assert global_residual < 2e-9
        np.testing.assert_allclose(global_norm, 1.0, atol=2e-11)
        trace = cupy.asarray([info.iterations, info.h_v_count], dtype=cupy.int64)
        minimum = runtime.collective.allreduce(trace, op="min")
        maximum = runtime.collective.allreduce(trace, op="max")
        np.testing.assert_array_equal(cupy.asnumpy(minimum), trace.get())
        np.testing.assert_array_equal(cupy.asnumpy(maximum), trace.get())
        operator_calls = info.h_v_count + 3
        assert counters["broadcast_calls"] == 2 * operator_calls
        assert counters["execution_calls"] == 2 * operator_calls
        assert counters["allreduce_calls"] == 4 * operator_calls + 4
        assert counters["allgather_calls"] == 0
    finally:
        runtime.close()
