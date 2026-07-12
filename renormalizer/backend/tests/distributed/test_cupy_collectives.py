import gc
import os
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.context import DistributedContext


cupy = pytest.importorskip("cupy")


class FakeNcclBackend:
    def __init__(self, size, rank):
        self.size = size
        self.rank = rank
        self.calls = []
        self.stop_calls = 0

    def barrier(self):
        self.calls.append(("barrier",))

    def broadcast(self, array, root=0, stream=None):
        self.calls.append(("broadcast", root, stream))
        array[...] = root + 10

    def all_reduce(self, in_array, out_array, op="sum", stream=None):
        self.calls.append(("all_reduce", op, stream))
        out_array[...] = in_array * self.size

    def all_gather(self, in_array, out_array, count, stream=None):
        self.calls.append(("all_gather", count, stream))
        for rank in range(self.size):
            out_array[rank * count : (rank + 1) * count] = in_array + rank * 100

    def reduce_scatter(self, in_array, out_array, count, op="sum", stream=None):
        self.calls.append(("reduce_scatter", count, op, stream))
        start = self.rank * count
        out_array[...] = in_array[start : start + count] * self.size

    def stop(self):
        self.stop_calls += 1

    def close(self):
        self.stop()


class FailingOnceNcclBackend(FakeNcclBackend):
    def stop(self):
        self.stop_calls += 1
        if self.stop_calls == 1:
            raise RuntimeError("injected stop failure")


@pytest.fixture
def context():
    return DistributedContext(rank=1, local_rank=0, world_size=2, local_world_size=2)


@pytest.fixture
def collective(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FakeNcclBackend(context.world_size, context.rank)
    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=lambda *args, **kwargs: backend,
        host="127.0.0.1",
        port=23456,
    )
    try:
        yield collective, backend
    finally:
        collective.close()


def test_constructor_uses_global_rank_and_selects_local_device(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    calls = []

    def initialize(*args, **kwargs):
        calls.append((args, kwargs, cupy.cuda.runtime.getDevice()))
        return FakeNcclBackend(context.world_size, context.rank)

    cupy.cuda.Device(1).use()
    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=initialize,
        host="127.0.0.9",
        port=23456,
    )
    try:
        assert calls == [
            (
                (context.world_size, context.rank),
                {"backend": "nccl", "host": "127.0.0.9", "port": 23456},
                context.local_rank,
            )
        ]
        assert collective.rank == context.rank
        assert collective.size == context.world_size
    finally:
        collective.close()


def test_constructor_stops_acquired_backend_when_validation_fails(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FakeNcclBackend(context.world_size, rank=0)

    with pytest.raises(RuntimeError, match="rank does not match"):
        CupyNcclCollective(
            context,
            cupy_module=cupy,
            init_process_group=lambda *args, **kwargs: backend,
            host="127.0.0.1",
            port=23456,
        )

    assert backend.stop_calls == 1


@pytest.mark.parametrize(
    "options, message",
    [
        ({"host": "", "port": 23456}, "host must be a non-empty string"),
        (
            {"host": "127.0.0.1", "port": 0},
            "port must be an integer between 1 and 65535",
        ),
        (
            {"host": "127.0.0.1", "port": True},
            "port must be an integer between 1 and 65535",
        ),
    ],
)
def test_constructor_rejects_invalid_rendezvous_before_initialization(
    context, options, message
):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    called = False

    def initialize(*args, **kwargs):
        nonlocal called
        called = True

    with pytest.raises((TypeError, ValueError), match=message):
        CupyNcclCollective(
            context,
            cupy_module=cupy,
            init_process_group=initialize,
            **options,
        )

    assert called is False


@pytest.mark.parametrize(
    "options, message",
    [
        ({"host": None, "port": 23456}, "host must be a non-empty string"),
        (
            {"host": "127.0.0.1", "port": None},
            "port must be an integer between 1 and 65535",
        ),
    ],
)
def test_constructor_requires_concrete_rendezvous_before_initialization(
    context, options, message
):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    called = False

    def initialize(*args, **kwargs):
        nonlocal called
        called = True

    with pytest.raises((TypeError, ValueError), match=message):
        CupyNcclCollective(
            context,
            cupy_module=cupy,
            init_process_group=initialize,
            **options,
        )

    assert called is False


def test_constructor_accepts_backend_without_private_size_attribute(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FakeNcclBackend(context.world_size, context.rank)

    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=lambda *args, **kwargs: backend,
        host="127.0.0.1",
        port=23456,
    )
    collective.close()

    assert backend.stop_calls == 1


def test_close_is_idempotent_and_rejects_later_calls(collective):
    wrapper, backend = collective

    wrapper.close()
    wrapper.close()

    assert backend.stop_calls == 1
    with pytest.raises(RuntimeError, match="collective is closed"):
        wrapper.barrier()


def test_collective_close_retries_without_dropping_backend(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FailingOnceNcclBackend(context.world_size, context.rank)
    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=lambda *args, **kwargs: backend,
        host="127.0.0.1",
        port=23456,
    )

    with pytest.raises(RuntimeError, match="injected stop failure"):
        collective.close()

    assert collective._backend is backend
    assert collective._closed is False
    collective.barrier()

    collective.close()
    collective.close()

    assert backend.stop_calls == 2
    assert collective._backend is None
    assert collective._closed is True


def test_runtime_close_retries_without_marking_runtime_closed():
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    class FailingOnceCollective:
        def __init__(self):
            self.close_calls = 0
            self.live = True

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected collective close failure")
            self.live = False

    context = DistributedContext(0, 0, 1, 1)
    collective = FailingOnceCollective()
    runtime = CupyDistributedRuntime(
        backend=object(),
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=collective,
    )

    with pytest.raises(RuntimeError, match="injected collective close failure"):
        runtime.close()

    assert runtime._closed is False
    assert collective.live is True

    runtime.close()
    runtime.close()

    assert runtime._closed is True
    assert collective.live is False
    assert collective.close_calls == 2


def test_runtime_execution_config_is_a_borrowed_frozen_view():
    from dataclasses import FrozenInstanceError

    from renormalizer.backend._distributed.collectives import SingleProcessCollective
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend._distributed.providers import DeviceResidentProvider
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    context = DistributedContext(0, 0, 1, 1)
    collective = SingleProcessCollective()
    runtime = CupyDistributedRuntime(
        backend=object(),
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=collective,
    )

    config = runtime.execution_config(
        device_memory_budget_bytes=1024,
        host_memory_budget_bytes=2048,
        prefetch_depth=2,
    )

    assert config.context is context
    assert config.mesh is runtime.mesh
    assert config.collective is collective
    assert isinstance(config.provider, DeviceResidentProvider)
    assert config.device_memory_budget_bytes == 1024
    assert config.host_memory_budget_bytes == 2048
    assert config.prefetch_depth == 2
    with pytest.raises(FrozenInstanceError):
        config.prefetch_depth = 3

    runtime.close()
    with pytest.raises(RuntimeError, match="distributed runtime is closed"):
        runtime.execution_config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "numpy"),
        ("device", "cuda:1"),
        ("precision", 32),
    ],
)
def test_runtime_execution_config_synchronizes_active_backend_mismatch(
    field, value, monkeypatch
):
    import renormalizer.backend.distributed_runtime as runtime_module
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, array, *, op="sum"):
            self.trace.append((op, np.asarray(array).dtype.str, np.asarray(array).size))
            return np.array(array, copy=True)

    expected = SimpleNamespace(
        name="cupy",
        device="cuda:0",
        config=SimpleNamespace(precision=64),
        array_namespace=np,
        asarray=np.asarray,
    )
    active = SimpleNamespace(
        name="cupy",
        device="cuda:0",
        config=SimpleNamespace(precision=64),
    )
    if field == "precision":
        active.config.precision = value
    else:
        setattr(active, field, value)
    monkeypatch.setattr("renormalizer.cons.get_backend", lambda: active)
    collective = Collective()
    context = DistributedContext(0, 0, 2, 2)
    runtime = runtime_module.CupyDistributedRuntime(
        backend=expected,
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((2,), ("rank",), 0),
        collective=collective,
    )

    with pytest.raises(RuntimeError, match="distributed runtime backend validation failed"):
        runtime.execution_config()

    assert collective.trace == [
        ("max", np.dtype(np.int32).str, 1),
        ("min", np.dtype(np.uint64).str, 4),
        ("max", np.dtype(np.uint64).str, 4),
    ]


def test_runtime_backend_digest_normalizes_valid_rank_local_devices(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh

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

        selected = SimpleNamespace(
            name="cupy",
            device="cuda:{}".format(rank),
            config=SimpleNamespace(precision=64),
            array_namespace=np,
            asarray=np.asarray,
        )
        monkeypatch.setattr("renormalizer.cons.get_backend", lambda: selected)
        collective = Collective()
        context = DistributedContext(rank, rank, 2, 2)
        runtime = runtime_module.CupyDistributedRuntime(
            backend=selected,
            context=context,
            rendezvous=DistributedRendezvous("127.0.0.1", 23456),
            mesh=DeviceMesh((2,), ("rank",), rank),
            collective=collective,
        )
        config = runtime.execution_config(
            device_memory_budget_bytes=1024,
            host_memory_budget_bytes=2048,
        )
        assert config.backend_device == "cuda:{}".format(rank)
        return collective.digest

    np.testing.assert_array_equal(digest_for_rank(0), digest_for_rank(1))


def test_fake_collectives_have_explicit_numerical_and_mutation_semantics(collective):
    wrapper, backend = collective
    source = cupy.arange(12, dtype=cupy.float64).reshape(3, 4)
    original = source.copy()

    reduced = wrapper.allreduce(source)
    gathered = wrapper.allgather(source, axis=1)
    scattered = wrapper.reduce_scatter(source, axis=1)
    broadcast = wrapper.broadcast(source, root=0)

    cupy.testing.assert_array_equal(reduced, original * 2)
    cupy.testing.assert_array_equal(
        gathered,
        cupy.concatenate((original, original + 100), axis=1),
    )
    cupy.testing.assert_array_equal(scattered, original[:, 2:] * 2)
    assert broadcast is source
    cupy.testing.assert_array_equal(source, cupy.full_like(source, 10))
    assert [call[0] for call in backend.calls] == [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "broadcast",
    ]


def test_inplace_allreduce_reuses_control_array_without_staging_allocation(
    collective, monkeypatch
):
    wrapper, backend = collective
    source = cupy.ones((1,), dtype=cupy.int32)
    allocation_calls = []
    original_empty_like = cupy.empty_like
    original_ascontiguousarray = cupy.ascontiguousarray

    def recording_empty_like(*args, **kwargs):
        allocation_calls.append("empty_like")
        return original_empty_like(*args, **kwargs)

    def recording_ascontiguousarray(*args, **kwargs):
        allocation_calls.append("ascontiguousarray")
        return original_ascontiguousarray(*args, **kwargs)

    monkeypatch.setattr(cupy, "empty_like", recording_empty_like)
    monkeypatch.setattr(cupy, "ascontiguousarray", recording_ascontiguousarray)

    result = wrapper.allreduce_inplace(source, op="max")

    assert result is source
    assert allocation_calls == []
    cupy.testing.assert_array_equal(source, cupy.asarray([2], dtype=cupy.int32))
    assert backend.calls == [("all_reduce", "max", cupy.cuda.Stream.null)]


def test_inplace_allreduce_uses_nondefault_stream_and_restores_active_device(
    collective,
):
    wrapper, backend = collective
    with cupy.cuda.Device(0):
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            source = cupy.ones((1,), dtype=cupy.int32)
            cupy.cuda.Device(1).use()

            result = wrapper.allreduce_inplace(source, op="max")

            assert int(cupy.cuda.runtime.getDevice()) == 1

    assert result is source
    assert backend.calls == [("all_reduce", "max", stream)]


def test_noncontiguous_arrays_are_staged_without_host_conversion(
    collective, monkeypatch
):
    wrapper, _ = collective
    source = cupy.arange(24, dtype=cupy.complex128).reshape(4, 6)[:, ::2]
    original = source.copy()
    monkeypatch.setattr(
        cupy,
        "asnumpy",
        lambda *args, **kwargs: pytest.fail("collective converted through the host"),
    )

    result = wrapper.allreduce(source, op="sum")

    assert bool(cupy.all(result == original * 2))
    assert bool(cupy.all(source == original))


def test_collectives_use_selected_device_current_stream(collective):
    wrapper, backend = collective
    with cupy.cuda.Device(0):
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            source = cupy.arange(4, dtype=cupy.float32)
            cupy.cuda.Device(1).use()
            wrapper.allreduce(source)

    operation, _, used_stream = backend.calls[-1]
    assert operation == "all_reduce"
    assert used_stream is stream


@pytest.mark.parametrize(
    "invoke, message",
    [
        (lambda c, a: c.broadcast(a, root=2), "root 2 is out of range"),
        (lambda c, a: c.allreduce(a, op="mean"), "unsupported reduction op"),
        (lambda c, a: c.allgather(a, axis=2), "axis 2 is out of range"),
        (
            lambda c, a: c.reduce_scatter(a, axis=1),
            "axis length 3 is not divisible by collective size 2",
        ),
    ],
)
def test_illegal_arguments_are_rejected_before_nccL(collective, invoke, message):
    wrapper, backend = collective
    source = cupy.ones((2, 3), dtype=cupy.float64)

    with pytest.raises(ValueError, match=message):
        invoke(wrapper, source)

    assert backend.calls == []


def test_unsupported_dtype_and_complex_op_are_rejected_before_nccl(collective):
    wrapper, backend = collective

    with pytest.raises(TypeError, match="dtype bool is not supported"):
        wrapper.allreduce(cupy.ones(2, dtype=cupy.bool_))
    with pytest.raises(ValueError, match="complex arrays only support sum"):
        wrapper.allreduce(cupy.ones(2, dtype=cupy.complex64), op="max")

    assert backend.calls == []


def test_wrong_device_array_is_rejected_before_nccl(collective):
    if cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible CUDA devices")
    wrapper, backend = collective
    with cupy.cuda.Device(1):
        source = cupy.ones(2)

    with pytest.raises(ValueError, match="selected CUDA device"):
        wrapper.allreduce(source)

    assert backend.calls == []


def test_runtime_parses_environment_once_and_preflights_world_size(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    parsed = []
    original = DistributedContext.from_environ.__func__

    def recording_parser(cls, environ):
        parsed.append(dict(environ))
        return original(cls, environ)

    monkeypatch.setattr(
        DistributedContext, "from_environ", classmethod(recording_parser)
    )
    backend_calls = []

    class FakeBackend:
        def create_collective(self, supplied_context, **kwargs):
            backend_calls.append((supplied_context, kwargs))
            return FakeNcclBackend(supplied_context.world_size, supplied_context.rank)

    monkeypatch.setattr(
        runtime_module,
        "create_backend",
        lambda *args, **kwargs: FakeBackend(),
    )
    environ = {
        "RANK": "9",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "16",
        "LOCAL_WORLD_SIZE": "8",
    }

    runtime = runtime_module.create_cupy_distributed_runtime(
        expected_world_size=16, environ=environ, host="host", port=1234
    )
    environ["RANK"] = "3"
    try:
        assert runtime.rank == 9
        assert runtime.local_rank == 1
        assert len(parsed) == 1
        assert backend_calls == [(runtime.context, {"host": "host", "port": 1234})]
    finally:
        runtime.close()


def test_runtime_snapshots_master_rendezvous_before_backend_creation(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module
    import cupyx.distributed as cupyx_distributed

    initializer_calls = []
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "snapshot-host",
        "MASTER_PORT": "29500",
    }
    original_create_backend = runtime_module.create_backend

    def initialize(*args, **kwargs):
        initializer_calls.append((args, kwargs))
        return FakeNcclBackend(size=args[0], rank=args[1])

    def create_backend_after_environment_mutation(*args, **kwargs):
        environ["MASTER_ADDR"] = "mutated-supplied-host"
        environ["MASTER_PORT"] = "31000"
        monkeypatch.setenv("CUPYX_DISTRIBUTED_HOST", "mutated-ambient-host")
        monkeypatch.setenv("CUPYX_DISTRIBUTED_PORT", "32000")
        return original_create_backend(*args, **kwargs)

    monkeypatch.setattr(cupyx_distributed, "init_process_group", initialize)
    monkeypatch.setattr(
        runtime_module, "create_backend", create_backend_after_environment_mutation
    )

    runtime = runtime_module.create_cupy_distributed_runtime(environ=environ)
    try:
        assert runtime.rendezvous.host == "snapshot-host"
        assert runtime.rendezvous.port == 30500
        assert initializer_calls == [
            (
                (1, 0),
                {
                    "backend": "nccl",
                    "host": "snapshot-host",
                    "port": 30500,
                },
            )
        ]
    finally:
        runtime.close()


def test_runtime_uses_cupyx_rendezvous_override(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    backend_calls = []
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "CUPYX_DISTRIBUTED_HOST": "cupyx-host",
        "CUPYX_DISTRIBUTED_PORT": "24567",
        "MASTER_ADDR": "master-host",
        "MASTER_PORT": "29500",
    }

    class FakeBackend:
        def create_collective(self, supplied_context, **kwargs):
            backend_calls.append((supplied_context, kwargs))
            return FakeNcclBackend(supplied_context.world_size, supplied_context.rank)

    monkeypatch.setattr(
        runtime_module, "create_backend", lambda *args, **kwargs: FakeBackend()
    )

    runtime = runtime_module.create_cupy_distributed_runtime(environ=environ)
    try:
        assert backend_calls == [
            (runtime.context, {"host": "cupyx-host", "port": 24567})
        ]
    finally:
        runtime.close()


def test_runtime_rejects_invalid_rendezvous_before_backend_creation(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime_module, "create_backend", fail_if_called)
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "master-host",
        "MASTER_PORT": "invalid",
    }

    with pytest.raises(ValueError, match="MASTER_PORT must be an integer"):
        runtime_module.create_cupy_distributed_runtime(environ=environ)

    assert called is False


def test_runtime_context_closes_collective_after_body_failure(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    created = []

    class FakeBackend:
        def create_collective(self, supplied_context, **kwargs):
            collective = FakeNcclBackend(
                supplied_context.world_size, supplied_context.rank
            )
            created.append(collective)
            return collective

    monkeypatch.setattr(
        runtime_module, "create_backend", lambda *args, **kwargs: FakeBackend()
    )
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "master-host",
        "MASTER_PORT": "29500",
    }

    with pytest.raises(RuntimeError, match="injected body failure"):
        with runtime_module.cupy_distributed_runtime(environ=environ):
            raise RuntimeError("injected body failure")

    assert len(created) == 1
    assert created[0].stop_calls == 1


def test_runtime_rejects_expected_world_size_before_backend_creation(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime_module, "create_backend", fail_if_called)
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }

    with pytest.raises(ValueError, match="expected world size 4, got 2"):
        runtime_module.create_cupy_distributed_runtime(
            expected_world_size=4, environ=environ
        )

    assert called is False


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires a two-rank torchrun launcher",
)
@pytest.mark.parametrize("dtype", [cupy.float64, cupy.complex128])
def test_real_two_rank_cupy_nccl_collectives_and_cleanup(dtype):
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    assert "CUPYX_DISTRIBUTED_HOST" not in os.environ
    assert "CUPYX_DISTRIBUTED_PORT" not in os.environ
    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    rank = runtime.rank
    local_rank = runtime.local_rank
    master_port = int(os.environ["MASTER_PORT"])
    assert runtime.rendezvous.host == os.environ["MASTER_ADDR"]
    assert runtime.rendezvous.port not in {master_port, 13333}
    collective = runtime.collective
    raw_backend = collective._backend
    backend_reference = weakref.ref(raw_backend)
    store_process = raw_backend._store._process if rank == 0 else None
    del raw_backend

    try:
        with cupy.cuda.Device(local_rank):
            stream = cupy.cuda.Stream(non_blocking=True)
            with stream:
                assert cupy.cuda.runtime.getDevice() == local_rank
                assert cupy.cuda.get_current_stream() is stream

                broadcast_source = cupy.arange(6, dtype=dtype).reshape(2, 3)
                if rank != 0:
                    broadcast_source.fill(-1)
                broadcast = collective.broadcast(broadcast_source, root=0)

                reduced_source = cupy.full((2, 3), rank + 1, dtype=dtype)
                if dtype == cupy.complex128:
                    reduced_source *= 1 + 2j
                reduced = collective.allreduce(reduced_source)

                gathered_source = cupy.arange(4, dtype=dtype).reshape(2, 2) + rank * 10
                if dtype == cupy.complex128:
                    gathered_source *= 1 + 1j
                gathered = collective.allgather(gathered_source, axis=1)

                scattered_source = (
                    cupy.arange(8, dtype=dtype).reshape(2, 4) + rank * 100
                )
                if dtype == cupy.complex128:
                    scattered_source *= 1 - 1j
                scattered = collective.reduce_scatter(scattered_source, axis=1)

                completion = cupy.cuda.Event()
                completion.record(stream)
                assert cupy.cuda.get_current_stream() is stream
            completion.synchronize()

            expected_broadcast = cupy.arange(6, dtype=dtype).reshape(2, 3)
            cupy.testing.assert_array_equal(broadcast, expected_broadcast)
            expected_reduced = cupy.full((2, 3), 3, dtype=dtype)
            if dtype == cupy.complex128:
                expected_reduced *= 1 + 2j
            cupy.testing.assert_array_equal(reduced, expected_reduced)

            gathered_parts = []
            for source_rank in range(2):
                part = cupy.arange(4, dtype=dtype).reshape(2, 2) + source_rank * 10
                if dtype == cupy.complex128:
                    part *= 1 + 1j
                gathered_parts.append(part)
            cupy.testing.assert_array_equal(
                gathered, cupy.concatenate(gathered_parts, axis=1)
            )

            reduced_full = cupy.arange(8, dtype=dtype).reshape(2, 4) * 2 + 100
            if dtype == cupy.complex128:
                reduced_full *= 1 - 1j
            cupy.testing.assert_array_equal(
                scattered, reduced_full[:, rank * 2 : (rank + 1) * 2]
            )

            uuid_bytes = cupy.cuda.runtime.getDeviceProperties(local_rank)["uuid"]
            uuid_hex = bytes(uuid_bytes).hex()
            gpu_uuid = "GPU-{}-{}-{}-{}-{}".format(
                uuid_hex[:8],
                uuid_hex[8:12],
                uuid_hex[12:16],
                uuid_hex[16:20],
                uuid_hex[20:],
            )

        runtime.barrier()
    finally:
        runtime.close()
    runtime.close()
    gc.collect()

    assert collective._backend is None
    assert backend_reference() is None
    if store_process is not None:
        assert not store_process.is_alive()
    print(
        "TASK13_GPU rank={} local_rank={} visible_index={} uuid={} "
        "master_port={} cupyx_port={} stream_ptr={} cleanup=closed".format(
            rank,
            local_rank,
            local_rank,
            gpu_uuid,
            master_port,
            runtime.rendezvous.port,
            stream.ptr,
        ),
        flush=True,
    )
