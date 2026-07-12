import importlib
import threading
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from renormalizer.backend._distributed.residency import (
    ForeignHostTensorRefError,
    HostTensorError,
    HostTensorRef,
    HostTensorStore,
    HostTensorStoreClosedError,
    HostTensorVersionConflictError,
    StaleHostTensorRefError,
)


def test_put_owns_a_read_only_c_copy_and_read_returns_an_independent_snapshot():
    source = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    expected = np.ascontiguousarray(source)
    store = HostTensorStore(store_id="run-a")

    ref = store.put("center", source)
    source.fill(-1)
    snapshot = store.read(ref)
    snapshot.fill(99)

    assert ref == HostTensorRef(
        key="center",
        version=0,
        shape=(3, 2),
        dtype="float64",
        nbytes=48,
        store_id="run-a",
        generation=0,
        layout="C",
    )
    assert snapshot.flags.c_contiguous
    assert snapshot.flags.owndata
    np.testing.assert_array_equal(store.read(ref), expected)


@pytest.mark.parametrize(
    "value",
    [
        np.array([object()], dtype=object),
        np.array(["x"]),
        np.array([1], dtype="datetime64[D]"),
        np.array([(1, 2.0)], dtype=[("x", "i4"), ("y", "f8")]),
        np.array([1], dtype=">i4"),
    ],
)
def test_put_rejects_unsupported_or_non_native_dtype_without_mutation(value):
    store = HostTensorStore(store_id="run-a")

    with pytest.raises((TypeError, ValueError), match="dtype"):
        store.put("bad", value)

    assert store.current_bytes == 0
    assert store.keys() == ()


def test_scalars_and_empty_numeric_arrays_are_supported():
    store = HostTensorStore(store_id="run-a")

    scalar = store.put("scalar", np.array(3 + 2j))
    empty = store.put("empty", np.empty((0, 3), dtype=np.int16))

    assert scalar.shape == ()
    assert scalar.nbytes == np.dtype(np.complex128).itemsize
    assert empty.shape == (0, 3)
    assert empty.nbytes == 0
    assert store.read(empty).shape == (0, 3)


def test_update_is_shape_and_dtype_changing_cas_and_old_ref_is_stale():
    store = HostTensorStore(store_id="run-a")
    old = store.put("center", np.zeros((2, 3), dtype=np.float32))

    new = store.update("center", np.ones((4,), dtype=np.complex128), expected_version=0)

    assert new.version == 1
    assert new.generation == old.generation
    assert new.shape == (4,)
    assert new.dtype == "complex128"
    with pytest.raises(StaleHostTensorRefError):
        store.read(old)
    np.testing.assert_array_equal(store.read(new), np.ones(4, dtype=np.complex128))


def test_failed_update_cas_and_budget_preflight_leave_value_and_version_unchanged():
    store = HostTensorStore(store_id="run-a", host_budget_bytes=64)
    current = store.put("center", np.arange(4, dtype=np.float64))

    with pytest.raises(HostTensorVersionConflictError):
        store.update("center", np.ones(4), expected_version=8)
    with pytest.raises(MemoryError, match="host memory budget"):
        store.update("center", np.ones(5), expected_version=0)

    assert store.ref("center") == current
    np.testing.assert_array_equal(store.read(current), np.arange(4, dtype=np.float64))


def test_put_and_update_budget_preflight_precede_owning_copy(monkeypatch):
    residency = importlib.import_module("renormalizer.backend._distributed.residency")
    put_store = HostTensorStore(store_id="put-budget", host_budget_bytes=8)
    update_store = HostTensorStore(store_id="update-budget", host_budget_bytes=32)
    current = update_store.put("center", np.ones(2, dtype=np.float64))
    copy_calls = []
    original_array = residency.np.array

    def recording_array(*args, **kwargs):
        copy_calls.append((args, kwargs))
        return original_array(*args, **kwargs)

    monkeypatch.setattr(residency.np, "array", recording_array)

    with pytest.raises(MemoryError, match="host memory budget"):
        put_store.put("center", np.ones(2, dtype=np.float64))
    with pytest.raises(MemoryError, match="host memory budget"):
        update_store.update(
            "center", np.ones(3, dtype=np.float64), expected_version=current.version
        )

    assert copy_calls == []
    assert put_store.current_bytes == 0
    assert update_store.ref("center") == current


def test_wave8_exact_copy_rejects_source_resize_without_oversized_allocation(
    monkeypatch,
):
    residency = importlib.import_module("renormalizer.backend._distributed.residency")
    source = np.arange(2, dtype=np.float64)
    store = HostTensorStore(store_id="resize-race")
    allocations = []
    original_empty = residency.np.empty
    original_copyto = residency.np.copyto

    def recording_empty(shape, dtype=None, order="C"):
        allocations.append((tuple(shape), np.dtype(dtype).name, order))
        return original_empty(shape, dtype=dtype, order=order)

    def resizing_copyto(destination, value, *, casting):
        value.resize((1024,), refcheck=False)
        return original_copyto(destination, value, casting=casting)

    monkeypatch.setattr(residency.np, "empty", recording_empty)
    monkeypatch.setattr(residency.np, "copyto", resizing_copyto)

    with pytest.raises(RuntimeError, match="changed during.*copy"):
        store.put("center", source)

    assert allocations == [((2,), "float64", "C")]
    assert store.current_bytes == 0
    assert store.keys() == ()


def test_store_rejects_non_ndarray_without_coercing_or_mutating():
    store = HostTensorStore(store_id="run-a")

    with pytest.raises(TypeError, match="NumPy ndarray"):
        store.put("center", [1.0, 2.0])

    assert store.current_bytes == 0
    assert store.keys() == ()


def test_store_rejects_ndarray_subclass_before_spoofed_budget_metadata_is_read(
    monkeypatch,
):
    residency = importlib.import_module("renormalizer.backend._distributed.residency")

    class SpoofedMetadataArray(np.ndarray):
        @property
        def size(self):
            return 1

        @property
        def nbytes(self):
            return np.dtype(np.float64).itemsize

    value = np.ones(128, dtype=np.float64).view(SpoofedMetadataArray)
    store = HostTensorStore(store_id="subclass-budget", host_budget_bytes=8)
    copy_calls = []
    original_array = residency.np.array

    def recording_array(*args, **kwargs):
        copy_calls.append((args, kwargs))
        return original_array(*args, **kwargs)

    monkeypatch.setattr(residency.np, "array", recording_array)

    with pytest.raises(TypeError, match="NumPy ndarray"):
        store.put("center", value)

    assert copy_calls == []
    assert store.current_bytes == 0
    assert store.keys() == ()


def test_store_snapshot_and_writeback_allocation_are_canonical_immutable_metadata():
    residency = importlib.import_module("renormalizer.backend._distributed.residency")
    store = HostTensorStore(store_id="run-a")
    second = store.put("z", np.ones((2, 3), dtype=np.float32))
    first = store.put("a", np.ones(4, dtype=np.float64))

    snapshot = store.snapshot()
    allocation = residency.HostTensorAllocation(
        key="a", shape=(3, 2), dtype="complex128"
    )

    assert snapshot.store_id == "run-a"
    assert snapshot.refs == (first, second)
    assert snapshot.current_bytes == first.nbytes + second.nbytes
    assert snapshot.namespace_revision == 2
    assert allocation.nbytes == 3 * 2 * np.dtype(np.complex128).itemsize
    assert allocation.layout == "C"
    with pytest.raises(FrozenInstanceError):
        snapshot.current_bytes = 0
    with pytest.raises(FrozenInstanceError):
        allocation.nbytes = 0


def test_remove_and_reinsert_advance_generation_and_never_revive_an_old_ref():
    store = HostTensorStore(store_id="run-a")
    first = store.put("center", np.array([1]))

    store.remove("center", expected_version=0)
    second = store.put("center", np.array([2]))

    assert second.version == 0
    assert second.generation == 1
    with pytest.raises(StaleHostTensorRefError):
        store.read(first)
    np.testing.assert_array_equal(store.read(second), [2])


def test_foreign_and_integrity_modified_refs_are_rejected():
    store = HostTensorStore(store_id="run-a")
    ref = store.put("center", np.arange(3))
    foreign = HostTensorRef(**{**ref.__dict__, "store_id": "run-b"})
    malformed = HostTensorRef(**{**ref.__dict__, "nbytes": ref.nbytes + 1})

    with pytest.raises(ForeignHostTensorRefError):
        store.read(foreign)
    with pytest.raises(StaleHostTensorRefError, match="metadata"):
        store.read(malformed)


def test_read_validates_canonical_slice_and_returns_c_contiguous_copy():
    store = HostTensorStore(store_id="run-a")
    ref = store.put("center", np.arange(24).reshape(4, 6))

    snapshot = store.read(ref, (slice(1, 3, 1), slice(2, 6, 1)))

    assert snapshot.flags.c_contiguous
    assert snapshot.flags.owndata
    np.testing.assert_array_equal(snapshot, np.arange(24).reshape(4, 6)[1:3, 2:6])
    with pytest.raises(ValueError, match="slice"):
        store.read(ref, (slice(None), slice(0, 6, 2)))


def test_copy_into_writes_exact_destination_without_snapshot_allocation(monkeypatch):
    residency = importlib.import_module("renormalizer.backend._distributed.residency")
    store = HostTensorStore(store_id="copy-into")
    value = np.arange(24, dtype=np.float64).reshape(6, 4)
    ref = store.put("tensor", value)
    destination = np.empty((2, 4), dtype=np.float64, order="C")

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("copy_into must not allocate an owning snapshot")

    monkeypatch.setattr(residency, "_canonical_copy", fail_snapshot)
    result = store.copy_into(
        ref,
        destination,
        (slice(2, 4, 1), slice(0, 4, 1)),
    )

    assert result is None
    np.testing.assert_array_equal(destination, value[2:4])
    with pytest.raises(ValueError, match="destination shape"):
        store.copy_into(ref, np.empty((1, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="destination dtype"):
        store.copy_into(ref, np.empty((6, 4), dtype=np.float32))


def test_store_reservation_freezes_complete_snapshot_and_commits_one_owned_cas():
    store = HostTensorStore(store_id="reserved")
    ref = store.put("center", np.zeros(4, dtype=np.float64))
    unrelated = store.put("unrelated", np.ones(2, dtype=np.float64))
    snapshot = store.snapshot()
    reservation = store.reserve(snapshot, dirty_ref=ref)

    with pytest.raises(HostTensorError, match="reserved"):
        store.put("late", np.ones(1, dtype=np.float64))
    with pytest.raises(HostTensorError, match="reserved"):
        store.update("center", np.ones(4), expected_version=ref.version)
    with pytest.raises(HostTensorError, match="reserved"):
        store.update("unrelated", np.zeros(2), expected_version=unrelated.version)
    with pytest.raises(HostTensorError, match="reserved"):
        store.remove("unrelated", expected_version=unrelated.version)
    with pytest.raises(HostTensorError, match="reservations"):
        store.close()

    updated = reservation.commit(ref, np.arange(4.0))
    assert updated.version == ref.version + 1
    assert reservation.snapshot == store.snapshot()
    assert reservation.snapshot.namespace_revision == snapshot.namespace_revision + 1
    np.testing.assert_array_equal(store.read(updated), np.arange(4.0))
    with pytest.raises(HostTensorError, match="already committed"):
        reservation.commit(updated, np.ones(4))
    reservation.close()
    reservation.close()
    store.close()


def test_store_reservation_atomically_rejects_a_stale_complete_snapshot():
    store = HostTensorStore(store_id="snapshot-race")
    dirty = store.put("center", np.zeros(2, dtype=np.float64))
    snapshot = store.snapshot()
    store.put("unrelated", np.ones(1, dtype=np.float64))

    with pytest.raises(HostTensorError, match="complete snapshot"):
        store.reserve(snapshot, dirty_ref=dirty)

    store.close()


def test_close_is_idempotent_invalidates_refs_and_blocks_all_operations():
    store = HostTensorStore(store_id="run-a")
    ref = store.put("center", np.arange(3))

    store.close()
    store.close()

    assert store.current_bytes == 0
    for operation in (
        lambda: store.read(ref),
        lambda: store.ref("center"),
        lambda: store.put("other", np.arange(2)),
        lambda: store.update("center", np.arange(2), expected_version=0),
        lambda: store.remove("center", expected_version=0),
    ):
        with pytest.raises(HostTensorStoreClosedError):
            operation()


def test_context_manager_closes_store():
    with HostTensorStore(store_id="run-a") as store:
        ref = store.put("center", np.arange(2))

    with pytest.raises(HostTensorStoreClosedError):
        store.read(ref)


def test_concurrent_reads_observe_only_complete_versions():
    store = HostTensorStore(store_id="run-a")
    ref = store.put("center", np.zeros(10000, dtype=np.int64))
    barrier = threading.Barrier(2)
    snapshots = []

    def reader():
        barrier.wait()
        while True:
            try:
                snapshots.append(store.read(ref))
            except StaleHostTensorRefError:
                return

    thread = threading.Thread(target=reader)
    thread.start()
    barrier.wait()
    updated = store.update("center", np.ones(10000, dtype=np.int64), expected_version=0)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert all(np.all(snapshot == 0) for snapshot in snapshots)
    assert np.all(store.read(updated) == 1)
