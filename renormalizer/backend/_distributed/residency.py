"""Host tensor ownership and immutable active-working-set metadata."""

import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
import threading
import uuid

import numpy as np

from renormalizer.backend._distributed.planner import DistributedPlan
from renormalizer.backend._distributed.sharding import ShardingSpec
from renormalizer.backend._distributed.solvers import (
    KrylovCoefficientIdentity,
    SolverMemoryProfile,
    canonical_solver_memory_profile,
    krylov_coefficient_payload,
)
from renormalizer.backend._distributed.terminal import (
    _publish_lease_construction_resource,
    _publish_lease_construction_resource_direct,
    _require_managed_resource_admission,
)
from renormalizer.backend._distributed.transfer import TransferProfile


_NUMERIC_DTYPE_KINDS = frozenset({"b", "i", "u", "f", "c"})
_MAX_COLLECTIVE_MEMORY_BYTES = int(np.iinfo(np.int64).max)


def _checked_memory_bytes(value, name):
    if type(value) is not int:
        raise TypeError("{} must be a Python integer".format(name))
    if value < 0:
        raise ValueError("{} must be non-negative".format(name))
    if value > _MAX_COLLECTIVE_MEMORY_BYTES:
        raise OverflowError("{} exceeds the supported int64 memory range".format(name))
    return value


def _checked_memory_sum(values, name):
    total = 0
    for value in values:
        total += _checked_memory_bytes(value, name)
        if total > _MAX_COLLECTIVE_MEMORY_BYTES:
            raise OverflowError(
                "{} exceeds the supported int64 memory range".format(name)
            )
    return total


def _checked_memory_product(values, name):
    product = 1
    for value in values:
        value = _checked_memory_bytes(value, name)
        product *= value
        if product > _MAX_COLLECTIVE_MEMORY_BYTES:
            raise OverflowError(
                "{} exceeds the supported int64 memory range".format(name)
            )
    return product


def _checked_shape_elements(shape, name):
    return _checked_memory_product(tuple(shape), "{} elements".format(name))


class HostTensorError(RuntimeError):
    """Base error for versioned host tensor storage."""


class HostTensorStoreClosedError(HostTensorError):
    """Raised when an operation targets a closed host tensor store."""


class ForeignHostTensorRefError(HostTensorError):
    """Raised when a reference belongs to another logical store."""


class StaleHostTensorRefError(HostTensorError):
    """Raised when a reference no longer identifies the current entry."""


class HostTensorVersionConflictError(HostTensorError):
    """Raised when a compare-and-swap version does not match."""


@dataclass(frozen=True)
class HostTensorRef:
    key: str
    version: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    store_id: str
    generation: int
    layout: str = "C"

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("host tensor key must be a non-empty string")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("host tensor version must be a non-negative integer")
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise TypeError("host tensor shape must be an iterable") from error
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError(
                "host tensor shape dimensions must be non-negative integers"
            )
        if not isinstance(self.dtype, str):
            raise TypeError("host tensor dtype must be a string")
        _checked_memory_bytes(self.nbytes, "host tensor nbytes")
        if not isinstance(self.store_id, str) or not self.store_id:
            raise ValueError("host tensor store_id must be a non-empty string")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("host tensor generation must be a non-negative integer")
        if self.layout != "C":
            raise ValueError("host tensor layout must be 'C'")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class _HostTensorEntry:
    ref: HostTensorRef
    array: np.ndarray


def _validate_key(key):
    if not isinstance(key, str):
        raise TypeError("host tensor key must be a string")
    if not key:
        raise ValueError("host tensor key must not be empty")


def _validate_expected_version(expected_version):
    if type(expected_version) is not int or expected_version < 0:
        raise ValueError("expected_version must be a non-negative integer")


def _canonical_array_metadata(value):
    if type(value) is not np.ndarray:
        raise TypeError("host tensor value must be a NumPy ndarray")
    dtype = value.dtype
    if dtype.fields is not None or dtype.kind not in _NUMERIC_DTYPE_KINDS:
        raise TypeError("host tensor dtype must be a numeric TensorSpec dtype")
    if not dtype.isnative:
        raise ValueError("host tensor dtype must use native byte order")
    shape = tuple(value.shape)
    nbytes = _checked_memory_product(
        (int(value.size), int(dtype.itemsize)), "host tensor nbytes"
    )
    if int(value.nbytes) != nbytes:
        raise ValueError("host tensor nbytes do not match shape and dtype")
    return shape, dtype, nbytes


def _canonical_copy(value, metadata):
    shape, dtype, nbytes = metadata
    destination = np.empty(shape, dtype=dtype, order="C")
    try:
        np.copyto(destination, value, casting="no")
    except Exception as error:
        try:
            current = _canonical_array_metadata(value)
        except Exception as metadata_error:
            raise RuntimeError(
                "host tensor changed during exact copy"
            ) from metadata_error
        if current != metadata:
            raise RuntimeError("host tensor changed during exact copy") from error
        raise
    if _canonical_array_metadata(value) != metadata:
        raise RuntimeError("host tensor changed during exact copy")
    if (
        destination.base is not None
        or not destination.flags.owndata
        or tuple(destination.shape) != shape
        or destination.dtype != dtype
        or int(destination.nbytes) != nbytes
        or not destination.flags.c_contiguous
    ):
        raise RuntimeError("exact host tensor destination is inconsistent")
    destination.setflags(write=False)
    return destination


def _canonical_read_slice(local_slice, shape):
    if local_slice is None:
        return None
    if not isinstance(local_slice, tuple):
        raise TypeError("local_slice must be a tuple of slices or None")
    if len(local_slice) != len(shape):
        raise ValueError("local_slice rank does not match tensor shape")
    canonical = []
    for item, dimension in zip(local_slice, shape):
        if not isinstance(item, slice):
            raise TypeError("local_slice entries must be slices")
        if (
            type(item.start) is not int
            or type(item.stop) is not int
            or type(item.step) is not int
            or item.step != 1
            or item.start < 0
            or item.stop > dimension
            or item.start >= item.stop
        ):
            raise ValueError("local_slice must contain canonical non-empty slices")
        canonical.append(slice(item.start, item.stop, 1))
    return tuple(canonical)


class HostTensorStore:
    """Linearizable owner of private, immutable C-contiguous NumPy arrays."""

    def __init__(self, *, store_id=None, host_budget_bytes=None):
        if store_id is None:
            store_id = uuid.uuid4().hex
        if not isinstance(store_id, str) or not store_id:
            raise ValueError("store_id must be a non-empty string")
        if host_budget_bytes is not None and (
            type(host_budget_bytes) is not int or host_budget_bytes <= 0
        ):
            raise ValueError("host_budget_bytes must be a positive integer or None")
        self._store_id = store_id
        self._host_budget_bytes = host_budget_bytes
        self._entries = {}
        self._generations = {}
        self._current_bytes = 0
        self._namespace_revision = 0
        self._reservations = {}
        self._closed = False
        self._lock = threading.RLock()

    @property
    def store_id(self):
        return self._store_id

    @property
    def host_budget_bytes(self):
        return self._host_budget_bytes

    @property
    def current_bytes(self):
        with self._lock:
            return self._current_bytes

    @property
    def closed(self):
        with self._lock:
            return self._closed

    def keys(self):
        with self._lock:
            self._require_open()
            return tuple(sorted(self._entries))

    def snapshot(self):
        with self._lock:
            self._require_open()
            return self._snapshot_locked()

    def _snapshot_locked(self):
        refs = tuple(self._entries[key].ref for key in sorted(self._entries))
        return HostTensorStoreSnapshot(
            self._store_id,
            refs,
            namespace_revision=self._namespace_revision,
        )

    def _require_unreserved_namespace(self):
        if self._reservations:
            raise HostTensorError("host tensor store is reserved by an active lease")

    def _require_open(self):
        if self._closed:
            raise HostTensorStoreClosedError("host tensor store is closed")

    def _check_budget(self, committed_bytes, pending_bytes):
        required = _checked_memory_sum(
            (committed_bytes, pending_bytes), "host tensor store requirement"
        )
        if self._host_budget_bytes is not None and required > self._host_budget_bytes:
            raise MemoryError(
                "host memory budget exceeded: required {} bytes, budget {} bytes".format(
                    required, self._host_budget_bytes
                )
            )

    def _make_ref(self, key, version, generation, array):
        return HostTensorRef(
            key=key,
            version=version,
            shape=tuple(array.shape),
            dtype=array.dtype.name,
            nbytes=int(array.nbytes),
            store_id=self._store_id,
            generation=generation,
            layout="C",
        )

    def put(self, key, value):
        _validate_key(key)
        with self._lock:
            self._require_open()
            self._require_unreserved_namespace()
            if key in self._entries:
                raise KeyError("host tensor key already exists: {!r}".format(key))
            metadata = _canonical_array_metadata(value)
            self._check_budget(self._current_bytes, metadata[2])
            array = _canonical_copy(value, metadata)
            generation = self._generations.get(key, 0)
            ref = self._make_ref(key, 0, generation, array)
            self._entries[key] = _HostTensorEntry(ref, array)
            self._current_bytes = _checked_memory_sum(
                (self._current_bytes, ref.nbytes), "host tensor store bytes"
            )
            self._namespace_revision += 1
            return ref

    def ref(self, key):
        _validate_key(key)
        with self._lock:
            self._require_open()
            try:
                return self._entries[key].ref
            except KeyError:
                raise KeyError("unknown host tensor key: {!r}".format(key)) from None

    def _validated_entry(self, ref):
        if not isinstance(ref, HostTensorRef):
            raise TypeError("ref must be a HostTensorRef")
        if ref.store_id != self._store_id:
            raise ForeignHostTensorRefError(
                "host tensor ref belongs to a foreign store"
            )
        entry = self._entries.get(ref.key)
        if entry is None:
            raise StaleHostTensorRefError("host tensor ref is stale")
        current = entry.ref
        if (ref.generation, ref.version) != (current.generation, current.version):
            raise StaleHostTensorRefError("host tensor ref is stale")
        if ref != current:
            raise StaleHostTensorRefError("host tensor ref metadata is stale")
        return entry

    def read(self, ref, local_slice=None):
        with self._lock:
            self._require_open()
            entry = self._validated_entry(ref)
            canonical_slice = _canonical_read_slice(local_slice, entry.ref.shape)
            source = (
                entry.array if canonical_slice is None else entry.array[canonical_slice]
            )
            snapshot = np.array(source, dtype=source.dtype, order="C", copy=True)
            if snapshot.base is not None or not snapshot.flags.owndata:
                snapshot = snapshot.copy(order="C")
            return snapshot

    def copy_into(self, ref, destination, local_slice=None):
        """Copy one exact retained value directly into caller-owned host storage."""
        if type(destination) is not np.ndarray:
            raise TypeError("destination must be a NumPy ndarray")
        with self._lock:
            self._require_open()
            entry = self._validated_entry(ref)
            canonical_slice = _canonical_read_slice(local_slice, entry.ref.shape)
            source = (
                entry.array if canonical_slice is None else entry.array[canonical_slice]
            )
            if tuple(destination.shape) != tuple(source.shape):
                raise ValueError(
                    "destination shape does not match the requested tensor"
                )
            if destination.dtype != source.dtype:
                raise ValueError(
                    "destination dtype does not match the requested tensor"
                )
            if int(destination.nbytes) != int(source.nbytes):
                raise ValueError("destination nbytes do not match the requested tensor")
            if not destination.flags.writeable:
                raise ValueError("destination must be writable")
            np.copyto(destination, source, casting="no")

    def reserve(
        self,
        snapshot,
        *,
        dirty_ref,
        _construction_slot=None,
        _managed_guard=None,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_managed_resource_admission(
            _managed_guard,
            _admission_token,
            _admission_validator,
            allowed_scopes=("construction",),
            allowed_operations=("lease_construction",),
        )
        if not isinstance(snapshot, HostTensorStoreSnapshot):
            raise TypeError("snapshot must be a HostTensorStoreSnapshot")
        if not isinstance(dirty_ref, HostTensorRef):
            raise TypeError("dirty_ref must be a HostTensorRef")
        with self._lock:
            self._require_open()
            self._require_unreserved_namespace()
            if snapshot != self._snapshot_locked():
                raise HostTensorError(
                    "host tensor store does not match the complete snapshot"
                )
            self._validated_entry(dirty_ref)
            if dirty_ref not in snapshot.refs:
                raise HostTensorError(
                    "authorized dirty ref is outside the complete snapshot"
                )
            reservation = _HostTensorReservation(
                self,
                snapshot,
                dirty_ref,
                _managed_guard=_managed_guard,
                _managed_epoch=(
                    None if _admission_token is None else _admission_token.epoch
                ),
            )
            self._reservations[id(reservation)] = reservation
        _publish_lease_construction_resource_direct(
            _construction_slot,
            reservation,
        )
        _publish_lease_construction_resource(
            _construction_slot,
            reservation,
        )
        return reservation

    def _release_reservation(self, reservation):
        with self._lock:
            self._reservations.pop(id(reservation), None)

    def _commit_reservation(
        self,
        reservation,
        ref,
        value,
        *,
        completion_capability=None,
    ):
        with self._lock:
            self._require_open()
            retained = self._reservations.get(id(reservation))
            if retained is not reservation:
                raise HostTensorError("host tensor reservation is closed")
            if reservation._committed:
                if (
                    completion_capability is not None
                    and reservation._commit_capability
                    is completion_capability
                    and reservation._commit_source_ref == ref
                ):
                    return reservation._commit_result
                raise HostTensorError("host tensor reservation already committed")
            if self._snapshot_locked() != reservation._snapshot:
                raise HostTensorError(
                    "host tensor store changed outside its complete snapshot"
                )
            if ref != reservation._dirty_ref:
                raise HostTensorError(
                    "host tensor ref is not owned by this reservation"
                )
            current = self._validated_entry(ref)
            metadata = _canonical_array_metadata(value)
            self._check_budget(self._current_bytes, metadata[2])
            array = _canonical_copy(value, metadata)
            updated = self._make_ref(
                ref.key,
                current.ref.version + 1,
                current.ref.generation,
                array,
            )
            self._entries[ref.key] = _HostTensorEntry(updated, array)
            self._current_bytes = _checked_memory_sum(
                (self._current_bytes - current.ref.nbytes, updated.nbytes),
                "host tensor store bytes",
            )
            self._namespace_revision += 1
            reservation._promote(
                ref,
                updated,
                self._snapshot_locked(),
                completion_capability=completion_capability,
            )
            return updated

    def validate(self, ref):
        """Validate identity and integrity metadata without copying tensor values."""
        with self._lock:
            self._require_open()
            self._validated_entry(ref)

    def update(self, key, value, *, expected_version):
        _validate_key(key)
        _validate_expected_version(expected_version)
        with self._lock:
            self._require_open()
            self._require_unreserved_namespace()
            try:
                current = self._entries[key]
            except KeyError:
                raise KeyError("unknown host tensor key: {!r}".format(key)) from None
            if current.ref.version != expected_version:
                raise HostTensorVersionConflictError(
                    "host tensor version conflict for {!r}: expected {}, current {}".format(
                        key, expected_version, current.ref.version
                    )
                )
            metadata = _canonical_array_metadata(value)
            self._check_budget(self._current_bytes, metadata[2])
            array = _canonical_copy(value, metadata)
            ref = self._make_ref(
                key,
                current.ref.version + 1,
                current.ref.generation,
                array,
            )
            self._entries[key] = _HostTensorEntry(ref, array)
            self._current_bytes = _checked_memory_sum(
                (self._current_bytes - current.ref.nbytes, ref.nbytes),
                "host tensor store bytes",
            )
            self._namespace_revision += 1
            return ref

    def remove(self, key, *, expected_version):
        _validate_key(key)
        _validate_expected_version(expected_version)
        with self._lock:
            self._require_open()
            self._require_unreserved_namespace()
            try:
                current = self._entries[key]
            except KeyError:
                raise KeyError("unknown host tensor key: {!r}".format(key)) from None
            if current.ref.version != expected_version:
                raise HostTensorVersionConflictError(
                    "host tensor version conflict for {!r}: expected {}, current {}".format(
                        key, expected_version, current.ref.version
                    )
                )
            del self._entries[key]
            self._current_bytes -= current.ref.nbytes
            self._generations[key] = current.ref.generation + 1
            self._namespace_revision += 1
            return current.ref

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._reservations:
                raise HostTensorError("host tensor store has active reservations")
            self._entries.clear()
            self._current_bytes = 0
            self._closed = True

    def __enter__(self):
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                pass
        return False


class _HostTensorReservation:
    def __init__(
        self,
        store,
        snapshot,
        dirty_ref,
        *,
        _managed_guard=None,
        _managed_epoch=None,
    ):
        self._store = store
        self._snapshot = snapshot
        self._dirty_ref = dirty_ref
        self._committed = False
        self._commit_capability = None
        self._commit_source_ref = None
        self._commit_result = None
        self._closed = False
        self._managed_guard = _managed_guard
        self._managed_epoch = _managed_epoch

    def _require_admission(
        self,
        token=None,
        validator=None,
        *,
        allowed_scopes,
        allowed_operations,
    ):
        return _require_managed_resource_admission(
            self._managed_guard,
            token,
            validator,
            allowed_scopes=allowed_scopes,
            allowed_operations=allowed_operations,
            epoch=lambda _token: self._managed_epoch,
        )

    @property
    def refs(self):
        self._require_admission(
            allowed_scopes=("construction", "lease", "lease_close"),
            allowed_operations=(
                "lease_construction",
                "operator_call",
                "resource_state",
                "schedule_writeback",
                "store_reservation_close",
                "d2h_completion",
            ),
        )
        if self._closed:
            return ()
        return self._snapshot.refs

    @property
    def snapshot(self):
        self._require_admission(
            allowed_scopes=("construction", "lease", "lease_close"),
            allowed_operations=(
                "lease_construction",
                "operator_call",
                "resource_state",
                "schedule_writeback",
                "store_reservation_close",
                "d2h_completion",
            ),
        )
        if self._closed:
            raise HostTensorError("host tensor reservation is closed")
        return self._snapshot

    def _promote(
        self,
        source_ref,
        updated,
        snapshot,
        *,
        completion_capability,
    ):
        self._dirty_ref = updated
        self._snapshot = snapshot
        self._committed = True
        self._commit_capability = completion_capability
        self._commit_source_ref = source_ref
        self._commit_result = updated

    def commit(
        self,
        ref,
        value,
        *,
        _admission_token=None,
        _admission_validator=None,
        _completion_capability=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_scopes=("lease", "lease_close"),
            allowed_operations=("d2h_completion",),
        )
        if self._closed:
            raise HostTensorError("host tensor reservation is closed")
        return self._store._commit_reservation(
            self,
            ref,
            value,
            completion_capability=_completion_capability,
        )

    def close(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_scopes=("construction", "lease_close"),
            allowed_operations=(
                "lease_construction",
                "store_reservation_close",
            ),
        )
        if self._closed:
            return
        store = self._store
        try:
            store._release_reservation(self)
        finally:
            self._store = None
            self._snapshot = None
            self._dirty_ref = None
            self._commit_capability = None
            self._commit_source_ref = None
            self._commit_result = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise HostTensorError("host tensor reservation is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                pass
        return False


def _require_sha256(value, name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("{} must be a SHA-256 hexadecimal digest".format(name))
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            "{} must be a SHA-256 hexadecimal digest".format(name)
        ) from error


def _validate_nonnegative_rank_values(values, name, world_size):
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError("{} must be an iterable".format(name)) from error
    if len(normalized) != world_size:
        raise ValueError("{} must contain one value per rank".format(name))
    for value in normalized:
        _checked_memory_bytes(value, "{} values".format(name))
    return normalized


def _ref_identity(ref):
    return ref.store_id, ref.key, ref.generation, ref.version


def _ref_payload(ref):
    return {
        "key": ref.key,
        "version": ref.version,
        "shape": list(ref.shape),
        "dtype": ref.dtype,
        "nbytes": ref.nbytes,
        "store_id": ref.store_id,
        "generation": ref.generation,
        "layout": ref.layout,
    }


def _store_snapshot_payload(snapshot):
    return {
        "store_id": snapshot.store_id,
        "refs": [_ref_payload(ref) for ref in snapshot.refs],
        "namespace_revision": snapshot.namespace_revision,
        "current_bytes": snapshot.current_bytes,
    }


def _allocation_payload(allocation):
    return {
        "key": allocation.key,
        "shape": list(allocation.shape),
        "dtype": allocation.dtype,
        "layout": allocation.layout,
        "nbytes": allocation.nbytes,
    }


def _metadata_nbytes(shape, dtype, name):
    try:
        shape = tuple(shape)
    except TypeError as error:
        raise TypeError("{} shape must be an iterable".format(name)) from error
    if any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise ValueError(
            "{} shape dimensions must be non-negative integers".format(name)
        )
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("{} dtype is unsupported".format(name)) from error
    if (
        dtype.fields is not None
        or dtype.kind not in _NUMERIC_DTYPE_KINDS
        or not dtype.isnative
    ):
        raise ValueError("{} dtype is unsupported".format(name))
    elements = _checked_shape_elements(shape, name)
    nbytes = _checked_memory_product(
        (elements, int(dtype.itemsize)), "{} nbytes".format(name)
    )
    return shape, dtype, nbytes


@dataclass(frozen=True)
class HostTensorStoreSnapshot:
    store_id: str
    refs: tuple[HostTensorRef, ...]
    namespace_revision: int = 0
    current_bytes: int = field(init=False)

    def __post_init__(self):
        if not isinstance(self.store_id, str) or not self.store_id:
            raise ValueError("snapshot store_id must be a non-empty string")
        if type(self.namespace_revision) is not int or self.namespace_revision < 0:
            raise ValueError("snapshot namespace_revision must be non-negative")
        try:
            refs = tuple(self.refs)
        except TypeError as error:
            raise TypeError("snapshot refs must be an iterable") from error
        if any(not isinstance(ref, HostTensorRef) for ref in refs):
            raise TypeError("snapshot refs must contain HostTensorRef values")
        if refs != tuple(sorted(refs, key=_ref_identity)):
            raise ValueError("snapshot refs must be canonical")
        if len({ref.key for ref in refs}) != len(refs):
            raise ValueError("snapshot refs must contain unique keys")
        current_bytes = 0
        for ref in refs:
            if ref.store_id != self.store_id:
                raise ValueError("snapshot refs must share its store_id")
            shape, dtype, nbytes = _metadata_nbytes(
                ref.shape, ref.dtype, "host tensor ref"
            )
            if (
                shape != ref.shape
                or dtype.name != ref.dtype
                or nbytes != ref.nbytes
                or ref.layout != "C"
            ):
                raise ValueError("snapshot ref metadata is inconsistent")
            current_bytes = _checked_memory_sum(
                (current_bytes, nbytes), "host tensor snapshot bytes"
            )
        object.__setattr__(self, "refs", refs)
        object.__setattr__(self, "current_bytes", current_bytes)


@dataclass(frozen=True)
class HostTensorAllocation:
    key: str
    shape: tuple[int, ...]
    dtype: str
    layout: str = "C"
    nbytes: int = field(init=False)

    def __post_init__(self):
        _validate_key(self.key)
        shape, dtype, nbytes = _metadata_nbytes(
            self.shape, self.dtype, "host tensor allocation"
        )
        if not isinstance(self.dtype, str) or self.dtype != dtype.name:
            raise ValueError("host tensor allocation dtype must be canonical")
        if self.layout != "C":
            raise ValueError("host tensor allocation layout must be 'C'")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.name)
        object.__setattr__(self, "nbytes", nbytes)


@dataclass(frozen=True)
class MemoryBudgetResolution:
    requested_bytes: int | None
    resolved_bytes: int
    source: str
    available_snapshot_bytes: int | None
    resource: str

    def __post_init__(self):
        if self.requested_bytes is not None:
            _checked_memory_bytes(self.requested_bytes, "requested_bytes")
            if self.requested_bytes == 0:
                raise ValueError("requested_bytes must be a positive integer or None")
        _checked_memory_bytes(self.resolved_bytes, "resolved_bytes")
        if self.resolved_bytes == 0:
            raise ValueError("resolved_bytes must be a positive integer")
        if self.source not in {"explicit", "auto"}:
            raise ValueError("budget source must be 'explicit' or 'auto'")
        if self.resource not in {"device", "host"}:
            raise ValueError("budget resource must be 'device' or 'host'")
        if self.available_snapshot_bytes is not None:
            _checked_memory_bytes(
                self.available_snapshot_bytes, "available_snapshot_bytes"
            )
            if self.available_snapshot_bytes == 0:
                raise ValueError(
                    "available_snapshot_bytes must be a positive integer or None"
                )
        if self.source == "explicit":
            if (
                self.requested_bytes is None
                or self.resolved_bytes != self.requested_bytes
                or self.available_snapshot_bytes is not None
            ):
                raise ValueError("explicit budget metadata is inconsistent")
        else:
            if (
                self.requested_bytes is not None
                or self.available_snapshot_bytes is None
            ):
                raise ValueError("auto budget metadata is inconsistent")
            ratio = 85 if self.resource == "device" else 80
            expected = self.available_snapshot_bytes * ratio // 100
            if self.resolved_bytes != expected:
                raise ValueError(
                    "auto {} budget does not match its availability snapshot".format(
                        self.resource
                    )
                )


def budget_resolution_hash(value):
    value = _bound_budget_resolution(value, value.resource)
    return _sha256(_budget_payload(value))


@dataclass(frozen=True)
class ResidencyPreflightReceipt:
    runtime_id: str
    rank: int
    local_device: str
    request_hash: str
    plan_hash: str
    device_budget_hash: str
    host_budget_hash: str
    receipt_hash: str

    def __post_init__(self):
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise ValueError("receipt runtime_id must be a non-empty string")
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("receipt rank must be a non-negative integer")
        if not isinstance(self.local_device, str) or not self.local_device:
            raise ValueError("receipt local_device must be a non-empty string")
        for name in (
            "request_hash",
            "plan_hash",
            "device_budget_hash",
            "host_budget_hash",
            "receipt_hash",
        ):
            _require_sha256(getattr(self, name), name)
        if self.receipt_hash != _sha256(_receipt_payload(self)):
            raise ValueError("receipt_hash does not match canonical preflight metadata")

    @classmethod
    def create(
        cls,
        *,
        runtime_id,
        rank,
        local_device,
        request_hash,
        plan_hash,
        device_budget_hash,
        host_budget_hash,
    ):
        fields = {
            "runtime_id": runtime_id,
            "rank": rank,
            "local_device": local_device,
            "request_hash": request_hash,
            "plan_hash": plan_hash,
            "device_budget_hash": device_budget_hash,
            "host_budget_hash": host_budget_hash,
        }
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        fields["receipt_hash"] = _sha256(_receipt_payload(provisional))
        return cls(**fields)


def _receipt_payload(value):
    return {
        "schema": "renormalizer.residency.receipt.v1",
        "runtime_id": value.runtime_id,
        "rank": value.rank,
        "local_device": value.local_device,
        "request_hash": value.request_hash,
        "plan_hash": value.plan_hash,
        "device_budget_hash": value.device_budget_hash,
        "host_budget_hash": value.host_budget_hash,
    }


@dataclass(frozen=True)
class ResidencyRuntimeIdentity:
    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    rank_to_node: tuple[int, ...]
    rank_to_local_rank: tuple[int, ...]
    rank_to_device: tuple[str, ...]
    backend_name: str
    device_budget_resource: str
    host_budget_resource: str
    device_budget_bindings: tuple[str, ...]
    host_budget_bindings: tuple[str, ...]

    def __post_init__(self):
        mesh_shape = tuple(self.mesh_shape)
        mesh_axis_names = tuple(self.mesh_axis_names)
        rank_to_node = tuple(self.rank_to_node)
        rank_to_local_rank = tuple(self.rank_to_local_rank)
        rank_to_device = tuple(self.rank_to_device)
        device_bindings = tuple(self.device_budget_bindings)
        host_bindings = tuple(self.host_budget_bindings)
        if not mesh_shape or any(
            type(value) is not int or value <= 0 for value in mesh_shape
        ):
            raise ValueError("residency runtime identity mesh shape is invalid")
        if (
            len(mesh_axis_names) != len(mesh_shape)
            or any(not isinstance(value, str) or not value for value in mesh_axis_names)
            or len(set(mesh_axis_names)) != len(mesh_axis_names)
        ):
            raise ValueError("residency runtime identity mesh axes are invalid")
        world_size = 1
        for value in mesh_shape:
            world_size *= value
        if any(
            len(values) != world_size
            for values in (rank_to_node, rank_to_local_rank, rank_to_device)
        ):
            raise ValueError("residency runtime identity rank mapping is incomplete")
        if rank_to_node != (0,) * world_size:
            raise NotImplementedError(
                "residency runtime identity currently requires one node"
            )
        if rank_to_local_rank != tuple(range(world_size)):
            raise ValueError(
                "residency runtime identity local-rank mapping is noncanonical"
            )
        if self.backend_name not in {"numpy", "cupy"}:
            raise ValueError("residency runtime identity backend is unsupported")
        expected_devices = (
            ("cpu",) * world_size
            if self.backend_name == "numpy"
            else tuple("cuda:{}".format(rank) for rank in range(world_size))
        )
        if rank_to_device != expected_devices:
            raise ValueError(
                "residency runtime identity device mapping is noncanonical"
            )
        if self.device_budget_resource != "device":
            raise ValueError(
                "residency runtime identity device budget resource is invalid"
            )
        if self.host_budget_resource != "host":
            raise ValueError(
                "residency runtime identity host budget resource is invalid"
            )
        expected_device_bindings = tuple(
            "device:{}:{}".format(self.backend_name, device)
            for device in rank_to_device
        )
        if device_bindings != expected_device_bindings:
            raise ValueError(
                "residency runtime identity device budget bindings are invalid"
            )
        if host_bindings != ("host:node:0",):
            raise ValueError(
                "residency runtime identity host budget bindings are invalid"
            )
        object.__setattr__(self, "mesh_shape", mesh_shape)
        object.__setattr__(self, "mesh_axis_names", mesh_axis_names)
        object.__setattr__(self, "rank_to_node", rank_to_node)
        object.__setattr__(self, "rank_to_local_rank", rank_to_local_rank)
        object.__setattr__(self, "rank_to_device", rank_to_device)
        object.__setattr__(self, "device_budget_bindings", device_bindings)
        object.__setattr__(self, "host_budget_bindings", host_bindings)

    @classmethod
    def one_node(
        cls,
        *,
        world_size,
        backend_name,
        mesh_shape=None,
        mesh_axis_names=None,
    ):
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        shape = (world_size,) if mesh_shape is None else tuple(mesh_shape)
        axes = ("rank",) if mesh_axis_names is None else tuple(mesh_axis_names)
        devices = (
            ("cpu",) * world_size
            if backend_name == "numpy"
            else tuple("cuda:{}".format(rank) for rank in range(world_size))
        )
        return cls(
            mesh_shape=shape,
            mesh_axis_names=axes,
            rank_to_node=(0,) * world_size,
            rank_to_local_rank=tuple(range(world_size)),
            rank_to_device=devices,
            backend_name=backend_name,
            device_budget_resource="device",
            host_budget_resource="host",
            device_budget_bindings=tuple(
                "device:{}:{}".format(backend_name, device) for device in devices
            ),
            host_budget_bindings=("host:node:0",),
        )

    def validate_request(
        self,
        *,
        world_size,
        local_world_size,
        backend_name,
        device_budget,
        host_budget,
    ):
        if world_size != len(self.rank_to_node):
            raise ValueError(
                "residency runtime identity world_size does not match the request"
            )
        if local_world_size != world_size:
            raise ValueError(
                "residency runtime identity local_world_size does not match world_size"
            )
        if backend_name != self.backend_name:
            raise ValueError(
                "residency runtime identity backend does not match the request"
            )
        if (
            device_budget.resource != self.device_budget_resource
            or host_budget.resource != self.host_budget_resource
        ):
            raise ValueError(
                "residency runtime identity does not match budget resources"
            )

    def validate_runtime(self, context, mesh, backend):
        from renormalizer.backend._distributed.context import DistributedContext
        from renormalizer.backend._distributed.mesh import DeviceMesh

        if not isinstance(context, DistributedContext):
            raise TypeError("residency runtime identity requires DistributedContext")
        if not isinstance(mesh, DeviceMesh):
            raise TypeError("residency runtime identity requires DeviceMesh")
        if (
            context.world_size != len(self.rank_to_node)
            or context.local_world_size != context.world_size
            or context.local_rank != self.rank_to_local_rank[context.rank]
            or mesh.shape != self.mesh_shape
            or mesh.axis_names != self.mesh_axis_names
            or mesh.rank != context.rank
        ):
            raise ValueError(
                "residency runtime identity does not match context or mesh"
            )
        backend_name = getattr(backend, "name", None)
        current_device = getattr(backend, "current_device", None)
        device = current_device() if callable(current_device) else None
        if (
            backend_name != self.backend_name
            or str(device) != self.rank_to_device[context.rank]
        ):
            raise ValueError(
                "residency runtime identity does not match backend or device"
            )


def _bound_budget_resolution(value, resource):
    if not isinstance(value, MemoryBudgetResolution):
        raise TypeError("budget must use MemoryBudgetResolution metadata")
    if resource not in {"device", "host"}:
        raise ValueError("budget resource must be 'device' or 'host'")
    if value.resource != resource:
        raise ValueError("{} budget metadata has the wrong resource".format(resource))
    return value


@dataclass(frozen=True, order=True)
class SliceRange:
    start: int
    stop: int
    step: int = 1

    def __post_init__(self):
        if any(type(value) is not int for value in (self.start, self.stop, self.step)):
            raise TypeError("slice range values must be integers")
        if self.start < 0 or self.stop <= self.start or self.step != 1:
            raise ValueError("slice range must be canonical, non-empty, and unit-step")


@dataclass(frozen=True, order=True)
class TensorPlacement:
    key: str
    rank: int
    source_rank: int
    ranges: tuple[SliceRange, ...]
    nbytes: int
    layout: str

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("placement key must be a non-empty string")
        if type(self.rank) is not int or type(self.source_rank) is not int:
            raise TypeError("placement ranks must be integers")
        if self.rank < 0 or self.source_rank < 0:
            raise ValueError("placement ranks must be non-negative")
        try:
            ranges = tuple(self.ranges)
        except TypeError as error:
            raise TypeError("placement ranges must be an iterable") from error
        if any(not isinstance(value, SliceRange) for value in ranges):
            raise TypeError("placement ranges must contain SliceRange values")
        if type(self.nbytes) is not int or self.nbytes < 0:
            raise ValueError("placement nbytes must be a non-negative integer")
        if self.layout not in {"C", "F", "strided"}:
            raise ValueError("placement layout is unsupported")
        object.__setattr__(self, "ranges", ranges)


def _canonical_refs(refs, name):
    try:
        normalized = tuple(refs)
    except TypeError as error:
        raise TypeError("{} must be an iterable".format(name)) from error
    if any(not isinstance(ref, HostTensorRef) for ref in normalized):
        raise TypeError("{} must contain HostTensorRef values".format(name))
    identities = tuple(_ref_identity(ref) for ref in normalized)
    if len(set(identities)) != len(identities):
        raise ValueError("{} must not contain duplicate identities".format(name))
    return tuple(sorted(normalized, key=_ref_identity))


@dataclass(frozen=True)
class FutureResidencyPlan:
    source_plan_hash: str
    required_refs: tuple[HostTensorRef, ...]
    local_slices: tuple[TensorPlacement, ...]
    plan_hash: str = field(init=False)

    def __post_init__(self):
        _require_sha256(self.source_plan_hash, "source_plan_hash")
        refs = _canonical_refs(self.required_refs, "required_refs")
        ref_keys = tuple(ref.key for ref in refs)
        if len(set(ref_keys)) != len(ref_keys):
            raise ValueError("required_refs must not contain duplicate keys")
        try:
            placements = tuple(self.local_slices)
        except TypeError as error:
            raise TypeError("local_slices must be an iterable") from error
        if any(not isinstance(value, TensorPlacement) for value in placements):
            raise TypeError("local_slices must contain TensorPlacement values")
        placements = tuple(sorted(placements))
        refs_by_key = {ref.key: ref for ref in refs}
        if any(placement.key not in refs_by_key for placement in placements):
            raise ValueError("future placement has no matching host ref")
        for placement in placements:
            ref = refs_by_key[placement.key]
            if len(placement.ranges) != len(ref.shape) or any(
                selected.stop > dimension
                for selected, dimension in zip(placement.ranges, ref.shape)
            ):
                raise ValueError("future placement ranges exceed host ref bounds")
            elements = _checked_shape_elements(
                tuple(selected.stop - selected.start for selected in placement.ranges),
                "future placement",
            )
            expected_nbytes = _checked_memory_product(
                (elements, int(np.dtype(ref.dtype).itemsize)),
                "future placement nbytes",
            )
            if placement.nbytes != expected_nbytes:
                raise ValueError("future placement nbytes do not match its ranges")
        object.__setattr__(self, "required_refs", refs)
        object.__setattr__(self, "local_slices", placements)
        object.__setattr__(self, "plan_hash", _sha256(_future_plan_payload(self)))


def _future_plan_payload(value):
    return {
        "schema": "renormalizer.residency.future.v1",
        "source_plan_hash": value.source_plan_hash,
        "required_refs": [_ref_payload(ref) for ref in value.required_refs],
        "local_slices": [
            _placement_payload(placement) for placement in value.local_slices
        ],
    }


def _placement_payload(value):
    return {
        "key": value.key,
        "rank": value.rank,
        "source_rank": value.source_rank,
        "ranges": [
            [selected.start, selected.stop, selected.step] for selected in value.ranges
        ],
        "nbytes": value.nbytes,
        "layout": value.layout,
    }


def _budget_payload(value):
    return {
        "requested_bytes": value.requested_bytes,
        "resolved_bytes": value.resolved_bytes,
        "source": value.source,
        "available_snapshot_bytes": value.available_snapshot_bytes,
        "resource": value.resource,
    }


def _runtime_identity_payload(value):
    return {
        "mesh_shape": list(value.mesh_shape),
        "mesh_axis_names": list(value.mesh_axis_names),
        "rank_to_node": list(value.rank_to_node),
        "rank_to_local_rank": list(value.rank_to_local_rank),
        "rank_to_device": list(value.rank_to_device),
        "backend_name": value.backend_name,
        "device_budget_resource": value.device_budget_resource,
        "host_budget_resource": value.host_budget_resource,
        "device_budget_bindings": list(value.device_budget_bindings),
        "host_budget_bindings": list(value.host_budget_bindings),
    }


def _solver_rank_payload(value):
    return {
        field.name: getattr(value, field.name) for field in dataclasses.fields(value)
    }


def _solver_payload(value):
    return {
        "solver_kind": value.solver_kind,
        "global_shape": list(value.global_shape),
        "sharding": _sharding_payload(value.sharding),
        "dtype": value.dtype,
        "itemsize": value.itemsize,
        "scalar_dtype": value.scalar_dtype,
        "scalar_itemsize": value.scalar_itemsize,
        "result_dtype": value.result_dtype,
        "result_itemsize": value.result_itemsize,
        "coefficient_identity": krylov_coefficient_payload(value.coefficient_identity),
        "global_vector_count": value.global_vector_count,
        "rank_counts": list(value.rank_counts),
        "allocation_vectors": value.allocation_vectors,
        "projected_host_bytes": value.projected_host_bytes,
        "block_size": value.block_size,
        "max_krylov_vectors": value.max_krylov_vectors,
        "max_space": value.max_space,
        "rank_estimates": [
            _solver_rank_payload(estimate) for estimate in value.rank_estimates
        ],
        "bounded": value.bounded,
        "basis_vectors": value.basis_vectors,
    }


def _derive_transfer_profile(
    distributed_plan,
    host_refs,
    future_plans,
    dirty_writeback_bytes,
    world_size,
):
    refs = dict(host_refs)
    variable_key = distributed_plan.variable_key
    current = [0] * world_size
    for block in distributed_plan.block_plans:
        block_refs = {ref.key: ref for ref in block.execution_plan.inputs}
        for key in block.operand_slices:
            if key == variable_key:
                continue
            if key not in refs or key not in block_refs:
                raise ValueError("transfer profile current placement is incomplete")
            current[block.rank] = max(current[block.rank], block_refs[key].spec.nbytes)

    future = [0] * world_size
    for plan in future_plans:
        for placement in plan.local_slices:
            if placement.rank >= world_size:
                raise ValueError("future transfer rank exceeds world_size")
            future[placement.rank] = max(future[placement.rank], placement.nbytes)
    return TransferProfile.build(current, future, dirty_writeback_bytes)


def _transfer_profile_payload(value):
    return {
        "lanes": value.lanes,
        "rank_staging_bytes": list(value.rank_staging_bytes),
        "current_h2d_bytes": list(value.current_h2d_bytes),
        "future_h2d_bytes": list(value.future_h2d_bytes),
        "dirty_d2h_bytes": list(value.dirty_d2h_bytes),
        "profile_hash": value.profile_hash,
    }


def _sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResidencyRequest:
    distributed_plan: DistributedPlan
    host_refs: object
    world_size: int
    local_world_size: int
    backend_name: str
    store_bytes: tuple[int, ...] | None
    external_host_bytes: tuple[int, ...]
    transfer_staging_host_bytes: tuple[int, ...] | None
    dirty_writeback_bytes: tuple[int, ...] | None
    solver_input_sharding: ShardingSpec
    solver_output_sharding: ShardingSpec
    mapped_local_counts: tuple[int, ...]
    solver_profile: SolverMemoryProfile
    materialization_policy: str
    complete_center_bytes: int | None
    prefetch_depth: int
    future_plans: tuple[FutureResidencyPlan, ...]
    device_budget: MemoryBudgetResolution
    host_budget: MemoryBudgetResolution
    center_profile: object = None
    store_snapshots: object = ()
    writeback_allocations: object = ()
    qn_mask_present: bool = False
    qn_mask_identity: object = None
    runtime_identity: ResidencyRuntimeIdentity | None = None
    transactional_store_bytes: tuple[int, ...] = field(init=False)
    complete_input_center_bytes: int = field(init=False)
    complete_diagonal_center_bytes: int = field(init=False)
    transfer_profile: TransferProfile = field(init=False)
    request_hash: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.distributed_plan, DistributedPlan):
            raise TypeError("distributed_plan must be a DistributedPlan")
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if type(self.local_world_size) is not int or self.local_world_size <= 0:
            raise ValueError("local_world_size must be a positive integer")
        if self.world_size != self.distributed_plan.world_size:
            raise ValueError("world_size does not match distributed plan")
        if self.backend_name not in {"numpy", "cupy"}:
            raise ValueError("backend_name must be 'numpy' or 'cupy'")

        try:
            items = tuple(self.host_refs.items())
        except AttributeError:
            try:
                items = tuple(self.host_refs)
            except TypeError as error:
                raise TypeError(
                    "host_refs must be a mapping or pair iterable"
                ) from error
        if any(not isinstance(item, tuple) or len(item) != 2 for item in items):
            raise TypeError("host_refs must contain key/ref pairs")
        refs = []
        seen_keys = set()
        for key, ref in items:
            if not isinstance(key, str) or not key:
                raise ValueError("host_refs keys must be non-empty strings")
            if key in seen_keys:
                raise ValueError("host_refs must not contain duplicate keys")
            if not isinstance(ref, HostTensorRef) or ref.key != key:
                raise ValueError("host_refs keys must match HostTensorRef keys")
            seen_keys.add(key)
            refs.append((key, ref))
        refs = tuple(sorted(refs))
        static_keys = {
            ref.key
            for ref in self.distributed_plan.execution_plan.inputs
            if ref.key != self.distributed_plan.variable_key
        }
        required_keys = static_keys | {self.distributed_plan.execution_plan.output.key}
        if not required_keys <= seen_keys:
            raise ValueError("host_refs do not cover static operands and dirty output")
        expected_refs = {
            ref.key: ref.spec
            for ref in self.distributed_plan.execution_plan.inputs
            if ref.key != self.distributed_plan.variable_key
        }
        output_ref = self.distributed_plan.execution_plan.output
        expected_refs[output_ref.key] = output_ref.spec
        refs_by_key = dict(refs)
        for key, spec in expected_refs.items():
            ref = refs_by_key[key]
            if (
                ref.shape != spec.shape
                or ref.dtype != spec.dtype
                or ref.nbytes != spec.nbytes
                or ref.layout != "C"
            ):
                raise ValueError("host ref metadata does not match execution plan")
        if len({ref.store_id for _, ref in refs}) != 1:
            raise ValueError("current host refs must share one logical store_id")
        logical_store_id = refs[0][1].store_id

        try:
            store_snapshots = tuple(self.store_snapshots)
        except TypeError as error:
            raise TypeError("store_snapshots must be an iterable") from error
        if len(store_snapshots) != self.world_size or any(
            not isinstance(value, HostTensorStoreSnapshot) for value in store_snapshots
        ):
            raise ValueError(
                "store_snapshots must contain one complete snapshot per rank"
            )
        if any(snapshot.store_id != logical_store_id for snapshot in store_snapshots):
            raise ValueError("store snapshots must match the logical store_id")
        current_ref_set = {ref for _, ref in refs}
        if any(
            not current_ref_set <= set(snapshot.refs) for snapshot in store_snapshots
        ):
            raise ValueError("store snapshot does not cover current host refs")
        derived_store_bytes = tuple(
            snapshot.current_bytes for snapshot in store_snapshots
        )
        if self.store_bytes is not None:
            supplied_store_bytes = _validate_nonnegative_rank_values(
                self.store_bytes, "store_bytes", self.world_size
            )
            if supplied_store_bytes != derived_store_bytes:
                raise ValueError("store_bytes do not match canonical store snapshots")

        try:
            writeback_allocations = tuple(
                tuple(values) for values in self.writeback_allocations
            )
        except TypeError as error:
            raise TypeError(
                "writeback_allocations must contain one iterable per rank"
            ) from error
        if len(writeback_allocations) != self.world_size:
            raise ValueError("writeback_allocations must contain one value per rank")
        writeback_dtype = (
            self.solver_profile.result_dtype
            if isinstance(self.solver_profile, SolverMemoryProfile)
            and self.solver_profile.result_dtype is not None
            else output_ref.spec.dtype
        )
        expected_allocation = HostTensorAllocation(
            output_ref.key,
            output_ref.spec.shape,
            writeback_dtype,
            "C",
        )
        for allocations in writeback_allocations:
            if any(
                not isinstance(value, HostTensorAllocation) for value in allocations
            ):
                raise TypeError(
                    "writeback_allocations must contain HostTensorAllocation values"
                )
            if allocations != (expected_allocation,):
                raise ValueError(
                    "writeback allocation does not match distributed output"
                )
        derived_dirty_bytes = tuple(
            _checked_memory_sum(
                tuple(value.nbytes for value in allocations),
                "dirty writeback bytes",
            )
            for allocations in writeback_allocations
        )
        if self.dirty_writeback_bytes is not None:
            supplied_dirty_bytes = _validate_nonnegative_rank_values(
                self.dirty_writeback_bytes,
                "dirty_writeback_bytes",
                self.world_size,
            )
            if supplied_dirty_bytes != derived_dirty_bytes:
                raise ValueError(
                    "dirty_writeback_bytes do not match canonical allocations"
                )
        transactional_store_bytes = tuple(
            _checked_memory_sum(
                (store_bytes, dirty_bytes),
                "transactional host tensor store bytes",
            )
            for store_bytes, dirty_bytes in zip(
                derived_store_bytes, derived_dirty_bytes
            )
        )

        rank_fields = {}
        for name in ("external_host_bytes", "mapped_local_counts"):
            rank_fields[name] = _validate_nonnegative_rank_values(
                getattr(self, name), name, self.world_size
            )
        if not isinstance(self.solver_input_sharding, ShardingSpec) or not isinstance(
            self.solver_output_sharding, ShardingSpec
        ):
            raise TypeError("solver sharding must use ShardingSpec")
        if self.solver_input_sharding != self.solver_output_sharding:
            raise ValueError("solver input and output sharding must be equal")
        if self.solver_input_sharding.parts != self.world_size:
            raise ValueError("solver sharding parts must match world_size")
        if any(count <= 0 for count in rank_fields["mapped_local_counts"]):
            raise NotImplementedError("empty mapped shard is unsupported")
        if not isinstance(self.solver_profile, SolverMemoryProfile):
            raise TypeError("solver_profile must be a SolverMemoryProfile")
        if len(self.solver_profile.rank_estimates) != self.world_size:
            raise ValueError("solver profile must contain one estimate per rank")
        if self.solver_profile.sharding != self.solver_input_sharding:
            raise ValueError("solver profile sharding does not match request")
        if self.solver_profile.rank_counts != rank_fields["mapped_local_counts"]:
            raise ValueError("solver profile rank counts do not match mapped counts")
        variable_ref = next(
            ref
            for ref in self.distributed_plan.execution_plan.inputs
            if ref.key == self.distributed_plan.variable_key
        )
        if (
            variable_ref.spec.shape != output_ref.spec.shape
            or variable_ref.spec.dtype != output_ref.spec.dtype
            or variable_ref.spec.nbytes != output_ref.spec.nbytes
        ):
            raise ValueError("distributed center input/output metadata must be equal")
        if (
            self.distributed_plan.input_sharding
            != self.distributed_plan.output_sharding
        ):
            raise ValueError("distributed center input/output sharding must be equal")
        if self.solver_profile.dtype != variable_ref.spec.dtype:
            raise ValueError("solver profile dtype does not match distributed center")
        if self.solver_profile.itemsize != np.dtype(variable_ref.spec.dtype).itemsize:
            raise ValueError(
                "solver profile itemsize does not match distributed center"
            )
        if type(self.qn_mask_present) is not bool:
            raise TypeError("qn_mask_present must be a boolean")
        if self.materialization_policy not in {"device", "host"}:
            raise ValueError("materialization_policy must be 'device' or 'host'")
        complete_input_center_bytes = variable_ref.spec.nbytes
        complete_diagonal_center_bytes = (
            complete_input_center_bytes
            if self.solver_profile.solver_kind == "davidson"
            else 0
        )
        derived_center_bytes = _checked_memory_product(
            (
                _checked_shape_elements(variable_ref.spec.shape, "complete center"),
                self.solver_profile.result_itemsize,
            ),
            "complete center bytes",
        )
        if self.complete_center_bytes is not None and (
            type(self.complete_center_bytes) is not int
            or self.complete_center_bytes != derived_center_bytes
        ):
            raise ValueError(
                "complete_center_bytes do not match the distributed center"
            )
        center_profile = self.center_profile
        if self.materialization_policy == "device":
            from renormalizer.backend._distributed.center import (
                CenterMaterializationMemoryProfile,
                QnMaskIdentity,
            )

            control_bytes = np.dtype(np.int32).itemsize
            receive_bytes = _checked_memory_product(
                (
                    max(rank_fields["mapped_local_counts"]),
                    self.solver_profile.result_itemsize,
                ),
                "materialization receive bytes",
            )
            dense_rank_counts = tuple(
                _checked_shape_elements(
                    self.distributed_plan.input_sharding.local_shape(rank),
                    "dense center shard",
                )
                for rank in range(self.world_size)
            )
            if any(
                packed > dense
                for packed, dense in zip(
                    rank_fields["mapped_local_counts"], dense_rank_counts
                )
            ):
                raise ValueError("mapped counts exceed dense center shard sizes")
            mask_identity = self.qn_mask_identity
            if mask_identity is None:
                if self.qn_mask_present:
                    raise ValueError("qn_mask_present requires exact QN mask identity")
                if rank_fields["mapped_local_counts"] != dense_rank_counts:
                    raise ValueError("mapped counts require exact QN mask identity")
            else:
                if not isinstance(mask_identity, QnMaskIdentity):
                    raise TypeError("qn_mask_identity must be canonical metadata")
                if not self.qn_mask_present:
                    raise ValueError(
                        "QN mask identity requires qn_mask_present metadata"
                    )
                if (
                    mask_identity.global_shape != variable_ref.spec.shape
                    or mask_identity.rank_counts != rank_fields["mapped_local_counts"]
                ):
                    raise ValueError("QN mask identity does not match the request")
            complete_elements = _checked_shape_elements(
                variable_ref.spec.shape, "complete center"
            )
            local_mask_bytes = (
                dense_rank_counts if self.qn_mask_present else (0,) * self.world_size
            )
            materialization_mask_bytes = (
                (max(dense_rank_counts),) * self.world_size
                if self.qn_mask_present
                else (0,) * self.world_size
            )
            derived_profile = CenterMaterializationMemoryProfile(
                policy="device",
                global_shape=variable_ref.spec.shape,
                input_sharding=self.distributed_plan.input_sharding,
                output_sharding=self.distributed_plan.output_sharding,
                solver_sharding=self.solver_input_sharding,
                dtype=self.solver_profile.result_dtype,
                itemsize=self.solver_profile.result_itemsize,
                input_dtype=variable_ref.spec.dtype,
                input_itemsize=np.dtype(variable_ref.spec.dtype).itemsize,
                rank_counts=rank_fields["mapped_local_counts"],
                complete_center_bytes=derived_center_bytes,
                receive_buffer_bytes=(receive_bytes,) * self.world_size,
                control_device_bytes=(3 * control_bytes,) * self.world_size,
                control_host_bytes=(control_bytes,) * self.world_size,
                qn_mask_present=self.qn_mask_present,
                setup_qn_mask_bytes=(complete_elements if self.qn_mask_present else 0),
                extract_qn_mask_bytes=local_mask_bytes,
                hv_qn_mask_bytes=local_mask_bytes,
                materialization_qn_mask_bytes=materialization_mask_bytes,
                writeback_qn_mask_bytes=(
                    complete_elements if self.qn_mask_present else 0
                ),
                writeback_packed_result_bytes=(
                    _checked_memory_product(
                        (
                            _checked_memory_sum(
                                rank_fields["mapped_local_counts"],
                                "mapped center count",
                            ),
                            self.solver_profile.result_itemsize,
                        ),
                        "packed writeback bytes",
                    )
                    if self.qn_mask_present
                    else 0
                ),
                state_digest_host_bytes=2 * derived_center_bytes,
                state_agreement_device_bytes=(104,) * self.world_size,
                state_agreement_host_bytes=(96,) * self.world_size,
                qn_mask_identity=mask_identity,
                persistent_qn_mask_host_bytes=(
                    2 * complete_elements if mask_identity is not None else 0
                ),
                setup_qn_digest_host_bytes=(
                    complete_elements if mask_identity is not None else 0
                ),
            )
            if center_profile is not None and center_profile != derived_profile:
                raise ValueError("center profile does not match residency request")
            center_profile = derived_profile
        elif center_profile is not None:
            raise ValueError(
                "host materialization does not accept a device center profile"
            )
        if type(self.prefetch_depth) is not int or self.prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be a positive integer")
        try:
            futures = tuple(self.future_plans)
        except TypeError as error:
            raise TypeError("future_plans must be an iterable") from error
        if len(futures) > self.prefetch_depth:
            raise ValueError("future_plans exceed prefetch_depth")
        if any(not isinstance(value, FutureResidencyPlan) for value in futures):
            raise TypeError("future_plans must contain FutureResidencyPlan values")
        if any(
            ref.store_id != logical_store_id
            for future in futures
            for ref in future.required_refs
        ):
            raise ValueError("future host refs must share the current logical store_id")
        future_ref_set = {ref for future in futures for ref in future.required_refs}
        required_snapshot_refs = current_ref_set | future_ref_set
        if any(
            not required_snapshot_refs <= set(snapshot.refs)
            for snapshot in store_snapshots
        ):
            raise ValueError(
                "store snapshot does not cover current and future host refs"
            )
        transfer_profile = _derive_transfer_profile(
            self.distributed_plan,
            refs,
            futures,
            derived_dirty_bytes,
            self.world_size,
        )
        if self.transfer_staging_host_bytes is not None:
            supplied_staging = _validate_nonnegative_rank_values(
                self.transfer_staging_host_bytes,
                "transfer_staging_host_bytes",
                self.world_size,
            )
            if supplied_staging != transfer_profile.rank_staging_bytes:
                raise ValueError(
                    "transfer staging bytes do not match the canonical profile"
                )
        rank_fields["transfer_staging_host_bytes"] = transfer_profile.rank_staging_bytes
        if not isinstance(self.device_budget, MemoryBudgetResolution) or not isinstance(
            self.host_budget, MemoryBudgetResolution
        ):
            raise TypeError("device_budget and host_budget must be resolved metadata")
        device_budget = _bound_budget_resolution(self.device_budget, "device")
        host_budget = _bound_budget_resolution(self.host_budget, "host")
        runtime_identity = self.runtime_identity
        if runtime_identity is None:
            runtime_identity = ResidencyRuntimeIdentity.one_node(
                world_size=self.world_size,
                backend_name=self.backend_name,
            )
        if not isinstance(runtime_identity, ResidencyRuntimeIdentity):
            raise TypeError("runtime_identity must be ResidencyRuntimeIdentity or None")
        runtime_identity.validate_request(
            world_size=self.world_size,
            local_world_size=self.local_world_size,
            backend_name=self.backend_name,
            device_budget=device_budget,
            host_budget=host_budget,
        )

        object.__setattr__(self, "host_refs", refs)
        object.__setattr__(self, "store_snapshots", store_snapshots)
        object.__setattr__(self, "writeback_allocations", writeback_allocations)
        object.__setattr__(self, "store_bytes", derived_store_bytes)
        object.__setattr__(self, "dirty_writeback_bytes", derived_dirty_bytes)
        object.__setattr__(self, "transactional_store_bytes", transactional_store_bytes)
        object.__setattr__(self, "complete_center_bytes", derived_center_bytes)
        object.__setattr__(
            self, "complete_input_center_bytes", complete_input_center_bytes
        )
        object.__setattr__(
            self,
            "complete_diagonal_center_bytes",
            complete_diagonal_center_bytes,
        )
        object.__setattr__(self, "center_profile", center_profile)
        object.__setattr__(
            self,
            "qn_mask_identity",
            None if center_profile is None else center_profile.qn_mask_identity,
        )
        for name, values in rank_fields.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "future_plans", futures)
        object.__setattr__(self, "device_budget", device_budget)
        object.__setattr__(self, "host_budget", host_budget)
        object.__setattr__(self, "runtime_identity", runtime_identity)
        object.__setattr__(self, "transfer_profile", transfer_profile)
        object.__setattr__(self, "request_hash", _sha256(_request_payload(self)))


def _request_payload(request):
    return {
        "schema": "renormalizer.residency.request.v1",
        "source_plan_hash": request.distributed_plan.execution_plan.plan_hash,
        "placement_hash": request.distributed_plan.placement_hash,
        "host_refs": [_ref_payload(ref) for _, ref in request.host_refs],
        "world_size": request.world_size,
        "local_world_size": request.local_world_size,
        "backend_name": request.backend_name,
        "store_bytes": list(request.store_bytes),
        "store_snapshots": [
            _store_snapshot_payload(value) for value in request.store_snapshots
        ],
        "external_host_bytes": list(request.external_host_bytes),
        "transfer_staging_host_bytes": list(request.transfer_staging_host_bytes),
        "transfer_profile": _transfer_profile_payload(request.transfer_profile),
        "dirty_writeback_bytes": list(request.dirty_writeback_bytes),
        "writeback_allocations": [
            [_allocation_payload(value) for value in allocations]
            for allocations in request.writeback_allocations
        ],
        "transactional_store_bytes": list(request.transactional_store_bytes),
        "solver_input_sharding": _sharding_payload(request.solver_input_sharding),
        "solver_output_sharding": _sharding_payload(request.solver_output_sharding),
        "mapped_local_counts": list(request.mapped_local_counts),
        "solver_profile": _solver_payload(request.solver_profile),
        "center_profile": _center_profile_payload(request.center_profile),
        "qn_mask_present": request.qn_mask_present,
        "qn_mask_identity": (
            None
            if request.center_profile is None
            else _qn_mask_payload(request.center_profile.qn_mask_identity)
        ),
        "materialization_policy": request.materialization_policy,
        "complete_center_bytes": request.complete_center_bytes,
        "complete_input_center_bytes": request.complete_input_center_bytes,
        "complete_diagonal_center_bytes": request.complete_diagonal_center_bytes,
        "original_packed_guess_bytes": (
            _checked_memory_product(
                (
                    request.solver_profile.global_vector_count,
                    request.solver_profile.itemsize,
                ),
                "original packed guess bytes",
            )
            if request.solver_profile.solver_kind == "davidson"
            else 0
        ),
        "original_packed_diagonal_bytes": (
            _checked_memory_product(
                (
                    request.solver_profile.global_vector_count,
                    request.solver_profile.itemsize,
                ),
                "original packed diagonal bytes",
            )
            if request.solver_profile.solver_kind == "davidson"
            else 0
        ),
        "prefetch_depth": request.prefetch_depth,
        "future_plan_hashes": [value.plan_hash for value in request.future_plans],
        "device_budget": _budget_payload(request.device_budget),
        "host_budget": _budget_payload(request.host_budget),
        "runtime_identity": _runtime_identity_payload(request.runtime_identity),
    }


def _sharding_payload(value):
    return {
        "global_shape": list(value.global_shape),
        "axis": value.axis,
        "local_slices": [
            [[selected.start, selected.stop, selected.step] for selected in local_slice]
            for local_slice in value.local_slices
        ],
    }


def _center_profile_payload(value):
    if value is None:
        return None
    return {
        "policy": value.policy,
        "global_shape": list(value.global_shape),
        "input_sharding": _sharding_payload(value.input_sharding),
        "output_sharding": _sharding_payload(value.output_sharding),
        "solver_sharding": _sharding_payload(value.solver_sharding),
        "dtype": value.dtype,
        "itemsize": value.itemsize,
        "input_dtype": value.input_dtype,
        "input_itemsize": value.input_itemsize,
        "rank_counts": list(value.rank_counts),
        "complete_center_bytes": value.complete_center_bytes,
        "receive_buffer_bytes": list(value.receive_buffer_bytes),
        "control_device_bytes": list(value.control_device_bytes),
        "control_host_bytes": list(value.control_host_bytes),
        "qn_mask_present": value.qn_mask_present,
        "qn_mask_identity": _qn_mask_payload(value.qn_mask_identity),
        "setup_qn_mask_bytes": value.setup_qn_mask_bytes,
        "extract_qn_mask_bytes": list(value.extract_qn_mask_bytes),
        "hv_qn_mask_bytes": list(value.hv_qn_mask_bytes),
        "materialization_qn_mask_bytes": list(value.materialization_qn_mask_bytes),
        "writeback_qn_mask_bytes": value.writeback_qn_mask_bytes,
        "writeback_packed_result_bytes": value.writeback_packed_result_bytes,
        "state_digest_host_bytes": value.state_digest_host_bytes,
        "state_agreement_device_bytes": list(value.state_agreement_device_bytes),
        "state_agreement_host_bytes": list(value.state_agreement_host_bytes),
        "persistent_qn_mask_host_bytes": value.persistent_qn_mask_host_bytes,
        "setup_qn_digest_host_bytes": value.setup_qn_digest_host_bytes,
    }


def _qn_mask_payload(value):
    from renormalizer.backend._distributed.center import qn_mask_identity_payload

    return qn_mask_identity_payload(value)


_RANK_COMPONENT_NAMES = (
    "current_static_bytes",
    "prefetched_static_bytes",
    "dense_input_bytes",
    "receive_buffer_bytes",
    "output_accumulator_bytes",
    "output_contribution_bytes",
    "execution_workspace_bytes",
    "task14_hash_control_device_bytes",
    "residency_hash_control_device_bytes",
    "residency_hash_control_host_bytes",
    "task14_preflight_status_device_bytes",
    "task14_execution_status_device_bytes",
    "mapped_packed_output_bytes",
    "solver_input_bytes",
    "solver_result_bytes",
    "solver_diagonal_bytes",
    "complete_input_center_bytes",
    "complete_diagonal_center_bytes",
    "original_packed_guess_bytes",
    "original_packed_diagonal_bytes",
    "solver_hv_retained_bytes",
    "solver_hv_transient_bytes",
    "solver_la_retained_bytes",
    "solver_la_transient_bytes",
    "solver_communication_device_bytes",
    "complete_center_bytes",
    "materialization_receive_bytes",
    "materialization_control_device_bytes",
    "materialization_control_host_bytes",
    "qn_setup_mask_device_bytes",
    "qn_extract_mask_device_bytes",
    "qn_hv_mask_device_bytes",
    "qn_materialization_mask_device_bytes",
    "qn_writeback_mask_device_bytes",
    "qn_writeback_packed_result_bytes",
    "qn_persistent_mask_host_bytes",
    "qn_setup_digest_host_bytes",
    "state_digest_host_bytes",
    "state_agreement_device_bytes",
    "state_agreement_host_bytes",
    "store_bytes",
    "external_host_bytes",
    "transfer_staging_host_bytes",
    "dirty_writeback_bytes",
    "task14_host_control_bytes",
    "task14_persistent_execution_status_host_bytes",
    "solver_host_peak_bytes",
)


@dataclass(frozen=True)
class RankResidencyEstimate:
    current_static_bytes: int
    prefetched_static_bytes: int
    dense_input_bytes: int
    receive_buffer_bytes: int
    output_accumulator_bytes: int
    output_contribution_bytes: int
    execution_workspace_bytes: int
    task14_hash_control_device_bytes: int
    residency_hash_control_device_bytes: int
    residency_hash_control_host_bytes: int
    task14_preflight_status_device_bytes: int
    task14_execution_status_device_bytes: int
    mapped_packed_output_bytes: int
    solver_input_bytes: int
    solver_result_bytes: int
    solver_diagonal_bytes: int
    complete_input_center_bytes: int
    complete_diagonal_center_bytes: int
    original_packed_guess_bytes: int
    original_packed_diagonal_bytes: int
    solver_hv_retained_bytes: int
    solver_hv_transient_bytes: int
    solver_la_retained_bytes: int
    solver_la_transient_bytes: int
    solver_communication_device_bytes: int
    complete_center_bytes: int
    materialization_receive_bytes: int
    materialization_control_device_bytes: int
    materialization_control_host_bytes: int
    qn_setup_mask_device_bytes: int
    qn_extract_mask_device_bytes: int
    qn_hv_mask_device_bytes: int
    qn_materialization_mask_device_bytes: int
    qn_writeback_mask_device_bytes: int
    qn_writeback_packed_result_bytes: int
    qn_persistent_mask_host_bytes: int
    qn_setup_digest_host_bytes: int
    state_digest_host_bytes: int
    state_agreement_device_bytes: int
    state_agreement_host_bytes: int
    store_bytes: int
    external_host_bytes: int
    transfer_staging_host_bytes: int
    dirty_writeback_bytes: int
    task14_host_control_bytes: int
    task14_persistent_execution_status_host_bytes: int
    solver_host_peak_bytes: int
    setup_device_peak_bytes: int
    preflight_device_peak_bytes: int
    hv_device_peak_bytes: int
    solver_la_device_peak_bytes: int
    materialization_device_peak_bytes: int
    state_digest_device_peak_bytes: int
    state_agreement_device_peak_bytes: int
    writeback_device_peak_bytes: int
    cleanup_device_peak_bytes: int
    setup_host_peak_bytes: int
    hv_host_peak_bytes: int
    solver_la_host_peak_bytes: int
    materialization_host_peak_bytes: int
    state_digest_host_peak_bytes: int
    state_agreement_host_peak_bytes: int
    writeback_host_peak_bytes: int
    cleanup_host_peak_bytes: int
    device_peak_bytes: int
    host_peak_bytes: int

    def __post_init__(self):
        for field_info in dataclasses.fields(self):
            value = getattr(self, field_info.name)
            _checked_memory_bytes(value, field_info.name)
        expected = _rank_peaks(
            {name: getattr(self, name) for name in _RANK_COMPONENT_NAMES}
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    "{} does not match deterministic components".format(name)
                )


def _rank_peaks(c):
    static = c["current_static_bytes"] + c["prefetched_static_bytes"]
    complete_inputs = (
        c["complete_input_center_bytes"]
        + c["complete_diagonal_center_bytes"]
        + c["original_packed_guess_bytes"]
        + c["original_packed_diagonal_bytes"]
    )
    local_inputs = c["solver_input_bytes"] + c["solver_diagonal_bytes"]
    setup = (
        static
        + complete_inputs
        + local_inputs
        + c["dense_input_bytes"]
        + c["mapped_packed_output_bytes"]
        + max(
            c["task14_hash_control_device_bytes"],
            c["residency_hash_control_device_bytes"],
            c["task14_preflight_status_device_bytes"],
            c["solver_communication_device_bytes"],
            c["qn_setup_mask_device_bytes"],
            c["qn_extract_mask_device_bytes"],
        )
    )
    execution_base = (
        static
        + c["dense_input_bytes"]
        + c["receive_buffer_bytes"]
        + c["output_accumulator_bytes"]
        + c["task14_execution_status_device_bytes"]
        + c["mapped_packed_output_bytes"]
    )
    hv = (
        execution_base
        + complete_inputs
        + local_inputs
        + c["solver_hv_retained_bytes"]
        + max(
            c["output_contribution_bytes"]
            + c["execution_workspace_bytes"]
            + c["solver_hv_transient_bytes"],
            c["task14_preflight_status_device_bytes"],
            c["solver_communication_device_bytes"],
            c["qn_hv_mask_device_bytes"],
        )
    )
    solver_la = (
        execution_base
        + complete_inputs
        + local_inputs
        + c["solver_la_retained_bytes"]
        + c["solver_la_transient_bytes"]
        + c["solver_communication_device_bytes"]
    )
    post_solver = (
        execution_base
        + complete_inputs
        + local_inputs
        + c["solver_result_bytes"]
        + c["complete_center_bytes"]
    )
    materialize = (
        post_solver
        + c["materialization_receive_bytes"]
        + c["materialization_control_device_bytes"]
        + c["qn_materialization_mask_device_bytes"]
    )
    state_digest = post_solver
    state_agreement = post_solver + c["state_agreement_device_bytes"]
    writeback = (
        post_solver
        + c["qn_writeback_mask_device_bytes"]
        + c["qn_writeback_packed_result_bytes"]
    )
    cleanup = static + c["complete_center_bytes"]
    host_base = (
        c["store_bytes"]
        + c["external_host_bytes"]
        + c["transfer_staging_host_bytes"]
        + c["qn_persistent_mask_host_bytes"]
    )
    host_setup = (
        host_base
        + c["qn_setup_digest_host_bytes"]
        + max(
            c["task14_host_control_bytes"],
            c["residency_hash_control_host_bytes"],
        )
    )
    host_persistent = host_base + c["task14_persistent_execution_status_host_bytes"]
    host_hv = host_persistent + c["solver_host_peak_bytes"]
    host_solver_la = host_persistent + c["solver_host_peak_bytes"]
    host_materialize = host_persistent + c["materialization_control_host_bytes"]
    host_state_digest = host_persistent + c["state_digest_host_bytes"]
    host_state_agreement = host_persistent + c["state_agreement_host_bytes"]
    host_writeback = (
        host_persistent
        + c["dirty_writeback_bytes"]
        + max(
            c["solver_host_peak_bytes"],
            c["materialization_control_host_bytes"],
        )
    )
    host_cleanup = host_persistent
    peaks = {
        "setup_device_peak_bytes": setup,
        "preflight_device_peak_bytes": setup,
        "hv_device_peak_bytes": hv,
        "solver_la_device_peak_bytes": solver_la,
        "materialization_device_peak_bytes": materialize,
        "state_digest_device_peak_bytes": state_digest,
        "state_agreement_device_peak_bytes": state_agreement,
        "writeback_device_peak_bytes": writeback,
        "cleanup_device_peak_bytes": cleanup,
        "setup_host_peak_bytes": host_setup,
        "hv_host_peak_bytes": host_hv,
        "solver_la_host_peak_bytes": host_solver_la,
        "materialization_host_peak_bytes": host_materialize,
        "state_digest_host_peak_bytes": host_state_digest,
        "state_agreement_host_peak_bytes": host_state_agreement,
        "writeback_host_peak_bytes": host_writeback,
        "cleanup_host_peak_bytes": host_cleanup,
        "device_peak_bytes": max(
            setup,
            hv,
            solver_la,
            materialize,
            state_digest,
            state_agreement,
            writeback,
            cleanup,
        ),
        "host_peak_bytes": max(
            host_setup,
            host_hv,
            host_solver_la,
            host_materialize,
            host_state_digest,
            host_state_agreement,
            host_writeback,
            host_cleanup,
        ),
    }
    for name, value in peaks.items():
        _checked_memory_bytes(value, name)
    return peaks


def _validate_plan_qn_identity(identity, estimates, refs):
    qn_fields = (
        "qn_setup_mask_device_bytes",
        "qn_extract_mask_device_bytes",
        "qn_hv_mask_device_bytes",
        "qn_materialization_mask_device_bytes",
        "qn_writeback_mask_device_bytes",
        "qn_writeback_packed_result_bytes",
        "qn_persistent_mask_host_bytes",
        "qn_setup_digest_host_bytes",
    )
    if identity is None:
        if any(getattr(estimate, name) for estimate in estimates for name in qn_fields):
            raise ValueError("QN plan components require exact mask identity")
        return
    from renormalizer.backend._distributed.center import QnMaskIdentity

    if not isinstance(identity, QnMaskIdentity):
        raise TypeError("plan QN mask identity must be canonical metadata")
    if len(identity.rank_counts) != len(estimates):
        raise ValueError("plan QN mask identity rank counts are inconsistent")
    complete_elements = _checked_shape_elements(identity.global_shape, "plan QN mask")
    complete_input_bytes = {
        estimate.complete_input_center_bytes for estimate in estimates
    }
    complete_result_bytes = {estimate.complete_center_bytes for estimate in estimates}
    if len(complete_input_bytes) != 1 or len(complete_result_bytes) != 1:
        raise ValueError("plan QN mask identity complete centers are inconsistent")
    complete_input_bytes = complete_input_bytes.pop()
    complete_result_bytes = complete_result_bytes.pop()
    if (
        complete_input_bytes % complete_elements
        or complete_result_bytes % complete_elements
    ):
        raise ValueError("plan QN mask identity shape is inconsistent")
    input_itemsize = complete_input_bytes // complete_elements
    result_itemsize = complete_result_bytes // complete_elements
    if input_itemsize not in {1, 2, 4, 8, 16} or result_itemsize not in {
        1,
        2,
        4,
        8,
        16,
    }:
        raise ValueError("plan QN mask identity itemsize is inconsistent")
    if not any(
        ref.shape == identity.global_shape and ref.nbytes == complete_input_bytes
        for ref in refs
    ):
        raise ValueError("plan QN mask identity shape has no matching center ref")
    dense_rank_counts = tuple(
        estimate.qn_extract_mask_device_bytes for estimate in estimates
    )
    if sum(dense_rank_counts) != complete_elements:
        raise ValueError("plan QN mask identity shape is inconsistent")
    for count, dense_count, estimate in zip(
        identity.rank_counts, dense_rank_counts, estimates
    ):
        if estimate.solver_input_bytes != _checked_memory_product(
            (count, input_itemsize), "QN solver input bytes"
        ) or (
            estimate.solver_result_bytes
            != _checked_memory_product(
                (count, result_itemsize), "QN solver result bytes"
            )
        ):
            raise ValueError("plan QN mask identity counts are inconsistent")
        expected = {
            "qn_setup_mask_device_bytes": complete_elements,
            "qn_extract_mask_device_bytes": dense_count,
            "qn_hv_mask_device_bytes": dense_count,
            "qn_materialization_mask_device_bytes": max(dense_rank_counts),
            "qn_writeback_mask_device_bytes": complete_elements,
            "qn_persistent_mask_host_bytes": 2 * complete_elements,
            "qn_setup_digest_host_bytes": complete_elements,
        }
        if any(getattr(estimate, name) != value for name, value in expected.items()):
            raise ValueError("plan QN mask components are inconsistent")
    expected_packed_result = _checked_memory_product(
        (
            _checked_memory_sum(identity.rank_counts, "QN packed result count"),
            result_itemsize,
        ),
        "QN packed result bytes",
    )
    if any(
        estimate.qn_writeback_packed_result_bytes != expected_packed_result
        for estimate in estimates
    ):
        raise ValueError("plan QN packed result is inconsistent")


@dataclass(frozen=True)
class ResidencyPlan:
    world_size: int
    required_refs: tuple[HostTensorRef, ...]
    local_slices: tuple[TensorPlacement, ...]
    rank_estimates: tuple[RankResidencyEstimate, ...]
    device_peak_bytes: tuple[int, ...]
    host_peak_bytes: tuple[int, ...]
    host_required_bytes: int
    prefetch_keys: tuple[str, ...]
    prefetch_depth: int
    source_plan_hash: str
    placement_hash: str
    request_hash: str
    backend_name: str
    solver_kind: str
    device_budget: MemoryBudgetResolution
    host_budget: MemoryBudgetResolution
    transfer_profile: TransferProfile
    plan_hash: str
    current_refs: tuple[HostTensorRef, ...] = ()
    future_plans: tuple[FutureResidencyPlan, ...] = ()
    future_plan_hashes: tuple[str, ...] = ()
    krylov_coefficient_identity: KrylovCoefficientIdentity | None = None
    qn_mask_identity: object = None
    runtime_identity: ResidencyRuntimeIdentity | None = None

    def __post_init__(self):
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        refs = tuple(self.required_refs)
        if refs != _canonical_refs(refs, "required_refs"):
            raise ValueError("required_refs must be canonical")
        placements = tuple(self.local_slices)
        if placements != tuple(sorted(placements)):
            raise ValueError("local_slices must be canonical")
        if any(not isinstance(value, TensorPlacement) for value in placements):
            raise TypeError("local_slices must contain TensorPlacement values")
        estimates = tuple(self.rank_estimates)
        if len(estimates) != self.world_size or any(
            not isinstance(value, RankResidencyEstimate) for value in estimates
        ):
            raise ValueError("rank_estimates must contain one estimate per rank")
        expected_device = tuple(value.device_peak_bytes for value in estimates)
        expected_host = tuple(value.host_peak_bytes for value in estimates)
        if tuple(self.device_peak_bytes) != expected_device:
            raise ValueError("device_peak_bytes do not match rank estimates")
        if tuple(self.host_peak_bytes) != expected_host:
            raise ValueError("host_peak_bytes do not match rank estimates")
        host_required = _checked_memory_sum(expected_host, "node host requirement")
        if self.backend_name == "numpy":
            host_required = _checked_memory_sum(
                (host_required, *expected_device),
                "NumPy node host requirement",
            )
        elif self.backend_name != "cupy":
            raise ValueError("backend_name must be 'numpy' or 'cupy'")
        if self.host_required_bytes != host_required:
            raise ValueError(
                "host_required_bytes does not match physical host requirements"
            )
        keys = tuple(self.prefetch_keys)
        if len(set(keys)) != len(keys) or any(
            not isinstance(key, str) or not key for key in keys
        ):
            raise ValueError("prefetch_keys must be unique non-empty strings")
        if not set(keys) <= {ref.key for ref in refs}:
            raise ValueError("prefetch_keys must identify required refs")
        if type(self.prefetch_depth) is not int or self.prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be a positive integer")
        _require_sha256(self.source_plan_hash, "source_plan_hash")
        _require_sha256(self.placement_hash, "placement_hash")
        _require_sha256(self.request_hash, "request_hash")
        if self.solver_kind not in {"krylov", "davidson"}:
            raise ValueError("solver_kind is unsupported")
        if self.solver_kind == "krylov":
            if not isinstance(
                self.krylov_coefficient_identity, KrylovCoefficientIdentity
            ):
                raise ValueError("Krylov residency plan requires coefficient identity")
        elif self.krylov_coefficient_identity is not None:
            raise ValueError(
                "Davidson residency plan must not define a Krylov coefficient"
            )
        _validate_plan_qn_identity(self.qn_mask_identity, estimates, refs)
        if not isinstance(self.device_budget, MemoryBudgetResolution) or not isinstance(
            self.host_budget, MemoryBudgetResolution
        ):
            raise TypeError("plan budgets must use MemoryBudgetResolution")
        device_budget = _bound_budget_resolution(self.device_budget, "device")
        host_budget = _bound_budget_resolution(self.host_budget, "host")
        if not isinstance(self.transfer_profile, TransferProfile):
            raise TypeError("plan transfer_profile must be a TransferProfile")
        if len(self.transfer_profile.rank_staging_bytes) != self.world_size:
            raise ValueError("plan transfer profile world size is inconsistent")
        if self.transfer_profile.rank_staging_bytes != tuple(
            estimate.transfer_staging_host_bytes for estimate in estimates
        ):
            raise ValueError("plan transfer profile does not match rank estimates")
        runtime_identity = self.runtime_identity
        if runtime_identity is None:
            runtime_identity = ResidencyRuntimeIdentity.one_node(
                world_size=self.world_size,
                backend_name=self.backend_name,
            )
        if not isinstance(runtime_identity, ResidencyRuntimeIdentity):
            raise TypeError("runtime_identity must be ResidencyRuntimeIdentity or None")
        runtime_identity.validate_request(
            world_size=self.world_size,
            local_world_size=self.world_size,
            backend_name=self.backend_name,
            device_budget=device_budget,
            host_budget=host_budget,
        )
        future_hashes = tuple(self.future_plan_hashes)
        for value in future_hashes:
            _require_sha256(value, "future_plan_hash")
        supplied_current_refs = tuple(self.current_refs)
        current_refs = _canonical_refs(supplied_current_refs, "current_refs")
        if supplied_current_refs != current_refs:
            raise ValueError("current_refs must be canonical")
        if not set(current_refs) <= set(refs):
            raise ValueError("current_refs must be a subset of required_refs")
        future_plans = tuple(self.future_plans)
        if any(not isinstance(value, FutureResidencyPlan) for value in future_plans):
            raise TypeError("future_plans must contain FutureResidencyPlan values")
        if len(future_plans) > self.prefetch_depth:
            raise ValueError("future_plans exceed prefetch_depth")
        expected_future_hashes = tuple(value.plan_hash for value in future_plans)
        if future_hashes != expected_future_hashes:
            raise ValueError("future_plan_hashes do not match retained future plans")
        object.__setattr__(self, "device_budget", device_budget)
        object.__setattr__(self, "host_budget", host_budget)
        object.__setattr__(self, "runtime_identity", runtime_identity)
        expected_hash = _sha256(_plan_payload(self))
        if self.plan_hash != expected_hash:
            raise ValueError("plan_hash does not match canonical residency metadata")
        self.validate_capacity()
        object.__setattr__(self, "required_refs", refs)
        object.__setattr__(self, "local_slices", placements)
        object.__setattr__(self, "rank_estimates", estimates)
        object.__setattr__(self, "device_peak_bytes", expected_device)
        object.__setattr__(self, "host_peak_bytes", expected_host)
        object.__setattr__(self, "prefetch_keys", keys)
        object.__setattr__(self, "current_refs", current_refs)
        object.__setattr__(self, "future_plans", future_plans)
        object.__setattr__(self, "future_plan_hashes", future_hashes)

    def validate_capacity(self):
        if self.backend_name == "cupy":
            for rank, required in enumerate(self.device_peak_bytes):
                if required > self.device_budget.resolved_bytes:
                    raise ResidencyBudgetError(
                        "device",
                        required,
                        self.device_budget.resolved_bytes,
                        rank,
                        self.plan_hash,
                    )
        if self.host_required_bytes > self.host_budget.resolved_bytes:
            raise ResidencyBudgetError(
                "host",
                self.host_required_bytes,
                self.host_budget.resolved_bytes,
                None,
                self.plan_hash,
            )

    def validate_store(self, store):
        if not isinstance(store, HostTensorStore):
            raise TypeError("store must be a HostTensorStore")
        for ref in self.required_refs:
            store.validate(ref)

    def validate_request(self, request):
        if not isinstance(request, ResidencyRequest):
            raise TypeError("request must be a ResidencyRequest")
        regenerated = ResidencyPlanner().plan(request)
        if self != regenerated:
            raise ValueError(
                "residency request hash or deterministic residency plan does not match"
            )

    def metadata(self):
        version_payload = [
            {
                "store_id": ref.store_id,
                "key": ref.key,
                "generation": ref.generation,
                "version": ref.version,
            }
            for ref in self.required_refs
        ]
        return {
            "plan_hash": self.plan_hash,
            "source_plan_hash": self.source_plan_hash,
            "placement_hash": self.placement_hash,
            "request_hash": self.request_hash,
            "backend_name": self.backend_name,
            "world_size": self.world_size,
            "solver_kind": self.solver_kind,
            "krylov_coefficient_identity": krylov_coefficient_payload(
                self.krylov_coefficient_identity
            ),
            "qn_mask_identity": _qn_mask_payload(self.qn_mask_identity),
            "device_budget": _budget_payload(self.device_budget),
            "host_budget": _budget_payload(self.host_budget),
            "transfer_profile": _transfer_profile_payload(self.transfer_profile),
            "runtime_identity": _runtime_identity_payload(self.runtime_identity),
            "device_peak_bytes": list(self.device_peak_bytes),
            "host_peak_bytes": list(self.host_peak_bytes),
            "host_required_bytes": self.host_required_bytes,
            "rank_estimates": [
                {
                    field_info.name: getattr(value, field_info.name)
                    for field_info in dataclasses.fields(value)
                }
                for value in self.rank_estimates
            ],
            "prefetch_depth": self.prefetch_depth,
            "prefetch_key_count": len(self.prefetch_keys),
            "future_plan_count": len(self.future_plan_hashes),
            "store_ref_count": len(self.required_refs),
            "store_version_digest": _sha256(version_payload),
            "full_replica_prediction": _predicts_full_replica(self),
        }


def _predicts_full_replica(plan):
    per_identity_rank_layout = {}
    seen = set()

    def retain(refs, placements):
        refs_by_key = {ref.key: ref for ref in refs}
        for placement in placements:
            ref = refs_by_key[placement.key]
            cache_identity = (
                _ref_identity(ref),
                placement.rank,
                placement.ranges,
                placement.layout,
            )
            if cache_identity in seen:
                continue
            seen.add(cache_identity)
            identity_rank_layout = (
                _ref_identity(ref),
                placement.rank,
                placement.layout,
            )
            per_identity_rank_layout.setdefault(identity_rank_layout, []).append(
                placement.ranges
            )

    retain(plan.current_refs, plan.local_slices)
    for future in plan.future_plans:
        retain(future.required_refs, future.local_slices)

    def covered_elements(ranges, shape, axis=0):
        if not ranges:
            return 0
        if axis == len(shape):
            return 1
        boundaries = {0, shape[axis]}
        for selected in ranges:
            boundaries.add(selected[axis].start)
            boundaries.add(selected[axis].stop)
        boundaries = sorted(boundaries)
        covered = 0
        for start, stop in zip(boundaries, boundaries[1:]):
            active = [
                selected
                for selected in ranges
                if selected[axis].start <= start and selected[axis].stop >= stop
            ]
            covered += (stop - start) * covered_elements(active, shape, axis + 1)
        return covered

    def rank_has_full_ref(ref, rank):
        expected = _checked_shape_elements(ref.shape, "replica prediction")
        return any(
            identity == _ref_identity(ref)
            and candidate_rank == rank
            and covered_elements(ranges, ref.shape) == expected
            for (
                identity,
                candidate_rank,
                _,
            ), ranges in per_identity_rank_layout.items()
        )

    return any(
        ref.nbytes > 0
        and all(rank_has_full_ref(ref, rank) for rank in range(plan.world_size))
        for ref in plan.required_refs
    )


def _plan_payload(plan):
    return {
        "schema": "renormalizer.residency.v1",
        "world_size": plan.world_size,
        "required_refs": [_ref_payload(ref) for ref in plan.required_refs],
        "local_slices": [_placement_payload(value) for value in plan.local_slices],
        "rank_estimates": [
            {
                field_info.name: getattr(value, field_info.name)
                for field_info in dataclasses.fields(value)
            }
            for value in plan.rank_estimates
        ],
        "device_peak_bytes": list(plan.device_peak_bytes),
        "host_peak_bytes": list(plan.host_peak_bytes),
        "host_required_bytes": plan.host_required_bytes,
        "prefetch_keys": list(plan.prefetch_keys),
        "prefetch_depth": plan.prefetch_depth,
        "source_plan_hash": plan.source_plan_hash,
        "placement_hash": plan.placement_hash,
        "request_hash": plan.request_hash,
        "backend_name": plan.backend_name,
        "solver_kind": plan.solver_kind,
        "krylov_coefficient_identity": krylov_coefficient_payload(
            plan.krylov_coefficient_identity
        ),
        "qn_mask_identity": _qn_mask_payload(plan.qn_mask_identity),
        "device_budget": _budget_payload(plan.device_budget),
        "host_budget": _budget_payload(plan.host_budget),
        "transfer_profile": _transfer_profile_payload(plan.transfer_profile),
        "runtime_identity": _runtime_identity_payload(plan.runtime_identity),
        "current_refs": [_ref_payload(ref) for ref in plan.current_refs],
        "future_plans": [_future_plan_payload(value) for value in plan.future_plans],
        "future_plan_hashes": list(plan.future_plan_hashes),
    }


class ResidencyBudgetError(MemoryError):
    def __init__(
        self, resource, required_bytes, resolved_budget_bytes, rank, plan_hash
    ):
        if resource not in {"device", "host"}:
            raise ValueError("resource must be 'device' or 'host'")
        self.resource = resource
        self.required_bytes = required_bytes
        self.resolved_budget_bytes = resolved_budget_bytes
        self.rank = rank
        self.plan_hash = plan_hash
        label = "{} memory budget".format(resource)
        location = "" if rank is None else " on rank {}".format(rank)
        super().__init__(
            "{} exceeded{}: required {} bytes, budget {} bytes, plan {}".format(
                label, location, required_bytes, resolved_budget_bytes, plan_hash
            )
        )


def _normalized_ranges(local_slice, shape):
    if len(local_slice) != len(shape):
        raise ValueError("placement slice rank does not match host ref")
    ranges = []
    for selected, dimension in zip(local_slice, shape):
        if selected == slice(None):
            start, stop, step = 0, dimension, 1
        else:
            start, stop, step = selected.start, selected.stop, selected.step
            if step is None:
                step = 1
        if dimension == 0:
            return None
        ranges.append(SliceRange(start, stop, step))
    return tuple(ranges)


def _current_placements(request):
    refs = dict(request.host_refs)
    variable_key = request.distributed_plan.variable_key
    placements = []
    for block in request.distributed_plan.block_plans:
        block_refs = {ref.key: ref for ref in block.execution_plan.inputs}
        for key, local_slice in block.operand_slices.items():
            if key == variable_key:
                continue
            host_ref = refs[key]
            ranges = _normalized_ranges(local_slice, host_ref.shape)
            if ranges is None:
                continue
            placements.append(
                TensorPlacement(
                    key=key,
                    rank=block.rank,
                    source_rank=block.source_rank,
                    ranges=ranges,
                    nbytes=block_refs[key].spec.nbytes,
                    layout=block_refs[key].spec.layout,
                )
            )
    return tuple(sorted(placements))


def _cache_identity(ref, placement):
    return (
        _ref_identity(ref),
        placement.ranges,
        ref.dtype,
        placement.layout,
        placement.rank,
    )


def _static_bytes_by_rank(request, current):
    current_refs = dict(request.host_refs)
    current_identities = [set() for _ in range(request.world_size)]
    current_bytes = [0] * request.world_size
    for placement in current:
        identity = _cache_identity(current_refs[placement.key], placement)
        if identity not in current_identities[placement.rank]:
            current_identities[placement.rank].add(identity)
            current_bytes[placement.rank] = _checked_memory_sum(
                (current_bytes[placement.rank], placement.nbytes),
                "current static residency bytes",
            )

    future_identities = [set() for _ in range(request.world_size)]
    future_bytes = [0] * request.world_size
    prefetch_keys = []
    for future in request.future_plans[: request.prefetch_depth]:
        future_refs = {ref.key: ref for ref in future.required_refs}
        for placement in future.local_slices:
            if (
                placement.rank >= request.world_size
                or placement.source_rank >= request.world_size
            ):
                raise ValueError("future placement rank exceeds world_size")
            if placement.key not in prefetch_keys:
                prefetch_keys.append(placement.key)
            identity = _cache_identity(future_refs[placement.key], placement)
            if (
                identity in current_identities[placement.rank]
                or identity in future_identities[placement.rank]
            ):
                continue
            future_identities[placement.rank].add(identity)
            future_bytes[placement.rank] = _checked_memory_sum(
                (future_bytes[placement.rank], placement.nbytes),
                "prefetched static residency bytes",
            )
    return tuple(current_bytes), tuple(future_bytes), tuple(prefetch_keys)


def _rank_estimate(request, rank, current_static, prefetched_static):
    task14 = request.distributed_plan.memory_estimates[rank]
    solver = request.solver_profile.rank_estimates[rank]
    variable_ref = next(
        ref
        for ref in request.distributed_plan.execution_plan.inputs
        if ref.key == request.distributed_plan.variable_key
    )
    itemsize = np.dtype(variable_ref.spec.dtype).itemsize
    solver_input = _checked_memory_product(
        (request.mapped_local_counts[rank], itemsize),
        "solver input bytes",
    )
    solver_result = _checked_memory_product(
        (
            request.mapped_local_counts[rank],
            request.solver_profile.result_itemsize,
        ),
        "solver result bytes",
    )
    if request.materialization_policy == "host":
        raise NotImplementedError("host center materialization is not implemented")
    if request.world_size > 1:
        hash_device_bytes = 3 * 8 * np.dtype(np.uint64).itemsize
        hash_host_bytes = hash_device_bytes
        policy_device_bytes = 3 * 2 * np.dtype(np.int32).itemsize
        policy_host_bytes = 2 * 2 * np.dtype(np.int32).itemsize
        requirement_bytes = (
            3 * (2 * request.world_size + 3) * np.dtype(np.int64).itemsize
        )
        residency_hash_device_bytes = max(
            hash_device_bytes + policy_device_bytes,
            requirement_bytes,
        )
        residency_hash_host_bytes = max(
            hash_host_bytes + policy_host_bytes,
            requirement_bytes,
        )
    else:
        residency_hash_device_bytes = 0
        residency_hash_host_bytes = 0
    components = {
        "current_static_bytes": current_static,
        "prefetched_static_bytes": prefetched_static,
        "dense_input_bytes": task14.local_input_bytes,
        "receive_buffer_bytes": task14.receive_buffer_bytes,
        "output_accumulator_bytes": task14.output_accumulator_bytes,
        "output_contribution_bytes": task14.output_contribution_bytes,
        "execution_workspace_bytes": task14.workspace_bytes,
        "task14_hash_control_device_bytes": task14.preflight_hash_device_bytes,
        "residency_hash_control_device_bytes": residency_hash_device_bytes,
        "residency_hash_control_host_bytes": residency_hash_host_bytes,
        "task14_preflight_status_device_bytes": task14.preflight_status_device_bytes,
        "task14_execution_status_device_bytes": task14.control_status_bytes,
        "mapped_packed_output_bytes": solver_input,
        "solver_input_bytes": solver_input,
        "solver_result_bytes": solver_result,
        "solver_diagonal_bytes": (
            solver_input if request.solver_profile.solver_kind == "davidson" else 0
        ),
        "complete_input_center_bytes": request.complete_input_center_bytes,
        "complete_diagonal_center_bytes": request.complete_diagonal_center_bytes,
        "original_packed_guess_bytes": (
            _checked_memory_product(
                (request.solver_profile.global_vector_count, itemsize),
                "original packed guess bytes",
            )
            if request.solver_profile.solver_kind == "davidson"
            else 0
        ),
        "original_packed_diagonal_bytes": (
            _checked_memory_product(
                (request.solver_profile.global_vector_count, itemsize),
                "original packed diagonal bytes",
            )
            if request.solver_profile.solver_kind == "davidson"
            else 0
        ),
        "solver_hv_retained_bytes": solver.hv_retained_bytes,
        "solver_hv_transient_bytes": solver.hv_transient_bytes,
        "solver_la_retained_bytes": solver.la_retained_bytes,
        "solver_la_transient_bytes": solver.la_transient_bytes,
        "solver_communication_device_bytes": solver.communication_device_bytes,
        "complete_center_bytes": request.center_profile.complete_center_bytes,
        "materialization_receive_bytes": request.center_profile.receive_buffer_bytes[
            rank
        ],
        "materialization_control_device_bytes": request.center_profile.control_device_bytes[
            rank
        ],
        "materialization_control_host_bytes": request.center_profile.control_host_bytes[
            rank
        ],
        "qn_setup_mask_device_bytes": request.center_profile.setup_qn_mask_bytes,
        "qn_extract_mask_device_bytes": request.center_profile.extract_qn_mask_bytes[
            rank
        ],
        "qn_hv_mask_device_bytes": request.center_profile.hv_qn_mask_bytes[rank],
        "qn_materialization_mask_device_bytes": request.center_profile.materialization_qn_mask_bytes[
            rank
        ],
        "qn_writeback_mask_device_bytes": request.center_profile.writeback_qn_mask_bytes,
        "qn_writeback_packed_result_bytes": request.center_profile.writeback_packed_result_bytes,
        "qn_persistent_mask_host_bytes": request.center_profile.persistent_qn_mask_host_bytes,
        "qn_setup_digest_host_bytes": request.center_profile.setup_qn_digest_host_bytes,
        "state_digest_host_bytes": request.center_profile.state_digest_host_bytes,
        "state_agreement_device_bytes": request.center_profile.state_agreement_device_bytes[
            rank
        ],
        "state_agreement_host_bytes": request.center_profile.state_agreement_host_bytes[
            rank
        ],
        "store_bytes": request.store_bytes[rank],
        "external_host_bytes": request.external_host_bytes[rank],
        "transfer_staging_host_bytes": request.transfer_staging_host_bytes[rank],
        "dirty_writeback_bytes": request.dirty_writeback_bytes[rank],
        "task14_host_control_bytes": task14.host_bytes,
        "task14_persistent_execution_status_host_bytes": (
            task14.host_control_status_bytes
        ),
        "solver_host_peak_bytes": solver.host_peak_bytes,
    }
    return RankResidencyEstimate(**components, **_rank_peaks(components))


class ResidencyPlanner:
    """Pure deterministic active-working-set preflight planner."""

    def plan(self, request):
        if not isinstance(request, ResidencyRequest):
            raise TypeError("request must be a ResidencyRequest")
        if request.local_world_size != request.world_size:
            raise ValueError("local_world_size must equal world_size")
        canonical_solver = canonical_solver_memory_profile(request.solver_profile)
        if canonical_solver != request.solver_profile:
            raise ValueError("solver profile does not match canonical solver profile")
        if any(count <= 0 for count in request.mapped_local_counts):
            raise NotImplementedError("empty mapped shard is unsupported")
        if not request.solver_profile.bounded:
            raise NotImplementedError("unbounded solver profile is unsupported")
        if tuple(request.mapped_local_counts) != tuple(
            request.solver_input_sharding.local_shape(rank)[0]
            for rank in range(request.world_size)
        ):
            raise ValueError("mapped_local_counts do not match solver sharding")

        current = _current_placements(request)
        current_bytes, future_bytes, prefetch_keys = _static_bytes_by_rank(
            request, current
        )
        estimates = tuple(
            _rank_estimate(request, rank, current_bytes[rank], future_bytes[rank])
            for rank in range(request.world_size)
        )
        all_refs = tuple(ref for _, ref in request.host_refs) + tuple(
            ref for future in request.future_plans for ref in future.required_refs
        )
        unique_refs = {_ref_identity(ref): ref for ref in all_refs}
        required_refs = tuple(unique_refs[key] for key in sorted(unique_refs))
        plan_fields = {
            "world_size": request.world_size,
            "required_refs": required_refs,
            "local_slices": current,
            "rank_estimates": estimates,
            "device_peak_bytes": tuple(value.device_peak_bytes for value in estimates),
            "host_peak_bytes": tuple(value.host_peak_bytes for value in estimates),
            "host_required_bytes": _checked_memory_sum(
                (
                    *tuple(value.host_peak_bytes for value in estimates),
                    *(
                        tuple(value.device_peak_bytes for value in estimates)
                        if request.backend_name == "numpy"
                        else ()
                    ),
                ),
                "node host requirement",
            ),
            "prefetch_keys": prefetch_keys,
            "prefetch_depth": request.prefetch_depth,
            "source_plan_hash": request.distributed_plan.execution_plan.plan_hash,
            "placement_hash": request.distributed_plan.placement_hash,
            "request_hash": request.request_hash,
            "backend_name": request.backend_name,
            "solver_kind": request.solver_profile.solver_kind,
            "krylov_coefficient_identity": (
                request.solver_profile.coefficient_identity
                if request.solver_profile.solver_kind == "krylov"
                else None
            ),
            "qn_mask_identity": request.qn_mask_identity,
            "device_budget": request.device_budget,
            "host_budget": request.host_budget,
            "transfer_profile": request.transfer_profile,
            "runtime_identity": request.runtime_identity,
            "current_refs": tuple(ref for _, ref in request.host_refs),
            "future_plans": request.future_plans,
            "future_plan_hashes": tuple(
                future.plan_hash for future in request.future_plans
            ),
        }
        provisional = object.__new__(ResidencyPlan)
        for name, value in plan_fields.items():
            object.__setattr__(provisional, name, value)
        plan_fields["plan_hash"] = _sha256(_plan_payload(provisional))
        plan = ResidencyPlan(**plan_fields)
        return plan


__all__ = [
    "FutureResidencyPlan",
    "ForeignHostTensorRefError",
    "HostTensorError",
    "HostTensorAllocation",
    "HostTensorRef",
    "HostTensorStore",
    "HostTensorStoreClosedError",
    "HostTensorStoreSnapshot",
    "HostTensorVersionConflictError",
    "MemoryBudgetResolution",
    "RankResidencyEstimate",
    "ResidencyBudgetError",
    "ResidencyPlan",
    "ResidencyPlanner",
    "ResidencyPreflightReceipt",
    "ResidencyRequest",
    "ResidencyRuntimeIdentity",
    "SliceRange",
    "StaleHostTensorRefError",
    "TensorPlacement",
    "TransferProfile",
    "budget_resolution_hash",
]
