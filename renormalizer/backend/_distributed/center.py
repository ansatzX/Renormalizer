"""Dense and quantum-number-packed solver vector boundaries."""

from dataclasses import dataclass, field, replace
import hashlib
import json
from time import perf_counter

import numpy as np

from renormalizer.backend._distributed.sharding import ShardingSpec, shard_axis


_MAX_METADATA_INT64 = int(np.iinfo(np.int64).max)


def _checked_metadata_product(values, name):
    product = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError(
                "{} values must be non-negative Python integers".format(name)
            )
        product *= value
        if product > _MAX_METADATA_INT64:
            raise OverflowError("{} exceeds the supported int64 range".format(name))
    return product


def _checked_shape_elements(shape, name):
    return _checked_metadata_product(tuple(shape), "{} elements".format(name))


class MeasuredCollective:
    """Borrow a collective while accounting its actual scheduled buffers."""

    def __init__(self, collective):
        self._collective = collective
        self.rank = collective.rank
        self.size = collective.size
        self.metrics = {
            "broadcast_calls": 0,
            "allreduce_calls": 0,
            "reduce_scatter_calls": 0,
            "allgather_calls": 0,
            "broadcast_bytes": 0,
            "allreduce_bytes": 0,
            "reduce_scatter_bytes": 0,
            "allgather_bytes": 0,
            "collective_bytes": 0,
            "collective_s": 0.0,
        }

    def __getattr__(self, name):
        return getattr(self._collective, name)

    @staticmethod
    def _nbytes(array):
        nbytes = getattr(array, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        return int(array.size) * int(np.dtype(array.dtype).itemsize)

    def _call(self, kind, array, operation):
        byte_count = self._nbytes(array)
        started = perf_counter()
        try:
            return operation()
        finally:
            elapsed = perf_counter() - started
            self.metrics[kind + "_calls"] += 1
            self.metrics[kind + "_bytes"] += byte_count
            self.metrics["collective_bytes"] += byte_count
            self.metrics["collective_s"] += elapsed

    def barrier(self):
        return self._collective.barrier()

    def broadcast(self, array, *, root):
        return self._call(
            "broadcast",
            array,
            lambda: self._collective.broadcast(array, root=root),
        )

    def allreduce(self, array, *, op="sum"):
        return self._call(
            "allreduce",
            array,
            lambda: self._collective.allreduce(array, op=op),
        )

    def allreduce_inplace(self, array, *, op="sum"):
        return self._call(
            "allreduce",
            array,
            lambda: self._collective.allreduce_inplace(array, op=op),
        )

    def reduce_scatter(self, array, *, axis, op="sum"):
        return self._call(
            "reduce_scatter",
            array,
            lambda: self._collective.reduce_scatter(array, axis=axis, op=op),
        )

    def allgather(self, array, *, axis):
        return self._call(
            "allgather",
            array,
            lambda: self._collective.allgather(array, axis=axis),
        )

    def close(self):
        return self._collective.close()


def _backend_mask(backend, mask):
    return backend.asarray(mask)


def _contiguous(backend, array):
    return backend.array_namespace.ascontiguousarray(array)


def _native_mask(array, mask):
    if type(array).__module__.split(".", 1)[0] == "cupy":
        import cupy

        return cupy.asarray(mask)
    return mask


@dataclass(frozen=True, order=True)
class QnMaskIdentity:
    """Bounded identity for one immutable complete Boolean center mask."""

    global_shape: tuple[int, ...]
    rank_counts: tuple[int, ...]
    digest: str

    def __post_init__(self):
        try:
            global_shape = tuple(self.global_shape)
            rank_counts = tuple(self.rank_counts)
        except TypeError as error:
            raise TypeError("QN mask identity shape/counts must be iterable") from error
        if not global_shape or any(
            type(value) is not int or value <= 0 for value in global_shape
        ):
            raise ValueError("QN mask identity shape must be positive")
        if not rank_counts or any(
            type(value) is not int or value <= 0 for value in rank_counts
        ):
            raise ValueError("QN mask identity rank counts must be positive")
        if sum(rank_counts) > _checked_shape_elements(global_shape, "QN mask identity"):
            raise ValueError("QN mask identity counts exceed the dense center")
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or self.digest != self.digest.lower()
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("QN mask identity digest must be canonical SHA-256")
        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "rank_counts", rank_counts)


def qn_mask_identity_payload(value):
    if value is None:
        return None
    if not isinstance(value, QnMaskIdentity):
        raise TypeError("value must be a QnMaskIdentity or None")
    return {
        "global_shape": list(value.global_shape),
        "rank_counts": list(value.rank_counts),
        "digest": value.digest,
    }


@dataclass(frozen=True)
class CenterVectorMap:
    input_sharding: ShardingSpec
    output_sharding: ShardingSpec
    qn_mask: object = None
    rank_counts: tuple[int, ...] = field(init=False)
    solver_sharding: ShardingSpec = field(init=False)
    qn_mask_identity: QnMaskIdentity | None = field(init=False)
    _local_masks: tuple[object, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.input_sharding, ShardingSpec) or not isinstance(
            self.output_sharding, ShardingSpec
        ):
            raise TypeError("center sharding must use ShardingSpec values")
        if self.input_sharding != self.output_sharding:
            raise ValueError(
                "center input and output require identical dense ownership"
            )

        mask = self.qn_mask
        if mask is not None:
            mask = np.array(mask, dtype=bool, copy=True, order="C")
            if tuple(mask.shape) != self.input_sharding.global_shape:
                raise ValueError("qn_mask shape must match the complete center")
            mask.flags.writeable = False

        local_masks = []
        counts = []
        for rank in range(self.input_sharding.parts):
            local_slice = self.input_sharding.local_slices[rank]
            if mask is None:
                local_mask = None
                count = _checked_shape_elements(
                    self.input_sharding.local_shape(rank), "dense center shard"
                )
            else:
                local_mask = np.array(mask[local_slice], copy=True, order="C")
                local_mask.flags.writeable = False
                count = int(np.count_nonzero(local_mask))
            if count == 0:
                raise ValueError("every rank must own at least one allowed entry")
            local_masks.append(local_mask)
            counts.append(count)

        slices = []
        start = 0
        for count in counts:
            slices.append((slice(start, start + count),))
            start += count
        mask_identity = None
        if mask is not None:
            mask_identity = QnMaskIdentity(
                self.input_sharding.global_shape,
                tuple(counts),
                hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
            )
        object.__setattr__(self, "qn_mask", mask)
        object.__setattr__(self, "rank_counts", tuple(counts))
        object.__setattr__(self, "qn_mask_identity", mask_identity)
        object.__setattr__(self, "_local_masks", tuple(local_masks))
        object.__setattr__(
            self,
            "solver_sharding",
            ShardingSpec((start,), 0, tuple(slices)),
        )

    def extract_local(self, full_center, rank, backend):
        local = full_center[self.input_sharding.local_slices[rank]]
        local_mask = self._local_masks[rank]
        if local_mask is not None:
            local = local[_backend_mask(backend, local_mask)]
        return _contiguous(backend, local.reshape(-1))

    def expand_local(self, local_1d, rank, dense_buffer):
        if tuple(local_1d.shape) != (self.rank_counts[rank],):
            raise ValueError("local solver vector shape does not match center map")
        if tuple(dense_buffer.shape) != self.input_sharding.local_shape(rank):
            raise ValueError("dense buffer shape does not match center map")
        local_mask = self._local_masks[rank]
        if local_mask is None:
            dense_buffer[...] = local_1d.reshape(dense_buffer.shape)
        else:
            dense_buffer[...] = 0
            dense_buffer[_native_mask(dense_buffer, local_mask)] = local_1d
        return dense_buffer

    def pack_local(self, dense_local_slab, rank, packed_buffer):
        if tuple(dense_local_slab.shape) != self.output_sharding.local_shape(rank):
            raise ValueError("dense output shape does not match center map")
        if tuple(packed_buffer.shape) != (self.rank_counts[rank],):
            raise ValueError("packed output shape does not match center map")
        local_mask = self._local_masks[rank]
        if local_mask is None:
            packed_buffer[...] = dense_local_slab.reshape(-1)
        else:
            packed_buffer[...] = dense_local_slab[
                _native_mask(dense_local_slab, local_mask)
            ]
        return packed_buffer

    def materialize(self, local_1d, collective, backend):
        rank = int(collective.rank)
        namespace = backend.array_namespace
        full_center = None
        receive = None
        allocation_error = None
        try:
            if tuple(local_1d.shape) != (self.rank_counts[rank],):
                raise ValueError("local solver vector shape does not match center map")
            full_center = namespace.zeros(
                self.input_sharding.global_shape, dtype=local_1d.dtype
            )
            receive = namespace.empty(max(self.rank_counts), dtype=local_1d.dtype)
        except BaseException as error:
            allocation_error = error
        status = _control_array(
            collective, [int(allocation_error is not None)], np.int32
        )
        failed = collective.allreduce(status, op="max")
        if int(_host_control(failed).reshape(-1)[0]):
            raise RuntimeError(
                "center materialization allocation failed"
            ) from allocation_error
        for source_rank, count in enumerate(self.rank_counts):
            shard = receive[:count]
            staging_error = None
            try:
                if source_rank == rank:
                    shard[...] = local_1d
            except BaseException as error:
                staging_error = error
            status = _control_array(
                collective, [int(staging_error is not None)], np.int32
            )
            failed = collective.allreduce(status, op="max")
            if int(_host_control(failed).reshape(-1)[0]):
                raise RuntimeError(
                    "center materialization source staging failed"
                ) from staging_error
            collective.broadcast(shard, root=source_rank)
            unpack_error = None
            try:
                dense = full_center[self.input_sharding.local_slices[source_rank]]
                self.expand_local(shard, source_rank, dense)
            except BaseException as error:
                unpack_error = error
            status = _control_array(
                collective, [int(unpack_error is not None)], np.int32
            )
            failed = collective.allreduce(status, op="max")
            if int(_host_control(failed).reshape(-1)[0]):
                raise RuntimeError(
                    "center materialization unpack failed"
                ) from unpack_error
        return full_center

    def pack_baseline(self, full_center):
        if tuple(full_center.shape) != self.input_sharding.global_shape:
            raise ValueError("complete center shape does not match center map")
        if self.qn_mask is None:
            return full_center.ravel()
        return full_center[_native_mask(full_center, self.qn_mask)]


@dataclass(frozen=True)
class CenterMaterializationMemoryProfile:
    policy: str
    complete_center_bytes: int
    receive_buffer_bytes: tuple[int, ...]
    control_device_bytes: tuple[int, ...]
    global_shape: tuple[int, ...] | None = None
    input_sharding: ShardingSpec | None = None
    output_sharding: ShardingSpec | None = None
    solver_sharding: ShardingSpec | None = None
    dtype: str | None = None
    itemsize: int | None = None
    rank_counts: tuple[int, ...] | None = None
    control_host_bytes: tuple[int, ...] | None = None
    input_dtype: str | None = None
    input_itemsize: int | None = None
    qn_mask_present: bool | None = None
    setup_qn_mask_bytes: int | None = None
    extract_qn_mask_bytes: tuple[int, ...] | None = None
    hv_qn_mask_bytes: tuple[int, ...] | None = None
    materialization_qn_mask_bytes: tuple[int, ...] | None = None
    writeback_qn_mask_bytes: int | None = None
    writeback_packed_result_bytes: int | None = None
    state_digest_host_bytes: int | None = None
    state_agreement_device_bytes: tuple[int, ...] | None = None
    state_agreement_host_bytes: tuple[int, ...] | None = None
    qn_mask_identity: QnMaskIdentity | None = None
    persistent_qn_mask_host_bytes: int | None = None
    setup_qn_digest_host_bytes: int | None = None

    def __post_init__(self):
        if self.policy != "device":
            raise ValueError("center materialization profile policy must be 'device'")
        if (
            type(self.complete_center_bytes) is not int
            or self.complete_center_bytes <= 0
        ):
            raise ValueError("complete_center_bytes must be a positive integer")
        receive = tuple(self.receive_buffer_bytes)
        controls = tuple(self.control_device_bytes)
        if not receive or len(receive) != len(controls):
            raise ValueError("center materialization ranks must be complete")
        if any(type(value) is not int or value < 0 for value in receive + controls):
            raise ValueError(
                "center materialization bytes must be non-negative integers"
            )
        object.__setattr__(self, "receive_buffer_bytes", receive)
        object.__setattr__(self, "control_device_bytes", controls)
        identity = (
            self.global_shape,
            self.input_sharding,
            self.output_sharding,
            self.solver_sharding,
            self.dtype,
            self.itemsize,
            self.rank_counts,
            self.control_host_bytes,
        )
        extensions = (
            self.input_dtype,
            self.input_itemsize,
            self.qn_mask_present,
            self.setup_qn_mask_bytes,
            self.extract_qn_mask_bytes,
            self.hv_qn_mask_bytes,
            self.materialization_qn_mask_bytes,
            self.writeback_qn_mask_bytes,
            self.writeback_packed_result_bytes,
            self.state_digest_host_bytes,
            self.state_agreement_device_bytes,
            self.state_agreement_host_bytes,
            self.persistent_qn_mask_host_bytes,
            self.setup_qn_digest_host_bytes,
        )
        if all(value is None for value in identity + extensions) and (
            self.qn_mask_identity is None
        ):
            return
        shardings = (
            self.input_sharding,
            self.output_sharding,
            self.solver_sharding,
        )
        if all(isinstance(value, ShardingSpec) for value in shardings) and (
            self.solver_sharding.parts != self.input_sharding.parts
        ):
            raise ValueError("center dense and solver sharding parts must be equal")
        if any(value is None for value in identity):
            raise ValueError(
                "center materialization identity metadata must be complete"
            )
        derive_extensions = all(value is None for value in extensions) and (
            self.qn_mask_identity is None
        )
        if not derive_extensions and any(value is None for value in extensions):
            raise ValueError(
                "center materialization identity metadata must be complete"
            )
        if not all(
            isinstance(value, ShardingSpec)
            for value in (
                self.input_sharding,
                self.output_sharding,
                self.solver_sharding,
            )
        ):
            raise TypeError("center materialization sharding must use ShardingSpec")
        if self.input_sharding != self.output_sharding:
            raise ValueError("center input and output sharding must be equal")
        if self.solver_sharding.parts != self.input_sharding.parts:
            raise ValueError("center dense and solver sharding parts must be equal")
        if (
            len(self.solver_sharding.global_shape) != 1
            or self.solver_sharding.axis != 0
        ):
            raise ValueError("center solver sharding must describe one packed vector")
        global_shape = tuple(self.global_shape)
        if global_shape != self.input_sharding.global_shape:
            raise ValueError("center profile global shape must match dense sharding")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("center materialization dtype is unsupported") from error
        if dtype.fields is not None or dtype.kind not in "biufc" or not dtype.isnative:
            raise ValueError("center materialization dtype is unsupported")
        if self.dtype != dtype.name:
            raise ValueError("center materialization dtype must be canonical")
        if type(self.itemsize) is not int or self.itemsize != dtype.itemsize:
            raise ValueError("center materialization itemsize does not match dtype")
        if derive_extensions:
            object.__setattr__(self, "input_dtype", dtype.name)
            object.__setattr__(self, "input_itemsize", dtype.itemsize)
        try:
            input_dtype = np.dtype(self.input_dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("center input dtype is unsupported") from error
        if (
            input_dtype.fields is not None
            or input_dtype.kind not in "biufc"
            or not input_dtype.isnative
        ):
            raise ValueError("center input dtype is unsupported")
        if self.input_dtype != input_dtype.name:
            raise ValueError("center input dtype must be canonical")
        if (
            type(self.input_itemsize) is not int
            or self.input_itemsize != input_dtype.itemsize
        ):
            raise ValueError("center input itemsize does not match dtype")
        rank_counts = tuple(self.rank_counts)
        expected_counts = tuple(
            self.solver_sharding.local_shape(rank)[0]
            for rank in range(self.solver_sharding.parts)
        )
        if rank_counts != expected_counts:
            raise ValueError("center profile rank counts do not match solver sharding")
        if sum(rank_counts) != self.solver_sharding.global_shape[0]:
            raise ValueError("center profile global packed count is inconsistent")
        complete_elements = _checked_shape_elements(global_shape, "complete center")
        expected_complete = _checked_metadata_product(
            (complete_elements, dtype.itemsize), "complete center bytes"
        )
        if self.complete_center_bytes != expected_complete:
            raise ValueError("complete_center_bytes do not match shape and dtype")
        dense_rank_counts = tuple(
            _checked_shape_elements(
                self.input_sharding.local_shape(rank), "dense center shard"
            )
            for rank in range(self.input_sharding.parts)
        )
        if derive_extensions:
            qn_mask_present = rank_counts != dense_rank_counts
            if qn_mask_present:
                raise ValueError(
                    "packed center profile requires exact QN mask identity"
                )
            local_mask_bytes = (
                dense_rank_counts
                if qn_mask_present
                else (0,) * self.input_sharding.parts
            )
            materialization_mask_bytes = (
                (max(dense_rank_counts),) * self.input_sharding.parts
                if qn_mask_present
                else (0,) * self.input_sharding.parts
            )
            object.__setattr__(self, "qn_mask_present", qn_mask_present)
            object.__setattr__(
                self,
                "setup_qn_mask_bytes",
                complete_elements if qn_mask_present else 0,
            )
            object.__setattr__(self, "extract_qn_mask_bytes", local_mask_bytes)
            object.__setattr__(self, "hv_qn_mask_bytes", local_mask_bytes)
            object.__setattr__(
                self,
                "materialization_qn_mask_bytes",
                materialization_mask_bytes,
            )
            object.__setattr__(
                self,
                "writeback_qn_mask_bytes",
                complete_elements if qn_mask_present else 0,
            )
            object.__setattr__(
                self,
                "writeback_packed_result_bytes",
                sum(rank_counts) * dtype.itemsize if qn_mask_present else 0,
            )
            object.__setattr__(self, "state_digest_host_bytes", 2 * expected_complete)
            object.__setattr__(
                self,
                "state_agreement_device_bytes",
                (104,) * self.input_sharding.parts,
            )
            object.__setattr__(
                self,
                "state_agreement_host_bytes",
                (96,) * self.input_sharding.parts,
            )
            object.__setattr__(self, "qn_mask_identity", None)
            object.__setattr__(self, "persistent_qn_mask_host_bytes", 0)
            object.__setattr__(self, "setup_qn_digest_host_bytes", 0)
        host_controls = tuple(self.control_host_bytes)
        if (
            not receive
            or len(receive) != self.input_sharding.parts
            or len(receive) != len(controls)
            or len(receive) != len(host_controls)
        ):
            raise ValueError("center materialization ranks must be complete")
        if any(
            type(value) is not int or value < 0
            for value in receive + controls + host_controls
        ):
            raise ValueError(
                "center materialization bytes must be non-negative integers"
            )
        expected_receive = max(rank_counts) * dtype.itemsize
        if receive != (expected_receive,) * self.input_sharding.parts:
            raise ValueError("center receive bytes do not match solver rank counts")
        if controls != (3 * np.dtype(np.int32).itemsize,) * self.input_sharding.parts:
            raise ValueError(
                "center device controls must include retained/input/output status"
            )
        if host_controls != (np.dtype(np.int32).itemsize,) * self.input_sharding.parts:
            raise ValueError("center host controls must include synchronized status")
        if type(self.qn_mask_present) is not bool:
            raise TypeError("center QN mask presence must be a boolean")
        if any(packed > dense for packed, dense in zip(rank_counts, dense_rank_counts)):
            raise ValueError("center packed counts exceed dense shard sizes")
        if self.qn_mask_identity is None:
            if self.qn_mask_present:
                raise ValueError("center QN presence requires exact mask identity")
            if rank_counts != dense_rank_counts:
                raise ValueError("center packed counts require QN mask identity")
        else:
            if not isinstance(self.qn_mask_identity, QnMaskIdentity):
                raise TypeError("center QN mask identity must be canonical metadata")
            if not self.qn_mask_present:
                raise ValueError("center QN mask identity requires mask presence")
            if (
                self.qn_mask_identity.global_shape != global_shape
                or self.qn_mask_identity.rank_counts != rank_counts
            ):
                raise ValueError("center QN mask identity does not match profile")
        expected_setup_mask = complete_elements if self.qn_mask_present else 0
        expected_local_masks = (
            dense_rank_counts
            if self.qn_mask_present
            else (0,) * self.input_sharding.parts
        )
        expected_materialization_masks = (
            (max(dense_rank_counts),) * self.input_sharding.parts
            if self.qn_mask_present
            else (0,) * self.input_sharding.parts
        )
        expected_writeback_mask = complete_elements if self.qn_mask_present else 0
        expected_packed_result = (
            sum(rank_counts) * dtype.itemsize if self.qn_mask_present else 0
        )
        expected_persistent_host_mask = (
            2 * complete_elements if self.qn_mask_present else 0
        )
        expected_setup_digest = complete_elements if self.qn_mask_present else 0
        qn_rank_fields = (
            ("extract", self.extract_qn_mask_bytes, expected_local_masks),
            ("H-v", self.hv_qn_mask_bytes, expected_local_masks),
            (
                "materialization",
                self.materialization_qn_mask_bytes,
                expected_materialization_masks,
            ),
        )
        for label, supplied, expected in qn_rank_fields:
            if tuple(supplied) != expected:
                raise ValueError(
                    "center {} QN mask bytes are inconsistent".format(label)
                )
        if self.setup_qn_mask_bytes != expected_setup_mask:
            raise ValueError("center setup QN mask bytes are inconsistent")
        if self.writeback_qn_mask_bytes != expected_writeback_mask:
            raise ValueError("center writeback QN mask bytes are inconsistent")
        if self.writeback_packed_result_bytes != expected_packed_result:
            raise ValueError("center QN packed result bytes are inconsistent")
        if self.state_digest_host_bytes != 2 * expected_complete:
            raise ValueError("center state digest host bytes are inconsistent")
        if self.persistent_qn_mask_host_bytes != expected_persistent_host_mask:
            raise ValueError("center persistent QN host mask bytes are inconsistent")
        if self.setup_qn_digest_host_bytes != expected_setup_digest:
            raise ValueError("center QN setup digest bytes are inconsistent")
        expected_state_device = (104,) * self.input_sharding.parts
        expected_state_host = (96,) * self.input_sharding.parts
        if tuple(self.state_agreement_device_bytes) != expected_state_device:
            raise ValueError("center state agreement device bytes are inconsistent")
        if tuple(self.state_agreement_host_bytes) != expected_state_host:
            raise ValueError("center state agreement host bytes are inconsistent")
        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "dtype", dtype.name)
        object.__setattr__(self, "input_dtype", input_dtype.name)
        object.__setattr__(self, "rank_counts", rank_counts)
        object.__setattr__(self, "control_host_bytes", host_controls)
        object.__setattr__(
            self, "extract_qn_mask_bytes", tuple(self.extract_qn_mask_bytes)
        )
        object.__setattr__(self, "hv_qn_mask_bytes", tuple(self.hv_qn_mask_bytes))
        object.__setattr__(
            self,
            "materialization_qn_mask_bytes",
            tuple(self.materialization_qn_mask_bytes),
        )
        object.__setattr__(
            self,
            "state_agreement_device_bytes",
            tuple(self.state_agreement_device_bytes),
        )
        object.__setattr__(
            self,
            "state_agreement_host_bytes",
            tuple(self.state_agreement_host_bytes),
        )


def center_materialization_memory_profile(
    vector_map, dtype, *, result_dtype=None, policy="device"
):
    """Describe the current complete-device-center broadcast schedule."""
    if not isinstance(vector_map, CenterVectorMap):
        raise TypeError("vector_map must be a CenterVectorMap")
    if policy == "host":
        raise NotImplementedError("host center materialization is not implemented")
    if policy != "device":
        raise ValueError("center materialization policy must be 'device' or 'host'")
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("center materialization dtype is unsupported") from error
    if dtype.fields is not None or dtype.kind not in "biufc" or not dtype.isnative:
        raise ValueError("center materialization dtype is unsupported")
    if result_dtype is None:
        result_dtype = dtype
    try:
        result_dtype = np.dtype(result_dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("center result dtype is unsupported") from error
    if (
        result_dtype.fields is not None
        or result_dtype.kind not in "biufc"
        or not result_dtype.isnative
    ):
        raise ValueError("center result dtype is unsupported")
    complete_elements = _checked_shape_elements(
        vector_map.input_sharding.global_shape, "complete center"
    )
    complete_bytes = _checked_metadata_product(
        (complete_elements, result_dtype.itemsize), "complete center bytes"
    )
    receive_bytes = _checked_metadata_product(
        (max(vector_map.rank_counts), result_dtype.itemsize),
        "center receive bytes",
    )
    rank_count = vector_map.input_sharding.parts
    control_bytes = np.dtype(np.int32).itemsize
    qn_mask_present = vector_map.qn_mask is not None
    dense_rank_counts = tuple(
        _checked_shape_elements(
            vector_map.input_sharding.local_shape(rank), "dense center shard"
        )
        for rank in range(rank_count)
    )
    local_mask_bytes = dense_rank_counts if qn_mask_present else (0,) * rank_count
    materialization_mask_bytes = (
        (max(dense_rank_counts),) * rank_count if qn_mask_present else (0,) * rank_count
    )
    return CenterMaterializationMemoryProfile(
        policy="device",
        global_shape=vector_map.input_sharding.global_shape,
        input_sharding=vector_map.input_sharding,
        output_sharding=vector_map.output_sharding,
        solver_sharding=vector_map.solver_sharding,
        dtype=result_dtype.name,
        itemsize=result_dtype.itemsize,
        input_dtype=dtype.name,
        input_itemsize=dtype.itemsize,
        rank_counts=vector_map.rank_counts,
        complete_center_bytes=complete_bytes,
        receive_buffer_bytes=(receive_bytes,) * rank_count,
        control_device_bytes=(3 * control_bytes,) * rank_count,
        control_host_bytes=(control_bytes,) * rank_count,
        qn_mask_present=qn_mask_present,
        setup_qn_mask_bytes=complete_elements if qn_mask_present else 0,
        extract_qn_mask_bytes=local_mask_bytes,
        hv_qn_mask_bytes=local_mask_bytes,
        materialization_qn_mask_bytes=materialization_mask_bytes,
        writeback_qn_mask_bytes=complete_elements if qn_mask_present else 0,
        writeback_packed_result_bytes=(
            sum(vector_map.rank_counts) * result_dtype.itemsize
            if qn_mask_present
            else 0
        ),
        state_digest_host_bytes=2 * complete_bytes,
        state_agreement_device_bytes=(104,) * rank_count,
        state_agreement_host_bytes=(96,) * rank_count,
        qn_mask_identity=vector_map.qn_mask_identity,
        persistent_qn_mask_host_bytes=(2 * complete_elements if qn_mask_present else 0),
        setup_qn_digest_host_bytes=(complete_elements if qn_mask_present else 0),
    )


@dataclass(frozen=True)
class KrylovRootFallbackMemoryProfile:
    dtype: str
    itemsize: int
    vector_count: int
    vector_bytes: int
    block_size: int
    max_krylov_vectors: int | None
    basis_vectors: int
    projected_host_bytes: int
    device_peak_bytes: int
    host_peak_bytes: int
    scalar_dtype: str | None = None
    scalar_itemsize: int | None = None
    result_dtype: str | None = None
    result_itemsize: int | None = None
    result_vector_bytes: int | None = None
    coefficient_identity: object = None

    def __post_init__(self):
        dtype = np.dtype(self.dtype)
        if self.dtype != dtype.name or self.itemsize != dtype.itemsize:
            raise ValueError("Krylov fallback dtype/itemsize metadata is inconsistent")
        extensions = (
            self.scalar_dtype,
            self.scalar_itemsize,
            self.result_dtype,
            self.result_itemsize,
            self.result_vector_bytes,
        )
        if all(value is None for value in extensions):
            scalar_dtype = np.dtype(np.float64)
            result_dtype = np.dtype(np.result_type(dtype, scalar_dtype))
            object.__setattr__(self, "scalar_dtype", scalar_dtype.name)
            object.__setattr__(self, "scalar_itemsize", scalar_dtype.itemsize)
            object.__setattr__(self, "result_dtype", result_dtype.name)
            object.__setattr__(self, "result_itemsize", result_dtype.itemsize)
            object.__setattr__(
                self,
                "result_vector_bytes",
                self.vector_count * result_dtype.itemsize,
            )
        elif any(value is None for value in extensions):
            raise ValueError("Krylov fallback identity metadata must be complete")
        result_dtype = np.dtype(self.result_dtype)
        scalar_dtype = np.dtype(self.scalar_dtype)
        from renormalizer.backend._distributed.solvers import (
            KrylovCoefficientIdentity,
            canonical_krylov_coefficient,
        )

        coefficient_identity = self.coefficient_identity
        if coefficient_identity is None:
            coefficient_identity = canonical_krylov_coefficient(1.0)
            object.__setattr__(self, "coefficient_identity", coefficient_identity)
        elif not isinstance(coefficient_identity, KrylovCoefficientIdentity):
            raise TypeError("Krylov fallback coefficient identity is invalid")
        expected_scalar_dtype = np.dtype(
            np.complex128 if coefficient_identity.kind == "complex" else np.float64
        )
        if scalar_dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
            raise ValueError("Krylov fallback scalar dtype is not canonical")
        if (
            self.result_dtype != result_dtype.name
            or self.result_itemsize != result_dtype.itemsize
            or self.scalar_dtype != scalar_dtype.name
            or self.scalar_itemsize != scalar_dtype.itemsize
            or result_dtype != np.dtype(np.result_type(dtype, scalar_dtype))
            or scalar_dtype != expected_scalar_dtype
        ):
            raise ValueError("Krylov fallback result dtype metadata is inconsistent")
        if (
            type(self.vector_count) is not int
            or self.vector_count <= 0
            or self.vector_bytes != self.vector_count * dtype.itemsize
        ):
            raise ValueError("Krylov fallback vector metadata is inconsistent")
        if self.result_vector_bytes != self.vector_count * result_dtype.itemsize:
            raise ValueError("Krylov fallback result vector metadata is inconsistent")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("Krylov fallback block_size must be positive")
        if self.max_krylov_vectors is not None and (
            type(self.max_krylov_vectors) is not int or self.max_krylov_vectors <= 0
        ):
            raise ValueError("Krylov fallback vector bound must be positive or None")
        expected_basis = min(
            self.vector_count,
            self.vector_count
            if self.max_krylov_vectors is None
            else self.max_krylov_vectors,
        )
        if self.basis_vectors != expected_basis:
            raise ValueError("Krylov fallback basis bound is inconsistent")
        for value in (
            self.projected_host_bytes,
            self.device_peak_bytes,
            self.host_peak_bytes,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("Krylov fallback memory bytes must be non-negative")
        if self.device_peak_bytes < 2 * self.vector_bytes:
            raise ValueError("Krylov fallback device peak omits persistent vectors")
        if self.host_peak_bytes < self.projected_host_bytes:
            raise ValueError("Krylov fallback host peak omits projected matrices")


def krylov_root_fallback_memory_profile(
    *,
    vector_bytes,
    vector_count,
    dtype,
    coefficient=1.0,
    block_size,
    max_krylov_vectors=None,
):
    """Bound the complete-vector fallback around the shared Krylov recurrence."""
    from renormalizer.backend._distributed.solvers import (
        build_krylov_memory_profile,
    )

    dtype = np.dtype(dtype)
    if type(vector_count) is not int or vector_count <= 0:
        raise ValueError("vector_count must be a positive integer")
    if type(vector_bytes) is not int or vector_bytes != vector_count * dtype.itemsize:
        raise ValueError("vector_bytes do not match count and dtype")
    solver_profile = build_krylov_memory_profile(
        shard_axis((vector_count,), 0, 1),
        dtype,
        coefficient=coefficient,
        block_size=block_size,
        max_krylov_vectors=max_krylov_vectors,
    )
    rank = solver_profile.rank_estimates[0]
    solver_peak = max(
        rank.hv_retained_bytes
        + max(rank.hv_transient_bytes, rank.communication_device_bytes),
        rank.la_retained_bytes
        + rank.la_transient_bytes
        + rank.communication_device_bytes,
    )
    metadata_bytes = np.dtype(np.int64).itemsize
    result_vector_bytes = vector_count * solver_profile.result_itemsize
    basis_vectors = (
        vector_count
        if solver_profile.basis_vectors is None
        else solver_profile.basis_vectors
    )
    return KrylovRootFallbackMemoryProfile(
        dtype=dtype.name,
        itemsize=dtype.itemsize,
        vector_count=vector_count,
        vector_bytes=vector_bytes,
        scalar_dtype=solver_profile.scalar_dtype,
        scalar_itemsize=solver_profile.scalar_itemsize,
        result_dtype=solver_profile.result_dtype,
        result_itemsize=solver_profile.result_itemsize,
        result_vector_bytes=result_vector_bytes,
        coefficient_identity=solver_profile.coefficient_identity,
        block_size=solver_profile.block_size,
        max_krylov_vectors=solver_profile.max_krylov_vectors,
        basis_vectors=basis_vectors,
        projected_host_bytes=solver_profile.projected_host_bytes,
        device_peak_bytes=max(
            vector_bytes + result_vector_bytes + metadata_bytes + solver_peak,
            vector_bytes + result_vector_bytes + metadata_bytes + 104,
        ),
        host_peak_bytes=max(
            rank.host_peak_bytes,
            metadata_bytes,
            2 * result_vector_bytes + 96,
        ),
    )


@dataclass(frozen=True)
class DavidsonRootFallbackMemoryProfile:
    basis_bytes: int
    device_peak_bytes: int
    host_peak_bytes: int
    dtype: str | None = None
    itemsize: int | None = None
    packed_vector_count: int | None = None
    packed_vector_bytes: int | None = None
    full_center_bytes: int | None = None
    concurrent_complete_center_bytes: int | None = None
    mask_bytes: int | None = None
    max_space: int | None = None
    result_dtype: str | None = None
    result_itemsize: int | None = None
    packed_result_bytes: int | None = None
    full_result_center_bytes: int | None = None
    original_packed_guess_bytes: int | None = None
    original_packed_diagonal_bytes: int | None = None

    def __post_init__(self):
        base_values = (self.basis_bytes, self.device_peak_bytes, self.host_peak_bytes)
        if any(type(value) is not int or value < 0 for value in base_values):
            raise ValueError("root fallback memory bytes must be non-negative integers")
        if self.device_peak_bytes < self.basis_bytes:
            raise ValueError("root fallback device peak must include both basis lists")
        identity = (
            self.dtype,
            self.itemsize,
            self.packed_vector_count,
            self.packed_vector_bytes,
            self.full_center_bytes,
            self.concurrent_complete_center_bytes,
            self.mask_bytes,
            self.max_space,
        )
        extensions = (
            self.result_dtype,
            self.result_itemsize,
            self.packed_result_bytes,
            self.full_result_center_bytes,
            self.original_packed_guess_bytes,
            self.original_packed_diagonal_bytes,
        )
        if all(value is None for value in identity + extensions):
            return
        if any(value is None for value in identity):
            raise ValueError("root fallback identity metadata must be complete")
        dtype = np.dtype(self.dtype)
        if self.dtype != dtype.name or self.itemsize != dtype.itemsize:
            raise ValueError("root fallback dtype/itemsize metadata is inconsistent")
        if all(value is None for value in extensions):
            result_dtype = np.dtype(np.result_type(dtype, np.float64))
            object.__setattr__(self, "result_dtype", result_dtype.name)
            object.__setattr__(self, "result_itemsize", result_dtype.itemsize)
            object.__setattr__(
                self,
                "packed_result_bytes",
                self.packed_vector_count * result_dtype.itemsize,
            )
            object.__setattr__(
                self,
                "full_result_center_bytes",
                self.full_center_bytes // dtype.itemsize * result_dtype.itemsize,
            )
            object.__setattr__(
                self, "original_packed_guess_bytes", self.packed_vector_bytes
            )
            object.__setattr__(
                self, "original_packed_diagonal_bytes", self.packed_vector_bytes
            )
        elif any(value is None for value in extensions):
            raise ValueError("root fallback identity metadata must be complete")
        result_dtype = np.dtype(self.result_dtype)
        if (
            self.result_dtype != result_dtype.name
            or self.result_itemsize != result_dtype.itemsize
            or result_dtype != np.dtype(np.result_type(dtype, np.float64))
        ):
            raise ValueError(
                "root fallback result dtype/itemsize metadata is inconsistent"
            )
        if (
            type(self.packed_vector_count) is not int
            or self.packed_vector_count <= 0
            or self.packed_vector_bytes != self.packed_vector_count * dtype.itemsize
        ):
            raise ValueError("root fallback packed vector metadata is inconsistent")
        if self.packed_result_bytes != self.packed_vector_count * result_dtype.itemsize:
            raise ValueError("root fallback packed result metadata is inconsistent")
        if (
            self.original_packed_guess_bytes != self.packed_vector_bytes
            or self.original_packed_diagonal_bytes != self.packed_vector_bytes
        ):
            raise ValueError(
                "root fallback original packed input metadata is inconsistent"
            )
        if self.full_center_bytes % dtype.itemsize:
            raise ValueError("root fallback full center metadata is inconsistent")
        expected_full_result = (
            self.full_center_bytes // dtype.itemsize * result_dtype.itemsize
        )
        if self.full_result_center_bytes != expected_full_result:
            raise ValueError("root fallback full result metadata is inconsistent")
        if self.concurrent_complete_center_bytes != (
            2 * self.full_center_bytes + 3 * self.full_result_center_bytes
        ):
            raise ValueError("root fallback concurrent center bound is inconsistent")
        if type(self.max_space) is not int or self.max_space < 2:
            raise ValueError("root fallback max_space must be at least two")
        identity_values = (
            self.full_center_bytes,
            self.concurrent_complete_center_bytes,
            self.mask_bytes,
        )
        if any(type(value) is not int or value < 0 for value in identity_values):
            raise ValueError("root fallback memory bytes must be non-negative integers")


def davidson_root_fallback_memory_profile(
    *,
    packed_vector_bytes,
    packed_vector_count,
    full_center_bytes,
    mask_bytes,
    dtype,
    max_space,
):
    """Bound the current single-rank fallback around the Davidson recurrence."""
    from renormalizer.backend._distributed.solvers import (
        build_davidson_memory_profile,
    )

    dtype = np.dtype(dtype)
    if type(packed_vector_count) is not int or packed_vector_count <= 0:
        raise ValueError("packed_vector_count must be a positive integer")
    for name, value in (
        ("packed_vector_bytes", packed_vector_bytes),
        ("full_center_bytes", full_center_bytes),
        ("mask_bytes", mask_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError("{} must be a non-negative integer".format(name))
    if packed_vector_bytes != packed_vector_count * dtype.itemsize:
        raise ValueError("packed_vector_bytes do not match count and dtype")
    solver_profile = build_davidson_memory_profile(
        shard_axis((packed_vector_count,), 0, 1),
        dtype,
        max_space=max_space,
    )
    rank = solver_profile.rank_estimates[0]
    packed_result_bytes = packed_vector_count * solver_profile.result_itemsize
    full_result_center_bytes = (
        full_center_bytes // dtype.itemsize * solver_profile.result_itemsize
    )
    basis_bytes = 2 * solver_profile.basis_vectors * packed_result_bytes
    solver_peak = max(
        rank.hv_retained_bytes
        + max(rank.hv_transient_bytes, rank.communication_device_bytes),
        rank.la_retained_bytes
        + rank.la_transient_bytes
        + rank.communication_device_bytes,
    )
    concurrent_complete_center_bytes = (
        2 * full_center_bytes + 3 * full_result_center_bytes
    )
    metadata_device_bytes = (
        2 * np.dtype(np.float64).itemsize + 5 * np.dtype(np.int64).itemsize
    )
    persistent_device = (
        4 * packed_vector_bytes
        + packed_result_bytes
        + concurrent_complete_center_bytes
        + mask_bytes
        + metadata_device_bytes
    )
    metadata_host_bytes = (
        2 * np.dtype(np.float64).itemsize + 5 * np.dtype(np.int64).itemsize
    )
    return DavidsonRootFallbackMemoryProfile(
        dtype=dtype.name,
        itemsize=dtype.itemsize,
        packed_vector_count=packed_vector_count,
        packed_vector_bytes=packed_vector_bytes,
        result_dtype=solver_profile.result_dtype,
        result_itemsize=solver_profile.result_itemsize,
        packed_result_bytes=packed_result_bytes,
        original_packed_guess_bytes=packed_vector_bytes,
        original_packed_diagonal_bytes=packed_vector_bytes,
        full_center_bytes=full_center_bytes,
        full_result_center_bytes=full_result_center_bytes,
        concurrent_complete_center_bytes=concurrent_complete_center_bytes,
        mask_bytes=mask_bytes,
        max_space=max_space,
        basis_bytes=basis_bytes,
        device_peak_bytes=max(
            persistent_device + solver_peak,
            2 * full_center_bytes
            + full_result_center_bytes
            + 2 * packed_vector_bytes
            + packed_result_bytes
            + mask_bytes
            + metadata_device_bytes
            + 104,
        ),
        host_peak_bytes=max(
            rank.host_peak_bytes + metadata_host_bytes,
            2 * full_result_center_bytes + 96,
        ),
    )


@dataclass
class MappedDistributedLocalOperator:
    local_operator: object
    vector_map: CenterVectorMap
    solver_input_sharding: ShardingSpec = field(init=False)
    solver_output_sharding: ShardingSpec = field(init=False)
    solver_dtype: np.dtype = field(init=False)
    _dense_input: object = field(init=False, default=None, repr=False)
    _packed_output: object = field(init=False, default=None, repr=False)

    def __post_init__(self):
        from renormalizer.backend._distributed.local_operator import (
            DistributedLocalOperator,
        )

        if not isinstance(self.local_operator, DistributedLocalOperator):
            raise TypeError("local_operator must be a DistributedLocalOperator")
        if not isinstance(self.vector_map, CenterVectorMap):
            raise TypeError("vector_map must be a CenterVectorMap")
        if self.local_operator.plan.input_sharding != self.vector_map.input_sharding:
            raise ValueError("center map input ownership does not match local operator")
        if self.local_operator.plan.output_sharding != self.vector_map.output_sharding:
            raise ValueError(
                "center map output ownership does not match local operator"
            )
        variable_dtype = self.local_operator.solver_dtype
        output_dtype = np.dtype(
            self.local_operator.plan.execution_plan.output.spec.dtype
        )
        if variable_dtype != output_dtype:
            raise ValueError("iterative operator input and output dtype must agree")
        self.solver_input_sharding = self.vector_map.solver_sharding
        self.solver_output_sharding = self.vector_map.solver_sharding
        self.solver_dtype = variable_dtype

    @property
    def backend(self):
        return self.local_operator.backend

    @property
    def context(self):
        return self.local_operator.context

    @property
    def collective(self):
        return self.local_operator.collective

    @property
    def counters(self):
        return self.local_operator.counters

    def solver_preflight(self):
        self.local_operator.solver_preflight()
        if self._dense_input is not None:
            return
        rank = self.context.rank
        packed_bytes = (
            2 * self.vector_map.rank_counts[rank] * self.solver_dtype.itemsize
        )
        estimate = self.local_operator.plan.memory_estimates[rank]
        over_budget = (
            self.local_operator.device_memory_budget_bytes is not None
            and estimate.device_bytes + packed_bytes
            > self.local_operator.device_memory_budget_bytes
        )
        if self.local_operator._allreduce_status(1 if over_budget else 0):
            raise ValueError("mapped solver capacity preflight failed")

        dense_input = None
        packed_output = None
        allocation_error = None
        try:
            namespace = self.backend.array_namespace
            dense_input = namespace.empty(
                self.vector_map.input_sharding.local_shape(rank),
                dtype=self.solver_dtype,
            )
            packed_output = namespace.empty(
                (self.vector_map.rank_counts[rank],), dtype=self.solver_dtype
            )
        except BaseException as error:
            allocation_error = error
        if self.local_operator._allreduce_status(
            1 if allocation_error is not None else 0
        ):
            if allocation_error is not None:
                raise ValueError(
                    "mapped solver allocation preflight failed"
                ) from allocation_error
            raise ValueError("mapped solver allocation preflight failed")
        self._dense_input = dense_input
        self._packed_output = packed_output

    def __call__(self, local_vector_1d):
        self.solver_preflight()
        rank = self.context.rank
        self.vector_map.expand_local(local_vector_1d, rank, self._dense_input)
        dense_output = self.local_operator(self._dense_input)
        return self.vector_map.pack_local(dense_output, rank, self._packed_output)


def _resolve_center_mapping(
    hop, full_center, distributed_execution, center_shape, qn_mask
):
    from renormalizer.backend._distributed.planner import (
        plan_distributed_execution,
    )

    resolver = getattr(hop, "resolve_execution_artifact", None)
    if not callable(resolver):
        raise NotImplementedError("local H-v has no resolved execution artifact")
    artifact = resolver(full_center)
    plan = artifact.execution_plan
    variable_ref = next(ref for ref in plan.inputs if ref.key == artifact.variable_key)
    center_shape = tuple(int(dimension) for dimension in center_shape)
    if (
        tuple(variable_ref.spec.shape) != center_shape
        or tuple(plan.output.spec.shape) != center_shape
    ):
        raise NotImplementedError("input and output center shapes must agree")

    chosen = None
    vector_map = None
    for axis, (output_mode, input_mode) in enumerate(
        zip(plan.output.spec.modes, variable_ref.spec.modes)
    ):
        if (
            output_mode == input_mode
            or center_shape[axis] < distributed_execution.context.world_size
        ):
            continue
        try:
            candidate = plan_distributed_execution(
                plan,
                variable_key=artifact.variable_key,
                world_size=distributed_execution.context.world_size,
                output_mode=output_mode,
                input_mode=input_mode,
            )
            candidate_map = CenterVectorMap(
                candidate.input_sharding,
                candidate.output_sharding,
                qn_mask,
            )
        except ValueError:
            continue
        if candidate.input_sharding != candidate.output_sharding:
            continue
        chosen = candidate
        vector_map = candidate_map
        break
    if chosen is None:
        raise NotImplementedError("center has no compatible distributed axis")
    return artifact, chosen, vector_map


def _host_control(value):
    getter = getattr(value, "get", None)
    if callable(getter):
        value = getter()
    return np.asarray(value)


def _control_array(collective, values, dtype):
    cupy = getattr(collective, "_cupy", None)
    if cupy is not None:
        with cupy.cuda.Device(int(collective._device_index)):
            return cupy.asarray(values, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def run_synchronized_setup_phase(
    distributed_execution, operation, phase, *, counters=None
):
    """Run rank-local setup with one fixed status collective."""
    result = None
    local_error = None
    try:
        result = operation()
    except BaseException as error:
        local_error = error
    collective = distributed_execution.collective
    status = _control_array(collective, [int(local_error is not None)], np.int32)
    failed = collective.allreduce(status, op="max")
    if counters is not None:
        counters["allreduce_calls"] = counters.get("allreduce_calls", 0) + 1
    if int(_host_control(failed).reshape(-1)[0]):
        raise RuntimeError("{} setup failed".format(phase)) from local_error
    return result


def normalize_distributed_backend_metadata(metadata, *, local_rank):
    """Canonicalize a valid rank-local device without hiding invalid devices."""
    name, device, precision = metadata
    normalized_name = None if name is None else str(name)
    normalized_device = None if device is None else str(device)
    normalized_precision = None if precision is None else int(precision)
    if normalized_name == "cupy" and normalized_device == "cuda:{}".format(
        int(local_rank)
    ):
        normalized_device = "cuda:<local_rank>"
    return normalized_name, normalized_device, normalized_precision


def validate_distributed_backend(distributed_execution, selected, *, network):
    """Synchronize active backend/runtime identity before adapter-local setup."""
    from renormalizer.backend.config import DistributedExecutionConfig

    if not isinstance(distributed_execution, DistributedExecutionConfig):
        raise TypeError("distributed_execution must be a DistributedExecutionConfig")
    local_error = None
    actual = (None, None, None)
    expected = (
        distributed_execution.backend_name,
        distributed_execution.backend_device,
        distributed_execution.backend_precision,
    )
    try:
        actual = (
            str(selected.name),
            str(selected.device),
            int(selected.config.precision),
        )
        if selected.name not in {"numpy", "cupy"}:
            raise NotImplementedError(
                "distributed execution supports only NumPy and CuPy backends"
            )
        if not selected.supports_execution_ir:
            raise NotImplementedError("distributed execution requires execution IR")
        if distributed_execution.residency_policy == "active_working_set" and (
            selected.config.execution_policy != "execution_ir"
            or selected.config.fallback_policy != "error"
        ):
            raise ValueError(
                "active_working_set requires execution_ir with fallback_policy='error'"
            )
        if expected[0] is not None and actual != expected:
            raise ValueError(
                "active backend name/device/precision does not match runtime"
            )
    except BaseException as error:
        local_error = error

    collective = distributed_execution.collective
    status = _control_array(collective, [int(local_error is not None)], np.int32)
    failed = collective.allreduce(status, op="max")
    payload = {
        "network": str(network),
        "actual": normalize_distributed_backend_metadata(
            actual, local_rank=distributed_execution.context.local_rank
        ),
        "expected": normalize_distributed_backend_metadata(
            expected, local_rank=distributed_execution.context.local_rank
        ),
        "error": None if local_error is None else type(local_error).__name__,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    hexdigest = hashlib.sha256(encoded).hexdigest()
    digest = _control_array(
        collective,
        [int(hexdigest[index : index + 16], 16) for index in range(0, 64, 16)],
        np.uint64,
    )
    minimum = collective.allreduce(digest, op="min")
    maximum = collective.allreduce(digest, op="max")
    if not np.array_equal(_host_control(minimum), _host_control(maximum)):
        raise RuntimeError("distributed backend/runtime metadata disagreement")
    if int(_host_control(failed).reshape(-1)[0]):
        raise RuntimeError(
            "distributed backend/runtime validation failed"
        ) from local_error
    return selected


def host_array(value):
    getter = getattr(value, "get", None)
    if callable(getter):
        value = getter()
    return np.asarray(value)


def result_dtype(*values):
    return np.dtype(np.result_type(*(np.dtype(value.dtype) for value in values)))


def measured_execution(distributed_execution):
    return replace(
        distributed_execution,
        collective=MeasuredCollective(distributed_execution.collective),
    )


def collective_elapsed(collective):
    """Return measured host time spent inside collective calls."""
    return float(getattr(collective, "metrics", {}).get("collective_s", 0.0))


def resolve_solver_config(distributed_execution, solver, solver_config):
    options = {} if solver_config is None else dict(solver_config)
    job_options = getattr(distributed_execution.collective, "solver_options", {})
    options.update(dict(job_options.get(solver, {})))
    return options


def copy_backend_metadata(target, values, selected):
    target[...] = selected.asarray(values, dtype=np.dtype(target.dtype))


def record_phase_summary(network, phase, operation, wall_s, *, operation_count=1):
    from renormalizer.utils import profiling

    if not profiling.enabled():
        return
    from renormalizer.backend._execution.profiling import phase_summary_payload

    profiling.record(
        "phase_summary",
        **phase_summary_payload(
            phase=str(phase),
            network=str(network),
            operation=str(operation),
            operation_count=int(operation_count),
            wall_s=float(wall_s),
        ),
    )


_UNBOUNDED_FALLBACK_REASON = "unbounded_device_resident_root_fallback"


def _mark_unbounded_root_fallback(
    distributed_execution,
    selected,
    *,
    device_bytes,
    host_bytes,
    counters,
):
    from renormalizer.backend._distributed.providers import (
        active_working_set_policy_error,
    )

    policy = getattr(distributed_execution, "residency_policy", "device_resident")
    policy_error = active_working_set_policy_error(policy, selected, None)
    if policy_error is not None:
        raise policy_error
    for value, name in (
        (device_bytes, "fallback estimated device bytes"),
        (host_bytes, "fallback estimated host bytes"),
    ):
        if type(value) is not int or value < 0:
            raise ValueError("{} must be a non-negative Python integer".format(name))
    counters["capacity_proven"] = False
    counters["capacity_reason"] = _UNBOUNDED_FALLBACK_REASON
    counters["fallback_estimated_device_bytes"] = device_bytes
    counters["fallback_estimated_host_bytes"] = host_bytes


def preflight_root_fallback_capacity(
    distributed_execution,
    selected,
    *,
    device_bytes,
    host_bytes,
    counters,
):
    """Compatibility shim: fallback estimates are telemetry, not authorization."""
    _mark_unbounded_root_fallback(
        distributed_execution,
        selected,
        device_bytes=device_bytes,
        host_bytes=host_bytes,
        counters=counters,
    )
    return None


def _resolved_execution_budget(distributed_execution, resource):
    if resource not in {"device", "host"}:
        raise ValueError("resource must be 'device' or 'host'")
    resolved_name = "resolved_{}_memory_budget_bytes".format(resource)
    resolved = getattr(distributed_execution, resolved_name, None)
    if resolved is not None:
        return resolved
    return getattr(
        distributed_execution, "{}_memory_budget_bytes".format(resource), None
    )


def _execution_budget_metadata(distributed_execution, resource):
    resolution = getattr(
        distributed_execution, "{}_budget_resolution".format(resource), None
    )
    if resolution is None:
        return _resolved_execution_budget(distributed_execution, resource)
    return {
        "requested_bytes": resolution.requested_bytes,
        "resolved_bytes": resolution.resolved_bytes,
        "source": resolution.source,
        "available_snapshot_bytes": resolution.available_snapshot_bytes,
        "resource": resolution.resource,
    }


def run_adapter_root_fallback(
    operation,
    distributed_execution,
    selected,
    receive_buffer,
    *,
    metadata_buffers=(),
    estimated_device_bytes,
    estimated_host_bytes,
    counters,
    capacity_approval=None,
):
    from renormalizer.backend._distributed.local_operator import run_root_fallback

    for key in (
        "fallback_count",
        "root_operation_count",
        "broadcast_calls",
        "allreduce_calls",
        "allgather_calls",
    ):
        counters.setdefault(key, 0)
    if capacity_approval is not None:
        raise TypeError("root fallback capacity approvals are unsupported")
    _mark_unbounded_root_fallback(
        distributed_execution,
        selected,
        device_bytes=estimated_device_bytes,
        host_bytes=estimated_host_bytes,
        counters=counters,
    )
    phase_start = perf_counter()
    root_compute_s = 0.0

    def root_operation():
        nonlocal root_compute_s
        counters["root_operation_count"] += 1
        compute_start = perf_counter()
        try:
            return operation()
        finally:
            root_compute_s += perf_counter() - compute_start

    result = run_root_fallback(
        root_operation,
        distributed_execution.collective,
        receive_buffer=receive_buffer,
    )
    counters["allreduce_calls"] += (
        2 if distributed_execution.context.world_size == 1 else 6
    )
    counters["broadcast_calls"] += 1
    for metadata in metadata_buffers:
        distributed_execution.collective.broadcast(metadata, root=0)
        counters["broadcast_calls"] += 1
    counters["fallback_count"] += 1
    counters["last_root_compute_s"] = root_compute_s
    counters["last_root_fallback_s"] = perf_counter() - phase_start
    counters["last_root_synchronization_s"] = max(
        counters["last_root_fallback_s"] - root_compute_s,
        0.0,
    )
    return result


def run_davidson_root_fallback(
    hop,
    distributed_execution,
    selected,
    mask,
    full_guess,
    full_diagonal,
    *,
    coefficient,
    solver_config,
    counters,
):
    from renormalizer.lib.davidson.backend import DavidsonInfo, davidson_backend
    from renormalizer.backend._distributed.providers import (
        active_working_set_policy_error,
    )

    policy_error = active_working_set_policy_error(
        getattr(distributed_execution, "residency_policy", "device_resident"),
        selected,
        None,
    )
    if policy_error is not None:
        raise policy_error

    def profile_fallback():
        options = {} if solver_config is None else dict(solver_config)
        dtype = np.dtype(full_guess.dtype)
        packed_count = int(np.count_nonzero(mask))
        dense_elements = _checked_shape_elements(mask.shape, "fallback center")
        memory_profile = davidson_root_fallback_memory_profile(
            packed_vector_bytes=packed_count * dtype.itemsize,
            packed_vector_count=packed_count,
            full_center_bytes=_checked_metadata_product(
                (dense_elements, dtype.itemsize), "fallback center bytes"
            ),
            mask_bytes=_checked_metadata_product(
                (dense_elements, np.dtype(bool).itemsize),
                "fallback mask bytes",
            ),
            dtype=dtype,
            max_space=int(options.get("max_space", 12)),
        )
        return (
            float(options.get("tol", 1e-12)),
            int(options.get("max_cycle", 50)),
            int(options.get("max_space", 12)),
            float(options.get("lindep", 1e-14)),
            bool(options.get("require_convergence", False)),
            getattr(hop, "legacy_fallback_expression", hop),
            memory_profile,
        )

    (
        tol,
        max_cycle,
        max_space,
        lindep,
        require_convergence,
        legacy_hop,
        memory_profile,
    ) = run_synchronized_setup_phase(
        distributed_execution,
        profile_fallback,
        "Davidson root fallback profile",
        counters=counters,
    )

    def setup():
        backend_mask = selected.asarray(mask)
        packed_guess = full_guess[backend_mask]
        packed_diagonal = full_diagonal[backend_mask]
        receive = selected.array_namespace.empty(
            packed_guess.shape, dtype=np.dtype(memory_profile.result_dtype)
        )
        float_metadata = selected.array_namespace.zeros(2, dtype=np.float64)
        int_metadata = selected.array_namespace.zeros(5, dtype=np.int64)
        return (
            backend_mask,
            packed_guess,
            packed_diagonal,
            receive,
            float_metadata,
            int_metadata,
        )

    (
        backend_mask,
        packed_guess,
        packed_diagonal,
        receive,
        float_metadata,
        int_metadata,
    ) = run_synchronized_setup_phase(
        distributed_execution,
        setup,
        "Davidson root fallback center allocation",
        counters=counters,
    )

    def packed_hop(value):
        full = selected.array_namespace.zeros(mask.shape, dtype=value.dtype)
        full[backend_mask] = value
        return (legacy_hop(full) * coefficient)[backend_mask]

    def operation():
        energy, vector, info = davidson_backend(
            packed_hop,
            packed_guess,
            packed_diagonal,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            lindep=lindep,
        )
        copy_backend_metadata(float_metadata, (energy, info.residual_norm), selected)
        copy_backend_metadata(
            int_metadata,
            (
                int(info.converged),
                info.iterations,
                info.h_v_count,
                info.subspace_size,
                info.restarts,
            ),
            selected,
        )
        return vector

    vector = run_adapter_root_fallback(
        operation,
        distributed_execution,
        selected,
        receive,
        metadata_buffers=(float_metadata, int_metadata),
        estimated_device_bytes=memory_profile.device_peak_bytes,
        estimated_host_bytes=memory_profile.host_peak_bytes,
        counters=counters,
    )
    floats = host_array(float_metadata)
    integers = host_array(int_metadata).astype(np.int64, copy=False)
    info = DavidsonInfo(
        converged=bool(integers[0]),
        iterations=int(integers[1]),
        h_v_count=int(integers[2]),
        subspace_size=int(integers[3]),
        restarts=int(integers[4]),
        residual_norm=float(floats[1]),
    )
    if require_convergence and not info.converged:
        raise RuntimeError("root fallback Davidson did not converge")
    return float(floats[0]), vector, info


def _canonical_decision_value(value):
    if isinstance(value, dict):
        return {
            str(key): _canonical_decision_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_decision_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.dtype):
        return value.str
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def canonicalize_compression_control(value):
    """Return stable controls for compression decisions without topology imports."""
    if not hasattr(value, "criteria"):
        return _canonical_decision_value(value)
    criteria = value.criteria
    criteria = getattr(criteria, "value", criteria)
    ofs = getattr(value, "ofs", None)
    ofs = getattr(ofs, "value", ofs)
    max_dims = getattr(value, "max_dims", None)
    if max_dims is not None:
        max_dims = np.asarray(max_dims).tolist()
    return {
        "criteria": str(criteria),
        "threshold": float(value.threshold),
        "bond_dim_max_value": int(value.bond_dim_max_value),
        "max_dims": max_dims,
        "ofs": ofs,
        "ofs_swap_jw": bool(getattr(value, "ofs_swap_jw", False)),
    }


def _adapter_decision_digest(
    *,
    error,
    artifact,
    plan,
    vector_map,
    network,
    operation,
    center_kind,
    center_shape,
    topology,
    qn_mask,
    solver_controls,
    selectors,
    fallback_policy,
    fallback_reason_code,
    world_size,
    local_world_size,
    mesh_shape,
    mesh_axis_names,
    device_budget,
    host_budget,
    prefetch_depth,
):
    qn_digest = None
    qn_count = None
    if qn_mask is not None:
        mask = np.ascontiguousarray(np.asarray(qn_mask, dtype=bool))
        qn_digest = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
        qn_count = int(np.count_nonzero(mask))
    payload = {
        "status": "ok" if error is None else type(error).__name__,
        "network": str(network),
        "operation": str(operation),
        "center_kind": str(center_kind),
        "center_shape": list(center_shape),
        "topology": _canonical_decision_value(topology),
        "qn_hash": qn_digest,
        "qn_count": qn_count,
        "solver_controls": _canonical_decision_value(solver_controls),
        "selectors": _canonical_decision_value(selectors),
        "fallback_policy": str(fallback_policy),
        "fallback_reason_code": str(fallback_reason_code),
        "world_size": int(world_size),
        "local_world_size": int(local_world_size),
        "mesh_shape": list(mesh_shape),
        "mesh_axis_names": list(mesh_axis_names),
        "device_budget": device_budget,
        "host_budget": host_budget,
        "prefetch_depth": int(prefetch_depth),
        "plan_hash": None if artifact is None else artifact.execution_plan.plan_hash,
        "placement_hash": None if plan is None else plan.placement_hash,
        "variable_key": None if artifact is None else artifact.variable_key,
        "output_mode": None if plan is None else plan.output_mode,
        "input_mode": None if plan is None else plan.input_mode,
        "solver_shape": (
            None
            if vector_map is None
            else list(vector_map.solver_sharding.global_shape)
        ),
        "solver_counts": None if vector_map is None else list(vector_map.rank_counts),
        "dtype": (
            None if artifact is None else np.dtype(artifact.variable_array.dtype).str
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    hexdigest = hashlib.sha256(encoded).hexdigest()
    return [int(hexdigest[index : index + 16], 16) for index in range(0, 64, 16)]


def _mapping_decision_digest(
    *,
    error,
    artifact,
    plan,
    vector_map,
    center_shape,
    qn_mask,
    distributed_execution,
    network="unspecified",
    operation="mapped_hv",
    center_kind="unspecified",
    topology=None,
    solver_controls=None,
    selectors=None,
    fallback_policy="error",
    fallback_reason_code="mapping",
):
    context = distributed_execution.context
    mesh = distributed_execution.mesh
    return _adapter_decision_digest(
        error=error,
        artifact=artifact,
        plan=plan,
        vector_map=vector_map,
        network=network,
        operation=operation,
        center_kind=center_kind,
        center_shape=center_shape,
        topology={} if topology is None else topology,
        qn_mask=qn_mask,
        solver_controls={} if solver_controls is None else solver_controls,
        selectors={} if selectors is None else selectors,
        fallback_policy=fallback_policy,
        fallback_reason_code=fallback_reason_code,
        world_size=context.world_size,
        local_world_size=context.local_world_size,
        mesh_shape=mesh.shape,
        mesh_axis_names=mesh.axis_names,
        device_budget=_execution_budget_metadata(distributed_execution, "device"),
        host_budget=_execution_budget_metadata(distributed_execution, "host"),
        prefetch_depth=distributed_execution.prefetch_depth,
    )


def coordinate_adapter_decision(
    distributed_execution,
    backend,
    *,
    network,
    operation,
    center_kind,
    center_shape,
    topology,
    qn_mask=None,
    solver_controls=None,
    selectors=None,
    supported=True,
    fallback_reason_code="none",
    artifact=None,
    plan=None,
    vector_map=None,
    counters=None,
):
    """Choose supported, fallback, or error with one fixed control schedule."""
    if counters is None:
        counters = {}
    from renormalizer.backend._distributed.providers import (
        active_working_set_policy_error,
    )

    counters.setdefault("allreduce_calls", 0)
    execution = distributed_execution
    context = execution.context
    collective = execution.collective
    mesh = execution.mesh
    configuration_error = None
    if (
        getattr(collective, "rank", None) != context.rank
        or getattr(collective, "size", None) != context.world_size
        or mesh.rank != context.rank
        or mesh.size != context.world_size
    ):
        configuration_error = ValueError("rank or world-size mismatch")
    policy_error = active_working_set_policy_error(
        getattr(execution, "residency_policy", "device_resident"),
        backend,
        plan if supported else None,
    )
    local_error = (
        configuration_error if configuration_error is not None else policy_error
    )
    if local_error is None and not supported:
        local_error = NotImplementedError(str(fallback_reason_code))

    status = _control_array(collective, [int(local_error is not None)], np.int32)
    failed = collective.allreduce(status, op="max")
    counters["allreduce_calls"] += 1
    fallback_policy = backend.config.fallback_policy
    digest = _control_array(
        collective,
        _mapping_decision_digest(
            error=local_error,
            artifact=artifact,
            plan=plan,
            vector_map=vector_map,
            network=network,
            operation=operation,
            center_kind=center_kind,
            center_shape=tuple(int(dimension) for dimension in center_shape),
            topology=topology,
            qn_mask=qn_mask,
            solver_controls={} if solver_controls is None else solver_controls,
            selectors={} if selectors is None else selectors,
            fallback_policy=fallback_policy,
            fallback_reason_code=fallback_reason_code,
            distributed_execution=execution,
        ),
        np.uint64,
    )
    minimum = collective.allreduce(digest, op="min")
    maximum = collective.allreduce(digest, op="max")
    counters["allreduce_calls"] += 2
    if not np.array_equal(_host_control(minimum), _host_control(maximum)):
        raise ValueError("distributed adapter decision digest disagreement")
    if not int(_host_control(failed).reshape(-1)[0]):
        return "supported"
    if configuration_error is not None:
        raise ValueError(
            "distributed adapter coordination preflight failed"
        ) from configuration_error
    if policy_error is not None:
        raise policy_error
    if fallback_policy == "legacy_oe":
        return "fallback"
    raise NotImplementedError("distributed adapter decision failed") from local_error


def build_mapped_local_operator(
    hop,
    full_center,
    distributed_execution,
    backend,
    center_shape,
    *,
    qn_mask=None,
    counters=None,
    network="unspecified",
    operation="mapped_hv",
    center_kind="unspecified",
    topology=None,
    solver_controls=None,
    selectors=None,
    fallback_reason_code="mapping",
):
    from renormalizer.backend._distributed.local_operator import (
        DistributedLocalOperator,
    )

    if counters is None:
        counters = {}
    for key in (
        "broadcast_calls",
        "allreduce_calls",
        "allgather_calls",
        "execution_calls",
        "boundary_materialization_broadcasts",
    ):
        counters.setdefault(key, 0)

    center_shape = tuple(int(dimension) for dimension in center_shape)

    def resolve():
        try:
            artifact, chosen, vector_map = _resolve_center_mapping(
                hop,
                full_center,
                distributed_execution,
                center_shape,
                qn_mask,
            )
            return artifact, chosen, vector_map, None
        except (NotImplementedError, TypeError, ValueError) as error:
            return None, None, None, error

    artifact, chosen, vector_map, local_error = run_synchronized_setup_phase(
        distributed_execution,
        resolve,
        "{} {} artifact resolution".format(network, operation),
        counters=counters,
    )

    try:
        route = coordinate_adapter_decision(
            distributed_execution,
            backend,
            network=network,
            operation=operation,
            center_kind=center_kind,
            center_shape=center_shape,
            topology={} if topology is None else topology,
            qn_mask=qn_mask,
            solver_controls={} if solver_controls is None else solver_controls,
            selectors={} if selectors is None else selectors,
            supported=local_error is None,
            fallback_reason_code=fallback_reason_code,
            artifact=artifact,
            plan=chosen,
            vector_map=vector_map,
            counters=counters,
        )
    except NotImplementedError as error:
        raise NotImplementedError("mapped adapter decision failed") from (
            local_error if local_error is not None else error
        )
    if route == "fallback":
        raise NotImplementedError("mapped adapter decision failed") from local_error

    def allocate():
        local_operator = DistributedLocalOperator(
            plan=chosen,
            provider=distributed_execution.provider,
            collective=distributed_execution.collective,
            counters=counters,
            backend=backend,
            context=distributed_execution.context,
            mesh=distributed_execution.mesh,
            source_bindings=artifact.source_bindings,
            device_memory_budget_bytes=_resolved_execution_budget(
                distributed_execution, "device"
            ),
            host_memory_budget_bytes=_resolved_execution_budget(
                distributed_execution, "host"
            ),
            residency_request=distributed_execution.residency_request,
            residency_plan=distributed_execution.residency_plan,
            residency_receipt=distributed_execution.residency_receipt,
        )
        return (
            MappedDistributedLocalOperator(local_operator, vector_map),
            vector_map,
            artifact.variable_array,
        )

    return run_synchronized_setup_phase(
        distributed_execution,
        allocate,
        "{} {} operator allocation".format(network, operation),
        counters=counters,
    )


def _state_digest(tensors, qn_arrays, metadata):
    digest = hashlib.sha256()
    for category, arrays in ((b"tensor", tensors), (b"qn", qn_arrays)):
        for value in arrays:
            host = value
            getter = getattr(host, "get", None)
            if callable(getter):
                host = getter()
            host = np.asarray(host)
            contiguous = np.ascontiguousarray(host)
            digest.update(category)
            digest.update(
                json.dumps(
                    {
                        "shape": list(contiguous.shape),
                        "dtype": contiguous.dtype.str,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            )
            digest.update(contiguous.tobytes(order="C"))
    digest.update(
        json.dumps(
            list(metadata),
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        ).encode("ascii")
    )
    hexdigest = digest.hexdigest()
    words = np.asarray(
        [int(hexdigest[index : index + 16], 16) for index in range(0, 64, 16)],
        dtype=np.uint64,
    )
    return hexdigest, words


def run_synchronized_state_update(
    distributed_execution,
    operation,
    state_payload,
    *,
    metadata=(),
    _return_digest=False,
):
    """Run a local state mutation, synchronize failure, then verify exact state."""
    if not callable(operation):
        raise TypeError("operation must be callable")
    if not callable(state_payload):
        raise TypeError("state_payload must be callable")
    result = None
    local_error = None
    hexdigest = None
    words = None
    resolved_metadata = metadata
    from renormalizer.utils import profiling

    profile_enabled = profiling.enabled()
    phase_start = perf_counter() if profile_enabled else None
    try:
        result = operation()
        tensors, qn_arrays = state_payload()
        if callable(metadata):
            resolved_metadata = metadata()
        hexdigest, words = _state_digest(tensors, qn_arrays, resolved_metadata)
    except BaseException as error:
        local_error = error

    collective = distributed_execution.collective
    status = _control_array(collective, [int(local_error is not None)], np.int32)
    failed = collective.allreduce(status, op="max")
    if int(_host_control(failed).reshape(-1)[0]):
        raise RuntimeError("distributed state update failed") from local_error

    control = _control_array(collective, words, np.uint64)
    minimum = collective.allreduce(control, op="min")
    maximum = collective.allreduce(control, op="max")
    if not np.array_equal(_host_control(minimum), _host_control(maximum)):
        raise RuntimeError(
            "distributed state digest disagreement after {!r}; local={}".format(
                tuple(resolved_metadata), hexdigest
            )
        )
    if profile_enabled:
        metadata_values = tuple(resolved_metadata)
        network = (
            str(metadata_values[0])
            if metadata_values and metadata_values[0] in {"mps", "ttns"}
            else "distributed"
        )
        record_phase_summary(
            network,
            "state_update",
            "synchronized_state_transition",
            perf_counter() - phase_start,
        )
    if _return_digest:
        return result, hexdigest
    return result


def synchronize_state_update(
    distributed_execution, tensors, *, qn_arrays=(), metadata=()
):
    _, hexdigest = run_synchronized_state_update(
        distributed_execution,
        lambda: None,
        lambda: (tensors, qn_arrays),
        metadata=metadata,
        _return_digest=True,
    )
    return hexdigest


def record_distributed_solve(
    operator,
    vector_map,
    *,
    network,
    center_kind,
    solver,
    hv_count,
    fallback_count=0,
    compute_s=0.0,
    synchronization_s=0.0,
    solver_residual_norm=0.0,
):
    from renormalizer.utils import profiling

    if not profiling.enabled():
        return
    from renormalizer.backend._distributed.profiling import (
        distributed_solve_payload,
    )

    rank = operator.context.rank
    counters = operator.counters
    metrics = getattr(operator.collective, "metrics", {})
    broadcast_calls = int(
        metrics.get("broadcast_calls", counters.get("broadcast_calls", 0))
    )
    allreduce_calls = int(
        metrics.get("allreduce_calls", counters.get("allreduce_calls", 0))
    )
    allgather_calls = int(
        metrics.get("allgather_calls", counters.get("allgather_calls", 0))
    )
    reduce_scatter_calls = int(metrics.get("reduce_scatter_calls", 0))
    collective_calls = (
        broadcast_calls + allreduce_calls + allgather_calls + reduce_scatter_calls
    )
    boundary = int(counters.get("boundary_materialization_broadcasts", 0))
    collective_bytes = int(metrics.get("collective_bytes", 0))
    collective_s = float(metrics.get("collective_s", synchronization_s))
    payload = distributed_solve_payload(
        global_shape=vector_map.solver_sharding.global_shape,
        local_shape=vector_map.solver_sharding.local_shape(rank),
        sharding_axis=0,
        hv_count=int(hv_count),
        collective_calls=collective_calls,
        collective_bytes=collective_bytes,
        collective_s=collective_s,
        compute_s=float(compute_s),
        synchronization_s=float(synchronization_s),
        fallback_count=int(fallback_count),
    )
    payload.update(
        {
            "network": str(network),
            "center_kind": str(center_kind),
            "solver": str(solver),
            "plan_hash": operator.local_operator.plan.execution_plan.plan_hash,
            "placement_hash": operator.local_operator.plan.placement_hash,
            "packed_qn": vector_map.qn_mask is not None,
            "packed_count": int(sum(vector_map.rank_counts)),
            "dense_local_elements": int(
                _checked_shape_elements(
                    vector_map.input_sharding.local_shape(rank),
                    "dense local center",
                )
            ),
            "broadcast_calls": broadcast_calls,
            "allreduce_calls": allreduce_calls,
            "allgather_calls": allgather_calls,
            "reduce_scatter_calls": reduce_scatter_calls,
            "boundary_materialization_broadcasts": boundary,
            "solver_residual_norm": float(solver_residual_norm),
        }
    )
    profiling.record("distributed_solve_summary", **payload)


def record_fallback_solve(
    distributed_execution,
    *,
    solver_dtype,
    global_count,
    packed_qn,
    network,
    center_kind,
    solver,
    hv_count,
    counters,
    synchronization_s,
    compute_s=0.0,
    solver_residual_norm=0.0,
):
    from renormalizer.utils import profiling

    if not profiling.enabled():
        return
    from renormalizer.backend._distributed.profiling import (
        distributed_solve_payload,
    )

    global_count = int(global_count)
    metrics = getattr(distributed_execution.collective, "metrics", {})
    broadcast_calls = int(
        metrics.get("broadcast_calls", counters.get("broadcast_calls", 0))
    )
    allreduce_calls = int(
        metrics.get("allreduce_calls", counters.get("allreduce_calls", 0))
    )
    allgather_calls = int(
        metrics.get("allgather_calls", counters.get("allgather_calls", 0))
    )
    reduce_scatter_calls = int(metrics.get("reduce_scatter_calls", 0))
    collective_calls = (
        broadcast_calls + allreduce_calls + allgather_calls + reduce_scatter_calls
    )
    collective_bytes = int(metrics.get("collective_bytes", 0))
    collective_s = float(metrics.get("collective_s", synchronization_s))
    payload = distributed_solve_payload(
        global_shape=(global_count,),
        local_shape=(global_count,),
        sharding_axis=0,
        hv_count=int(hv_count),
        collective_calls=collective_calls,
        collective_bytes=collective_bytes,
        collective_s=collective_s,
        compute_s=float(compute_s),
        synchronization_s=float(synchronization_s),
        fallback_count=1,
    )
    payload.update(
        {
            "network": str(network),
            "center_kind": str(center_kind),
            "solver": str(solver),
            "plan_hash": "fallback",
            "placement_hash": "fallback",
            "packed_qn": bool(packed_qn),
            "packed_count": global_count,
            "dense_local_elements": global_count,
            "broadcast_calls": broadcast_calls,
            "allreduce_calls": allreduce_calls,
            "allgather_calls": allgather_calls,
            "reduce_scatter_calls": reduce_scatter_calls,
            "boundary_materialization_broadcasts": 0,
            "solver_residual_norm": float(solver_residual_norm),
            "root_operation_count": int(counters.get("root_operation_count", 0)),
            "capacity_proven": False,
            "capacity_reason": str(
                counters.get("capacity_reason", _UNBOUNDED_FALLBACK_REASON)
            ),
        }
    )
    profiling.record("distributed_solve_summary", **payload)


__all__ = [
    "CenterMaterializationMemoryProfile",
    "CenterVectorMap",
    "DavidsonRootFallbackMemoryProfile",
    "KrylovRootFallbackMemoryProfile",
    "MappedDistributedLocalOperator",
    "QnMaskIdentity",
    "build_mapped_local_operator",
    "canonicalize_compression_control",
    "center_materialization_memory_profile",
    "davidson_root_fallback_memory_profile",
    "krylov_root_fallback_memory_profile",
    "collective_elapsed",
    "coordinate_adapter_decision",
    "copy_backend_metadata",
    "host_array",
    "measured_execution",
    "normalize_distributed_backend_metadata",
    "preflight_root_fallback_capacity",
    "qn_mask_identity_payload",
    "record_phase_summary",
    "resolve_solver_config",
    "run_synchronized_setup_phase",
    "result_dtype",
    "run_adapter_root_fallback",
    "run_davidson_root_fallback",
    "validate_distributed_backend",
    "run_synchronized_state_update",
    "synchronize_state_update",
    "record_distributed_solve",
    "record_fallback_solve",
]
