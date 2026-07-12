import math

from renormalizer.utils._profiling.events import (
    EVENT_NAMES,
    MAX_INTEGER_DIGITS,
    normalize_bounded_metadata,
)

_EVENT_INDEX = {name: index for index, name in enumerate(EVENT_NAMES)}
_STATE_SLOTS = (
    "_event_counts",
    "_local_hv_count",
    "_local_hv_ir_count",
    "_local_hv_legacy_count",
    "_local_hv_fallback_count",
    "_local_hv_wall_s",
    "_grouped_execution_count",
    "_grouped_task_count",
    "_distributed_hv_count",
    "_collective_calls",
    "_collective_bytes",
    "_collective_s",
    "_distributed_compute_s",
    "_synchronization_s",
    "_distributed_fallback_count",
    "_h2d_bytes",
    "_d2h_bytes",
    "_h2d_s",
    "_d2h_s",
    "_dirty_writeback_bytes",
    "_dirty_writeback_s",
    "_dirty_writeback_count",
    "_working_set_wall_s",
    "_cache_hits",
    "_cache_misses",
    "_prefetch_overlap_s",
    "_prefetch_wait_s",
    "_peak_device_bytes",
    "_planned_device_peak_bytes",
    "_observed_device_peak_bytes",
    "_planned_host_peak_bytes",
    "_observed_host_peak_bytes",
    "_planned_cache_peak_bytes",
    "_observed_cache_peak_bytes",
    "_planned_pinned_peak_bytes",
    "_observed_pinned_peak_bytes",
    "_pageable_fallback_count",
    "_pageable_fallback_bytes",
    "_full_replica_detected",
    "_full_replica_count",
)


class _AggregateDelta:
    __slots__ = (
        "owner",
        "version",
        "next_version",
        "next_state",
        "snapshot",
        "consumed",
    )

    def __init__(self, owner, version, next_version, next_state, snapshot):
        self.owner = owner
        self.version = version
        self.next_version = next_version
        self.next_state = next_state
        self.snapshot = snapshot
        self.consumed = False


def _required_non_negative_int(payload, field):
    if field not in payload:
        raise ValueError(f"{field} is required")
    value = payload[field]
    if type(value) is not int:
        raise TypeError(f"{field} must be a Python integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _required_non_negative_number(payload, field):
    if field not in payload:
        raise ValueError(f"{field} is required")
    value = payload[field]
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be a Python number")
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{field} must be finite and non-negative")
    return value


def _required_bool(payload, field):
    if field not in payload:
        raise ValueError(f"{field} is required")
    value = payload[field]
    if type(value) is not bool:
        raise TypeError(f"{field} must be a Python boolean")
    return value


def _optional_non_negative_int(payload, field, default=0):
    if field not in payload:
        return default
    return _required_non_negative_int(payload, field)


def _optional_non_negative_number(payload, field, default=0.0):
    if field not in payload:
        return default
    return _required_non_negative_number(payload, field)


def _optional_bool(payload, field, default=False):
    if field not in payload:
        return default
    return _required_bool(payload, field)


class RunAggregate:
    """Fixed-size totals with 0.0 for empty ratios and absent working-set times.

    ``working_set_wall_s`` sums only explicit ``wall_s`` measurements. It is never
    inferred from component times because transfer, prefetch, and writeback can overlap.
    """

    __slots__ = _STATE_SLOTS + ("_version",)

    def __init__(self):
        self._event_counts = [0] * len(EVENT_NAMES)
        self._local_hv_count = 0
        self._local_hv_ir_count = 0
        self._local_hv_legacy_count = 0
        self._local_hv_fallback_count = 0
        self._local_hv_wall_s = 0.0
        self._grouped_execution_count = 0
        self._grouped_task_count = 0
        self._distributed_hv_count = 0
        self._collective_calls = 0
        self._collective_bytes = 0
        self._collective_s = 0.0
        self._distributed_compute_s = 0.0
        self._synchronization_s = 0.0
        self._distributed_fallback_count = 0
        self._h2d_bytes = 0
        self._d2h_bytes = 0
        self._h2d_s = 0.0
        self._d2h_s = 0.0
        self._dirty_writeback_bytes = 0
        self._dirty_writeback_s = 0.0
        self._dirty_writeback_count = 0
        self._working_set_wall_s = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._prefetch_overlap_s = 0.0
        self._prefetch_wait_s = 0.0
        self._peak_device_bytes = 0
        self._planned_device_peak_bytes = 0
        self._observed_device_peak_bytes = 0
        self._planned_host_peak_bytes = 0
        self._observed_host_peak_bytes = 0
        self._planned_cache_peak_bytes = 0
        self._observed_cache_peak_bytes = 0
        self._planned_pinned_peak_bytes = 0
        self._observed_pinned_peak_bytes = 0
        self._pageable_fallback_count = 0
        self._pageable_fallback_bytes = 0
        self._full_replica_detected = False
        self._full_replica_count = 0
        self._version = 0

    def prepare(self, payload):
        if type(payload) is not dict:
            raise TypeError("aggregate events must be dictionaries")
        normalize_bounded_metadata(payload)
        event = payload.get("event")
        if type(event) is not str:
            raise TypeError("event must be a Python string")
        if event not in _EVENT_INDEX:
            raise ValueError(f"unsupported profiling event: {event}")
        event_index = _EVENT_INDEX[event]
        values = self._validate_event(event, payload)
        prospective = RunAggregate()
        prospective._install_state(self._export_state())
        prospective._version = self._version
        prospective._apply(event_index, values)
        prospective._version += 1
        prospective._validate_fixed_state()
        return _AggregateDelta(
            self,
            self._version,
            prospective._version,
            prospective._export_state(),
            prospective.summary(),
        )

    def commit(self, delta):
        self._validate_delta(delta)
        self._install_state(delta.next_state)
        self._version = delta.next_version
        delta.consumed = True

    def _apply(self, event_index, values):
        event = EVENT_NAMES[event_index]

        self._event_counts[event_index] += 1
        if event == "local_hv_execute":
            actual_policy, wall_s, fallback = values
            self._local_hv_count += 1
            self._local_hv_ir_count += actual_policy == "execution_ir"
            self._local_hv_legacy_count += actual_policy == "legacy_oe"
            self._local_hv_fallback_count += fallback is not None
            self._local_hv_wall_s += wall_s
        elif event == "grouped_gemm_execute":
            grouped_execution, task_count = values
            if grouped_execution:
                self._grouped_execution_count += 1
                self._grouped_task_count += task_count
        elif event == "distributed_solve_summary":
            (
                hv_count,
                collective_calls,
                collective_bytes,
                collective_s,
                compute_s,
                synchronization_s,
                fallback_count,
            ) = values
            self._distributed_hv_count += hv_count
            self._collective_calls += collective_calls
            self._collective_bytes += collective_bytes
            self._collective_s += collective_s
            self._distributed_compute_s += compute_s
            self._synchronization_s += synchronization_s
            self._distributed_fallback_count += fallback_count
        elif event == "working_set_transfer":
            (
                h2d_bytes,
                d2h_bytes,
                h2d_s,
                d2h_s,
                cache_hits,
                cache_misses,
                prefetch_overlap_s,
                prefetch_wait_s,
                dirty_writeback_bytes,
                dirty_writeback_s,
                dirty_writeback_count,
                peak_device_bytes,
                planned_device_peak_bytes,
                observed_device_peak_bytes,
                planned_host_peak_bytes,
                observed_host_peak_bytes,
                planned_cache_peak_bytes,
                observed_cache_peak_bytes,
                planned_pinned_peak_bytes,
                observed_pinned_peak_bytes,
                pageable_fallback_count,
                pageable_fallback_bytes,
                full_replica,
                wall_s,
            ) = values
            self._h2d_bytes += h2d_bytes
            self._d2h_bytes += d2h_bytes
            self._h2d_s += h2d_s
            self._d2h_s += d2h_s
            self._cache_hits += cache_hits
            self._cache_misses += cache_misses
            self._prefetch_overlap_s += prefetch_overlap_s
            self._prefetch_wait_s += prefetch_wait_s
            self._dirty_writeback_bytes += dirty_writeback_bytes
            self._dirty_writeback_s += dirty_writeback_s
            self._dirty_writeback_count += dirty_writeback_count
            self._working_set_wall_s += wall_s
            self._peak_device_bytes = max(self._peak_device_bytes, peak_device_bytes)
            self._planned_device_peak_bytes = max(
                self._planned_device_peak_bytes, planned_device_peak_bytes
            )
            self._observed_device_peak_bytes = max(
                self._observed_device_peak_bytes, observed_device_peak_bytes
            )
            self._planned_host_peak_bytes = max(
                self._planned_host_peak_bytes, planned_host_peak_bytes
            )
            self._observed_host_peak_bytes = max(
                self._observed_host_peak_bytes, observed_host_peak_bytes
            )
            self._planned_cache_peak_bytes = max(
                self._planned_cache_peak_bytes, planned_cache_peak_bytes
            )
            self._observed_cache_peak_bytes = max(
                self._observed_cache_peak_bytes, observed_cache_peak_bytes
            )
            self._planned_pinned_peak_bytes = max(
                self._planned_pinned_peak_bytes, planned_pinned_peak_bytes
            )
            self._observed_pinned_peak_bytes = max(
                self._observed_pinned_peak_bytes, observed_pinned_peak_bytes
            )
            self._pageable_fallback_count += pageable_fallback_count
            self._pageable_fallback_bytes += pageable_fallback_bytes
            self._full_replica_detected = self._full_replica_detected or full_replica
            self._full_replica_count += full_replica

    def add(self, payload):
        self.commit(self.prepare(payload))

    def preview(self, delta):
        self._validate_delta(delta)
        return self._copy_snapshot(delta.snapshot)

    def _validate_delta(self, delta):
        if type(delta) is not _AggregateDelta:
            raise TypeError("aggregate operations require a prepared delta")
        if delta.owner is not self:
            raise ValueError("prepared delta belongs to a different origin aggregate")
        if delta.consumed:
            raise ValueError("prepared delta has already been committed")
        if delta.version != self._version:
            raise ValueError("prepared delta has a stale aggregate version")

    def _export_state(self):
        return tuple(
            tuple(value) if type(value) is list else value
            for value in (getattr(self, slot) for slot in _STATE_SLOTS)
        )

    def _install_state(self, state):
        for slot, value in zip(_STATE_SLOTS, state):
            setattr(self, slot, list(value) if slot == "_event_counts" else value)

    @staticmethod
    def _copy_snapshot(snapshot):
        copied = dict(snapshot)
        copied["event_counts"] = dict(snapshot["event_counts"])
        return copied

    def _validate_fixed_state(self):
        for value in self._event_counts:
            self._validate_state_number(value)
        for slot in _STATE_SLOTS[1:]:
            self._validate_state_number(getattr(self, slot))
        self._validate_state_number(self._version)
        snapshot = self.summary()
        normalize_bounded_metadata(snapshot)
        for field in ("ir_call_coverage", "cache_hit_rate"):
            ratio = snapshot[field]
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{field} must remain finite and bounded")

    @staticmethod
    def _validate_state_number(value):
        if type(value) is bool:
            return
        if type(value) is int:
            if value < 0:
                raise ValueError("aggregate state integers must remain non-negative")
            if len(str(value)) > MAX_INTEGER_DIGITS:
                raise ValueError("aggregate state integers exceed the digit limit")
            return
        if type(value) is float:
            if value < 0 or not math.isfinite(value):
                raise ValueError(
                    "aggregate state floats must remain finite and non-negative"
                )
            return
        raise TypeError("aggregate state must contain fixed numeric values only")

    @staticmethod
    def _validate_event(event, payload):
        if event == "phase_summary":
            return (_required_non_negative_number(payload, "wall_s"),)
        if event == "local_hv_execute":
            if "actual_policy" not in payload:
                raise ValueError("actual_policy is required")
            actual_policy = payload["actual_policy"]
            if type(actual_policy) is not str:
                raise TypeError("actual_policy must be a Python string")
            if actual_policy not in {"execution_ir", "legacy_oe"}:
                raise ValueError("actual_policy is unsupported")
            wall_s = _required_non_negative_number(payload, "wall_s")
            fallback = payload.get("fallback")
            if fallback is not None and type(fallback) is not str:
                raise TypeError("fallback must be a Python string or None")
            return actual_policy, wall_s, fallback
        if event == "grouped_gemm_execute":
            grouped_execution = _required_bool(payload, "grouped_execution")
            task_count = _required_non_negative_int(payload, "task_count")
            if grouped_execution and task_count < 2:
                raise ValueError("grouped execution requires at least two tasks")
            return grouped_execution, task_count
        if event == "distributed_solve_summary":
            values = (
                tuple(
                    _required_non_negative_int(payload, field)
                    for field in ("hv_count", "collective_calls", "collective_bytes")
                )
                + tuple(
                    _required_non_negative_number(payload, field)
                    for field in ("collective_s", "compute_s", "synchronization_s")
                )
                + (_required_non_negative_int(payload, "fallback_count"),)
            )
            if values[-1] > values[0]:
                raise ValueError("fallback_count must not exceed hv_count")
            return values
        if event == "working_set_transfer":
            return (
                tuple(
                    _optional_non_negative_int(payload, field)
                    for field in ("h2d_bytes", "d2h_bytes")
                )
                + tuple(
                    _optional_non_negative_number(payload, field)
                    for field in ("h2d_s", "d2h_s")
                )
                + tuple(
                    _optional_non_negative_int(payload, field)
                    for field in ("cache_hits", "cache_misses")
                )
                + tuple(
                    _optional_non_negative_number(payload, field)
                    for field in ("prefetch_overlap_s", "prefetch_wait_s")
                )
                + (
                    _optional_non_negative_int(payload, "dirty_writeback_bytes"),
                    _optional_non_negative_number(payload, "dirty_writeback_s"),
                    _optional_non_negative_int(payload, "dirty_writeback_count"),
                    _optional_non_negative_int(payload, "peak_device_bytes"),
                    _optional_non_negative_int(payload, "planned_device_peak_bytes"),
                    _optional_non_negative_int(payload, "observed_device_peak_bytes"),
                    _optional_non_negative_int(payload, "planned_host_peak_bytes"),
                    _optional_non_negative_int(payload, "observed_host_peak_bytes"),
                    _optional_non_negative_int(payload, "planned_cache_peak_bytes"),
                    _optional_non_negative_int(payload, "observed_cache_peak_bytes"),
                    _optional_non_negative_int(payload, "planned_pinned_peak_bytes"),
                    _optional_non_negative_int(payload, "observed_pinned_peak_bytes"),
                    _optional_non_negative_int(payload, "pageable_fallback_count"),
                    _optional_non_negative_int(payload, "pageable_fallback_bytes"),
                    _optional_bool(payload, "full_replica"),
                    _optional_non_negative_number(payload, "wall_s"),
                )
            )
        return ()

    def summary(self):
        local_hv_count = self._local_hv_count
        cache_accesses = self._cache_hits + self._cache_misses
        return {
            "local_hv_count": local_hv_count,
            "local_hv_ir_count": self._local_hv_ir_count,
            "local_hv_legacy_count": self._local_hv_legacy_count,
            "local_hv_fallback_count": self._local_hv_fallback_count,
            "local_hv_wall_s": self._local_hv_wall_s,
            "ir_call_coverage": (
                self._local_hv_ir_count / local_hv_count if local_hv_count else 0.0
            ),
            "grouped_execution_count": self._grouped_execution_count,
            "grouped_task_count": self._grouped_task_count,
            "distributed_hv_count": self._distributed_hv_count,
            "collective_calls": self._collective_calls,
            "collective_bytes": self._collective_bytes,
            "collective_s": self._collective_s,
            "distributed_compute_s": self._distributed_compute_s,
            "synchronization_s": self._synchronization_s,
            "distributed_fallback_count": self._distributed_fallback_count,
            "h2d_bytes": self._h2d_bytes,
            "d2h_bytes": self._d2h_bytes,
            "h2d_s": self._h2d_s,
            "d2h_s": self._d2h_s,
            "dirty_writeback_bytes": self._dirty_writeback_bytes,
            "dirty_writeback_s": self._dirty_writeback_s,
            "dirty_writeback_count": self._dirty_writeback_count,
            "working_set_wall_s": self._working_set_wall_s,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / cache_accesses
            if cache_accesses
            else 0.0,
            "prefetch_overlap_s": self._prefetch_overlap_s,
            "prefetch_wait_s": self._prefetch_wait_s,
            "peak_device_bytes": self._peak_device_bytes,
            "planned_device_peak_bytes": self._planned_device_peak_bytes,
            "observed_device_peak_bytes": self._observed_device_peak_bytes,
            "planned_host_peak_bytes": self._planned_host_peak_bytes,
            "observed_host_peak_bytes": self._observed_host_peak_bytes,
            "planned_cache_peak_bytes": self._planned_cache_peak_bytes,
            "observed_cache_peak_bytes": self._observed_cache_peak_bytes,
            "planned_pinned_peak_bytes": self._planned_pinned_peak_bytes,
            "observed_pinned_peak_bytes": self._observed_pinned_peak_bytes,
            "pageable_fallback_count": self._pageable_fallback_count,
            "pageable_fallback_bytes": self._pageable_fallback_bytes,
            "full_replica_detected": self._full_replica_detected,
            "full_replica_count": self._full_replica_count,
            "event_counts": {
                name: self._event_counts[index]
                for index, name in enumerate(EVENT_NAMES)
            },
        }
