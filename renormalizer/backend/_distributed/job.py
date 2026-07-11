"""Topology-neutral helpers for distributed workflow evidence jobs."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np


SCHEMA = "renormalizer.stage4.workflow.v1"
RANK_SCHEMA = "renormalizer.stage4.workflow.rank.v1"


class _NoAllgatherCollective:
    def __init__(self, collective, solver_options=None):
        self._collective = collective
        self.rank = collective.rank
        self.size = collective.size
        options = {} if solver_options is None else solver_options
        self.solver_options = MappingProxyType(
            {
                str(name): MappingProxyType(dict(values))
                for name, values in options.items()
            }
        )

    def __getattr__(self, name):
        return getattr(self._collective, name)

    def allgather(self, array, *, axis):
        raise RuntimeError("complete-vector allgather is forbidden in Stage 4 jobs")


def _arm_allgather_trap(collective, solver_options=None):
    return _NoAllgatherCollective(collective, solver_options)


def _trapped_execution_config(execution, solver_options=None):
    return replace(
        execution,
        collective=_arm_allgather_trap(execution.collective, solver_options),
    )


def _add_common_arguments(parser):
    parser.add_argument("--mode", choices=("tdvp", "davidson", "both"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--model-size", type=int, default=8)
    parser.add_argument("--bond-dimension", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--precision", type=int, choices=(64,), default=64)
    parser.add_argument("--shard-solver-vectors", action="store_true")
    parser.add_argument(
        "--fallback-policy", choices=("error", "legacy_oe"), default="error"
    )
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--krylov-block-size", type=int, default=50)
    parser.add_argument("--davidson-tol", type=float, default=1e-6)
    parser.add_argument("--davidson-max-cycle", type=int, default=100)
    parser.add_argument("--davidson-max-space", type=int, default=24)
    parser.add_argument("--davidson-lindep", type=float, default=1e-14)
    parser.add_argument("--davidson-sweeps", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--rank-atol", type=float, default=0.0)
    return parser


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _prepare_output(path, runtime):
    path = Path(path)

    def prepare():
        if path.exists() and any(path.iterdir()):
            raise FileExistsError("output directory must be absent or empty")
        path.mkdir(parents=True, exist_ok=True)

    _synchronized_root_action(runtime, prepare, "output setup")
    return path


def _control_array(collective, value):
    cupy = getattr(collective, "_cupy", None)
    if cupy is not None:
        with cupy.cuda.Device(int(collective._device_index)):
            return cupy.asarray([value], dtype=np.int32)
    return np.asarray([value], dtype=np.int32)


def _synchronized_action(runtime, operation, phase, *, root_only=False):
    result = None
    local_error = None
    rank = getattr(runtime, "rank", None)
    if rank is None:
        rank = runtime.context.rank
    if not root_only or rank == 0:
        try:
            result = operation()
        except BaseException as error:
            local_error = error
    status = _control_array(runtime.collective, int(local_error is not None))
    failed = runtime.collective.allreduce(status, op="max")
    failed = int(_host(failed).reshape(-1)[0])
    if failed:
        raise RuntimeError("{} failed".format(phase)) from local_error
    return result


def _synchronized_root_action(runtime, operation, phase):
    return _synchronized_action(runtime, operation, phase, root_only=True)


_FORMAL_TOLERANCE_FIELDS = (
    "atol",
    "rtol",
    "rank_atol",
    "davidson_tol",
    "davidson_lindep",
)


def _validate_job_tolerances(runtime, args):
    def validate():
        for name in _FORMAL_TOLERANCE_FIELDS:
            value = getattr(args, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("{} must be finite and non-negative".format(name))
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "{} must be finite and non-negative".format(name)
                ) from error
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))

    return _synchronized_action(runtime, validate, "formal tolerance setup")


def _run_root_reference(execution, operation, receive_buffer):
    from renormalizer.backend._distributed.local_operator import run_root_fallback

    return run_root_fallback(
        operation,
        execution.collective,
        receive_buffer=receive_buffer,
    )


def _host(value):
    getter = getattr(value, "get", None)
    if callable(getter):
        value = getter()
    return np.asarray(value)


def _profile_distribution(path):
    events = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "distributed_solve_summary":
                events.append(event)
    supported = [
        event
        for event in events
        if _genuine_distributed_hash(event.get("plan_hash"))
        and _genuine_distributed_hash(event.get("placement_hash"))
    ]
    supported_plan_hashes = sorted(
        {str(event["plan_hash"]) for event in supported}
    )
    supported_placement_hashes = sorted(
        {str(event["placement_hash"]) for event in supported}
    )
    return {
        "h_v_count": sum(int(event["hv_count"]) for event in events),
        "broadcast_calls": sum(
            int(event.get("broadcast_calls", 0)) for event in events
        ),
        "allreduce_calls": sum(
            int(event.get("allreduce_calls", 0)) for event in events
        ),
        "allgather_calls": sum(
            int(event.get("allgather_calls", 0)) for event in events
        ),
        "boundary_materialization_broadcasts": sum(
            int(event.get("boundary_materialization_broadcasts", 0))
            for event in events
        ),
        "fallback_count": sum(
            int(event.get("fallback_count", 0)) for event in events
        ),
        "root_operation_count": sum(
            int(event.get("root_operation_count", 0)) for event in events
        ),
        "event_count": len(events),
        "supported_event_count": len(supported),
        "supported_plan_hashes": supported_plan_hashes,
        "supported_placement_hashes": supported_placement_hashes,
        "strictly_sharded": bool(supported)
        and all(
            int(event["local_shard_extent"])
            < int(event["global_shard_extent"])
            for event in supported
        ),
        "plan_hashes": sorted(
            {str(event.get("plan_hash", "")) for event in events}
        ),
        "placement_hashes": sorted(
            {str(event.get("placement_hash", "")) for event in events}
        ),
        "max_solver_residual_norm": _failure_propagating_max(
            [
                float(event.get("solver_residual_norm", 0.0))
                for event in events
            ],
            default=0.0,
        ),
    }


def _genuine_distributed_hash(value):
    return type(value) is str and value not in {"", "fallback"}


_MODE_NORM_ERROR_FIELDS = {
    "tdvp": ("tdvp_norm_error",),
    "davidson": ("davidson_norm_error",),
    "both": ("tdvp_norm_error", "davidson_norm_error"),
}


_MODE_NUMERICAL_FIELDS = {
    "tdvp": ("max_abs_error", "tdvp_norm_error", "norm_error"),
    "davidson": (
        "energy",
        "energy_error",
        "davidson_norm_error",
        "davidson_reference_norm",
        "norm_error",
        "residual_norm",
    ),
    "both": (
        "max_abs_error",
        "energy",
        "energy_error",
        "tdvp_norm_error",
        "davidson_norm_error",
        "davidson_reference_norm",
        "norm_error",
        "residual_norm",
    ),
}


def _required_numerical_fields(mode):
    try:
        return _MODE_NUMERICAL_FIELDS[str(mode)]
    except KeyError as error:
        raise ValueError("unknown formal numerical mode: {!r}".format(mode)) from error


def _required_norm_error_fields(mode):
    try:
        return _MODE_NORM_ERROR_FIELDS[str(mode)]
    except KeyError as error:
        raise ValueError("unknown formal numerical mode: {!r}".format(mode)) from error


def _numerical_values(numerical, fields):
    values = {}
    try:
        for field in fields:
            value = numerical[field]
            if isinstance(value, (bool, np.bool_)):
                return None
            values[field] = float(value)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return values


def _finite_numerical_values(numerical, fields):
    values = _numerical_values(numerical, fields)
    if values is None or not all(
        np.isfinite(value) for value in values.values()
    ):
        return None
    return values


def _failure_propagating_max(values, *, default=None):
    values = tuple(float(value) for value in values)
    if not values:
        if default is None:
            raise ValueError("maximum requires at least one value")
        return float(default)
    if not all(np.isfinite(value) for value in values):
        return float("nan")
    return max(values)


def _aggregate_norm_error(numerical, mode):
    fields = _required_norm_error_fields(mode)
    values = _numerical_values(numerical, fields)
    if values is None:
        raise ValueError("mode-specific norm evidence is missing or nonnumeric")
    return _failure_propagating_max(values.values())


def _numerical_passed(numerical, args):
    fields = _required_numerical_fields(args.mode)
    values = _finite_numerical_values(numerical, fields)
    if values is None:
        return False
    if any(
        value < 0.0
        for field, value in values.items()
        if field != "energy"
    ):
        return False
    tolerance = float(args.atol) + float(args.rtol)
    if values.get("max_abs_error", 0.0) > tolerance:
        return False
    if values.get("energy_error", 0.0) > tolerance:
        return False
    norm_errors = [
        values[field] for field in _required_norm_error_fields(args.mode)
    ]
    if values["norm_error"] != _failure_propagating_max(norm_errors):
        return False
    if any(norm_error > tolerance for norm_error in norm_errors):
        return False
    if values.get("residual_norm", 0.0) > float(args.davidson_tol):
        return False
    return True


def _summary_numerical(records, args):
    fields = _required_numerical_fields(args.mode)
    summary = {}
    complete = bool(records)
    for field in fields:
        values = []
        for record in records:
            numerical = record.get("numerical", {})
            converted = _numerical_values(numerical, (field,))
            if converted is None:
                complete = False
                break
            values.append(converted[field])
        summary[field] = (
            _failure_propagating_max(values)
            if len(values) == len(records) and values
            else None
        )
    summary["passed"] = bool(
        complete
        and all(
            bool(record.get("numerical", {}).get("passed", False))
            and _numerical_passed(record.get("numerical", {}), args)
            for record in records
        )
    )
    return {"passed": summary.pop("passed"), **summary}


def _finalize_profile(runtime, profile_path):
    from renormalizer.utils import profiling

    def flush():
        profiling.record_run_summary()
        profiling.flush_event_output()

    _synchronized_action(runtime, flush, "profile flush")
    return _synchronized_action(
        runtime,
        lambda: _profile_distribution(profile_path),
        "profile parse",
    )


def _validate_rank_records(records, expected, world_size):
    reasons = []
    if len(records) != world_size:
        reasons.append("missing or duplicate rank records")
    if any(record.get("schema") != RANK_SCHEMA for record in records):
        reasons.append("rank record schema mismatch")
    ranks = [record.get("rank") for record in records]
    local_ranks = [record.get("local_rank") for record in records]
    if (
        any(type(rank) is not int for rank in ranks)
        or sorted(ranks) != list(range(world_size))
    ):
        reasons.append("rank record identity mismatch")
    if (
        any(type(rank) is not int for rank in local_ranks)
        or sorted(local_ranks) != list(range(world_size))
    ):
        reasons.append("local rank assignment mismatch")
    if any(
        record.get("run_id") != expected.get("run_id")
        or record.get("world_size") != world_size
        or record.get("git_commit") != expected.get("git_commit")
        for record in records
    ):
        reasons.append("rank run identity mismatch")
    if len({record.get("state_hash") for record in records}) != 1:
        reasons.append("final state hash mismatch")
    if len(
        {json.dumps(record.get("solver"), sort_keys=True) for record in records}
    ) != 1:
        reasons.append("solver control mismatch")
    distribution_keys = (
        "h_v_count",
        "broadcast_calls",
        "allreduce_calls",
        "allgather_calls",
        "boundary_materialization_broadcasts",
        "allgather_trap_armed",
        "strictly_sharded",
        "supported_event_count",
        "supported_plan_hashes",
        "supported_placement_hashes",
        "plan_hashes",
        "placement_hashes",
        "max_solver_residual_norm",
    )
    evidence = {
        json.dumps(
            {
                key: record.get("distribution", {}).get(key)
                for key in distribution_keys
            },
            sort_keys=True,
        )
        for record in records
    }
    if len(evidence) != 1:
        reasons.append("distributed counter or plan mismatch")
    if any(
        type(
            record.get("distribution", {}).get(
                "boundary_materialization_broadcasts"
            )
        )
        is not int
        or record["distribution"]["boundary_materialization_broadcasts"] < 0
        for record in records
    ):
        reasons.append("boundary materialization broadcast evidence invalid")
    if any(
        type(
            record.get("distribution", {}).get("max_solver_residual_norm")
        )
        not in (int, float)
        or isinstance(
            record.get("distribution", {}).get("max_solver_residual_norm"),
            bool,
        )
        or not np.isfinite(
            float(record["distribution"]["max_solver_residual_norm"])
        )
        or float(record["distribution"]["max_solver_residual_norm"]) < 0.0
        for record in records
    ):
        reasons.append("solver residual evidence invalid")
    if any(
        record.get("distribution", {}).get("allgather_calls") != 0
        for record in records
    ):
        reasons.append("complete-vector allgather observed")
    if any(
        not record.get("distribution", {}).get("allgather_trap_armed")
        for record in records
    ):
        reasons.append("allgather trap was not armed")
    if world_size > 1 and any(
        not record.get("distribution", {}).get("strictly_sharded")
        for record in records
    ):
        reasons.append("solver shard was not smaller than the global vector")
    if world_size > 1 and any(
        type(record.get("distribution", {}).get("supported_event_count")) is not int
        or record["distribution"]["supported_event_count"] < 1
        for record in records
    ):
        reasons.append("supported distributed solve evidence missing")

    def has_supported_hash(record, supported_key, observed_key):
        distribution = record.get("distribution", {})
        supported_values = distribution.get(supported_key)
        observed_values = distribution.get(observed_key)
        if not isinstance(supported_values, list) or not isinstance(
            observed_values, list
        ):
            return False
        supported_hashes = {
            value for value in supported_values if _genuine_distributed_hash(value)
        }
        observed_hashes = {
            value for value in observed_values if _genuine_distributed_hash(value)
        }
        return bool(supported_hashes & observed_hashes)

    if world_size > 1 and any(
        not has_supported_hash(record, "supported_plan_hashes", "plan_hashes")
        for record in records
    ):
        reasons.append("supported distributed plan hash evidence missing")
    if world_size > 1 and any(
        not has_supported_hash(
            record, "supported_placement_hashes", "placement_hashes"
        )
        for record in records
    ):
        reasons.append("supported distributed placement hash evidence missing")
    fallback_counts = {
        record.get("fallback", {}).get("count") for record in records
    }
    if len(fallback_counts) != 1:
        reasons.append("fallback count mismatch")
    root_operations = sum(
        int(record.get("fallback", {}).get("root_operation_count", 0))
        for record in records
    )
    fallback_count = next(iter(fallback_counts), 0)
    if root_operations != fallback_count:
        reasons.append("root fallback operation count mismatch")
    root_operations_by_rank = {
        record.get("rank"): int(
            record.get("fallback", {}).get("root_operation_count", 0)
        )
        for record in records
        if type(record.get("rank")) is int
    }
    if (
        root_operations_by_rank.get(0, 0) != fallback_count
        or any(
            count != 0
            for rank, count in root_operations_by_rank.items()
            if rank != 0
        )
    ):
        reasons.append("root fallback operation attribution mismatch")
    return reasons


def _load_rank_records(output_dir):
    records = []
    reasons = []
    for path in sorted(Path(output_dir).glob("rank-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise TypeError("rank record must be a JSON object")
        except (OSError, TypeError, ValueError) as error:
            reasons.append(
                "malformed rank record {}: {}".format(
                    path.name, type(error).__name__
                )
            )
            continue
        records.append(record)
    return records, reasons


def _max_scalar_spread(records):
    numerical = [record.get("numerical", {}) for record in records]
    keys = (
        set().union(*(values.keys() for values in numerical))
        if numerical
        else set()
    )
    maximum = 0.0
    for key in keys:
        values = [record.get(key) for record in numerical]
        if (
            len(values) != len(records)
            or any(type(value) not in (int, float) for value in values)
        ):
            continue
        converted = [float(value) for value in values]
        if not all(np.isfinite(value) for value in converted):
            return float("nan")
        maximum = max(maximum, max(converted) - min(converted))
    return maximum


def _summary_distribution(records, shard_solver_vectors):
    distributions = [record.get("distribution", {}) for record in records]
    supported_counts = [
        int(values.get("supported_event_count", 0)) for values in distributions
    ]
    return {
        "sharded_solver_vectors": bool(shard_solver_vectors),
        "h_v_count": sum(int(values.get("h_v_count", 0)) for values in distributions),
        "broadcast_calls": sum(
            int(values.get("broadcast_calls", 0)) for values in distributions
        ),
        "allreduce_calls": sum(
            int(values.get("allreduce_calls", 0)) for values in distributions
        ),
        "allgather_calls": sum(
            int(values.get("allgather_calls", 0)) for values in distributions
        ),
        "boundary_materialization_broadcasts": sum(
            int(values.get("boundary_materialization_broadcasts", 0))
            for values in distributions
        ),
        "max_solver_residual_norm": _failure_propagating_max(
            [
                values.get("max_solver_residual_norm", 0.0)
                for values in distributions
            ],
            default=0.0,
        ),
        "allgather_trap_armed": bool(distributions)
        and all(bool(values.get("allgather_trap_armed")) for values in distributions),
        "strictly_sharded": bool(distributions)
        and all(bool(values.get("strictly_sharded")) for values in distributions),
        "supported_event_count": min(supported_counts, default=0),
        "supported_plan_hashes": sorted(
            {
                str(plan_hash)
                for values in distributions
                for plan_hash in values.get("supported_plan_hashes", ())
            }
        ),
        "supported_placement_hashes": sorted(
            {
                str(placement_hash)
                for values in distributions
                for placement_hash in values.get(
                    "supported_placement_hashes", ()
                )
            }
        ),
        "plan_hashes": sorted(
            {
                str(plan_hash)
                for values in distributions
                for plan_hash in values.get("plan_hashes", ())
            }
        ),
        "placement_hashes": sorted(
            {
                str(placement_hash)
                for values in distributions
                for placement_hash in values.get("placement_hashes", ())
            }
        ),
    }
