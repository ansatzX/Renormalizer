import math

import numpy as np
import pytest

from renormalizer.backend._distributed.profiling import (
    distributed_solve_payload,
    working_set_payload,
)
from renormalizer.utils._profiling.aggregate import RunAggregate
from renormalizer.utils._profiling.events import MAX_INTEGER_DIGITS


def test_distributed_solve_payload_includes_bounded_cost_metadata():
    payload = distributed_solve_payload(
        global_shape=tuple(range(1, 21)),
        local_shape=(1,) + tuple(range(2, 20)) + (10,),
        sharding_axis=19,
        hv_count=7,
        collective_calls=3,
        collective_bytes=4096,
        collective_s=0.25,
        compute_s=1.5,
        synchronization_s=0.125,
        fallback_count=1,
    )

    assert payload == {
        "global_shape": list(range(1, 17)),
        "local_shape": list(range(1, 17)),
        "shape_rank": 20,
        "shapes_truncated": True,
        "sharding_axis": 19,
        "global_shard_extent": 20,
        "local_shard_extent": 10,
        "hv_count": 7,
        "collective_calls": 3,
        "collective_bytes": 4096,
        "collective_s": 0.25,
        "compute_s": 1.5,
        "synchronization_s": 0.125,
        "fallback_count": 1,
    }


@pytest.mark.parametrize("local_extent", [3, 2])
def test_distributed_solve_payload_accepts_uneven_rank_local_extents(local_extent):
    payload = distributed_solve_payload(
        global_shape=(4, 10),
        local_shape=(4, local_extent),
        sharding_axis=1,
        hv_count=1,
        collective_calls=0,
        collective_bytes=0,
        collective_s=0.0,
        compute_s=0.1,
        synchronization_s=0.0,
        fallback_count=0,
    )

    assert payload["global_shard_extent"] == 10
    assert payload["local_shard_extent"] == local_extent


def test_distributed_solve_payload_validates_all_shape_values_before_bounding():
    invalid_global = tuple(range(1, 20)) + (np.int64(20),)
    with pytest.raises(TypeError, match="shape dimensions"):
        distributed_solve_payload(
            global_shape=invalid_global,
            local_shape=tuple(range(1, 21)),
            sharding_axis=19,
            hv_count=1,
            collective_calls=0,
            collective_bytes=0,
            collective_s=0.0,
            compute_s=0.0,
            synchronization_s=0.0,
            fallback_count=0,
        )


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"global_shape": np.array([4, 8])}, TypeError),
        ({"hv_count": np.int64(1)}, TypeError),
        ({"collective_s": np.float64(0.1)}, TypeError),
        ({"sharding_axis": True}, TypeError),
        ({"fallback_count": -1}, ValueError),
        ({"compute_s": math.inf}, ValueError),
        ({"global_shape": (4, 8), "local_shape": (4,)}, ValueError),
        ({"global_shape": (4, 8), "local_shape": (2, 8), "sharding_axis": 1}, ValueError),
        ({"global_shape": (4, 8), "local_shape": (4, 9)}, ValueError),
        ({"sharding_axis": 2}, ValueError),
    ],
)
def test_distributed_solve_payload_rejects_malformed_metadata(updates, error):
    kwargs = {
        "global_shape": (4, 8),
        "local_shape": (4, 2),
        "sharding_axis": 1,
        "hv_count": 1,
        "collective_calls": 0,
        "collective_bytes": 0,
        "collective_s": 0.0,
        "compute_s": 0.0,
        "synchronization_s": 0.0,
        "fallback_count": 0,
    }
    kwargs.update(updates)

    with pytest.raises(error):
        distributed_solve_payload(**kwargs)


def test_working_set_payload_includes_residency_and_dirty_writeback_costs():
    payload = working_set_payload(
        h2d_bytes=100,
        d2h_bytes=20,
        h2d_s=0.1,
        d2h_s=0.2,
        cache_hits=8,
        cache_misses=2,
        prefetch_overlap_s=0.3,
        prefetch_wait_s=0.4,
        dirty_writeback_bytes=12,
        dirty_writeback_s=0.05,
        peak_device_bytes=2048,
        full_replica=True,
        wall_s=0.75,
    )

    assert payload == {
        "h2d_bytes": 100,
        "d2h_bytes": 20,
        "h2d_s": 0.1,
        "d2h_s": 0.2,
        "cache_hits": 8,
        "cache_misses": 2,
        "prefetch_overlap_s": 0.3,
        "prefetch_wait_s": 0.4,
        "dirty_writeback_bytes": 12,
        "dirty_writeback_s": 0.05,
        "peak_device_bytes": 2048,
        "full_replica": True,
        "wall_s": 0.75,
    }


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("h2d_bytes", np.int64(1), TypeError),
        ("d2h_s", np.float64(0.1), TypeError),
        ("full_replica", np.bool_(True), TypeError),
        ("peak_device_bytes", np.array(1), TypeError),
        ("cache_misses", -1, ValueError),
        ("dirty_writeback_s", math.nan, ValueError),
    ],
)
def test_working_set_payload_rejects_non_python_or_negative_metadata(field, value, error):
    kwargs = {
        "h2d_bytes": 1,
        "d2h_bytes": 1,
        "h2d_s": 0.1,
        "d2h_s": 0.1,
        "cache_hits": 1,
        "cache_misses": 1,
        "prefetch_overlap_s": 0.1,
        "prefetch_wait_s": 0.1,
        "dirty_writeback_bytes": 1,
        "dirty_writeback_s": 0.1,
        "peak_device_bytes": 1,
        "full_replica": False,
    }
    kwargs[field] = value

    with pytest.raises(error):
        working_set_payload(**kwargs)


def test_run_aggregate_has_fixed_schema_and_exact_arithmetic():
    aggregate = RunAggregate()
    aggregate.add({"event": "run_start", "backend": "numpy"})
    aggregate.add({"event": "phase_summary", "wall_s": 0.5})
    aggregate.add(
        {
            "event": "local_hv_execute",
            "actual_policy": "execution_ir",
            "wall_s": 2.0,
            "fallback": None,
        }
    )
    aggregate.add(
        {
            "event": "local_hv_execute",
            "actual_policy": "legacy_oe",
            "wall_s": 1.0,
            "fallback": "legacy_oe",
        }
    )
    aggregate.add(
        {
            "event": "grouped_gemm_execute",
            "grouped_execution": True,
            "task_count": 6,
        }
    )
    aggregate.add(
        {
            "event": "distributed_solve_summary",
            "hv_count": 5,
            "collective_calls": 4,
            "collective_bytes": 1000,
            "collective_s": 0.4,
            "compute_s": 3.0,
            "synchronization_s": 0.2,
            "fallback_count": 2,
        }
    )
    aggregate.add(
        {
            "event": "working_set_transfer",
            "h2d_bytes": 100,
            "d2h_bytes": 20,
            "h2d_s": 0.1,
            "d2h_s": 0.2,
            "cache_hits": 8,
            "cache_misses": 2,
            "prefetch_overlap_s": 0.3,
            "prefetch_wait_s": 0.4,
            "dirty_writeback_bytes": 12,
            "dirty_writeback_s": 0.05,
            "peak_device_bytes": 2048,
            "full_replica": False,
            "wall_s": 0.75,
        }
    )
    aggregate.add(
        {
            "event": "working_set_transfer",
            "h2d_bytes": 50,
            "d2h_bytes": 10,
            "h2d_s": 0.05,
            "d2h_s": 0.1,
            "cache_hits": 1,
            "cache_misses": 1,
            "prefetch_overlap_s": 0.2,
            "prefetch_wait_s": 0.1,
            "dirty_writeback_bytes": 8,
            "dirty_writeback_s": 0.025,
            "peak_device_bytes": 1024,
            "full_replica": True,
        }
    )

    summary = aggregate.summary()

    assert summary == {
        "local_hv_count": 2,
        "local_hv_ir_count": 1,
        "local_hv_legacy_count": 1,
        "local_hv_fallback_count": 1,
        "local_hv_wall_s": 3.0,
        "ir_call_coverage": 0.5,
        "grouped_execution_count": 1,
        "grouped_task_count": 6,
        "distributed_hv_count": 5,
        "collective_calls": 4,
        "collective_bytes": 1000,
        "collective_s": 0.4,
        "distributed_compute_s": 3.0,
        "synchronization_s": 0.2,
        "distributed_fallback_count": 2,
        "h2d_bytes": 150,
        "d2h_bytes": 30,
        "h2d_s": pytest.approx(0.15),
        "d2h_s": pytest.approx(0.3),
        "dirty_writeback_bytes": 20,
        "dirty_writeback_s": pytest.approx(0.075),
        "working_set_wall_s": 0.75,
        "cache_hits": 9,
        "cache_misses": 3,
        "cache_hit_rate": 0.75,
        "prefetch_overlap_s": 0.5,
        "prefetch_wait_s": 0.5,
        "peak_device_bytes": 2048,
        "full_replica_detected": True,
        "full_replica_count": 1,
        "event_counts": {
            "run_start": 1,
            "phase_summary": 1,
            "local_hv_execute": 2,
            "grouped_gemm_execute": 1,
            "distributed_solve_summary": 1,
            "working_set_transfer": 2,
            "run_summary": 0,
        },
    }
    assert aggregate.summary() == summary
    aggregate.add({"event": "run_summary"})
    summary_after_event = aggregate.summary()
    assert summary_after_event["event_counts"]["run_summary"] == 1
    assert {key: value for key, value in summary_after_event.items() if key != "event_counts"} == {
        key: value for key, value in summary.items() if key != "event_counts"
    }
    assert len(RunAggregate().summary()) == len(summary)
    assert RunAggregate().summary()["ir_call_coverage"] == 0.0
    assert RunAggregate().summary()["cache_hit_rate"] == 0.0


def test_run_aggregate_accepts_brief_minimal_working_set_projection():
    aggregate = RunAggregate()

    aggregate.add(
        {
            "event": "working_set_transfer",
            "h2d_bytes": 100,
            "d2h_bytes": 20,
            "wall_s": 0.5,
        }
    )

    summary = aggregate.summary()
    assert summary["h2d_bytes"] == 100
    assert summary["d2h_bytes"] == 20
    assert summary["working_set_wall_s"] == 0.5
    assert summary["h2d_s"] == 0.0
    assert summary["dirty_writeback_bytes"] == 0


def test_run_aggregate_prepare_does_not_mutate_until_commit():
    aggregate = RunAggregate()
    before = aggregate.summary()

    delta = aggregate.prepare(
        {
            "event": "local_hv_execute",
            "actual_policy": "execution_ir",
            "wall_s": 1.5,
        }
    )

    assert aggregate.summary() == before
    aggregate.commit(delta)
    assert aggregate.summary()["local_hv_count"] == 1
    assert aggregate.summary()["local_hv_wall_s"] == 1.5


def test_prepared_delta_is_owned_single_use_and_preview_is_non_consuming():
    aggregate = RunAggregate()
    other = RunAggregate()
    before = aggregate.summary()
    other_before = other.summary()
    delta = aggregate.prepare(
        {
            "event": "local_hv_execute",
            "actual_policy": "execution_ir",
            "wall_s": 1.5,
        }
    )

    with pytest.raises(ValueError, match="origin"):
        other.preview(delta)
    with pytest.raises(ValueError, match="origin"):
        other.commit(delta)
    preview = aggregate.preview(delta)

    assert preview["local_hv_count"] == 1
    assert aggregate.summary() == before
    assert other.summary() == other_before
    aggregate.commit(delta)
    with pytest.raises(ValueError, match="committed"):
        aggregate.commit(delta)


def test_prepared_delta_rejects_stale_version_after_another_commit():
    aggregate = RunAggregate()
    first = aggregate.prepare({"event": "run_start"})
    stale = aggregate.prepare({"event": "phase_summary", "wall_s": 0.5})

    aggregate.commit(first)

    with pytest.raises(ValueError, match="stale"):
        aggregate.preview(stale)
    with pytest.raises(ValueError, match="stale"):
        aggregate.commit(stale)
    assert aggregate.summary()["event_counts"]["phase_summary"] == 0


def test_prepare_rejects_two_event_integer_overflow_without_mutation():
    aggregate = RunAggregate()
    largest = 10**MAX_INTEGER_DIGITS - 1
    aggregate.add({"event": "working_set_transfer", "h2d_bytes": largest})
    before = aggregate.summary()

    with pytest.raises(ValueError, match="integer"):
        aggregate.prepare({"event": "working_set_transfer", "h2d_bytes": 1})

    assert aggregate.summary() == before


def test_prepare_rejects_float_sum_to_infinity_without_mutation():
    aggregate = RunAggregate()
    aggregate.add(
        {
            "event": "local_hv_execute",
            "actual_policy": "legacy_oe",
            "wall_s": 1e308,
        }
    )
    before = aggregate.summary()

    with pytest.raises(ValueError, match="finite"):
        aggregate.prepare(
            {
                "event": "local_hv_execute",
                "actual_policy": "legacy_oe",
                "wall_s": 1e308,
            }
        )

    assert aggregate.summary() == before


def test_prepare_rejects_event_count_overflow_without_mutation():
    aggregate = RunAggregate()
    aggregate._event_counts[0] = 10**MAX_INTEGER_DIGITS - 1
    before = aggregate.summary()

    with pytest.raises(ValueError, match="integer"):
        aggregate.prepare({"event": "run_start"})

    assert aggregate.summary() == before


@pytest.mark.parametrize(
    "event,error",
    [
        ({"event": "unknown"}, ValueError),
        ({"event": np.str_("run_start")}, TypeError),
        ({"event": "local_hv_execute", "actual_policy": "execution_ir"}, ValueError),
        ({"event": "local_hv_execute", "actual_policy": "bad", "wall_s": 1.0}, ValueError),
        (
            {
                "event": "grouped_gemm_execute",
                "grouped_execution": True,
                "task_count": 1,
            },
            ValueError,
        ),
        (
            {
                "event": "distributed_solve_summary",
                "hv_count": 1,
                "collective_calls": 0,
                "collective_bytes": 0,
                "collective_s": 0.0,
                "compute_s": 0.0,
                "synchronization_s": 0.0,
                "fallback_count": 2,
            },
            ValueError,
        ),
        ({"event": "working_set_transfer", "h2d_bytes": np.int64(1)}, TypeError),
        ({"event": "run_start", "values": np.array([1])}, TypeError),
        ([], TypeError),
    ],
)
def test_run_aggregate_rejects_unknown_or_malformed_events(event, error):
    aggregate = RunAggregate()
    before = aggregate.summary()

    with pytest.raises(error):
        aggregate.add(event)

    assert aggregate.summary() == before
