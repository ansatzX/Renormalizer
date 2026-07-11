import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np
import pytest

from renormalizer.backend._distributed.profiling import (
    distributed_solve_payload,
    working_set_payload,
)
from renormalizer.backend._execution.profiling import local_hv_payload, phase_summary_payload
from renormalizer.backend._gemm.profiling import grouped_gemm_payload
from renormalizer.utils import profiling
from renormalizer.utils._profiling.events import (
    EVENT_NAMES,
    MAX_CONTAINER_ITEMS,
    MAX_INTEGER_DIGITS,
    MAX_KEY_BYTES,
    MAX_STRING_BYTES,
)
from renormalizer.utils._profiling.jsonl import JsonlEventWriter
from renormalizer.utils.log import DEBUG, PROFILING, init_log


@pytest.fixture(autouse=True)
def reset_profiling_runtime(monkeypatch):
    profiling.close_event_output()
    monkeypatch.setattr(profiling, "_runtime", None)
    init_log(DEBUG)
    yield
    profiling.close_event_output()
    init_log(DEBUG)


def _run_start_payload(**updates):
    payload = {"backend": "numpy", "rank": 0, "world_size": 1}
    payload.update(updates)
    return payload


def _local_payload():
    return local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=((2, 3, 4), (3, 5, 4, 6), (7, 6, 8), (4, 4, 8)),
        output_shape=(2, 5, 7),
        requested_policy="legacy_oe",
        actual_policy="legacy_oe",
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=("gemm", "tensordot", "tensordot"),
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
    )


def _distributed_payload():
    return distributed_solve_payload(
        global_shape=(4, 10),
        local_shape=(4, 3),
        sharding_axis=1,
        hv_count=2,
        collective_calls=1,
        collective_bytes=128,
        collective_s=0.1,
        compute_s=0.5,
        synchronization_s=0.05,
        fallback_count=0,
    )


def _working_set_payload():
    return working_set_payload(
        h2d_bytes=100,
        d2h_bytes=20,
        h2d_s=0.1,
        d2h_s=0.2,
        cache_hits=8,
        cache_misses=2,
        prefetch_overlap_s=0.05,
        prefetch_wait_s=0.01,
        dirty_writeback_bytes=10,
        dirty_writeback_s=0.02,
        peak_device_bytes=1024,
        full_replica=False,
        wall_s=0.5,
    )


def test_disabled_record_does_not_initialize_runtime(monkeypatch):
    init_log(DEBUG)
    called = False

    def fail():
        nonlocal called
        called = True
        raise AssertionError("runtime initialized")

    monkeypatch.setattr(profiling, "_load_runtime", fail)

    assert profiling.enabled() is False
    profiling.record("run_start", backend="numpy")
    assert called is False


def test_jsonl_output_contains_bounded_event(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"

    profiling.register_event_output(path)
    profiling.record("run_start", **_run_start_payload())
    profiling.close_event_output()

    payload = json.loads(path.read_text().splitlines()[0])
    assert payload == {
        "event": "run_start",
        "backend": "numpy",
        "rank": 0,
        "world_size": 1,
    }


def test_event_filter_and_nested_scope_precedence_are_deterministic(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"

    profiling.register_event_output(path, events={"run_start"})
    with profiling.scope(backend="numpy", rank=0, world_size=2):
        with profiling.scope(rank=1, stage="evolve"):
            profiling.record("run_start", backend="cupy")
            profiling.record(
                "phase_summary",
                **phase_summary_payload(
                    phase="optimization_sweep",
                    network="mps",
                    operation="dmrg_sweep",
                    operation_count=1,
                    wall_s=0.1,
                ),
            )
    profiling.close_event_output()

    payloads = [json.loads(line) for line in path.read_text().splitlines()]
    assert payloads == [
        {
            "event": "run_start",
                "backend": "cupy",
                "rank": 1,
                "world_size": 2,
                "stage": "evolve",
        }
    ]


def test_jsonl_converts_numpy_scalars_and_rejects_arrays(tmp_path):
    class ScalarTensor:
        shape = ()

        def item(self):
            return 1

    init_log(PROFILING)
    path = tmp_path / "events.jsonl"

    profiling.register_event_output(path)
    profiling.record(
        "run_start",
        **_run_start_payload(attempt=np.int64(3), ratio=np.float32(0.5)),
    )
    with pytest.raises(TypeError, match="array"):
        profiling.record("run_start", **_run_start_payload(values=np.array([1, 2])))
    with pytest.raises(TypeError, match="array"):
        profiling.record("run_start", **_run_start_payload(values=ScalarTensor()))
    profiling.close_event_output()

    payload = json.loads(path.read_text().splitlines()[0])
    assert payload == {
        "event": "run_start",
        "backend": "numpy",
        "rank": 0,
        "world_size": 1,
        "attempt": 3,
        "ratio": 0.5,
    }


def test_default_event_output_name_contains_timestamp_pid_and_rank(tmp_path, monkeypatch):
    init_log(PROFILING)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANK", "7")

    path = profiling.register_event_output()
    profiling.close_event_output()

    assert path.parent == tmp_path
    assert re.fullmatch(
        rf"profiling-\d{{8}}-\d{{6}}-p{os.getpid()}-r7\.jsonl", path.name
    )


def test_disabled_aggregate_apis_do_not_load_or_build_summary(monkeypatch):
    init_log(DEBUG)
    aggregate_module = "renormalizer.utils._profiling.aggregate"
    sys.modules.pop(aggregate_module, None)

    def fail():
        raise AssertionError("aggregate initialized while profiling is disabled")

    monkeypatch.setattr(profiling, "_load_run_aggregate", fail)

    profiling.record("run_start", backend="numpy")
    profiling.record_run_summary()

    assert aggregate_module not in sys.modules


def test_register_resets_run_aggregate_and_summary_is_explicit(tmp_path):
    init_log(PROFILING)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    profiling.register_event_output(first_path)
    profiling.record("local_hv_execute", **_local_payload())
    assert profiling.run_summary()["local_hv_count"] == 1

    profiling.register_event_output(second_path)
    assert profiling.run_summary()["local_hv_count"] == 0
    profiling.record("run_start", **_run_start_payload())
    profiling.close_event_output()

    assert [json.loads(line)["event"] for line in first_path.read_text().splitlines()] == [
        "local_hv_execute"
    ]
    assert [json.loads(line)["event"] for line in second_path.read_text().splitlines()] == [
        "run_start"
    ]


def test_record_run_summary_writes_once_without_aggregating_itself(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"

    profiling.register_event_output(path)
    profiling.record("run_start", **_run_start_payload())
    first = profiling.record_run_summary()
    second = profiling.record_run_summary()
    profiling.close_event_output()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "run_start",
        "run_summary",
        "run_summary",
    ]
    assert first["event_counts"]["run_summary"] == 1
    assert second["event_counts"]["run_summary"] == 2
    assert events[1] == {"event": "run_summary", "context": {}, **first}
    assert events[2] == {"event": "run_summary", "context": {}, **second}
    assert profiling.run_summary()["event_counts"]["run_summary"] == 2


def test_run_summary_captures_bounded_nested_scope_without_metric_override(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "summary.jsonl"
    profiling.register_event_output(path, events={"run_summary"})

    with profiling.scope(
        backend="cupy",
        device="cuda:0",
        rank=0,
        world_size=2,
        policy="legacy_oe",
        local_hv_count=999,
    ):
        with profiling.scope(rank=1, policy="execution_ir"):
            returned = profiling.record_run_summary()
    profiling.close_event_output()

    event = json.loads(path.read_text())
    assert event["context"] == {
        "backend": "cupy",
        "device": "cuda:0",
        "rank": 1,
        "world_size": 2,
        "policy": "execution_ir",
        "local_hv_count": 999,
    }
    assert event["local_hv_count"] == 0
    assert returned == profiling.run_summary()
    assert "context" not in returned


def test_invalid_summary_scope_does_not_increment_or_write(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "summary.jsonl"
    profiling.register_event_output(path, events={"run_summary"})

    with profiling.scope(note="x" * (MAX_STRING_BYTES + 1)):
        with pytest.raises(ValueError, match="UTF-8"):
            profiling.record_run_summary()

    profiling.close_event_output()
    assert path.read_text() == ""
    assert profiling.run_summary()["event_counts"]["run_summary"] == 0


def test_summary_count_tracks_accepted_calls_when_filtered_or_writer_absent(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "filtered.jsonl"
    profiling.register_event_output(path, events={"run_start"})

    filtered = profiling.record_run_summary()
    profiling.close_event_output()
    without_writer = profiling.record_run_summary()

    assert path.read_text() == ""
    assert filtered["event_counts"]["run_summary"] == 1
    assert without_writer["event_counts"]["run_summary"] == 2
    assert profiling.run_summary()["event_counts"]["run_summary"] == 2


def test_event_filter_does_not_filter_aggregate_evidence(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "summary-only.jsonl"

    profiling.register_event_output(path, events={"run_summary"})
    profiling.record("local_hv_execute", **_local_payload())
    profiling.record_run_summary()
    profiling.close_event_output()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "run_summary"
    assert events[0]["local_hv_count"] == 1
    assert events[0]["ir_call_coverage"] == 0.0


def test_reset_run_aggregate_is_explicit_and_fixed_size(tmp_path):
    init_log(PROFILING)
    profiling.register_event_output(tmp_path / "events.jsonl")
    baseline = profiling.run_summary()
    profiling.record(
        "phase_summary",
        **phase_summary_payload(
            phase="optimization_sweep",
            network="mps",
            operation="dmrg_sweep",
            operation_count=1,
            wall_s=0.5,
        ),
    )

    reset = profiling.reset_run_aggregate()

    assert reset == baseline
    assert profiling.run_summary() == baseline


def test_event_names_are_canonical_and_invalid_filter_preserves_writer(tmp_path):
    assert EVENT_NAMES == (
        "run_start",
        "phase_summary",
        "local_hv_execute",
        "grouped_gemm_execute",
        "distributed_solve_summary",
        "working_set_transfer",
        "run_summary",
    )
    init_log(PROFILING)
    first_path = tmp_path / "first.jsonl"
    replacement_path = tmp_path / "replacement.jsonl"
    profiling.register_event_output(first_path, events={"run_start"})
    profiling.record("run_start", **_run_start_payload())

    with pytest.raises(ValueError, match="unsupported"):
        profiling.register_event_output(replacement_path, events={"run_finish"})

    profiling.record("run_start", **_run_start_payload())
    profiling.close_event_output()
    assert len(first_path.read_text().splitlines()) == 2
    assert not replacement_path.exists()


def test_writer_replacement_open_failure_preserves_prior_writer(tmp_path):
    init_log(PROFILING)
    first_path = tmp_path / "first.jsonl"
    replacement_path = tmp_path / "missing" / "replacement.jsonl"
    profiling.register_event_output(first_path, events={"run_start"})
    profiling.record("run_start", **_run_start_payload())

    with pytest.raises(FileNotFoundError):
        profiling.register_event_output(replacement_path, events={"run_start"})

    profiling.record("run_start", **_run_start_payload())
    profiling.close_event_output()
    assert len(first_path.read_text().splitlines()) == 2
    assert not replacement_path.exists()


def test_failed_writer_replacement_cleans_new_artifact_and_preserves_prior(
    tmp_path, monkeypatch
):
    init_log(PROFILING)
    first_path = tmp_path / "first.jsonl"
    replacement_path = tmp_path / "replacement.jsonl"
    profiling.register_event_output(first_path, events={"run_start"})
    profiling.record("run_start", **_run_start_payload())
    real_open = Path.open

    def fail_after_create(path, *args, **kwargs):
        if path == replacement_path:
            descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(descriptor)
            raise PermissionError("simulated open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_after_create)

    with pytest.raises(PermissionError, match="simulated"):
        profiling.register_event_output(replacement_path, events={"run_start"})

    profiling.record("run_start", **_run_start_payload())
    profiling.close_event_output()
    assert len(first_path.read_text().splitlines()) == 2
    assert not replacement_path.exists()


@pytest.mark.parametrize("destination_existed", [False, True])
def test_writer_replacement_close_failure_rolls_back_to_prior_writer(
    tmp_path, monkeypatch, destination_existed
):
    init_log(PROFILING)
    first_path = tmp_path / "first.jsonl"
    replacement_path = tmp_path / "replacement.jsonl"
    if destination_existed:
        replacement_path.write_text("existing\n")
    profiling.register_event_output(first_path, events={"run_start"})
    profiling.record("run_start", **_run_start_payload())
    runtime = profiling._load_runtime()
    previous = runtime._event_output
    real_close = previous.close

    def fail_close():
        raise OSError("simulated close failure")

    monkeypatch.setattr(previous, "close", fail_close)
    with pytest.raises(OSError, match="simulated close failure"):
        profiling.register_event_output(replacement_path, events={"run_start"})

    published = runtime._event_output
    monkeypatch.setattr(previous, "close", real_close)
    if published is previous:
        profiling.record("run_start", **_run_start_payload())
        profiling.close_event_output()
    else:
        published.close()
        runtime._event_output = None
        previous.close()

    assert published is previous
    assert len(first_path.read_text().splitlines()) == 2
    assert replacement_path.exists() is destination_existed


def test_public_record_rejects_direct_run_summary(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    before = profiling.run_summary()

    with pytest.raises(ValueError, match="record_run_summary"):
        profiling.record("run_summary", **before)

    profiling.close_event_output()
    assert path.read_text() == ""
    assert profiling.run_summary() == before


@pytest.mark.parametrize(
    "event,payload",
    [
        ("run_start", {"backend": "numpy", "rank": 0}),
        ("phase_summary", {"wall_s": 0.1}),
        ("local_hv_execute", {"actual_policy": "legacy_oe", "wall_s": 0.1}),
        (
            "grouped_gemm_execute",
            {"operation": "gemm", "task_count": 2, "grouped_execution": True},
        ),
        (
            "distributed_solve_summary",
            {key: value for key, value in _distributed_payload().items() if key != "local_shape"},
        ),
        (
            "working_set_transfer",
            {"h2d_bytes": 100, "d2h_bytes": 20, "wall_s": 0.5},
        ),
    ],
)
def test_public_record_rejects_noncanonical_raw_source_events(tmp_path, event, payload):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    before = profiling.run_summary()

    with pytest.raises((TypeError, ValueError)):
        profiling.record(event, **payload)

    profiling.close_event_output()
    assert path.read_text() == ""
    assert profiling.run_summary() == before


def test_public_record_accepts_all_canonical_source_schemas(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    profiling.record("run_start", **_run_start_payload())
    profiling.record(
        "phase_summary",
        **phase_summary_payload(
            phase="optimization_sweep",
            network="mps",
            operation="dmrg_sweep",
            operation_count=1,
            wall_s=0.1,
        ),
    )
    profiling.record("local_hv_execute", **_local_payload())
    profiling.record(
        "grouped_gemm_execute",
        **grouped_gemm_payload(
            operation="gemm",
            task_count=2,
            shape_buckets={(2, 3, 4): 2},
            executed_grouped=True,
        ),
    )
    profiling.record("distributed_solve_summary", **_distributed_payload())
    profiling.record("working_set_transfer", **_working_set_payload())
    profiling.close_event_output()

    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == list(
        EVENT_NAMES[:-1]
    )


def test_writer_failure_leaves_source_aggregate_unchanged(tmp_path, monkeypatch):
    init_log(PROFILING)
    profiling.register_event_output(tmp_path / "events.jsonl")
    runtime = profiling._load_runtime()
    before = profiling.run_summary()

    def fail_write(payload):
        raise OSError("disk full")

    monkeypatch.setattr(runtime, "write_prepared", fail_write)

    with pytest.raises(OSError, match="disk full"):
        profiling.record("working_set_transfer", **_working_set_payload())

    assert profiling.run_summary() == before


def test_public_source_and_summary_transactions_are_serialized(tmp_path, monkeypatch):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    runtime = profiling._load_runtime()
    original_write = runtime.write_prepared
    start = threading.Barrier(3)
    state_guard = threading.Lock()
    active_writes = 0
    max_active_writes = 0
    errors = []

    def slow_write(payload):
        nonlocal active_writes, max_active_writes
        with state_guard:
            active_writes += 1
            max_active_writes = max(max_active_writes, active_writes)
        try:
            time.sleep(0.05)
            original_write(payload)
        finally:
            with state_guard:
                active_writes -= 1

    monkeypatch.setattr(runtime, "write_prepared", slow_write)

    def run(operation):
        start.wait()
        try:
            operation()
        except Exception as error:
            errors.append(error)

    phase = phase_summary_payload(
        phase="optimization_sweep",
        network="mps",
        operation="dmrg_sweep",
        operation_count=1,
        wall_s=0.5,
    )
    source_thread = threading.Thread(
        target=run,
        args=(lambda: profiling.record("phase_summary", **phase),),
    )
    summary_thread = threading.Thread(target=run, args=(profiling.record_run_summary,))
    source_thread.start()
    summary_thread.start()
    start.wait()
    source_thread.join()
    summary_thread.join()
    profiling.close_event_output()

    assert errors == []
    assert max_active_writes == 1
    assert len(path.read_text().splitlines()) == 2
    summary = profiling.run_summary()
    assert summary["event_counts"]["phase_summary"] == 1
    assert summary["event_counts"]["run_summary"] == 1


@pytest.mark.parametrize("outer", ["record", "summary"])
def test_reentrant_public_recording_is_rejected_before_nested_write(
    tmp_path, monkeypatch, outer
):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    runtime = profiling._load_runtime()
    original_write = runtime.write_prepared
    nested_errors = []
    triggered = False
    phase = phase_summary_payload(
        phase="optimization_sweep",
        network="mps",
        operation="dmrg_sweep",
        operation_count=1,
        wall_s=0.5,
    )

    def reentrant_write(payload):
        nonlocal triggered
        if not triggered:
            triggered = True
            nested = (
                profiling.record_run_summary
                if outer == "record"
                else lambda: profiling.record("phase_summary", **phase)
            )
            try:
                nested()
            except RuntimeError as error:
                nested_errors.append(error)
        original_write(payload)

    monkeypatch.setattr(runtime, "write_prepared", reentrant_write)

    if outer == "record":
        profiling.record("phase_summary", **phase)
    else:
        profiling.record_run_summary()
    profiling.close_event_output()

    assert len(nested_errors) == 1
    assert "reentrant" in str(nested_errors[0])
    assert len(path.read_text().splitlines()) == 1
    summary = profiling.run_summary()
    assert summary["event_counts"]["phase_summary"] == (outer == "record")
    assert summary["event_counts"]["run_summary"] == (outer == "summary")


def test_prepare_overflow_rejects_public_record_before_write(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    largest = 10**MAX_INTEGER_DIGITS - 1
    first = _working_set_payload()
    first["h2d_bytes"] = largest
    profiling.record("working_set_transfer", **first)
    before = profiling.run_summary()
    overflowing = _working_set_payload()
    overflowing["h2d_bytes"] = 1

    with pytest.raises(ValueError, match="integer"):
        profiling.record("working_set_transfer", **overflowing)

    profiling.close_event_output()
    assert len(path.read_text().splitlines()) == 1
    assert profiling.run_summary() == before


def test_invalid_scoped_metadata_leaves_output_and_aggregate_unchanged(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    before = profiling.run_summary()

    with profiling.scope(values=np.array([1, 2])):
        with pytest.raises(TypeError, match="array"):
            profiling.record("run_start", **_run_start_payload())

    profiling.close_event_output()
    assert path.read_text() == ""
    assert profiling.run_summary() == before


def test_failed_summary_write_does_not_increment_summary_count(tmp_path, monkeypatch):
    init_log(PROFILING)
    profiling.register_event_output(tmp_path / "events.jsonl")
    runtime = profiling._load_runtime()

    def fail_write(payload):
        raise OSError("disk full")

    monkeypatch.setattr(runtime, "write_prepared", fail_write)

    with pytest.raises(OSError, match="disk full"):
        profiling.record_run_summary()

    assert profiling.run_summary()["event_counts"]["run_summary"] == 0


class _WriteSpy:
    def __init__(self):
        self.calls = []

    def write(self, value):
        self.calls.append(value)

    def flush(self):
        pass

    def close(self):
        pass


def test_jsonl_serializes_before_one_stream_write(tmp_path):
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    writer._stream.close()
    stream = _WriteSpy()
    writer._stream = stream

    writer.write("run_start", {"event": "run_start", **_run_start_payload()})

    assert len(stream.calls) == 1
    assert stream.calls[0].endswith("\n")
    assert json.loads(stream.calls[0])["event"] == "run_start"


def test_jsonl_serialization_failure_performs_no_stream_write(tmp_path):
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    writer._stream.close()
    stream = _WriteSpy()
    writer._stream = stream

    with pytest.raises(TypeError, match="array"):
        writer.write("run_start", {"event": "run_start", "values": np.array([1])})

    assert stream.calls == []


@pytest.mark.parametrize(
    "extra,error",
    [
        ({"note": "x" * (MAX_STRING_BYTES + 1)}, ValueError),
        ({"k" * (MAX_KEY_BYTES + 1): 1}, ValueError),
        ({"attempt": 10**MAX_INTEGER_DIGITS}, ValueError),
        ({"items": list(range(MAX_CONTAINER_ITEMS + 1))}, ValueError),
        ({"nested": {"a": {"b": {"c": {"d": {"e": 1}}}}}}, ValueError),
    ],
)
def test_public_record_rejects_unbounded_optional_metadata(tmp_path, extra, error):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    before = profiling.run_summary()

    with pytest.raises(error):
        profiling.record("run_start", **_run_start_payload(**extra))

    profiling.close_event_output()
    assert path.read_text() == ""
    assert profiling.run_summary() == before


def test_numpy_scalar_does_not_satisfy_exact_schema_metric(tmp_path):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)

    with pytest.raises(TypeError, match="rank"):
        profiling.record("run_start", **_run_start_payload(rank=np.int64(0)))

    profiling.close_event_output()
    assert path.read_text() == ""
