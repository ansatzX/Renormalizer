import json
import os
import re

import numpy as np
import pytest

from renormalizer.utils import profiling
from renormalizer.utils.log import DEBUG, PROFILING, init_log


@pytest.fixture(autouse=True)
def reset_profiling_runtime(monkeypatch):
    profiling.close_event_output()
    monkeypatch.setattr(profiling, "_runtime", None)
    init_log(DEBUG)
    yield
    profiling.close_event_output()
    init_log(DEBUG)


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
    profiling.record("run_start", backend="numpy", rank=0, world_size=1)
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
    with profiling.scope(backend="numpy", rank=0):
        with profiling.scope(rank=1, stage="evolve"):
            profiling.record("run_start", backend="cupy")
            profiling.record("run_finish", backend="cupy")
    profiling.close_event_output()

    payloads = [json.loads(line) for line in path.read_text().splitlines()]
    assert payloads == [
        {
            "event": "run_start",
            "backend": "cupy",
            "rank": 1,
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
    profiling.record("run_start", rank=np.int64(3), ratio=np.float32(0.5))
    with pytest.raises(TypeError, match="array"):
        profiling.record("run_start", values=np.array([1, 2]))
    with pytest.raises(TypeError, match="array"):
        profiling.record("run_start", values=ScalarTensor())
    profiling.close_event_output()

    payload = json.loads(path.read_text().splitlines()[0])
    assert payload == {"event": "run_start", "rank": 3, "ratio": 0.5}


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
