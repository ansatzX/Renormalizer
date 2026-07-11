# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path

from renormalizer.utils.log import PROFILING, get_logger


_runtime = None
_run_aggregate = None


def enabled() -> bool:
    return get_logger().isEnabledFor(PROFILING)


def _load_runtime():
    global _runtime
    if _runtime is None:
        from renormalizer.utils._profiling.runtime import ProfilingRuntime

        _runtime = ProfilingRuntime()
    return _runtime


def _load_run_aggregate():
    global _run_aggregate
    if _run_aggregate is None:
        from renormalizer.utils._profiling.aggregate import RunAggregate

        _run_aggregate = RunAggregate()
    return _run_aggregate


def _default_event_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rank = os.environ.get("RANK", "0")
    return Path.cwd() / f"profiling-{timestamp}-p{os.getpid()}-r{rank}.jsonl"


def register_event_output(path=None, *, events=None):
    path = Path(path) if path is not None else _default_event_output_path()
    _load_runtime().register_event_output(path, events=events)
    reset_run_aggregate()
    return path


def record(event: str, **payload) -> None:
    if not enabled():
        return
    if event == "run_summary":
        raise ValueError("run_summary must be emitted with record_run_summary()")
    runtime = _load_runtime()
    with runtime.transaction():
        prepared = runtime.prepare_event(event, payload)
        aggregate = _load_run_aggregate()
        delta = aggregate.prepare(prepared)
        # The protected commit cannot stale after a successful public write.
        runtime.write_prepared(prepared)
        aggregate.commit(delta)


def run_summary():
    return _load_run_aggregate().summary()


def record_run_summary():
    """Emit a prospective snapshot, then count the accepted enabled call.

    Filtering or an absent writer still accepts the call. Serialization or I/O failure
    does not commit its ``run_summary`` event count.
    """
    if not enabled():
        return None
    runtime = _load_runtime()
    with runtime.transaction():
        aggregate = _load_run_aggregate()
        delta = aggregate.prepare({"event": "run_summary"})
        summary = aggregate.preview(delta)
        prepared = runtime.prepare_summary(summary)
        runtime.write_prepared(prepared)
        aggregate.commit(delta)
        return summary


def reset_run_aggregate():
    global _run_aggregate
    from renormalizer.utils._profiling.aggregate import RunAggregate

    _run_aggregate = RunAggregate()
    return _run_aggregate.summary()


def flush_event_output() -> None:
    if _runtime is not None:
        _runtime.flush_event_output()


def close_event_output() -> None:
    if _runtime is not None:
        _runtime.close_event_output()


@contextmanager
def scope(**context):
    if not enabled():
        yield
        return
    runtime = _load_runtime()
    token = runtime.push_scope(context)
    try:
        yield
    finally:
        runtime.pop_scope(token)
