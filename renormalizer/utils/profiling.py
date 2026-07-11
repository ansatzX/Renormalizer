# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path

from renormalizer.utils.log import PROFILING, get_logger


_runtime = None


def enabled() -> bool:
    return get_logger().isEnabledFor(PROFILING)


def _load_runtime():
    global _runtime
    if _runtime is None:
        from renormalizer.utils._profiling.runtime import ProfilingRuntime

        _runtime = ProfilingRuntime()
    return _runtime


def _default_event_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rank = os.environ.get("RANK", "0")
    return Path.cwd() / f"profiling-{timestamp}-p{os.getpid()}-r{rank}.jsonl"


def register_event_output(path=None, *, events=None):
    path = Path(path) if path is not None else _default_event_output_path()
    _load_runtime().register_event_output(path, events=events)
    return path


def record(event: str, **payload) -> None:
    if not enabled():
        return
    _load_runtime().record(event, **payload)


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
