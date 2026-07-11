from contextlib import contextmanager
import contextvars
import threading

from renormalizer.utils._profiling.events import (
    normalize_bounded_metadata,
    normalize_event_filter,
    validate_source_event,
)
from renormalizer.utils._profiling.jsonl import JsonlEventWriter


class ProfilingRuntime:
    def __init__(self):
        self._scope_stack = contextvars.ContextVar("renormalizer_profiling_scope", default=())
        self._event_output = None
        self._transaction_lock = threading.Lock()
        self._transaction_state = threading.local()

    @contextmanager
    def transaction(self):
        if getattr(self._transaction_state, "active", False):
            raise RuntimeError("reentrant profiling record transaction is not supported")
        with self._transaction_lock:
            self._transaction_state.active = True
            try:
                yield
            finally:
                self._transaction_state.active = False

    def register_event_output(self, path, *, events=None) -> None:
        events = normalize_event_filter(events)
        replacement = JsonlEventWriter(path, events=events)
        previous = self._event_output
        if previous is not None:
            try:
                previous.close()
            except Exception:
                try:
                    replacement.discard()
                except Exception:
                    pass
                raise
        self._event_output = replacement

    def push_scope(self, context):
        stack = self._scope_stack.get()
        return self._scope_stack.set(stack + (dict(context),))

    def pop_scope(self, token) -> None:
        self._scope_stack.reset(token)

    def _merged_scope(self):
        payload = {}
        for context in self._scope_stack.get():
            payload.update(context)
        return payload

    def prepare_event(self, event: str, fields):
        payload = self._merged_scope()
        payload.update(fields)
        payload["event"] = event
        return validate_source_event(payload)

    def prepare_summary(self, summary):
        context = normalize_bounded_metadata(self._merged_scope())
        return normalize_bounded_metadata(
            {"event": "run_summary", "context": context, **summary}
        )

    def write_prepared(self, payload) -> None:
        if self._event_output is None:
            return
        event = payload["event"]
        self._event_output.write(event, payload)

    def record(self, event: str, **fields) -> None:
        self.write_prepared(self.prepare_event(event, fields))

    def flush_event_output(self) -> None:
        if self._event_output is not None:
            self._event_output.flush()

    def close_event_output(self) -> None:
        if self._event_output is not None:
            self._event_output.close()
            self._event_output = None
