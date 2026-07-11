import contextvars

from renormalizer.utils._profiling.jsonl import JsonlEventWriter


class ProfilingRuntime:
    def __init__(self):
        self._scope_stack = contextvars.ContextVar("renormalizer_profiling_scope", default=())
        self._event_output = None

    def register_event_output(self, path, *, events=None) -> None:
        self.close_event_output()
        self._event_output = JsonlEventWriter(path, events=events)

    def push_scope(self, context):
        stack = self._scope_stack.get()
        return self._scope_stack.set(stack + (dict(context),))

    def pop_scope(self, token) -> None:
        self._scope_stack.reset(token)

    def record(self, event: str, **fields) -> None:
        if self._event_output is None:
            return
        payload = {}
        for context in self._scope_stack.get():
            payload.update(context)
        payload["event"] = event
        payload.update(fields)
        self._event_output.write(event, payload)

    def flush_event_output(self) -> None:
        if self._event_output is not None:
            self._event_output.flush()

    def close_event_output(self) -> None:
        if self._event_output is not None:
            self._event_output.close()
            self._event_output = None
