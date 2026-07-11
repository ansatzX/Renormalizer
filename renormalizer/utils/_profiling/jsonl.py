import json
from pathlib import Path

from renormalizer.utils._profiling.events import normalize_event_filter, should_write_event


def to_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "shape"):
        if value.__class__.__module__ == "numpy" and value.__class__.__name__ != "ndarray":
            return to_jsonable(value.item())
        raise TypeError("profiling metadata arrays are not supported")
    if hasattr(value, "item"):
        scalar = value.item()
        if scalar is not value:
            return to_jsonable(scalar)
    raise TypeError(f"profiling metadata is not JSON serializable: {type(value).__name__}")


class JsonlEventWriter:
    def __init__(self, path, *, events=None):
        self.path = Path(path)
        self.events = normalize_event_filter(events)
        self._stream = self.path.open("w", encoding="utf-8")

    def write(self, event, payload) -> None:
        if not should_write_event(event, self.events):
            return
        self._stream.write(
            json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        self._stream.write("\n")

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()
