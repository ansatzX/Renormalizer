import json
from pathlib import Path

from renormalizer.utils._profiling.events import (
    normalize_bounded_metadata,
    normalize_event_filter,
    should_write_event,
)


def to_jsonable(value):
    return normalize_bounded_metadata(value)


class JsonlEventWriter:
    def __init__(self, path, *, events=None):
        self.path = Path(path)
        self.events = normalize_event_filter(events)
        existed = self.path.exists()
        try:
            self._stream = self.path.open("w", encoding="utf-8")
        except Exception:
            if not existed:
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        self._created_path = not existed

    def write(self, event, payload) -> None:
        if not should_write_event(event, self.events):
            return
        line = (
            json.dumps(
                to_jsonable(payload),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        self._stream.write(line)

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def discard(self) -> None:
        try:
            self.close()
        finally:
            if self._created_path:
                self.path.unlink(missing_ok=True)
