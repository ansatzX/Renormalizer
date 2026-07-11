"""Validated process identity for distributed backend execution."""

from dataclasses import dataclass
from typing import Mapping


_ENVIRONMENT_FIELDS = (
    ("RANK", "rank"),
    ("LOCAL_RANK", "local_rank"),
    ("WORLD_SIZE", "world_size"),
    ("LOCAL_WORLD_SIZE", "local_world_size"),
)

_MIN_DEDICATED_PORT = 1024
_MAX_PORT = 65535
_PORT_RANGE = _MAX_PORT - _MIN_DEDICATED_PORT + 1
_CUPY_PORT_OFFSET = 1000
_CUPY_DEFAULT_PORT = 13333


def _validate_host(host, *, source="host"):
    if not isinstance(host, str) or not host or host.strip() != host:
        raise ValueError("{} must be a non-empty string".format(source))
    return host


def _validate_port(port, *, source="port"):
    if type(port) is not int:
        raise TypeError("{} must be an integer between 1 and 65535".format(source))
    if port < 1 or port > _MAX_PORT:
        raise ValueError("{} must be between 1 and 65535".format(source))
    return port


def _environment_port(environ, name):
    try:
        raw_port = environ[name]
    except KeyError:
        return None
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be an integer".format(name)) from error
    return _validate_port(port, source=name)


def _dedicated_cupy_port(master_port):
    port = _MIN_DEDICATED_PORT + (
        (master_port - _MIN_DEDICATED_PORT + _CUPY_PORT_OFFSET) % _PORT_RANGE
    )
    if port == _CUPY_DEFAULT_PORT:
        port += 1
    return port


@dataclass(frozen=True)
class DistributedRendezvous:
    """Concrete CuPy TCP endpoint resolved from one launcher snapshot.

    Explicit paired host/port values are preferred for concurrent production
    jobs. Without a dedicated CuPy port, the torchrun fallback deterministically
    offsets ``MASTER_PORT`` and never reuses torchrun's endpoint or CuPy's fixed
    default port.
    """

    host: str
    port: int

    def __post_init__(self):
        _validate_host(self.host)
        _validate_port(self.port)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str], *, host=None, port=None):
        if host is None:
            if "CUPYX_DISTRIBUTED_HOST" in environ:
                resolved_host = _validate_host(
                    environ["CUPYX_DISTRIBUTED_HOST"],
                    source="CUPYX_DISTRIBUTED_HOST",
                )
            elif "MASTER_ADDR" in environ:
                resolved_host = _validate_host(
                    environ["MASTER_ADDR"], source="MASTER_ADDR"
                )
            else:
                raise ValueError("distributed rendezvous host is missing")
        else:
            resolved_host = _validate_host(host)

        if port is not None:
            resolved_port = _validate_port(port)
        elif "CUPYX_DISTRIBUTED_PORT" in environ:
            resolved_port = _environment_port(environ, "CUPYX_DISTRIBUTED_PORT")
        elif "MASTER_PORT" in environ:
            master_port = _environment_port(environ, "MASTER_PORT")
            resolved_port = _dedicated_cupy_port(master_port)
        else:
            raise ValueError("distributed rendezvous port is missing")

        return cls(host=resolved_host, port=resolved_port)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    def __post_init__(self):
        for name in ("rank", "local_rank", "world_size", "local_world_size"):
            if type(getattr(self, name)) is not int:
                raise TypeError("{} must be an integer".format(name))
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if self.local_world_size <= 0:
            raise ValueError("local_world_size must be positive")
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")
        if self.local_rank >= self.local_world_size:
            raise ValueError("local_rank must be smaller than local_world_size")
        if self.local_world_size > self.world_size:
            raise ValueError("local_world_size must not exceed world_size")

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]):
        values = {}
        for environment_name, field_name in _ENVIRONMENT_FIELDS:
            try:
                raw_value = environ[environment_name]
            except KeyError as error:
                raise ValueError(
                    "missing distributed environment variable {}".format(
                        environment_name
                    )
                ) from error
            try:
                values[field_name] = int(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "{} must be an integer".format(environment_name)
                ) from error
        return cls(**values)
