# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Backend selection configuration."""

import re
from dataclasses import dataclass


_DEVICE_ALIASES = {
    "cpu": "cpu",
    "host": "cpu",
    "gpu": "gpu",
    "cuda": "gpu",
}

_EXECUTION_POLICIES = frozenset({"legacy_oe", "execution_ir"})
_FALLBACK_POLICIES = frozenset({"error", "legacy_oe"})


def _normalize_device(device):
    if not isinstance(device, str):
        raise ValueError("unsupported backend device: {!r}".format(device))
    normalized = device.lower().strip()
    if normalized in _DEVICE_ALIASES:
        return _DEVICE_ALIASES[normalized]
    match = re.fullmatch(r"cuda:([0-9]+)", normalized)
    if match is not None:
        return "cuda:{}".format(int(match.group(1)))
    raise ValueError("unsupported backend device: {!r}".format(device))


@dataclass(frozen=True)
class BackendConfig:
    device: str = "cpu"
    precision: int = 64
    seed: int | None = None
    execution_policy: str = "legacy_oe"
    fallback_policy: str = "error"
    experimental_oe_ir: bool = False

    def __post_init__(self):
        object.__setattr__(self, "device", _normalize_device(self.device))
        if self.precision not in {32, 64}:
            raise ValueError("unsupported backend precision: {!r}".format(self.precision))
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("backend seed must be an integer or None")
        if self.execution_policy not in _EXECUTION_POLICIES:
            raise ValueError(
                "unsupported execution_policy={!r}; expected one of {}".format(
                    self.execution_policy, ", ".join(sorted(_EXECUTION_POLICIES))
                )
            )
        if self.fallback_policy not in _FALLBACK_POLICIES:
            raise ValueError(
                "unsupported fallback_policy={!r}; expected one of {}".format(
                    self.fallback_policy, ", ".join(sorted(_FALLBACK_POLICIES))
                )
            )
        if type(self.experimental_oe_ir) is not bool:
            raise TypeError("experimental_oe_ir must be a boolean")


@dataclass(frozen=True)
class DistributedExecutionConfig:
    context: object
    mesh: object
    collective: object
    provider: object
    residency_policy: str = "device_resident"
    device_memory_budget_bytes: int | None = None
    host_memory_budget_bytes: int | None = None
    prefetch_depth: int = 1
    backend_name: str | None = None
    backend_device: str | None = None
    backend_precision: int | None = None

    def __post_init__(self):
        from renormalizer.backend._distributed.context import DistributedContext
        from renormalizer.backend._distributed.mesh import DeviceMesh

        if not isinstance(self.context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if not isinstance(self.mesh, DeviceMesh):
            raise TypeError("mesh must be a DeviceMesh")
        if self.residency_policy != "device_resident":
            raise ValueError(
                "Stage 4 supports only residency_policy='device_resident'"
            )
        if (
            self.mesh.size != self.context.world_size
            or self.mesh.rank != self.context.rank
        ):
            raise ValueError("mesh rank or size does not match distributed context")
        if (
            getattr(self.collective, "size", None) != self.context.world_size
            or getattr(self.collective, "rank", None) != self.context.rank
        ):
            raise ValueError("collective rank or size does not match distributed context")
        if not callable(getattr(self.provider, "acquire", None)):
            raise TypeError("provider must implement acquire")
        for name in ("device_memory_budget_bytes", "host_memory_budget_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("{} must be a positive integer or None".format(name))
        if type(self.prefetch_depth) is not int or self.prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be a positive integer")
        backend_metadata = (
            self.backend_name,
            self.backend_device,
            self.backend_precision,
        )
        if any(value is not None for value in backend_metadata):
            if any(value is None for value in backend_metadata):
                raise ValueError("distributed backend metadata must be complete")
            if self.backend_name not in {"numpy", "cupy"}:
                raise ValueError("distributed backend must be NumPy or CuPy")
            if not isinstance(self.backend_device, str):
                raise TypeError("distributed backend device must be a string")
            if self.backend_precision not in {32, 64}:
                raise ValueError("distributed backend precision must be 32 or 64")
