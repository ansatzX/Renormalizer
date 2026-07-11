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
