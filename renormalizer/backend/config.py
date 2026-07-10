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
        if self.execution_policy != "legacy_oe":
            raise ValueError("execution_policy={!r} is unavailable before Stage 3".format(self.execution_policy))
        if self.fallback_policy != "error":
            raise ValueError("fallback_policy={!r} is unavailable before Stage 3".format(self.fallback_policy))
        if self.experimental_oe_ir is not False:
            raise ValueError("experimental_oe_ir is unavailable before Stage 3")
