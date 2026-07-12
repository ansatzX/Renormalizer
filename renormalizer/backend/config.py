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
            raise ValueError(
                "unsupported backend precision: {!r}".format(self.precision)
            )
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
    device_budget_resolution: object = None
    host_budget_resolution: object = None
    residency_request: object = None
    residency_plan: object = None
    residency_receipt: object = None

    def __post_init__(self):
        from renormalizer.backend._distributed.context import DistributedContext
        from renormalizer.backend._distributed.mesh import DeviceMesh

        if not isinstance(self.context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if not isinstance(self.mesh, DeviceMesh):
            raise TypeError("mesh must be a DeviceMesh")
        if self.residency_policy not in {"device_resident", "active_working_set"}:
            raise ValueError("unsupported distributed residency_policy")
        if (
            self.mesh.size != self.context.world_size
            or self.mesh.rank != self.context.rank
        ):
            raise ValueError("mesh rank or size does not match distributed context")
        if (
            getattr(self.collective, "size", None) != self.context.world_size
            or getattr(self.collective, "rank", None) != self.context.rank
        ):
            raise ValueError(
                "collective rank or size does not match distributed context"
            )
        if not callable(getattr(self.provider, "acquire", None)):
            raise TypeError("provider must implement acquire")
        provider_policy = getattr(self.provider, "residency_policy", None)
        provider_role = getattr(self.provider, "provider_role", None)
        if (
            self.residency_policy == "active_working_set"
            and provider_policy != "active_working_set"
        ):
            raise ValueError(
                "provider policy mismatch; DeviceResidentProvider supports only "
                "residency_policy='device_resident'"
            )
        if self.residency_policy == "device_resident" and provider_policy not in {
            None,
            "device_resident",
        }:
            raise ValueError("provider policy does not match residency_policy")
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
        from renormalizer.backend._distributed.residency import (
            MemoryBudgetResolution,
        )

        for requested_name, resolution_name, resource in (
            (
                "device_memory_budget_bytes",
                "device_budget_resolution",
                "device",
            ),
            (
                "host_memory_budget_bytes",
                "host_budget_resolution",
                "host",
            ),
        ):
            requested = getattr(self, requested_name)
            resolution = getattr(self, resolution_name)
            if resolution is None:
                continue
            if not isinstance(resolution, MemoryBudgetResolution):
                raise TypeError(
                    "{} must be a MemoryBudgetResolution or None".format(
                        resolution_name
                    )
                )
            if resolution.requested_bytes != requested:
                raise ValueError("budget resolution does not match requested bytes")
            if resolution.resource != resource:
                raise ValueError(
                    "{} budget resolution has the wrong resource".format(resource)
                )

        residency_metadata = (
            self.residency_request,
            self.residency_plan,
            self.residency_receipt,
        )
        if self.residency_policy == "device_resident":
            if provider_role not in {None, "resident"}:
                raise ValueError("device-resident provider role is invalid")
            if any(value is not None for value in residency_metadata):
                raise ValueError("residency metadata requires active_working_set")
            return

        if self.device_budget_resolution is None or self.host_budget_resolution is None:
            raise ValueError(
                "active_working_set requires resolved device and host budgets"
            )
        if provider_role == "factory":
            if not callable(getattr(self.provider, "open_working_set", None)):
                raise TypeError("factory provider must implement open_working_set")
            if any(value is not None for value in residency_metadata):
                raise ValueError("factory config must not retain lease metadata")
            return
        if provider_role != "working_set":
            raise ValueError("active provider role must be 'factory' or 'working_set'")

        from renormalizer.backend._distributed.residency import (
            ResidencyPlan,
            ResidencyPreflightReceipt,
            ResidencyRequest,
        )

        if not isinstance(self.residency_request, ResidencyRequest):
            raise TypeError("working-set config requires a ResidencyRequest")
        if not isinstance(self.residency_plan, ResidencyPlan):
            raise TypeError("working-set config requires a ResidencyPlan")
        if not isinstance(self.residency_receipt, ResidencyPreflightReceipt):
            raise TypeError("working-set config requires a ResidencyPreflightReceipt")
        if (
            getattr(self.provider, "request", None) is not self.residency_request
            or getattr(self.provider, "plan", None) is not self.residency_plan
            or getattr(self.provider, "receipt", None) is not self.residency_receipt
        ):
            raise ValueError("working-set config metadata does not match its lease")
        self.residency_plan.validate_request(self.residency_request)
        if (
            self.residency_receipt.request_hash != self.residency_request.request_hash
            or self.residency_receipt.plan_hash != self.residency_plan.plan_hash
        ):
            raise ValueError("working-set receipt does not match request and plan")

    @property
    def resolved_device_memory_budget_bytes(self):
        resolution = self.device_budget_resolution
        return None if resolution is None else resolution.resolved_bytes

    @property
    def resolved_host_memory_budget_bytes(self):
        resolution = self.host_budget_resolution
        return None if resolution is None else resolution.resolved_bytes

    @property
    def device_memory_budget_source(self):
        resolution = self.device_budget_resolution
        return None if resolution is None else resolution.source

    @property
    def host_memory_budget_source(self):
        resolution = self.host_budget_resolution
        return None if resolution is None else resolution.source

    @property
    def device_available_snapshot_bytes(self):
        resolution = self.device_budget_resolution
        return None if resolution is None else resolution.available_snapshot_bytes

    @property
    def host_available_snapshot_bytes(self):
        resolution = self.host_budget_resolution
        return None if resolution is None else resolution.available_snapshot_bytes
