# -*- coding: utf-8 -*-

"""Explicit backend configuration helpers."""

from renormalizer.backend.execution import FallbackPolicy, legacy_device_kind, parse_device_spec


_PRECISION_ALIASES = {
    None: None,
    32: 32,
    64: 64,
    "32": 32,
    "32bit": 32,
    "32-bit": 32,
    "fp32": 32,
    "float32": 32,
    "single": 32,
    "64": 64,
    "64bit": 64,
    "64-bit": 64,
    "fp64": 64,
    "float64": 64,
    "double": 64,
}


def normalize_device(device):
    return legacy_device_kind(parse_device_spec(device))


def normalize_precision(precision):
    key = precision
    if isinstance(precision, str):
        key = precision.lower().replace("_", "-").strip()
    if key in _PRECISION_ALIASES:
        return _PRECISION_ALIASES[key]
    raise ValueError("Unknown backend precision {0!r}. Expected 32, 64, or None.".format(precision))


class BackendConfig:
    """Explicit configuration shared by backend adapters.

    Parameters
    ----------
    device
        Requested compute device kind. Accepted values are ``"cpu"``,
        ``"gpu"``/``"cuda"``, or ``None`` for the backend default.
    precision
        Requested floating precision. Accepted values are ``32``, ``64``,
        common string aliases such as ``"fp32"``, or ``None`` for the default
        environment-driven behavior.
    seed
        Optional backend seed. Public ``set_backend`` still applies the legacy
        seed when this is omitted.
    """

    def __init__(self, device=None, precision=None, seed=None, fallback_policy=None, **options):
        self.device_spec = parse_device_spec(device)
        self.device = legacy_device_kind(self.device_spec)
        self.precision = normalize_precision(precision)
        self.seed = seed
        self.fallback_policy = FallbackPolicy.from_value(fallback_policy)
        self.options = dict(options)
        if self.fallback_policy is FallbackPolicy.SILENT and not self.options.get("allow_silent_fallback", False):
            raise ValueError("silent fallback policy requires allow_silent_fallback=True")

    def replace(self, **updates):
        values = dict(self.options)
        values.update(updates.pop("options", {}))
        device = updates.pop("device", self.device_spec)
        precision = updates.pop("precision", self.precision)
        seed = updates.pop("seed", self.seed)
        fallback_policy = updates.pop("fallback_policy", self.fallback_policy)
        values.update(updates)
        return BackendConfig(device=device, precision=precision, seed=seed, fallback_policy=fallback_policy, **values)

    @classmethod
    def from_config(cls, config=None, **overrides):
        if config is None:
            return cls(**overrides)
        if isinstance(config, cls):
            if overrides:
                return config.replace(**overrides)
            return config
        if isinstance(config, dict):
            values = dict(config)
            values.update(overrides)
            return cls(**values)
        raise TypeError("backend config must be a BackendConfig, dict, or None")

    def __repr__(self):
        return (
            "BackendConfig(device={0!r}, device_spec={1!r}, precision={2!r}, "
            "seed={3!r}, fallback_policy={4!r}, options={5!r})"
            .format(
                self.device,
                self.device_spec,
                self.precision,
                self.seed,
                self.fallback_policy,
                self.options,
            )
        )
