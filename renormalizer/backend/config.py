# -*- coding: utf-8 -*-

"""Explicit backend configuration helpers."""


_DEVICE_ALIASES = {
    None: None,
    "auto": None,
    "default": None,
    "cpu": "cpu",
    "host": "cpu",
    "gpu": "gpu",
    "cuda": "gpu",
}

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
    key = device
    if isinstance(device, str):
        key = device.lower().strip()
    if key in _DEVICE_ALIASES:
        return _DEVICE_ALIASES[key]
    raise ValueError("Unknown backend device {0!r}. Expected 'cpu', 'gpu', or None.".format(device))


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

    def __init__(self, device=None, precision=None, seed=None, **options):
        self.device = normalize_device(device)
        self.precision = normalize_precision(precision)
        self.seed = seed
        self.options = dict(options)

    def replace(self, **updates):
        values = dict(self.options)
        values.update(updates.pop("options", {}))
        device = updates.pop("device", self.device)
        precision = updates.pop("precision", self.precision)
        seed = updates.pop("seed", self.seed)
        values.update(updates)
        return BackendConfig(device=device, precision=precision, seed=seed, **values)

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
            "BackendConfig(device={0!r}, precision={1!r}, seed={2!r}, options={3!r})"
            .format(self.device, self.precision, self.seed, self.options)
        )
