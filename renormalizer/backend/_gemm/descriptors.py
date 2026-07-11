"""Immutable metadata for keyed matrix multiplication tasks."""

import math
from dataclasses import dataclass


def _normalize_key(value, field):
    if not isinstance(value, str):
        raise TypeError("{} key must be a non-empty string".format(field))
    value = value.strip()
    if not value:
        raise ValueError("{} key must be a non-empty string".format(field))
    return value


def _normalize_dimension(value, field):
    if type(value) is not int:
        raise TypeError("{} must be a Python integer".format(field))
    if value < 0:
        raise ValueError("{} must be non-negative".format(field))
    return value


def _normalize_transpose(value, field):
    if not isinstance(value, str):
        raise TypeError("{} must be one of N, T, or C".format(field))
    value = value.upper()
    if value not in {"N", "T", "C"}:
        raise ValueError("{} must be one of N, T, or C".format(field))
    return value


def _normalize_scalar(value, field):
    if isinstance(value, bool) or type(value) not in {int, float, complex}:
        raise TypeError("{} must be a finite Python numeric scalar".format(field))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("{} must be finite".format(field))
    return value


@dataclass(frozen=True)
class MatmulDesc:
    a_key: str
    b_key: str
    c_key: str
    m: int
    n: int
    k: int
    trans_a: str = "N"
    trans_b: str = "N"
    alpha: complex | float = 1.0
    beta: complex | float = 0.0

    def __post_init__(self):
        for field in ("a_key", "b_key", "c_key"):
            object.__setattr__(
                self, field, _normalize_key(getattr(self, field), field)
            )
        for field in ("m", "n", "k"):
            object.__setattr__(
                self, field, _normalize_dimension(getattr(self, field), field)
            )
        for field in ("trans_a", "trans_b"):
            object.__setattr__(
                self, field, _normalize_transpose(getattr(self, field), field)
            )
        for field in ("alpha", "beta"):
            object.__setattr__(
                self, field, _normalize_scalar(getattr(self, field), field)
            )


__all__ = ["MatmulDesc"]
