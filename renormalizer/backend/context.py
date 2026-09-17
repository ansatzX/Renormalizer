"""Explicit numerical policy and per-call dispatch, separate from legacy globals."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from renormalizer.backend.factory import create_backend, normalize_backend_name
from renormalizer.backend.proxy import BackendProxy
from renormalizer.cons import runtime_backend

_active = ContextVar('renormalizer_algorithm_backend', default=None)


def current_backend():
    selected = _active.get()
    return runtime_backend() if selected is None else selected


@contextmanager
def capture_backend(selected=None):
    """Keep a real instance for this call; nesting and exceptions restore it.

    Unlike the public compatibility proxy, this dispatch is local to the current
    thread/async context. A raw legacy adapter retains its legacy precision rules;
    use make_context for an instance whose precision is locked.
    """
    captured = current_backend() if selected is None else selected
    if isinstance(captured, (BackendProxy, InternalBackendProxy)):
        captured = captured.current
    token = _active.set(captured)
    try:
        yield captured
    finally:
        _active.reset(token)


class InternalBackendProxy:
    @property
    def current(self):
        return current_backend()

    def __getattr__(self, name):
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        setattr(self.current, name, value)


internal_backend = InternalBackendProxy()


@dataclass(frozen=True)
class NumericalContext:
    """Factory-owned adapter plus immutable device, precision and host policy."""
    adapter: object
    device: str
    real_dtype: np.dtype
    complex_dtype: np.dtype
    host_policy: str

    @property
    def ops(self):
        from renormalizer.backend.contracts import StrictOperations
        return StrictOperations(self)


def make_context(name='numpy', *, device='cpu', real_dtype='float64', host_policy='forbid'):
    """Create a private instance without selecting it or seeding global RNGs.

    Only NumPy's strict CPU surface is implemented here. Optional raw adapters
    remain available through the legacy factory; they are not strict support
    claims until their device/precision contracts are implemented and verified.
    """
    from renormalizer.backend.contracts import CapabilityError, PrecisionError
    if host_policy not in ('forbid', 'explicit'):
        raise ValueError("host_policy must be 'forbid' or 'explicit'")
    dtype = np.dtype(real_dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('real_dtype must be float32 or float64')
    normalized = normalize_backend_name(name)
    if normalized != 'numpy':
        raise CapabilityError(f'strict context for {normalized} is not implemented')
    if device != 'cpu':
        raise ValueError('NumPy context device must be cpu')
    adapter = create_backend(normalized, explicit=True)
    complex_dtype = np.dtype('complex64' if dtype == np.dtype('float32') else 'complex128')
    # Explicit policy wins over the old presence-based RENO_FP32 default.
    adapter.dtypes = (dtype.type, complex_dtype.type)
    actual = adapter.zeros((0,), dtype=adapter.real_dtype).dtype
    if actual != dtype:
        raise PrecisionError(f'requested {dtype}, received {actual}')
    adapter.first_mp = True
    return NumericalContext(adapter, device, dtype, complex_dtype, host_policy)
