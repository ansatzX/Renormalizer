"""Captured hybrid algorithm calls and observable contraction boundaries.

Records describe logical array transfers, not allocator peaks or all library
workspace. Existing Matrix/tree storage and scientific solvers remain on host.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from math import prod
import numpy as np
import opt_einsum as oe
from .context import capture_backend, current_backend
from .contracts import CapabilityError, NUMERIC_DTYPES, AUXILIARY_DTYPES

_HOST_DTYPES = frozenset(NUMERIC_DTYPES + AUXILIARY_DTYPES)

_context = ContextVar('renormalizer_numerical_context', default=None)
_ledger = ContextVar('renormalizer_execution_ledger', default=None)

@dataclass
class ExecutionLedger:
    adapter_id: int | None = None
    operations: list = field(default_factory=list)
    transfers: list = field(default_factory=list)

@contextmanager
def record_execution(context):
    """Record calls explicitly bound to context; does not select a backend."""
    ledger = ExecutionLedger(adapter_id=id(context.adapter))
    token = _ledger.set(ledger)
    try:
        yield ledger
    finally:
        _ledger.reset(token)


def bind_backend(function):
    @wraps(function)
    def wrapped(*args, backend_context=None, **kwargs):
        context = backend_context or _context.get()
        if context is not None and context.host_policy != 'explicit':
            raise CapabilityError('legacy hybrid algorithm requires explicit host policy')
        selected = context.adapter if context is not None else current_backend()
        token = _context.set(context)
        try:
            # The device scope matters for CuPy, which launches every kernel on
            # the process-wide current device: an explicitly selected device
            # must govern the whole algorithm, not only adapter calls.
            with capture_backend(selected), selected.device_scope():
                return function(*args, **kwargs)
        finally:
            _context.reset(token)
    return wrapped


def _device(array):
    device = getattr(array, 'device', 'cpu')
    if callable(device):
        device = device()
    platform = getattr(device, 'platform', None)
    if platform == 'cpu':
        return 'cpu'
    if platform in ('gpu', 'cuda'):
        return f'cuda:{device.id}'
    if hasattr(device, 'id') and type(array).__module__.startswith('cupy'):
        return f'cuda:{device.id}'
    return str(device)


def _record_transfer(source, result, reason):
    ledger = _ledger.get()
    if ledger is None or source is result:
        return
    src, dst = _device(source), _device(result)
    source_host = src.split(':')[0].lower() == 'cpu'
    target_host = dst.split(':')[0].lower() == 'cpu'
    direction = ('host_to_host' if source_host and target_host else
                 'H2D' if source_host else 'D2H' if target_host else 'device_to_device')
    ledger.transfers.append(dict(direction=direction, logical_bytes=prod(result.shape) * np.dtype(str(result.dtype).removeprefix('torch.')).itemsize,
                                 source_device=src,target_device=dst,reason=reason,
                                 operation_id=len(ledger.operations)))


def to_backend(array):
    selected = current_backend()
    context, ledger = _context.get(), _ledger.get()
    # NumPy hot path: uploading a host ndarray to the NumPy adapter is the
    # identity, and without a ledger there is nothing to record. DMRG/TDVP issue
    # tens of thousands of small contractions, so skip the conversion chain.
    # Context uploads keep from_host's dtype domain; others take the slow path.
    if ledger is None and type(array) is np.ndarray and selected.name == 'numpy' and (
            context is None or array.dtype in _HOST_DTYPES):
        return array
    if context is not None and isinstance(array, np.ndarray):
        from .host_solver import from_host
        return from_host(array, context=context, ledger=None if ledger is None else ledger.transfers,
                         reason='algorithm operand', operation_id=0 if ledger is None else len(ledger.operations))
    result = selected.from_numpy(array) if isinstance(array,np.ndarray) else selected.asarray(array)
    _record_transfer(array,result,'algorithm operand')
    return result


def to_host(array):
    # Every adapter's numpy() returns host ndarrays unchanged, and an unchanged
    # array is never recorded as a transfer.
    if type(array) is np.ndarray:
        return array
    context, ledger = _context.get(), _ledger.get()
    if context is not None and not isinstance(array, np.ndarray):
        from .host_solver import to_host as download
        return download(array, context=context, ledger=None if ledger is None else ledger.transfers,
                        reason='host storage or solver boundary', operation_id=0 if ledger is None else len(ledger.operations))
    result = current_backend().numpy(array)
    _record_transfer(array,result,'host storage or solver boundary')
    return result


def _witness(result):
    ledger = _ledger.get()
    if ledger is not None:
        adapter = current_backend()
        if ledger.adapter_id != id(adapter):
            raise CapabilityError('execution witness differs from requested context')
        ledger.operations.append(dict(operation='contraction',backend=adapter.name,
            device=_device(result),dtype=str(result.dtype),adapter_id=id(adapter),shape=tuple(result.shape)))


def _contraction_dtype(values, adapter):
    from .contracts import PROMOTION
    arrays = [a for a in values if hasattr(a, 'shape') and hasattr(a, 'dtype')]
    if not arrays:
        return None
    dtype = adapter.dtype_of(arrays[0])
    for a in arrays[1:]:
        other = adapter.dtype_of(a)
        promoted = PROMOTION.get((dtype, other))
        dtype = np.result_type(dtype, other) if promoted is None else promoted
    return dtype


def _promote_operands(values, adapter, dtype=None):
    if adapter.name != 'torch':
        return tuple(values)
    dtype = _contraction_dtype(values, adapter) if dtype is None else dtype
    if dtype is None:
        return tuple(values)
    result = []
    for a in values:
        if hasattr(a, 'shape') and hasattr(a, 'dtype'):
            b = a if adapter.dtype_of(a) == dtype else adapter.asarray(a, dtype=dtype)
            _record_transfer(a, b, 'explicit contraction dtype promotion')
            result.append(b)
        else:
            result.append(a)
    return tuple(result)


def contract(*args, **kwargs):
    selected = current_backend()
    # Interleaved labels and shapes are metadata, never converted to tensors.
    args = tuple(to_backend(a) if hasattr(a,'shape') and hasattr(a,'dtype') else a for a in args)
    args = _promote_operands(args, selected)
    kwargs['backend'] = selected.opt_einsum_name
    context = _context.get()
    if (selected.name == 'jax' and hasattr(selected, 'fused_einsum')
            and args and isinstance(args[0], str) and all(hasattr(a, 'shape') for a in args[1:])
            and set(kwargs) <= {'backend', 'optimize'} and isinstance(kwargs.get('optimize', 'auto'), str)
            and (context is None or context.operators.policy == 'builtin')):
        # Same single-computation evaluation as _fused_expression, path cached by shape.
        shapes = tuple(tuple(map(int, a.shape)) for a in args[1:])
        path = _cached_path(args[0], shapes, (('optimize', kwargs.get('optimize', 'auto')),))
        result = selected.fused_einsum(args[0], path)(*args[1:])
    elif context is None or context.operators.policy == 'builtin':
        result = oe.contract(*args, **kwargs)
    else:
        from .operators import dispatch
        result = dispatch(context, 'contract', oe.contract, args, kwargs)
    _witness(result)
    return result


@lru_cache(maxsize=4096)
def _cached_path(subscripts, shapes, options):
    path, _ = oe.contract_path(subscripts, *shapes, shapes=True, **dict(options))
    return path


def _fused_expression(selected, converted, kwargs):
    # Eager JAX pays one dispatch per pairwise step of the expression, every
    # Krylov iteration. The adapter evaluates the same path as one jitted
    # computation; constants are passed as arguments rather than baked in, so
    # one compilation per shape serves every site and time step.
    subscripts, specs = converted[0], converted[1:]
    constants = {i: specs[i] for i in kwargs.get('constants', ())}
    fused = selected.fused_einsum(subscripts, kwargs['optimize'])
    def call(*operands, **options):
        if current_backend() is not selected:
            raise CapabilityError('contraction expression belongs to another backend instance')
        options.pop('backend', None)
        if options:
            raise TypeError(f'unsupported contraction options: {sorted(options)}')
        runtime = iter(to_backend(a) for a in operands)
        result = fused(*(constants[i] if i in constants else next(runtime) for i in range(len(specs))))
        _witness(result)
        return result
    return call


def contract_expression(*args, **kwargs):
    # Constants must be placed by the captured adapter, not opt_einsum's
    # process-default device conversion. Expressions belong to one instance.
    selected = current_backend()
    converted = tuple(to_backend(a) if hasattr(a, 'dtype') and hasattr(a, 'shape') else a for a in args)
    context = _context.get()
    if isinstance(kwargs.get('optimize', 'auto'), str) and converted and isinstance(converted[0], str):
        # Sweeps rebuild the same expression at every site; a named path search
        # depends only on subscripts, shapes and options, so reuse its result.
        shapes = tuple(tuple(map(int, a.shape)) if hasattr(a, 'shape') else tuple(map(int, a))
                       for a in converted[1:])
        options = tuple(sorted((k, v) for k, v in kwargs.items() if k != 'constants'))
        try:
            kwargs['optimize'] = _cached_path(converted[0], shapes, options)
        except TypeError:  # unhashable option: search the path as before
            pass
    if (selected.name == 'jax' and hasattr(selected, 'fused_einsum')
            and isinstance(converted[0], str) and isinstance(kwargs.get('optimize'), list)
            and set(kwargs) <= {'constants', 'optimize'}
            and (context is None or context.operators.policy == 'builtin')):
        return _fused_expression(selected, converted, kwargs)
    expr = oe.contract_expression(*converted, **kwargs)
    # Retain only the latest precision specialization and its constants, never
    # runtime operand contents. Metadata comparisons need no content hashes.
    specialization = None
    def call(*operands, **options):
        nonlocal specialization
        if current_backend() is not selected:
            raise CapabilityError('contraction expression belongs to another backend instance')
        options['backend'] = selected.opt_einsum_name
        native = tuple(to_backend(a) for a in operands)
        active_expr = expr
        if selected.name == 'torch':
            dtype = _contraction_dtype(converted + native, selected)
            cached = specialization
            if cached is None or cached[0] != dtype:
                constants = _promote_operands(converted, selected, dtype)
                if any(a is not b for a, b in zip(constants, converted)):
                    active_expr = oe.contract_expression(*constants, **kwargs)
                specialization = (dtype, active_expr)
            else:
                active_expr = cached[1]
            native = _promote_operands(native, selected, dtype)
        if context is not None and context.operators.policy != 'builtin' and _context.get() is not context:
            raise CapabilityError('contraction expression belongs to another numerical context')
        if context is None or context.operators.policy == 'builtin':
            result = active_expr(*native, **options)
        else:
            from .operators import dispatch
            def evaluate(*values, expression, **opts):
                return expression(*values, **opts)
            result = dispatch(context, 'contract_expression', evaluate, native,
                              dict(options, expression=active_expr),
                              metadata_values=converted + native)
        _witness(result)
        return result
    return call
