"""Captured hybrid algorithm calls and observable contraction boundaries.

Records describe logical array transfers, not allocator peaks or all library
workspace. Existing Matrix/tree storage and scientific solvers remain on host.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from math import prod
import numpy as np
import opt_einsum as oe
from .context import capture_backend, current_backend
from .contracts import CapabilityError

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
            with capture_backend(selected):
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
    if context is not None and isinstance(array, np.ndarray):
        from .host_solver import from_host
        return from_host(array, context=context, ledger=[] if ledger is None else ledger.transfers,
                         reason='algorithm operand', operation_id=0 if ledger is None else len(ledger.operations))
    result = selected.from_numpy(array) if isinstance(array,np.ndarray) else selected.asarray(array)
    _record_transfer(array,result,'algorithm operand')
    return result


def to_host(array):
    context, ledger = _context.get(), _ledger.get()
    if context is not None and not isinstance(array, np.ndarray):
        from .host_solver import to_host as download
        return download(array, context=context, ledger=[] if ledger is None else ledger.transfers,
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


def _promote_operands(values, adapter):
    if adapter.name != 'torch':
        return tuple(values)
    from .contracts import PROMOTION
    arrays = [a for a in values if hasattr(a, 'shape') and hasattr(a, 'dtype')]
    if not arrays:
        return tuple(values)
    dtype = np.dtype(str(arrays[0].dtype).removeprefix('torch.'))
    for a in arrays[1:]:
        other = np.dtype(str(a.dtype).removeprefix('torch.'))
        dtype = PROMOTION.get((dtype, other), np.result_type(dtype, other))
    result = []
    for a in values:
        if hasattr(a, 'shape') and hasattr(a, 'dtype'):
            b = adapter.asarray(a, dtype=dtype)
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
    result = oe.contract(*args,**kwargs)
    _witness(result)
    return result


def contract_expression(*args, **kwargs):
    # Constants must be placed by the captured adapter, not opt_einsum's
    # process-default device conversion. Expressions belong to one instance.
    selected = current_backend()
    converted = tuple(to_backend(a) if hasattr(a, 'dtype') and hasattr(a, 'shape') else a for a in args)
    expr = oe.contract_expression(*converted, **kwargs)
    def call(*operands, **options):
        if current_backend() is not selected:
            raise CapabilityError('contraction expression belongs to another backend instance')
        options['backend'] = selected.opt_einsum_name
        native = tuple(to_backend(a) for a in operands)
        active_expr = expr
        if selected.name == 'torch':
            promoted = _promote_operands(converted + native, selected)
            active_expr = oe.contract_expression(*promoted[:len(converted)], **kwargs)
            native = promoted[len(converted):]
        result = active_expr(*native, **options)
        _witness(result)
        return result
    return call
