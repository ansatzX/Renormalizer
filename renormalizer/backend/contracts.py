"""A deliberately named NumPy numerical surface; not a namespace proxy.

Creation/conversion may allocate as their copy policy permits. reshape may copy;
transpose and real/imag may alias. Callers must not rely on output aliasing. All
arithmetic and functional updates leave inputs untouched. Native NumPy views,
including negative strides, overlaps, read-only and empty views, are accepted.
"""
from dataclasses import dataclass
import numbers

import numpy as np


class CapabilityError(NotImplementedError):
    pass


class PrecisionError(ValueError):
    pass


class OwnershipError(TypeError):
    pass


NUMERIC_DTYPES = tuple(np.dtype(x) for x in ('float32', 'float64', 'complex64', 'complex128'))
AUXILIARY_DTYPES = tuple(np.dtype(x) for x in ('bool', 'int32', 'int64'))
_PROMOTION_ROWS = (
    ('float32', 'float64', 'complex64', 'complex128'),
    ('float64', 'float64', 'complex128', 'complex128'),
    ('complex64', 'complex128', 'complex64', 'complex128'),
    ('complex128', 'complex128', 'complex128', 'complex128'),
)
PROMOTION = {
    (a, b): np.dtype(_PROMOTION_ROWS[i][j])
    for i, a in enumerate(NUMERIC_DTYPES) for j, b in enumerate(NUMERIC_DTYPES)
}


def check_actual_dtype(actual, requested):
    if np.dtype(actual) != np.dtype(requested):
        raise PrecisionError(f'requested {requested}, received {actual}')


def check_solve_shapes(a, b):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise CapabilityError('solve requires a two-dimensional square matrix')
    if b.ndim not in (1, 2) or b.shape[0] != a.shape[0]:
        raise ValueError('solve RHS must have shape (n,) or (n,k)')


@dataclass(frozen=True)
class StrictOperations:
    context: object

    _OPERATIONS = frozenset((
        'array', 'zeros', 'ones', 'eye', 'from_numpy', 'to_numpy', 'reshape',
        'transpose', 'conj', 'astype', 'add', 'subtract', 'multiply', 'divide',
        'sum', 'max', 'min', 'real', 'imag', 'matmul', 'einsum', 'qr', 'svd',
        'eigh', 'solve', 'norm', 'at_set', 'at_add', 'at_sub', 'at_mul', 'scalar', 'sync',
    ))

    def capability(self, operation, dtype):
        """Implementation domain only; runtime evidence is recorded separately."""
        auxiliary = operation in ('array', 'zeros', 'ones', 'eye', 'from_numpy',
                                  'to_numpy', 'reshape', 'transpose', 'astype')
        domain = NUMERIC_DTYPES + (AUXILIARY_DTYPES if auxiliary else ())
        supported = operation in self._OPERATIONS and np.dtype(dtype) in domain
        return {'backend': 'numpy', 'version': np.__version__, 'device': 'cpu',
                'dtype': str(np.dtype(dtype)), 'operation': operation,
                'status': 'supported' if supported else 'unsupported',
                'layout': 'strided; reshape may copy; views may alias',
                'evidence': 'not supplied by capability query'}

    def _dtype(self, dtype=None, *, complex_input=False):
        if dtype is None:
            dtype = self.context.complex_dtype if complex_input else self.context.real_dtype
        dtype = np.dtype(dtype)
        if dtype not in NUMERIC_DTYPES + AUXILIARY_DTYPES:
            raise CapabilityError(f'unsupported dtype {dtype}')
        return dtype

    def _owned(self, x, *, numeric=True):
        if not isinstance(x, np.ndarray):
            raise OwnershipError('operation requires a NumPy array; use an explicit conversion')
        domain = NUMERIC_DTYPES if numeric else NUMERIC_DTYPES + AUXILIARY_DTYPES
        if x.dtype not in domain:
            raise CapabilityError(f'unsupported input dtype {x.dtype}')
        if numeric and not np.isfinite(x).all():
            raise ValueError('numerical input must be finite')
        return x

    def _result(self, result, dtype=None):
        out = np.asarray(result)
        if dtype is not None:
            check_actual_dtype(out.dtype, dtype)
        if not np.isfinite(out).all():
            raise ValueError('numerical output must be finite')
        return out

    def _operand(self, x):
        if isinstance(x, np.ndarray):
            return self._owned(x)
        if isinstance(x, np.generic):
            return self._owned(np.asarray(x))
        if isinstance(x, numbers.Number):
            return self._owned(self.array(x))
        raise OwnershipError('operation requires an array or scalar; use an explicit conversion')

    def _promote(self, *args):
        arrays = [self._operand(x) for x in args]
        dtype = arrays[0].dtype
        for x in arrays[1:]:
            dtype = PROMOTION[dtype, x.dtype]
        return [x.astype(dtype, copy=False) for x in arrays], dtype

    def array(self, data, dtype=None, *, copy=None):
        if not isinstance(data, (np.ndarray, np.generic, list, tuple, numbers.Number)):
            raise OwnershipError('array requires host data or a NumPy array')
        # Do not invoke a foreign device array's implicit host conversion.
        if isinstance(data, (list, tuple)):
            def validate(values):
                for value in values:
                    if isinstance(value, (list, tuple)):
                        validate(value)
                    elif not isinstance(value, (numbers.Number, np.generic)):
                        raise CapabilityError('unsupported input dtype')
            validate(data)
        host_dtype = data.dtype if isinstance(data, (np.ndarray, np.generic)) else None
        if host_dtype is not None and host_dtype.kind not in 'biufc':
            raise CapabilityError(f'unsupported input dtype {host_dtype}')
        dtype = self._dtype(dtype, complex_input=np.iscomplexobj(data))
        result = self.context.adapter.array(data, dtype=dtype, copy=copy)
        check_actual_dtype(result.dtype, dtype)
        return result

    def from_numpy(self, x, *, copy=None):
        if not isinstance(x, np.ndarray):
            raise OwnershipError('from_numpy requires a NumPy array')
        return self.array(x, dtype=x.dtype, copy=copy)

    def to_numpy(self, x, *, copy=None):
        self._owned(x, numeric=False)
        return self.context.adapter.to_numpy(x, copy=copy)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=self._dtype(dtype))

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=self._dtype(dtype))

    def eye(self, n, m=None, dtype=None):
        return np.eye(n, M=m, dtype=self._dtype(dtype))

    def reshape(self, x, shape):
        return self._owned(x, numeric=False).reshape(shape)

    def transpose(self, x, axes=None):
        return self._owned(x, numeric=False).transpose(axes)

    def conj(self, x):
        return self._result(np.conj(self._owned(x)), x.dtype)

    def astype(self, x, dtype, *, copy=True):
        self._owned(x, numeric=False)
        dtype = self._dtype(dtype)
        return self._result(x.astype(dtype, copy=copy), dtype)

    def _binary(self, operation, a, b):
        (a, b), dtype = self._promote(a, b)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return self._result(operation(a, b), dtype)

    def add(self, a, b):
        return self._binary(np.add, a, b)

    def subtract(self, a, b):
        return self._binary(np.subtract, a, b)

    def multiply(self, a, b):
        return self._binary(np.multiply, a, b)

    def divide(self, a, b):
        return self._binary(np.divide, a, b)

    def sum(self, x, axis=None, keepdims=False, dtype=None):
        self._owned(x)
        dtype = x.dtype if dtype is None else self._dtype(dtype)
        if dtype not in NUMERIC_DTYPES:
            raise CapabilityError('sum requires a floating or complex accumulation dtype')
        with np.errstate(over='raise', invalid='raise'):
            return self._result(np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims), dtype)

    def max(self, x, axis=None, keepdims=False):
        self._owned(x)
        return self._result(np.max(x, axis=axis, keepdims=keepdims), x.dtype)

    def min(self, x, axis=None, keepdims=False):
        self._owned(x)
        return self._result(np.min(x, axis=axis, keepdims=keepdims), x.dtype)

    def real(self, x):
        return self._result(np.real(self._owned(x)))

    def imag(self, x):
        return self._result(np.imag(self._owned(x)))

    def matmul(self, a, b):
        self._owned(a)
        self._owned(b)
        return self._binary(np.matmul, a, b)

    def einsum(self, subscripts, *operands):
        if not isinstance(subscripts, str) or '->' not in subscripts:
            raise ValueError('einsum requires explicit output subscripts')
        if not operands:
            raise ValueError('einsum requires operands')
        for operand in operands:
            self._owned(operand)
        arrays, dtype = self._promote(*operands)
        with np.errstate(over='raise', invalid='raise'):
            return self._result(np.einsum(subscripts, *arrays), dtype)

    def _matrix(self, a, *, square=False):
        self._owned(a)
        if a.ndim != 2 or (square and a.shape[0] != a.shape[1]):
            raise CapabilityError('operation requires a two-dimensional ' + ('square ' if square else '') + 'matrix')
        return a

    def qr(self, a, mode='reduced'):
        if mode != 'reduced':
            raise CapabilityError('only reduced QR is supported')
        self._matrix(a)
        return tuple(self._result(x, a.dtype) for x in np.linalg.qr(a, mode='reduced'))

    def svd(self, a, full_matrices=False):
        if full_matrices is not False:
            raise CapabilityError('only reduced SVD is supported')
        self._matrix(a)
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        real_dtype = np.empty((), dtype=a.dtype).real.dtype
        return self._result(u, a.dtype), self._result(s, real_dtype), self._result(vh, a.dtype)

    def eigh(self, a, UPLO='L'):
        if UPLO not in ('L', 'U'):
            raise ValueError('UPLO must be L or U')
        self._matrix(a, square=True)
        w, v = np.linalg.eigh(a, UPLO=UPLO)
        real_dtype = np.empty((), dtype=a.dtype).real.dtype
        return self._result(w, real_dtype), self._result(v, a.dtype)

    def solve(self, a, b):
        self._owned(a)
        self._owned(b)
        check_solve_shapes(a, b)
        (a, b), dtype = self._promote(a, b)
        return self._result(np.linalg.solve(a, b), dtype)

    def norm(self, x, ord=None, axis=None, keepdims=False):
        self._owned(x)
        if axis is None:
            valid = ord is None or (x.ndim == 1 and ord == 2) or (x.ndim == 2 and ord == 'fro')
        elif isinstance(axis, (int, np.integer)):
            valid = ord in (None, 2)
        elif isinstance(axis, tuple) and len(axis) == 2:
            valid = ord in (None, 'fro')
        else:
            valid = False
        if not valid:
            raise CapabilityError('norm supports vector 2-norm and Frobenius norm only')
        if x.ndim == 0 and axis is None:
            out = np.abs(x)
        else:
            out = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        return self._result(out)

    def _update(self, name, x, idx, value):
        self._owned(x)
        if isinstance(idx, np.ndarray):
            if x.ndim != 1 or idx.ndim != 1 or idx.dtype.kind not in 'iu':
                raise CapabilityError('advanced index must be one-dimensional integer indices on a vector')
            if np.any(idx >= x.size) or (idx.dtype.kind == 'i' and np.any(idx < -x.size)):
                raise IndexError('index out of bounds')
            idx = idx.astype(np.intp, copy=False) % max(x.size, 1)
            if name == 'set' and len(np.unique(idx)) != len(idx):
                raise CapabilityError('duplicate assignment indices')
        else:
            parts = idx if isinstance(idx, tuple) else (idx,)
            if not all(isinstance(part, (int, np.integer, slice)) and not isinstance(part, (bool, np.bool_)) for part in parts):
                raise CapabilityError('unsupported index pattern')
            x[idx]  # Validate basic indexing before allocating an output.
        (x, value), dtype = self._promote(x, value)
        y = x.copy()
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if name == 'set':
                y[idx] = value
            else:
                {'add': np.add, 'sub': np.subtract, 'mul': np.multiply}[name].at(y, idx, value)
        return self._result(y, dtype)

    def at_set(self, x, idx, value):
        return self._update('set', x, idx, value)

    def at_add(self, x, idx, value):
        return self._update('add', x, idx, value)

    def at_sub(self, x, idx, value):
        return self._update('sub', x, idx, value)

    def at_mul(self, x, idx, value):
        return self._update('mul', x, idx, value)

    def sync(self):
        return self.context.adapter.sync()

    def scalar(self, x):
        self._owned(x)
        if x.ndim != 0:
            raise ValueError('scalar requires a zero-dimensional array')
        self.sync()
        return x.item()
