# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import weakref
import logging
import time
from collections import OrderedDict
from dataclasses import replace
from typing import List, Union

from renormalizer.backend.boundary import eye_like, scalar_to_python as _scalar_to_python
from renormalizer.mps.backend import np, backend, xp
from renormalizer.utils import profiling

logger = logging.getLogger(__name__)

_PAIR_CONTRACTION_PLAN_CACHE_MAXSIZE = 512
_PAIR_CONTRACTION_PLAN_CACHE = OrderedDict()



def _to_numpy_dtype(dtype):
    """Convert a backend-specific dtype (e.g. torch.float64) to a numpy dtype."""
    if dtype is None:
        return None
    if isinstance(dtype, np.dtype):
        return dtype
    try:
        return np.dtype(dtype)
    except TypeError:
        pass
    # Fallback: convert via dtype name (works for torch dtypes like torch.float64)
    try:
        name = str(dtype).rsplit(".", 1)[-1]  # "torch.float64" -> "float64"
        return np.dtype(name)
    except (TypeError, ValueError, AttributeError):
        raise TypeError("Cannot convert {0!r} to a numpy dtype".format(dtype))


class Matrix:
    # Matrix is a legacy host-resident container; backend arrays are the hot-path execution tensors.
    is_legacy_host_container = True
    is_backend_execution_tensor = False
    execution_role = "legacy_host_container"

    def __init__(self, array, dtype=None):
        assert array is not None
        array = asnumpy(array)
        dtype = _to_numpy_dtype(dtype)
        np_real = _to_numpy_dtype(backend.real_dtype)
        np_complex = _to_numpy_dtype(backend.complex_dtype)
        if dtype is not None and dtype == np_real:
            # forbid unchecked casting
            assert not np.iscomplexobj(array)
        if dtype is None:
            if np.iscomplexobj(array):
                dtype = np_complex
            else:
                dtype = np_real
        self.array: np.ndarray = np.asarray(array, dtype=dtype)
        self.original_shape = self.array.shape
        self.sigmaqn = None
        pass  # backend.running removed

    def __getattr__(self, item):
        # use this way to obtain ``array`` to prevent infinite recursion during multi-processing
        # see https://stackoverflow.com/questions/22781872/python-pickle-got-acycle-recursion-with-getattr
        array = super().__getattribute__("array")
        res = getattr(array, item)
        if isinstance(res, np.ndarray):
            return Matrix(res)
        functiontype = type([].append)
        if isinstance(res, functiontype):

            def wrapped(*args, **kwargs):
                res2 = res(*args, **kwargs)
                if isinstance(res2, np.ndarray):
                    return Matrix(res2)
                return res2

            return wrapped
        return res

    # for debugging purpose (let it shown in debuggers)
    @property
    def dtype(self):
        return self.array.dtype

    def astype(self, dtype):
        np_dtype = _to_numpy_dtype(dtype)
        np_real = _to_numpy_dtype(backend.real_dtype)
        np_complex = _to_numpy_dtype(backend.complex_dtype)
        assert not (self.dtype == np_complex and np_dtype == np_real)
        self.array = np.asarray(self.array, dtype=np_dtype)
        return self

    def abs(self):
        return self.__class__(np.abs(self.array))

    def norm(self):
        return np.linalg.norm(self.array.flatten())

    # physical indices exclude first and last indices
    @property
    def pdim(self):
        return self.original_shape[1:-1]

    @property
    def pdim_prod(self):
        return np.prod(self.pdim)

    @property
    def bond_dim(self):
        return self.original_shape[0], self.original_shape[-1]

    @property
    def r_combine_shape(self):
        return self.original_shape[0], np.prod(self.original_shape[1:])

    @property
    def l_combine_shape(self):
        return np.prod(self.original_shape[:-1]), self.original_shape[-1]

    def r_combine(self):
        return self.reshape(self.r_combine_shape)

    def l_combine(self):
        return self.reshape(self.l_combine_shape)

    def check_lortho(self, rtol: float = None, atol: float = None):
        """
        check L-orthogonal
        """
        if atol is None:
            atol = backend.canonical_atol
        if rtol is None:
            rtol = backend.canonical_rtol
        tensm = asxp(self.array.reshape([np.prod(self.shape[:-1]), self.shape[-1]]))
        s = tensm.T.conj() @ tensm
        return xp.allclose(s, eye_like(s.shape[0], s, xp), rtol=rtol, atol=atol)

    def check_rortho(self, rtol: float = None, atol: float = None):
        """
        check R-orthogonal
        """
        if atol is None:
            atol = backend.canonical_atol
        if rtol is None:
            rtol = backend.canonical_rtol
        tensm = asxp(self.array.reshape([self.shape[0], np.prod(self.shape[1:])]))
        s = tensm @ tensm.T.conj()
        return xp.allclose(s, eye_like(s.shape[0], s, xp), rtol=rtol, atol=atol)

    def to_complex(self):
        # `xp.array` always creates new array, so to_complex means copy, which is
        # in accordance with NumPy
        return np.array(self.array, dtype=_to_numpy_dtype(backend.complex_dtype))

    def copy(self):
        new = self.__class__(self.array.copy(), self.array.dtype)
        new.original_shape = self.original_shape
        new.sigmaqn = self.sigmaqn
        return new

    def nearly_zero(self):
        if backend.is_32bits:
            atol = 1e-10
        else:
            atol = 1e-20
        return np.allclose(self.array, np.zeros_like(self.array), atol=atol)

    def __hash__(self):
        return hash((self.array.shape, self.array.tobytes()))

    def __getitem__(self, item):
        res = self.array.__getitem__(item)
        if res.ndim != 0:
            return self.__class__(res)
        else:
            return res

    def __setitem__(self, key, value):
        if isinstance(value, Matrix):
            value = value.array
        self.array[key] = value

    def __add__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__add__(other))

    def __radd__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__radd__(other))

    def __mul__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__mul__(other))

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__rmul__(other))

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__truediv__(other))

    def __repr__(self):
        return f"<Matrix at 0x{id(self):x} {self.shape} {self.dtype}>"

    def __str__(self):
        return str(self.array)

    def __float__(self):
        return self.array.__float__()

    def __complex__(self):
        return self.array.__complex__()


def zeros(shape, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(np.zeros(shape), dtype=dtype)


def eye(N, M=None, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(np.eye(N, M), dtype=dtype)


def ones(shape, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(np.ones(shape), dtype=dtype)


def einsum(subscripts, *operands):
    return Matrix(backend.contract(subscripts, *[asxp(operand) for operand in operands]))


def _normalized_tensordot_axes(axes, a_ndim, b_ndim):
    if isinstance(axes, int):
        return list(range(a_ndim - axes, a_ndim)), list(range(axes))
    left_axes, right_axes = axes
    if isinstance(left_axes, int):
        left_axes = [left_axes]
    else:
        left_axes = list(left_axes)
    if isinstance(right_axes, int):
        right_axes = [right_axes]
    else:
        right_axes = list(right_axes)
    return left_axes, right_axes


def tensordot(a: Union[Matrix, np.ndarray], b: Union[Matrix, np.ndarray, xp.ndarray], axes) -> xp.ndarray:
    a_arr = asxp(a)
    b_arr = asxp(b)
    left_axes, right_axes = _normalized_tensordot_axes(axes, a_arr.ndim, b_arr.ndim)
    result_ndim = (a_arr.ndim - len(left_axes)) + (b_arr.ndim - len(right_axes))
    if backend.name == "cupynumeric" and result_ndim > 4:
        raise NotImplementedError(
            "cupynumeric backend does not support tensor contractions producing rank > 4 "
            "for the current FMO workload. Legate raises a lower-level runtime error here; "
            "Renormalizer now fails fast at the backend boundary."
        )
    if not profiling.should_record_op():
        return backend.tensordot(a_arr, b_arr, axes)
    started = time.perf_counter()
    result = backend.tensordot(a_arr, b_arr, axes)
    profiling.record(
        "tensordot",
        backend=backend.name,
        input_shapes=[tuple(a_arr.shape), tuple(b_arr.shape)],
        operand_array_types=profiling.array_type_names((a_arr, b_arr)),
        operand_array_backends=profiling.array_backend_names((a_arr, b_arr)),
        axes=(left_axes, right_axes),
        output_shape=tuple(result.shape),
        **profiling.tensordot_compute_payload(a_arr, b_arr, axes, result),
        wall_s=time.perf_counter() - started,
    )
    return result


def moveaxis(a: Matrix, source, destination):
    return Matrix(np.moveaxis(a.array, source, destination))


def vstack(tup):
    return Matrix(np.vstack([m.array for m in tup]))


def dstack(tup):
    return Matrix(np.dstack([m.array for m in tup]))


def concatenate(arrays, axis=None):
    return Matrix(np.concatenate([m.array for m in arrays], axis))


# can only use numpy for now. see gh-cupy-1946
def allclose(a, b, rtol=1.0e-5, atol=1.0e-8):
    if isinstance(a, Matrix):
        a = a.array
    else:
        a = np.asarray(a)
    if isinstance(b, Matrix):
        b = b.array
    else:
        b = np.asarray(b)
    return np.allclose(a, b, rtol=rtol, atol=atol)


def multi_tensor_contract(path, *operands: [List[Union[Matrix, np.ndarray, xp.ndarray]]]):
    """
    ipath[0] is the index of the mat
    ipaht[1] is the contraction index
    oeprands is the arrays

    For example:  in mpompsmat.py
    path = [([0, 1],"fdla, abc -> fdlbc")   ,\
            ([2, 0],"fdlbc, gdeb -> flcge") ,\
            ([1, 0],"flcge, helc -> fgh")]
    outtensor = tensorlib.multi_tensor_contract(path, MPSconj[isite], intensor,
            MPO[isite], MPS[isite])
    """
    if not profiling.should_record_op():
        return _multi_tensor_contract_impl(path, operands)
    started = time.perf_counter()
    initial_operands = tuple(operands)
    with profiling.span("multi_tensor_contract", backend=backend.name, contraction_count=len(path)):
        result = _multi_tensor_contract_impl(path, operands)
        profiling.record(
            "multi_tensor_contract",
            backend=backend.name,
            **profiling.multi_tensor_contract_payload(path, initial_operands, result),
            wall_s=time.perf_counter() - started,
        )
        return result


def _multi_tensor_contract_impl(path, operands):
    operands = list(operands)
    for ipath in path:

        input_str, results_str = ipath[1].split("->")
        input_str = input_str.split(",")
        input_str = [x.replace(" ", "") for x in input_str]
        results_str = results_str.replace(" ", "")
        results_set = set(results_str)
        inputs_set = set(input_str[0] + input_str[1])
        idx_removed = inputs_set - (inputs_set & results_set)

        tmpmat = pair_tensor_contract(
            operands[ipath[0][0]],
            input_str[0],
            operands[ipath[0][1]],
            input_str[1],
            idx_removed,
            output_modes=results_str,
        )

        for x in sorted(ipath[0], reverse=True):
            del operands[x]

        operands.append(tmpmat)

    return operands[0]


def pair_tensor_contract(
    view_left: Union[Matrix, np.ndarray, xp.ndarray],
    input_left,
    view_right: Union[Matrix, np.ndarray, xp.ndarray],
    input_right,
    idx_removed,
    output_modes=None,
):
    left_array = asxp(view_left)
    right_array = asxp(view_right)
    removed = set(idx_removed)
    input_left, input_right, output_modes = _expand_pair_contract_modes(
        input_left,
        input_right,
        removed,
        left_array,
        right_array,
        output_modes=output_modes,
    )
    equation = "{0},{1}->{2}".format("".join(input_left), "".join(input_right), "".join(output_modes))
    spec = backend.parse_einsum(equation, left_array, right_array)
    if profiling.should_record_op():
        plan = backend.plan_contraction(spec)
    else:
        plan = _cached_pair_contraction_plan(equation, spec, left_array, right_array)
    return backend.execute(plan)


def _expand_pair_contract_modes(input_left, input_right, removed, left_array, right_array, output_modes=None):
    input_left = tuple(str(input_left).replace(" ", ""))
    input_right = tuple(str(input_right).replace(" ", ""))
    removed = set(removed)
    left_rank = len(getattr(left_array, "shape", ()))
    right_rank = len(getattr(right_array, "shape", ()))
    if len(input_left) > left_rank:
        raise ValueError("left contraction labels exceed array rank")
    if len(input_right) > right_rank:
        raise ValueError("right contraction labels exceed array rank")

    left_contract_positions = tuple(index for index, mode in enumerate(input_left) if mode in removed)
    right_contract_positions = tuple(index for index, mode in enumerate(input_right) if mode in removed)
    left_uncontracted_count = left_rank - len(left_contract_positions)
    right_uncontracted_count = right_rank - len(right_contract_positions)

    if output_modes is None:
        output_modes = tuple(mode for mode in input_left if mode not in removed)
        output_modes += tuple(mode for mode in input_right if mode not in removed)
    else:
        output_modes = tuple(str(output_modes).replace(" ", ""))
    expected_output_rank = left_uncontracted_count + right_uncontracted_count
    if len(output_modes) != expected_output_rank:
        raise ValueError(
            "pair contraction output labels have rank {0}, expected {1}".format(
                len(output_modes),
                expected_output_rank,
            )
        )

    left_output_modes = output_modes[:left_uncontracted_count]
    right_output_modes = output_modes[left_uncontracted_count:]

    def expand(modes, rank, output):
        output_iter = iter(output)
        expanded = []
        for axis in range(rank):
            if axis < len(modes) and modes[axis] in removed:
                expanded.append(modes[axis])
            else:
                expanded.append(next(output_iter))
        return tuple(expanded)

    return (
        expand(input_left, left_rank, left_output_modes),
        expand(input_right, right_rank, right_output_modes),
        output_modes,
    )


def _array_plan_cache_key(array):
    try:
        info = backend.array_info(array)
        return (
            info.shape,
            str(info.dtype),
            info.strides,
            info.order,
            info.contiguous,
            str(info.device),
        )
    except Exception:
        strides = getattr(array, "strides", None)
        if callable(strides):
            strides = strides()
        if strides is not None:
            strides = tuple(int(stride) for stride in strides)
        return (
            tuple(int(dim) for dim in getattr(array, "shape", ())),
            str(getattr(array, "dtype", None)),
            strides,
            "unknown",
            False,
            str(backend.current_device()),
        )


def _pair_contraction_plan_cache_key(equation, spec):
    return (
        backend.name,
        str(backend.current_device()),
        str(backend.fallback_policy),
        equation,
        tuple(operand.modes for operand in spec.operands),
        tuple(spec.output_modes),
        tuple(_array_plan_cache_key(operand.array) for operand in spec.operands),
    )


def _refresh_matmul_plan_arrays(plan, left_array, right_array):
    if not hasattr(plan, "descs"):
        return plan
    descs = []
    for index, desc in enumerate(plan.descs):
        if index == 0:
            descs.append(replace(desc, A=left_array, B=right_array, C=None))
        else:
            descs.append(desc)
    return replace(plan, descs=tuple(descs))


def _refresh_pair_contraction_plan_arrays(plan, spec, left_array, right_array):
    steps = []
    for step in plan.steps:
        steps.append(replace(step, plan=_refresh_matmul_plan_arrays(step.plan, left_array, right_array)))
    return replace(plan, steps=tuple(steps), input_specs=spec.operands)


def _cached_pair_contraction_plan(equation, spec, left_array, right_array):
    key = _pair_contraction_plan_cache_key(equation, spec)
    cached = _PAIR_CONTRACTION_PLAN_CACHE.get(key)
    if cached is None:
        plan = backend.plan_contraction(spec)
        _PAIR_CONTRACTION_PLAN_CACHE[key] = plan
        if len(_PAIR_CONTRACTION_PLAN_CACHE) > _PAIR_CONTRACTION_PLAN_CACHE_MAXSIZE:
            _PAIR_CONTRACTION_PLAN_CACHE.popitem(last=False)
        return plan
    _PAIR_CONTRACTION_PLAN_CACHE.move_to_end(key)
    return _refresh_pair_contraction_plan_arrays(cached, spec, left_array, right_array)


def asnumpy(array):
    if array is None:
        return None
    if isinstance(array, Matrix):
        return array.array
    if isinstance(array, list):
        return np.array(array)
    return backend.to_host(array)


def asxp(array):
    if array is None:
        return None
    if isinstance(array, Matrix):
        array = array.array
    return backend.to_backend(array)


def scalar_to_python(array):
    return _scalar_to_python(array, backend)


def asxp_oe_args(oe_args):
    # opt_einsum.contract args in interleaved format
    new_args = []
    # the last one is the output index
    for i in range(len(oe_args) - 1):
        if i % 2 == 0:
            new_args.append(asxp(oe_args[i]))
        else:
            new_args.append(oe_args[i])
    new_args.append(oe_args[-1])
    return new_args
