"""Sharded Krylov and Davidson solver adapters."""

from collections.abc import Mapping
import importlib
from numbers import Integral
import numpy as np

from renormalizer.backend._distributed.local_operator import DistributedLocalOperator
from renormalizer.backend._distributed.sharding import DistributedTensor
from renormalizer.lib.krylov.krylov import (
    _lanczos_expm,
    _projected_exponential_coefficients,
)


_davidson_backend = importlib.import_module("renormalizer.lib.davidson.backend")

_KRYLOV_KEYS = frozenset({"block_size"})
_DAVIDSON_KEYS = frozenset(
    {"diagonal", "tol", "max_cycle", "max_space", "lindep", "require_convergence"}
)


def _host_scalar(value):
    return _host_array(value).reshape(-1)[0].item()


def _host_array(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    getter = getattr(value, "get", None)
    if callable(getter):
        value = getter()
    return np.asarray(value)


def distributed_vdot(x_local, y_local, collective):
    """Return the conjugating global inner product as a host scalar."""
    if tuple(x_local.shape) != tuple(y_local.shape):
        raise ValueError("distributed vdot operands must have identical shapes")
    local = (x_local.conj() * y_local).sum().reshape(1)
    reduced = collective.allreduce(local, op="sum")
    return _host_scalar(reduced)


def distributed_norm(x_local, collective):
    """Return the global Euclidean norm using one scalar sum allreduce."""
    local = (x_local.conj() * x_local).real.sum().reshape(1)
    reduced = _host_scalar(collective.allreduce(local, op="sum"))
    return float(np.sqrt(max(0.0, reduced.real)))


def _validated_mapping(config, allowed, solver_name):
    if not isinstance(config, Mapping):
        raise TypeError(
            "{} config must be a read-only Mapping input".format(solver_name)
        )
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            "unknown {} config keys: {}".format(
                solver_name, ", ".join(sorted(map(str, unknown)))
            )
        )
    return config


def _positive_integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def _int64_control(value, name, *, minimum=1):
    value = _positive_integer(value, name, minimum=minimum)
    if value > np.iinfo(np.int64).max:
        raise ValueError("{} exceeds the supported integer range".format(name))
    return value


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be a positive real number".format(name))
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("{} must be a positive finite number".format(name))
    return value


def _finite_coefficient(value):
    if isinstance(value, bool):
        raise TypeError("coefficient must be a finite real or complex scalar")
    if isinstance(value, Integral) and abs(int(value)) > 2**53:
        raise ValueError(
            "integer coefficient exceeds the binary64 exact integer range"
        )
    try:
        normalized = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            "coefficient must be a finite real or complex scalar"
        ) from error
    if not np.isfinite(normalized.real) or not np.isfinite(normalized.imag):
        raise ValueError("coefficient must be finite")
    if np.iscomplexobj(value):
        return normalized
    return float(normalized.real)


def _float_control_bits(*values):
    normalized = [0.0 if value == 0 else value for value in values]
    return np.asarray(normalized, dtype=np.float64).view(np.int64)


def _synchronized_controls(parse, operator, collective, solver_name):
    error = None
    try:
        values, host_controls = parse()
    except BaseException as caught:
        error = caught
        values = None
        host_controls = None

    namespace = operator.backend.array_namespace
    status = namespace.asarray([1 if error is not None else 0], dtype=np.int32)
    failed = _host_scalar(collective.allreduce(status, op="max"))
    if failed:
        if error is not None:
            raise error
        raise ValueError(
            "distributed {} control validation failed".format(solver_name)
        )

    controls = namespace.asarray(host_controls)
    minimum = _host_array(collective.allreduce(controls, op="min"))
    maximum = _host_array(collective.allreduce(controls, op="max"))
    if not np.array_equal(minimum, maximum):
        raise ValueError(
            "distributed {} controls must agree across ranks".format(solver_name)
        )
    return values


def _vector_contract_error(operator, vector, collective):
    if not isinstance(vector, DistributedTensor):
        return TypeError("vector must be a DistributedTensor")
    if vector.spec != operator.plan.input_sharding:
        return ValueError("vector sharding spec does not match operator input")
    if operator.plan.input_sharding != operator.plan.output_sharding:
        return ValueError("operator input and output sharding must be equal")
    if len(vector.spec.global_shape) != 1 or vector.spec.axis != 0:
        return NotImplementedError(
            "Task 15 supports only dense contiguous axis-sharded vectors"
        )
    if vector.rank != operator.context.rank:
        return ValueError("distributed vector rank does not match operator context")
    if collective is not operator.collective:
        return ValueError("solver collective must be the operator collective")
    if collective.rank != vector.rank or collective.size != vector.spec.parts:
        return ValueError("collective rank or size does not match distributed vector")
    try:
        operator.backend._validate_execution_array(vector.local_array)
    except (TypeError, ValueError) as error:
        return error
    if not bool(vector.local_array.flags.c_contiguous):
        return ValueError("distributed vector local storage must be C contiguous")
    variable_ref = next(
        ref
        for ref in operator.plan.execution_plan.inputs
        if ref.key == operator.plan.variable_key
    )
    expected_dtype = np.dtype(variable_ref.spec.dtype)
    if np.dtype(vector.local_array.dtype) != expected_dtype:
        return ValueError(
            "distributed vector dtype does not match operator input dtype"
        )
    if np.dtype(operator.plan.execution_plan.output.spec.dtype) != expected_dtype:
        return ValueError("iterative operator input and output dtype must agree")
    return None


def _solver_preflight(operator, vector, collective):
    if not isinstance(operator, DistributedLocalOperator):
        raise TypeError("operator must be a DistributedLocalOperator")
    operator._preflight_setup()
    backend_name = getattr(operator.backend, "name", None)
    if backend_name not in {"numpy", "cupy"}:
        raise NotImplementedError(
            "distributed solvers support only NumPy and CuPy backends"
        )
    error = _vector_contract_error(operator, vector, collective)
    namespace = operator.backend.array_namespace
    status = namespace.asarray(
        [
            1 if error is not None else 0,
            1 if isinstance(error, NotImplementedError) else 0,
        ],
        dtype=np.int32,
    )
    global_status = _host_array(
        operator.collective.allreduce(status, op="max")
    ).reshape(-1)
    failed = global_status[0]
    if failed:
        if global_status[1]:
            if isinstance(error, NotImplementedError):
                raise error
            raise NotImplementedError(
                "Task 15 supports only dense contiguous axis-sharded vectors"
            )
        if error is not None:
            raise ValueError("distributed solver preflight failed") from error
        raise ValueError("distributed solver preflight failed")


class _DistributedVectorOps:
    def __init__(self, backend, collective):
        self.backend = backend
        self.namespace = backend.array_namespace
        self.collective = collective

    def asarray(self, vector):
        return self.backend.asarray(vector)

    def copy(self, vector):
        return vector.copy()

    def norm(self, vector):
        return distributed_norm(vector, self.collective)

    def vdot(self, left, right):
        return distributed_vdot(left, right, self.collective)

    def projected_exponential(self, alpha, beta, basis, vector_norm, coefficient):
        projected = _projected_exponential_coefficients(
            alpha, beta, vector_norm, coefficient
        )
        vectors = self.namespace.stack(basis, axis=1)
        return vectors @ self.backend.asarray(projected)

    def allclose(self, left, right):
        mismatch = self.namespace.logical_not(
            self.namespace.allclose(left, right)
        ).astype(np.int32).reshape(1)
        global_mismatch = self.collective.allreduce(mismatch, op="max")
        return _host_scalar(global_mismatch) == 0

    def linear_combination(self, coefficients, vectors):
        dtype = np.result_type(vectors[0].dtype, coefficients.dtype)
        result = self.namespace.zeros_like(vectors[0], dtype=dtype)
        for coefficient, vector in zip(coefficients, vectors):
            result += coefficient * vector
        return result


def run_sharded_krylov(operator, vector, coefficient, *, collective, config):
    """Apply a Hermitian exponential to a dense axis-sharded vector."""
    _solver_preflight(operator, vector, collective)

    def parse_controls():
        validated = _validated_mapping(config, _KRYLOV_KEYS, "Krylov")
        block_size = _int64_control(
            validated.get("block_size", 50), "block_size"
        )
        parsed_coefficient = _finite_coefficient(coefficient)
        coefficient_bits = _float_control_bits(
            complex(parsed_coefficient).real,
            complex(parsed_coefficient).imag,
        )
        controls = np.concatenate(
            (coefficient_bits, np.asarray([block_size], dtype=np.int64))
        )
        return (parsed_coefficient, block_size), controls

    coefficient, block_size = _synchronized_controls(
        parse_controls, operator, collective, "Krylov"
    )
    vector_ops = _DistributedVectorOps(operator.backend, collective)
    local_result, iterations = _lanczos_expm(
        lambda local: operator(local).copy(),
        coefficient,
        vector.local_array,
        block_size=block_size,
        vector_ops=vector_ops,
        global_size=int(np.prod(vector.global_shape)),
    )
    return (
        DistributedTensor(vector.spec, vector.rank, local_result.copy()),
        iterations,
    )


def _validated_davidson_config(config):
    config = _validated_mapping(config, _DAVIDSON_KEYS, "Davidson")
    if "diagonal" not in config:
        raise ValueError("Davidson config requires diagonal")
    require_convergence = config.get("require_convergence", False)
    if type(require_convergence) is not bool:
        raise TypeError("require_convergence must be a bool")
    return {
        "diagonal": config["diagonal"],
        "tol": _positive_float(config.get("tol", 1e-12), "tol"),
        "max_cycle": _positive_integer(config.get("max_cycle", 50), "max_cycle"),
        "max_space": _positive_integer(
            config.get("max_space", 12), "max_space", minimum=2
        ),
        "lindep": _davidson_backend._validated_lindep(
            config.get("lindep", 1e-14)
        ),
        "require_convergence": require_convergence,
    }


def _diagonal_contract_error(diagonal, vector, operator):
    if not isinstance(diagonal, DistributedTensor):
        return TypeError("Davidson diagonal must be a DistributedTensor")
    if diagonal.spec != vector.spec or diagonal.rank != vector.rank:
        return ValueError("Davidson diagonal sharding must match the vector")
    try:
        operator.backend._validate_execution_array(diagonal.local_array)
    except (TypeError, ValueError) as error:
        return error
    if not bool(diagonal.local_array.flags.c_contiguous):
        return ValueError("Davidson diagonal local storage must be C contiguous")
    if np.dtype(diagonal.local_array.dtype) != np.dtype(vector.local_array.dtype):
        return ValueError("Davidson diagonal dtype must match the vector dtype")
    return None


def run_sharded_davidson(operator, vector, *, collective, config):
    """Find the lowest eigenpair without materializing sharded basis vectors."""
    _solver_preflight(operator, vector, collective)

    def parse_controls():
        values = _validated_davidson_config(config)
        max_cycle = _int64_control(values["max_cycle"], "max_cycle")
        max_space = _int64_control(values["max_space"], "max_space", minimum=2)
        controls = np.asarray(
            [
                _float_control_bits(values["tol"])[0],
                max_cycle,
                max_space,
                _float_control_bits(values["lindep"])[0],
                int(values["require_convergence"]),
            ],
            dtype=np.int64,
        )
        return values, controls

    values = _synchronized_controls(
        parse_controls, operator, collective, "Davidson"
    )
    diagonal = values["diagonal"]
    require_convergence = values["require_convergence"]
    diagonal_error = _diagonal_contract_error(diagonal, vector, operator)
    namespace = operator.backend.array_namespace
    status = namespace.asarray(
        [1 if diagonal_error is not None else 0], dtype=np.int32
    )
    failed = _host_scalar(collective.allreduce(status, op="max"))
    if failed:
        if diagonal_error is not None:
            raise ValueError(
                "distributed Davidson diagonal preflight failed"
            ) from diagonal_error
        raise ValueError("distributed Davidson diagonal preflight failed")

    vector_ops = _DistributedVectorOps(operator.backend, collective)
    energy, local_result, info = _davidson_backend._davidson_single_root(
        lambda local: operator(local).copy(),
        vector.local_array,
        diagonal.local_array,
        vector_ops=vector_ops,
        global_size=int(np.prod(vector.global_shape)),
        tol=values["tol"],
        max_cycle=values["max_cycle"],
        max_space=values["max_space"],
        lindep=values["lindep"],
    )
    if require_convergence and not info.converged:
        raise RuntimeError("sharded Davidson did not converge")
    return energy, DistributedTensor(
        vector.spec, vector.rank, local_result.copy()
    ), info


__all__ = [
    "distributed_norm",
    "distributed_vdot",
    "run_sharded_davidson",
    "run_sharded_krylov",
]
