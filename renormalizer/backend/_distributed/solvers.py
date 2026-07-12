"""Sharded Krylov and Davidson solver adapters."""

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from numbers import Integral
import numpy as np
from typing import Protocol, runtime_checkable

from renormalizer.backend._distributed.sharding import DistributedTensor, ShardingSpec
from renormalizer.lib.krylov.krylov import (
    _lanczos_expm,
    _projected_exponential_coefficients,
)


_davidson_backend = importlib.import_module("renormalizer.lib.davidson.backend")

_KRYLOV_KEYS = frozenset({"block_size", "max_krylov_vectors"})
_DAVIDSON_KEYS = frozenset(
    {"diagonal", "tol", "max_cycle", "max_space", "lindep", "require_convergence"}
)
_MAX_METADATA_INT64 = int(np.iinfo(np.int64).max)


def _checked_shape_elements(shape, name):
    product = 1
    for value in tuple(shape):
        if type(value) is not int or value < 0:
            raise ValueError(
                "{} dimensions must be non-negative Python integers".format(name)
            )
        product *= value
        if product > _MAX_METADATA_INT64:
            raise OverflowError("{} exceeds the supported int64 range".format(name))
    return product


@dataclass(frozen=True)
class SolverRankMemoryEstimate:
    """Solver-owned bytes live in each temporal phase for one rank."""

    hv_retained_bytes: int
    hv_transient_bytes: int
    la_retained_bytes: int
    la_transient_bytes: int
    communication_device_bytes: int
    host_peak_bytes: int

    def __post_init__(self):
        for name in (
            "hv_retained_bytes",
            "hv_transient_bytes",
            "la_retained_bytes",
            "la_transient_bytes",
            "communication_device_bytes",
            "host_peak_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError("solver memory fields must be integers")
            if value < 0:
                raise ValueError("solver memory fields must be non-negative")


@dataclass(frozen=True, order=True)
class KrylovCoefficientIdentity:
    """Exact recurrence-significant binary64 coefficient identity."""

    kind: str
    real_bits: str
    imag_bits: str

    def __post_init__(self):
        if self.kind not in {"real", "complex"}:
            raise ValueError("Krylov coefficient kind must be 'real' or 'complex'")
        for name in ("real_bits", "imag_bits"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 16
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("Krylov coefficient bits must be canonical")
        real = _coefficient_float(self.real_bits)
        imag = _coefficient_float(self.imag_bits)
        if not np.isfinite(real) or not np.isfinite(imag):
            raise ValueError("Krylov coefficient identity must be finite")
        if self.kind == "real" and (self.imag_bits != "0000000000000000"):
            raise ValueError("real Krylov coefficient identity has imaginary bits")
        if self.kind == "complex" and imag == 0.0:
            raise ValueError(
                "complex Krylov coefficient identity requires nonzero imaginary part"
            )
        if (
            _coefficient_bits(real) != self.real_bits
            or _coefficient_bits(imag) != self.imag_bits
        ):
            raise ValueError("Krylov coefficient identity is not canonical")

    @property
    def coefficient(self):
        real = _coefficient_float(self.real_bits)
        if self.kind == "real":
            return real
        return complex(real, _coefficient_float(self.imag_bits))


@dataclass(frozen=True)
class SolverMemoryProfile:
    """Deterministic rank-local liveness profile for one solver recurrence."""

    solver_kind: str
    rank_estimates: tuple[SolverRankMemoryEstimate, ...]
    bounded: bool
    basis_vectors: int | None
    global_shape: tuple[int, ...] | None = None
    sharding: ShardingSpec | None = None
    dtype: str | None = None
    itemsize: int | None = None
    global_vector_count: int | None = None
    rank_counts: tuple[int, ...] | None = None
    allocation_vectors: int | None = None
    projected_host_bytes: int | None = None
    block_size: int | None = None
    max_krylov_vectors: int | None = None
    max_space: int | None = None
    scalar_dtype: str | None = None
    scalar_itemsize: int | None = None
    result_dtype: str | None = None
    result_itemsize: int | None = None
    coefficient_identity: KrylovCoefficientIdentity | None = None

    def __post_init__(self):
        if self.solver_kind not in {"krylov", "davidson"}:
            raise ValueError("solver_kind must be 'krylov' or 'davidson'")
        try:
            estimates = tuple(self.rank_estimates)
        except TypeError as error:
            raise TypeError("rank_estimates must be an iterable") from error
        if not estimates or any(
            not isinstance(estimate, SolverRankMemoryEstimate) for estimate in estimates
        ):
            raise ValueError(
                "rank_estimates must contain SolverRankMemoryEstimate values"
            )
        if type(self.bounded) is not bool:
            raise TypeError("bounded must be a boolean")
        if self.basis_vectors is not None and (
            type(self.basis_vectors) is not int or self.basis_vectors <= 0
        ):
            raise ValueError("basis_vectors must be a positive integer or None")
        if self.bounded and self.basis_vectors is None:
            raise ValueError("bounded solver profiles require basis_vectors")
        object.__setattr__(self, "rank_estimates", estimates)
        identity = (
            self.global_shape,
            self.sharding,
            self.dtype,
            self.itemsize,
            self.global_vector_count,
            self.rank_counts,
            self.allocation_vectors,
            self.projected_host_bytes,
        )
        extensions = (
            self.scalar_dtype,
            self.scalar_itemsize,
            self.result_dtype,
            self.result_itemsize,
        )
        if all(value is None for value in identity + extensions):
            return
        if any(value is None for value in identity):
            raise ValueError("solver profile identity metadata must be complete")
        if all(value is None for value in extensions):
            input_dtype = _profile_dtype(self.dtype)
            scalar_dtype = (
                np.dtype(np.float64)
                if self.solver_kind == "krylov"
                else np.dtype(np.result_type(input_dtype, np.float64))
            )
            result_dtype = np.dtype(np.result_type(input_dtype, scalar_dtype))
            object.__setattr__(self, "scalar_dtype", scalar_dtype.name)
            object.__setattr__(self, "scalar_itemsize", scalar_dtype.itemsize)
            object.__setattr__(self, "result_dtype", result_dtype.name)
            object.__setattr__(self, "result_itemsize", result_dtype.itemsize)
        elif any(value is None for value in extensions):
            raise ValueError("solver profile identity metadata must be complete")
        if not isinstance(self.sharding, ShardingSpec):
            raise TypeError("solver profile sharding must be a ShardingSpec")
        global_shape = tuple(self.global_shape)
        if global_shape != self.sharding.global_shape:
            raise ValueError("solver profile global shape must match sharding")
        normalized_dtype = _profile_dtype(self.dtype)
        if self.dtype != normalized_dtype.name:
            raise ValueError("solver profile dtype must be canonical")
        if type(self.itemsize) is not int or self.itemsize != normalized_dtype.itemsize:
            raise ValueError("solver profile itemsize does not match dtype")
        scalar_dtype = _profile_dtype(self.scalar_dtype)
        result_dtype = _profile_dtype(self.result_dtype)
        if self.scalar_dtype != scalar_dtype.name:
            raise ValueError("solver profile scalar dtype must be canonical")
        if (
            type(self.scalar_itemsize) is not int
            or self.scalar_itemsize != scalar_dtype.itemsize
        ):
            raise ValueError("solver profile scalar itemsize does not match dtype")
        if self.solver_kind == "krylov":
            if scalar_dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
                raise ValueError("Krylov profile scalar dtype is not canonical")
            coefficient_identity = self.coefficient_identity
            if coefficient_identity is None:
                coefficient_identity = canonical_krylov_coefficient(1.0)
                object.__setattr__(self, "coefficient_identity", coefficient_identity)
            elif not isinstance(coefficient_identity, KrylovCoefficientIdentity):
                raise TypeError(
                    "Krylov coefficient identity must be canonical metadata"
                )
            expected_scalar_dtype = np.dtype(
                np.complex128 if coefficient_identity.kind == "complex" else np.float64
            )
            if scalar_dtype != expected_scalar_dtype:
                raise ValueError(
                    "Krylov coefficient identity does not match scalar dtype"
                )
        else:
            if self.coefficient_identity is not None:
                raise ValueError(
                    "Davidson profile must not define a Krylov coefficient"
                )
            expected_scalar_dtype = np.dtype(
                np.result_type(normalized_dtype, np.float64)
            )
            if scalar_dtype != expected_scalar_dtype:
                raise ValueError("Davidson profile scalar dtype is not canonical")
        if self.result_dtype != result_dtype.name:
            raise ValueError("solver profile result dtype must be canonical")
        if (
            type(self.result_itemsize) is not int
            or self.result_itemsize != result_dtype.itemsize
        ):
            raise ValueError("solver profile result itemsize does not match dtype")
        expected_result_dtype = np.dtype(np.result_type(normalized_dtype, scalar_dtype))
        if result_dtype != expected_result_dtype:
            raise ValueError("solver profile result dtype does not follow scalar rules")
        if (
            type(self.global_vector_count) is not int
            or self.global_vector_count != self.sharding.global_shape[0]
        ):
            raise ValueError(
                "solver profile global vector count does not match sharding"
            )
        rank_counts = tuple(self.rank_counts)
        expected_counts = tuple(
            self.sharding.local_shape(rank)[0] for rank in range(self.sharding.parts)
        )
        if (
            rank_counts != expected_counts
            or sum(rank_counts) != self.global_vector_count
        ):
            raise ValueError("solver profile rank counts do not match sharding")
        if len(estimates) != self.sharding.parts:
            raise ValueError("solver profile estimates must contain one value per rank")
        if type(self.allocation_vectors) is not int or self.allocation_vectors <= 0:
            raise ValueError("solver profile allocation_vectors must be positive")
        if type(self.projected_host_bytes) is not int or self.projected_host_bytes < 0:
            raise ValueError("solver profile projected_host_bytes must be non-negative")
        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "dtype", normalized_dtype.name)
        object.__setattr__(self, "scalar_dtype", scalar_dtype.name)
        object.__setattr__(self, "result_dtype", result_dtype.name)
        object.__setattr__(self, "rank_counts", rank_counts)
        controls = (self.block_size, self.max_krylov_vectors, self.max_space)
        if all(value is None for value in controls):
            return
        if self.solver_kind == "krylov":
            _positive_integer(self.block_size, "block_size")
            if self.max_krylov_vectors is not None:
                _positive_integer(self.max_krylov_vectors, "max_krylov_vectors")
            if self.max_space is not None:
                raise ValueError("Krylov profile must not define max_space")
        else:
            _positive_integer(self.max_space, "max_space", minimum=2)
            if self.block_size is not None or self.max_krylov_vectors is not None:
                raise ValueError("Davidson profile must define only max_space")


def _profile_dtype(dtype):
    try:
        normalized = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("solver profile dtype is unsupported") from error
    if (
        normalized.fields is not None
        or normalized.kind not in "biufc"
        or not normalized.isnative
    ):
        raise ValueError("solver profile dtype is unsupported")
    return normalized


def _profile_sharding(sharding):
    if not isinstance(sharding, ShardingSpec):
        raise TypeError("solver profile sharding must be a ShardingSpec")
    if len(sharding.global_shape) != 1 or sharding.axis != 0:
        raise NotImplementedError(
            "solver memory profiles support only axis-sharded vectors"
        )
    return sharding


def _coefficient_bits(value):
    value = 0.0 if value == 0 else float(value)
    return "{:016x}".format(
        int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])
    )


def _coefficient_float(bits):
    return np.asarray([int(bits, 16)], dtype=np.uint64).view(np.float64)[0].item()


def canonical_krylov_coefficient(value):
    normalized = _finite_coefficient(value)
    real = complex(normalized).real
    imag = complex(normalized).imag
    if imag == 0.0:
        return KrylovCoefficientIdentity(
            "real", _coefficient_bits(real), "0000000000000000"
        )
    return KrylovCoefficientIdentity(
        "complex", _coefficient_bits(real), _coefficient_bits(imag)
    )


def krylov_coefficient_payload(value):
    if value is None:
        return None
    if not isinstance(value, KrylovCoefficientIdentity):
        raise TypeError("value must be a KrylovCoefficientIdentity or None")
    return {
        "kind": value.kind,
        "real_bits": value.real_bits,
        "imag_bits": value.imag_bits,
    }


def validate_krylov_profile_coefficient(profile, coefficient):
    if not isinstance(profile, SolverMemoryProfile) or profile.solver_kind != "krylov":
        raise TypeError("profile must be a Krylov SolverMemoryProfile")
    actual = canonical_krylov_coefficient(coefficient)
    if profile.coefficient_identity != actual:
        raise ValueError("Krylov coefficient does not match the memory profile")


def _reduction_scratch(local_bytes, itemsize):
    # Complex vdot materializes conjugate and product arrays before scalar reduction.
    return 2 * local_bytes + max(4 * np.dtype(np.int64).itemsize, itemsize)


def _communication_peak(local_bytes, itemsize, control_words):
    synchronized_controls = (
        3 * control_words * np.dtype(np.int64).itemsize
        + 2 * np.dtype(np.int32).itemsize
    )
    return max(_reduction_scratch(local_bytes, itemsize), synchronized_controls)


def build_krylov_memory_profile(
    sharding,
    dtype,
    *,
    coefficient=1.0,
    block_size=50,
    max_krylov_vectors=None,
):
    """Return the bounded Lanczos liveness model without allocating vectors."""
    sharding = _profile_sharding(sharding)
    dtype = _profile_dtype(dtype)
    coefficient_identity = canonical_krylov_coefficient(coefficient)
    coefficient = coefficient_identity.coefficient
    scalar_dtype = np.dtype(
        np.complex128 if coefficient_identity.kind == "complex" else np.float64
    )
    result_dtype = np.dtype(np.result_type(dtype, scalar_dtype))
    block_size = _positive_integer(block_size, "block_size")
    global_count = sharding.global_shape[0]
    if max_krylov_vectors is None:
        bounded = False
        basis_vectors = global_count
        public_bound = None
    else:
        max_krylov_vectors = _positive_integer(max_krylov_vectors, "max_krylov_vectors")
        bounded = True
        basis_vectors = min(max_krylov_vectors, global_count)
        public_bound = basis_vectors
    capacity = ((basis_vectors + block_size - 1) // block_size) * block_size
    host_projected = (
        3 * 8 * basis_vectors * basis_vectors + 48 * basis_vectors + 16 * capacity
    )
    host_peak = max(host_projected, 3 * 4 * np.dtype(np.int64).itemsize)
    estimates = []
    for rank in range(sharding.parts):
        local_count = sharding.local_shape(rank)[0]
        local_bytes = local_count * dtype.itemsize
        local_result_bytes = local_count * result_dtype.itemsize
        retained = (basis_vectors + 1) * local_bytes + local_result_bytes
        estimates.append(
            SolverRankMemoryEstimate(
                hv_retained_bytes=retained,
                hv_transient_bytes=2 * local_bytes,
                la_retained_bytes=retained,
                la_transient_bytes=(
                    basis_vectors * local_bytes
                    + basis_vectors * scalar_dtype.itemsize
                    + local_result_bytes
                ),
                communication_device_bytes=max(
                    _communication_peak(local_bytes, dtype.itemsize, 4),
                    _communication_peak(local_result_bytes, result_dtype.itemsize, 4),
                ),
                host_peak_bytes=host_peak,
            )
        )
    return SolverMemoryProfile(
        solver_kind="krylov",
        rank_estimates=tuple(estimates),
        bounded=bounded,
        basis_vectors=public_bound,
        global_shape=sharding.global_shape,
        sharding=sharding,
        dtype=dtype.name,
        itemsize=dtype.itemsize,
        global_vector_count=global_count,
        rank_counts=tuple(
            sharding.local_shape(rank)[0] for rank in range(sharding.parts)
        ),
        allocation_vectors=capacity,
        projected_host_bytes=host_projected,
        block_size=block_size,
        max_krylov_vectors=max_krylov_vectors,
        scalar_dtype=scalar_dtype.name,
        scalar_itemsize=scalar_dtype.itemsize,
        result_dtype=result_dtype.name,
        result_itemsize=result_dtype.itemsize,
        coefficient_identity=coefficient_identity,
    )


def build_davidson_memory_profile(sharding, dtype, *, max_space=12):
    """Return the bounded single-root Davidson recurrence liveness model."""
    sharding = _profile_sharding(sharding)
    dtype = _profile_dtype(dtype)
    max_space = _positive_integer(max_space, "max_space", minimum=2)
    basis_vectors = min(max_space, sharding.global_shape[0])
    initial_projected_dtype = np.dtype(np.result_type(dtype, dtype))
    scalar_dtype = np.dtype(np.result_type(dtype, np.dtype(np.float64)))
    projected_dtype = np.dtype(np.result_type(dtype, scalar_dtype))
    projected_itemsize = projected_dtype.itemsize
    real_itemsize = np.empty((), dtype=projected_dtype).real.dtype.itemsize
    eigensolve_peak = (
        max_space * max_space * projected_itemsize
        + 2 * basis_vectors * basis_vectors * projected_itemsize
        + basis_vectors * real_itemsize
    )
    promotion_peak = max_space * max_space * projected_itemsize
    if projected_dtype != initial_projected_dtype:
        promotion_peak += max_space * max_space * initial_projected_dtype.itemsize
    host_projected = max(eigensolve_peak, promotion_peak)
    synchronized_host_controls = 3 * 5 * np.dtype(np.int64).itemsize
    host_peak = max(host_projected, synchronized_host_controls)
    estimates = []
    for rank in range(sharding.parts):
        local_count = sharding.local_shape(rank)[0]
        local_bytes = local_count * dtype.itemsize
        local_result_bytes = local_count * projected_dtype.itemsize
        basis_bytes = 2 * basis_vectors * local_result_bytes
        estimates.append(
            SolverRankMemoryEstimate(
                hv_retained_bytes=(basis_bytes + 4 * local_result_bytes + local_count),
                hv_transient_bytes=2 * local_bytes,
                la_retained_bytes=(basis_bytes + 5 * local_result_bytes + local_count),
                la_transient_bytes=2 * local_result_bytes,
                communication_device_bytes=max(
                    _communication_peak(local_bytes, dtype.itemsize, 5),
                    _communication_peak(
                        local_result_bytes, projected_dtype.itemsize, 5
                    ),
                ),
                host_peak_bytes=host_peak,
            )
        )
    return SolverMemoryProfile(
        solver_kind="davidson",
        rank_estimates=tuple(estimates),
        bounded=True,
        basis_vectors=basis_vectors,
        global_shape=sharding.global_shape,
        sharding=sharding,
        dtype=dtype.name,
        itemsize=dtype.itemsize,
        global_vector_count=sharding.global_shape[0],
        rank_counts=tuple(
            sharding.local_shape(rank)[0] for rank in range(sharding.parts)
        ),
        allocation_vectors=max_space,
        projected_host_bytes=host_projected,
        max_space=max_space,
        scalar_dtype=scalar_dtype.name,
        scalar_itemsize=scalar_dtype.itemsize,
        result_dtype=projected_dtype.name,
        result_itemsize=projected_dtype.itemsize,
    )


def canonical_solver_memory_profile(profile):
    """Rebuild a bound profile solely from canonical identity and controls."""
    if not isinstance(profile, SolverMemoryProfile):
        raise TypeError("profile must be a SolverMemoryProfile")
    if profile.sharding is None or profile.dtype is None:
        raise ValueError("solver profile identity metadata is incomplete")
    if profile.solver_kind == "krylov":
        if profile.block_size is None:
            raise ValueError("Krylov profile controls are incomplete")
        return build_krylov_memory_profile(
            profile.sharding,
            profile.dtype,
            coefficient=profile.coefficient_identity.coefficient,
            block_size=profile.block_size,
            max_krylov_vectors=profile.max_krylov_vectors,
        )
    if profile.max_space is None:
        raise ValueError("Davidson profile controls are incomplete")
    return build_davidson_memory_profile(
        profile.sharding, profile.dtype, max_space=profile.max_space
    )


@runtime_checkable
class DistributedSolverOperator(Protocol):
    backend: object
    context: object
    collective: object
    solver_input_sharding: object
    solver_output_sharding: object
    solver_dtype: np.dtype

    def solver_preflight(self) -> None:
        ...

    def __call__(self, local_vector):
        ...


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
        raise ValueError("integer coefficient exceeds the binary64 exact integer range")
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
        raise ValueError("distributed {} control validation failed".format(solver_name))

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
    if vector.spec != operator.solver_input_sharding:
        return ValueError("vector sharding spec does not match operator input")
    if operator.solver_input_sharding != operator.solver_output_sharding:
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
    expected_dtype = np.dtype(operator.solver_dtype)
    if np.dtype(vector.local_array.dtype) != expected_dtype:
        return ValueError(
            "distributed vector dtype does not match operator input dtype"
        )
    return None


def _solver_preflight(operator, vector, collective):
    preflight = getattr(operator, "solver_preflight", None)
    if not callable(preflight):
        raise TypeError("operator must implement the distributed solver protocol")
    preflight()
    if not isinstance(operator, DistributedSolverOperator):
        raise TypeError("operator must implement the distributed solver protocol")
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
        mismatch = (
            self.namespace.logical_not(self.namespace.allclose(left, right))
            .astype(np.int32)
            .reshape(1)
        )
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
        block_size = _int64_control(validated.get("block_size", 50), "block_size")
        requested_bound = validated.get("max_krylov_vectors")
        max_krylov_vectors = (
            None
            if requested_bound is None
            else _int64_control(requested_bound, "max_krylov_vectors")
        )
        parsed_coefficient = _finite_coefficient(coefficient)
        expected_plan = getattr(operator, "residency_plan", None)
        if expected_plan is None:
            local_operator = getattr(operator, "local_operator", None)
            expected_plan = getattr(local_operator, "residency_plan", None)
        expected_identity = getattr(expected_plan, "krylov_coefficient_identity", None)
        if expected_identity is not None and expected_identity != (
            canonical_krylov_coefficient(parsed_coefficient)
        ):
            raise ValueError("Krylov coefficient does not match the residency plan")
        coefficient_bits = _float_control_bits(
            complex(parsed_coefficient).real,
            complex(parsed_coefficient).imag,
        )
        controls = np.concatenate(
            (
                coefficient_bits,
                np.asarray(
                    [
                        block_size,
                        0 if max_krylov_vectors is None else max_krylov_vectors,
                    ],
                    dtype=np.int64,
                ),
            )
        )
        return (parsed_coefficient, block_size, max_krylov_vectors), controls

    coefficient, block_size, max_krylov_vectors = _synchronized_controls(
        parse_controls, operator, collective, "Krylov"
    )
    vector_ops = _DistributedVectorOps(operator.backend, collective)
    recurrence_kwargs = {
        "block_size": block_size,
        "vector_ops": vector_ops,
        "global_size": _checked_shape_elements(
            vector.global_shape, "Krylov global shape"
        ),
    }
    if max_krylov_vectors is not None:
        recurrence_kwargs["max_vectors"] = max_krylov_vectors
    local_result, iterations = _lanczos_expm(
        lambda local: operator(local).copy(),
        coefficient,
        vector.local_array,
        **recurrence_kwargs,
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
        "lindep": _davidson_backend._validated_lindep(config.get("lindep", 1e-14)),
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

    values = _synchronized_controls(parse_controls, operator, collective, "Davidson")
    diagonal = values["diagonal"]
    require_convergence = values["require_convergence"]
    diagonal_error = _diagonal_contract_error(diagonal, vector, operator)
    namespace = operator.backend.array_namespace
    status = namespace.asarray([1 if diagonal_error is not None else 0], dtype=np.int32)
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
        global_size=_checked_shape_elements(
            vector.global_shape, "Davidson global shape"
        ),
        tol=values["tol"],
        max_cycle=values["max_cycle"],
        max_space=values["max_space"],
        lindep=values["lindep"],
    )
    if require_convergence and not info.converged:
        raise RuntimeError("sharded Davidson did not converge")
    return (
        energy,
        DistributedTensor(vector.spec, vector.rank, local_result.copy()),
        info,
    )


__all__ = [
    "build_davidson_memory_profile",
    "build_krylov_memory_profile",
    "canonical_solver_memory_profile",
    "DistributedSolverOperator",
    "SolverMemoryProfile",
    "SolverRankMemoryEstimate",
    "distributed_norm",
    "distributed_vdot",
    "run_sharded_davidson",
    "run_sharded_krylov",
]
