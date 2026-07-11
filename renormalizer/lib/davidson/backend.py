# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Backend-neutral single-root Davidson recurrence."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from renormalizer.mps.backend import xp


@dataclass(frozen=True)
class DavidsonInfo:
    converged: bool
    iterations: int
    h_v_count: int
    subspace_size: int
    restarts: int
    residual_norm: float


class _LocalVectorOps:
    def copy(self, vector):
        return vector.copy()

    def vdot(self, left, right):
        return xp.vdot(left, right).item()

    def norm(self, vector):
        return float(xp.linalg.norm(vector))

    def linear_combination(self, coefficients, vectors):
        dtype = np.result_type(vectors[0].dtype, coefficients.dtype)
        result = xp.zeros_like(vectors[0], dtype=dtype)
        for coefficient, vector in zip(coefficients, vectors):
            result += coefficient * vector
        return result


def _validated_lindep(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("lindep must be a positive real number less than 1")
    value = float(value)
    if not np.isfinite(value) or value <= 0 or value >= 1:
        raise ValueError("lindep must be positive, finite, and less than 1")
    return value


def _orthogonalize(vector, basis, vector_ops):
    trial = vector_ops.copy(vector)
    for base in basis:
        trial -= vector_ops.vdot(base, trial) * base
    return trial


def _davidson_single_root(
    aop,
    x0,
    diagonal,
    *,
    vector_ops,
    global_size,
    tol,
    max_cycle,
    max_space,
    lindep,
):
    """Run one Davidson recurrence over vectors owned by ``vector_ops``."""
    trial = vector_ops.copy(x0)
    trial_norm = vector_ops.norm(trial)
    if trial_norm == 0:
        raise ValueError("initial Davidson vector must be nonzero")
    trial /= trial_norm

    basis = []
    a_basis = []
    projected = np.empty(
        (max_space, max_space), dtype=np.result_type(x0.dtype, diagonal.dtype)
    )
    restarts = 0
    h_v_count = 0
    result = vector_ops.copy(trial)
    energy = 0.0
    residual_norm = float("inf")
    converged = False
    iterations = 0

    for iteration in range(1, max_cycle + 1):
        iterations = iteration
        trial = _orthogonalize(trial, basis, vector_ops)
        trial_norm = vector_ops.norm(trial)
        if basis and trial_norm * trial_norm <= lindep:
            break
        trial /= trial_norm
        basis.append(vector_ops.copy(trial))
        applied = vector_ops.copy(aop(trial))
        a_basis.append(applied)
        h_v_count += 1

        new_index = len(basis) - 1
        for index in range(new_index + 1):
            element = vector_ops.vdot(basis[index], applied)
            element_dtype = np.asarray(element).dtype
            projected_dtype = np.result_type(
                projected.dtype, applied.dtype, element_dtype
            )
            if projected_dtype != projected.dtype:
                projected = projected.astype(projected_dtype)
            projected[index, new_index] = element
            projected[new_index, index] = np.conjugate(element)
        projected[new_index, new_index] = projected[new_index, new_index].real

        eigenvalues, eigenvectors = np.linalg.eigh(
            projected[: new_index + 1, : new_index + 1]
        )
        energy = float(eigenvalues[0].real)
        coefficients = eigenvectors[:, 0]
        result = vector_ops.linear_combination(coefficients, basis)
        applied_result = vector_ops.linear_combination(coefficients, a_basis)
        residual = applied_result - energy * result
        residual_norm = vector_ops.norm(residual)
        if residual_norm <= tol:
            converged = True
            break

        denominator = diagonal - energy
        small = abs(denominator) < 1e-8
        denominator = denominator.copy()
        denominator[small] = 1e-8
        trial = residual / denominator

        if len(basis) >= min(max_space, global_size):
            trial = vector_ops.copy(result)
            basis = []
            a_basis = []
            restarts += 1

    result_norm = vector_ops.norm(result)
    if result_norm == 0:
        raise ValueError("Davidson recurrence produced a zero vector")
    result = vector_ops.copy(result / result_norm)
    if np.isfinite(residual_norm):
        residual_norm /= result_norm
    info = DavidsonInfo(
        converged=converged,
        iterations=iterations,
        h_v_count=h_v_count,
        subspace_size=len(basis),
        restarts=restarts,
        residual_norm=float(residual_norm),
    )
    return energy, result, info


def davidson_backend(
    aop,
    x0,
    diagonal,
    tol=1e-12,
    max_cycle=50,
    max_space=12,
    lindep=1e-14,
):
    """Find the lowest eigenpair with the backend Davidson recurrence."""
    if getattr(xp, "name", None) not in {"numpy", "cupy"}:
        raise NotImplementedError(
            "Davidson supports only NumPy and CuPy backends"
        )
    lindep = _validated_lindep(lindep)
    x0 = xp.asarray(x0)
    diagonal = xp.asarray(diagonal)
    if x0.ndim != 1 or diagonal.shape != x0.shape:
        raise ValueError("Davidson vectors and diagonal must be matching 1D arrays")
    return _davidson_single_root(
        aop,
        x0,
        diagonal,
        vector_ops=_LocalVectorOps(),
        global_size=x0.size,
        tol=tol,
        max_cycle=max_cycle,
        max_space=max_space,
        lindep=lindep,
    )


__all__ = ["DavidsonInfo", "davidson_backend"]
