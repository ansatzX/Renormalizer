# -*- coding: utf-8 -*-
# adopted from https://github.com/cmendl/pytenet/blob/master/pytenet/krylov.py

import logging

from scipy.linalg import eigh_tridiagonal
import numpy as np

from renormalizer.mps.backend import xp


logger = logging.getLogger(__name__)


def _projected_exponential_coefficients(alpha, beta, v_norm, dt):
    # diagonalize Hessenberg matrix (tridiagonal matrix for hermitian matrix A)
    try:
        w_hess, u_hess = eigh_tridiagonal(alpha, beta)
    except np.linalg.LinAlgError:
        logger.warning(f"Tridiagonal diagonalization in Krylov solver failed, size:{len(alpha)}. "
                       f"Usually this means: (1) Unphysical Hamiltonian OR (2) large step size.")
        h = np.diag(alpha) + np.diag(beta, k=-1) + np.diag(beta, k=1)
        w_hess, u_hess = np.linalg.eigh(h)

    return u_hess @ (v_norm * np.exp(dt*w_hess) * u_hess[0])


def _expm_krylov(alpha, beta, V, v_norm, dt):
    coefficients = _projected_exponential_coefficients(
        alpha, beta, v_norm, dt
    )

    return V @ xp.asarray(coefficients)


class _LocalLanczosVectorOps:
    def asarray(self, vector):
        return xp.asarray(vector)

    def copy(self, vector):
        return vector.copy()

    def norm(self, vector):
        return float(xp.linalg.norm(vector))

    def vdot(self, left, right):
        return xp.vdot(left, right).item()

    def projected_exponential(self, alpha, beta, basis, vector_norm, dt):
        vectors = xp.stack(basis, axis=0).T
        return _expm_krylov(alpha, beta, vectors, vector_norm, dt)

    def allclose(self, left, right):
        return bool(xp.allclose(left, right))


def _lanczos_expm(
    Afunc,
    dt,
    vstart,
    *,
    block_size,
    vector_ops,
    global_size,
    max_vectors=None,
):
    """Generic Lanczos recurrence over vectors managed by ``vector_ops``."""
    if not np.iscomplex(dt):
        dt = dt.real

    vstart = vector_ops.asarray(vstart)
    nrmv = vector_ops.norm(vstart)
    assert nrmv > 0
    vstart = vector_ops.copy(vstart / nrmv)

    alpha = np.zeros(block_size)
    beta = np.zeros(block_size - 1)
    basis = [vstart]
    res = None

    if max_vectors is None:
        iteration_limit = global_size
    else:
        if type(max_vectors) is not int or max_vectors <= 0:
            raise ValueError("max_vectors must be a positive integer or None")
        iteration_limit = min(global_size, max_vectors)

    for j in range(iteration_limit):
        w = vector_ops.copy(Afunc(basis[j]))
        alpha[j] = vector_ops.vdot(w, basis[j]).real

        if j == iteration_limit - 1:
            return (
                vector_ops.projected_exponential(
                    alpha[: j + 1], beta[:j], basis[: j + 1], nrmv, dt
                ),
                j + 1,
            )

        if len(alpha) == j + 1:
            alpha = np.concatenate([alpha, np.zeros(block_size)])
            beta = np.concatenate([beta, np.zeros(block_size)])

        w -= alpha[j] * basis[j] + (beta[j - 1] * basis[j - 1] if j > 0 else 0)
        beta[j] = vector_ops.norm(w)
        if beta[j] < 100 * global_size * np.finfo(float).eps:
            return (
                vector_ops.projected_exponential(
                    alpha[: j + 1], beta[:j], basis[: j + 1], nrmv, dt
                ),
                j + 1,
            )

        if 3 < j and j % 2 == 0:
            new_res = vector_ops.projected_exponential(
                alpha[: j + 1], beta[:j], basis[: j + 1], nrmv, dt
            )
            if res is not None and vector_ops.allclose(res, new_res):
                return new_res, j + 1
            res = new_res
        basis.append(vector_ops.copy(w / beta[j]))


def expm_krylov(
    Afunc,
    dt,
    vstart: xp.ndarray,
    block_size=50,
    *,
    max_krylov_vectors=None,
):
    """
    Compute Krylov subspace approximation of the matrix exponential
    applied to input vector: `expm(dt*A)*v`.
    A is a hermitian matrix.
    Reference:
        M. Hochbruck and C. Lubich
        On Krylov subspace approximations to the matrix exponential operator
        SIAM J. Numer. Anal. 34, 1911 (1997)
    """
    recurrence_kwargs = {
        "block_size": block_size,
        "vector_ops": _LocalLanczosVectorOps(),
        "global_size": int(np.prod(vstart.shape)),
    }
    if max_krylov_vectors is not None:
        if type(max_krylov_vectors) is not int or max_krylov_vectors <= 0:
            raise ValueError(
                "max_krylov_vectors must be a positive integer or None"
            )
        recurrence_kwargs["max_vectors"] = max_krylov_vectors
    return _lanczos_expm(Afunc, dt, vstart, **recurrence_kwargs)
