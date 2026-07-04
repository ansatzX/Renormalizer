import logging
import time
from typing import List, Union


import scipy

from renormalizer.backend import BackendFeatureError, FallbackPolicy
from renormalizer.lib import davidson
from renormalizer.mps.backend import backend, primme, IMPORT_PRIMME_EXCEPTION, np
from renormalizer.mps.matrix import asnumpy, asxp
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TTNS, TTNO, TTNEnviron
from renormalizer.tn.hop_expr import hop_expr2
from renormalizer.utils import profiling


logger = logging.getLogger(__name__)
_TN_RHS_LOOP_FALLBACK_REASON = "tn hop expression does not yet support batched RHS"


def optimize_ttns(ttns: TTNS, ttno: TTNO, procedure=None):
    if procedure is None:
        procedure = ttns.optimize_config.procedure
    ttne = TTNEnviron(ttns, ttno)
    e_list = []
    for m, percent in procedure:
        # todo: better converge condition
        micro_e = optimize_recursion(ttns.root, ttns, ttno, ttne, m, percent)
        logger.info(f"Micro e: {micro_e}")
        e_list.append(micro_e[-1])
    return e_list


def optimize_recursion(
    snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron, m: Union[int, List[int]], percent: float = 0
) -> List[float]:
    """Optimize snode and all of its children"""
    assert snode.children  # 2 site can't do only one node
    micro_e = []
    for ichild, child in enumerate(snode.children):
        if child.children:
            # optimize snode + child
            e, c = optimize_2site(child, ttns, ttno, ttne)
            micro_e.append(e)
            # cano to child
            ttns.update_2site(child, c, m, percent, cano_parent=False)
            # update env
            ttne.update_2site(child, ttns, ttno)
            # recursive optimization
            micro_e_child = optimize_recursion(child, ttns, ttno, ttne, m)
            micro_e.extend(micro_e_child)

        # optimize snode + child
        e, c = optimize_2site(child, ttns, ttno, ttne)
        micro_e.append(e)
        # cano to snode
        ttns.update_2site(child, c, m, percent, cano_parent=True)
        # update env
        ttne.update_2site(child, ttns, ttno)
    return micro_e


def optimize_2site(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron):
    cguess = ttns.merge_with_parent(snode)
    qn_mask = ttns.get_qnmask(snode, include_parent=True)
    cguess = cguess[qn_mask].ravel()
    expr, hdiag = hop_expr2(snode, ttns, ttno, ttne)
    hdiag = hdiag[qn_mask].ravel()

    def hop(x):
        return _apply_tn_hop_to_packed_vectors(x, qn_mask, expr)

    assert ttns.optimize_config.nroots == 1
    algo: str = ttns.optimize_config.algo
    e, c = eigh_iterative(hop, hdiag, cguess, algo)
    c = vec2tensor(c, qn_mask)
    return e, c


def eigh_iterative(hop, hdiag, cguess, algo):
    hdiag = asnumpy(hdiag)
    cguess = asnumpy(cguess)
    h_dim = len(hdiag)

    if algo == "davidson":
        precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

        e, c = davidson(hop, cguess, precond, max_cycle=100, nroots=1, max_memory=64000, verbose=0)
    elif algo == "primme":
        if primme is None:
            logger.error("can not import primme")
            raise IMPORT_PRIMME_EXCEPTION
        precond = lambda x: scipy.sparse.diags(1 / (hdiag + 1e-4)) @ x
        A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop, matmat=hop)
        M = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=precond, matmat=hop)
        e, c = primme.eigsh(
            A,
            k=1,
            which="SA",
            v0=np.array(cguess).reshape(-1, 1),
            OPinv=M,
            method="PRIMME_DYNAMIC",
            tol=1e-6,
        )
        c = c[:, 0]
        e = e[0]
    elif algo == "arpack":
        A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop)
        e, c = scipy.sparse.linalg.eigsh(A, k=1, which="SA", v0=cguess)
        e = e[0]
    elif algo == "direct":
        # direct algorithm. Poor performance, debugging only.
        a_list = []
        for i in range(h_dim):
            a = np.zeros(h_dim)
            a[i] = 1
            a_list.append(hop(a))
        a = np.array(a_list)
        assert np.allclose(a, a.conj().T)
        evals, evecs = np.linalg.eigh(a)
        e = evals[0]
        c = evecs[:, 0]
    else:
        assert False

    return e, c


def _apply_tn_hop_to_packed_vectors(x, qn_mask, expr):
    if x.ndim == 1:
        cstruct = vec2tensor(x, qn_mask)
        ret = expr(asxp(cstruct))[qn_mask].ravel()
        return asnumpy(ret)
    if x.ndim != 2:
        raise ValueError(f"TN hop expects 1D or 2D RHS, got shape {x.shape}")

    fallback_reason = _handle_tn_rhs_loop_fallback_policy()
    should_profile = profiling.should_record_op()
    started = time.perf_counter() if should_profile else None

    def apply_one(vector):
        cstruct = vec2tensor(vector, qn_mask)
        ret = expr(asxp(cstruct))[qn_mask].ravel()
        return asnumpy(ret)

    result = np.stack([apply_one(x[:, i]) for i in range(x.shape[1])], axis=1)
    if should_profile:
        read_bytes = int(getattr(x, "nbytes", 0) or 0)
        write_bytes = int(getattr(result, "nbytes", 0) or 0)
        profiling.record(
            "contraction_execute",
            backend=backend.name,
            **profiling.contraction_execute_compute_payload("fallback_rhs_loop"),
            equation=getattr(expr, "equation", None),
            lowering="fallback_rhs_loop",
            input_shapes=[tuple(x.shape)],
            input_dtypes=[str(getattr(x, "dtype", None))],
            operands=[
                profiling.array_operand_payload(backend, "packed_rhs", x, ("packed", "rhs")),
            ],
            output_shape=tuple(result.shape),
            output_strides=profiling.array_strides(result),
            output_order=profiling.array_order(result),
            output_contiguous=profiling.array_contiguous(result),
            output_backend=profiling.array_backend_name(result),
            output_device_kind=profiling.array_device_kind(result),
            output_location=profiling.array_location(result),
            output_is_host=profiling.array_is_host(result),
            output_is_device=profiling.array_is_device(result),
            output_is_distributed=profiling.array_is_distributed(result),
            dtype=str(getattr(result, "dtype", None)),
            **profiling.device_execution_payload(backend.current_device()),
            flops=0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            copy_bytes=0,
            workspace_bytes=0,
            largest_intermediate=max(read_bytes, write_bytes),
            num_gemm=0,
            num_batched_gemm=0,
            num_grouped_tasks=0,
            num_blocks=0,
            num_shape_buckets=0,
            num_rhs=int(x.shape[1]),
            num_rhs_loop_calls=int(x.shape[1]),
            execution_primitives=["rhs_loop"],
            execution_policies=["fallback_rhs_loop"],
            fallback_from="batched_rhs_hop",
            fallback_to="rhs_loop",
            fallback_policy=backend.fallback_policy.value,
            fallback_reason=fallback_reason,
            fallback_reasons=[fallback_reason],
            wall_s=time.perf_counter() - started,
        )
    return result


def _handle_tn_rhs_loop_fallback_policy():
    reason = _TN_RHS_LOOP_FALLBACK_REASON
    if backend.fallback_policy is FallbackPolicy.FORBID:
        raise BackendFeatureError(reason)
    if backend.fallback_policy is FallbackPolicy.WARN:
        import warnings

        warnings.warn(reason, RuntimeWarning, stacklevel=3)
    return reason


def vec2tensor(c, qn_mask):
    cstruct = np.zeros(qn_mask.shape, dtype=c.dtype)
    np.place(cstruct, qn_mask, c)
    return cstruct
