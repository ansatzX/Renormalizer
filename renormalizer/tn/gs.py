import logging
from typing import List, Union


import scipy

from renormalizer.lib import davidson
from renormalizer.mps.backend import primme, IMPORT_PRIMME_EXCEPTION, np
from renormalizer.mps.matrix import asnumpy, asxp
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TTNS, TTNO, TTNEnviron
from renormalizer.tn.hop_expr import hop_expr2
from renormalizer.utils import profiling


logger = logging.getLogger(__name__)


def optimize_ttns(ttns: TTNS, ttno: TTNO, procedure=None):
    if procedure is None:
        procedure = ttns.optimize_config.procedure
    distributed_execution = ttns.optimize_config.distributed_execution
    if distributed_execution is not None:
        from renormalizer.backend._distributed.center import (
            canonicalize_compression_control,
        )
        from renormalizer.tn.distributed import coordinate_ttns_workflow_entry

        effective_procedure = [
            {
                "compression": canonicalize_compression_control(compression),
                "percent": float(percent),
            }
            for compression, percent in procedure
        ]
        supported = (
            ttns.optimize_config.algo == "davidson"
            and ttns.optimize_config.nroots == 1
        )
        coordinate_ttns_workflow_entry(
            distributed_execution,
            operation="ground_state_workflow",
            node_count=len(ttns.node_list),
            center_kind="two_site_workflow",
            solver_controls={
                "algo": ttns.optimize_config.algo,
                "nroots": ttns.optimize_config.nroots,
                "effective_procedure": effective_procedure,
                "convergence": {
                    "rtol": ttns.optimize_config.e_rtol,
                    "atol": ttns.optimize_config.e_atol,
                },
            },
            selectors={"ground_state_topology": "existing_two_site"},
            supported=supported,
            fallback_reason_code="unsupported_ground_state_workflow",
        )
    profile_enabled = profiling.enabled()
    if profile_enabled:
        from time import perf_counter

        phase_start = perf_counter()
    if distributed_execution is None:
        ttne = TTNEnviron(ttns, ttno)
    else:
        ttne = _synchronize_distributed_state(
            ttns,
            ttns.root,
            "environment_setup",
            operation=lambda: TTNEnviron(ttns, ttno),
        )
    if profile_enabled:
        _record_phase_summary("environment_construction", "construct", perf_counter() - phase_start)
    e_list = []
    for m, percent in procedure:
        # todo: better converge condition
        if profile_enabled:
            phase_start = perf_counter()
        micro_e = optimize_recursion(ttns.root, ttns, ttno, ttne, m, percent)
        if profile_enabled:
            _record_phase_summary(
                "optimization_sweep", "ttns_optimization_sweep", perf_counter() - phase_start
            )
        logger.info(f"Micro e: {micro_e}")
        e_list.append(micro_e[-1])
    if distributed_execution is not None:
        _synchronize_distributed_state(
            ttns,
            ttns.root,
            "final_normalization",
            operation=lambda: ttns.normalize("ttns_only"),
        )
    return e_list


def _record_phase_summary(phase, operation, wall_s):
    from renormalizer.backend._execution.profiling import phase_summary_payload

    payload = phase_summary_payload(
        phase=phase,
        network="ttns",
        operation=operation,
        operation_count=1,
        wall_s=wall_s,
    )
    profiling.record("phase_summary", **payload)


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
            _synchronize_distributed_state(
                ttns,
                child,
                "to_child",
                operation=lambda: (
                    ttns.update_2site(
                        child, c, m, percent, cano_parent=False
                    ),
                    ttne.update_2site(child, ttns, ttno),
                ),
            )
            # recursive optimization
            micro_e_child = optimize_recursion(child, ttns, ttno, ttne, m)
            micro_e.extend(micro_e_child)

        # optimize snode + child
        e, c = optimize_2site(child, ttns, ttno, ttne)
        micro_e.append(e)
        # cano to snode
        _synchronize_distributed_state(
            ttns,
            child,
            "to_parent",
            operation=lambda: (
                ttns.update_2site(
                    child, c, m, percent, cano_parent=True
                ),
                ttne.update_2site(child, ttns, ttno),
            ),
        )
    return micro_e


def _synchronize_distributed_state(ttns, node, direction, *, operation):
    execution = ttns.optimize_config.distributed_execution
    if execution is None:
        return operation()
    from renormalizer.tn.distributed import _synchronize_ttns_state

    return _synchronize_ttns_state(
        execution,
        ttns,
        metadata=("davidson", ttns.node_idx[node], direction),
        operation=operation,
    )


def optimize_2site(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron):
    cguess = ttns.merge_with_parent(snode)
    qn_mask = ttns.get_qnmask(snode, include_parent=True)
    cguess = cguess[qn_mask].ravel()
    distributed_execution = ttns.optimize_config.distributed_execution
    if distributed_execution is not None and (
        ttns.optimize_config.nroots != 1
        or ttns.optimize_config.algo != "davidson"
    ):
        from renormalizer.tn.distributed import run_ttns_ground_state_fallback

        node_index = ttns.node_idx[snode]

        def local_ground_operation():
            local_expr, local_hdiag = hop_expr2(snode, ttns, ttno, ttne)
            local_hdiag = local_hdiag[qn_mask].ravel()

            def local_hop(x):
                cstruct = vec2tensor(x, qn_mask)
                ret = local_expr(asxp(cstruct))[qn_mask].ravel()
                return asnumpy(ret)

            return eigh_iterative(
                local_hop,
                local_hdiag,
                cguess,
                ttns.optimize_config.algo,
            )

        energy, vector = run_ttns_ground_state_fallback(
            local_ground_operation,
            distributed_execution=distributed_execution,
            qn_mask=qn_mask,
            initial_guess=cguess,
            node_index=node_index,
            parent_index=ttns.node_idx[snode.parent],
            child_index=snode.parent.children.index(snode),
            degree=len(snode.children) + 1,
            center_kind="two_site",
            solver_controls={
                "algo": ttns.optimize_config.algo,
                "nroots": ttns.optimize_config.nroots,
            },
            selectors={"ground_state_topology": "existing_two_site"},
        )
        return energy, vec2tensor(vector, qn_mask)

    expr, hdiag = hop_expr2(snode, ttns, ttno, ttne)
    hdiag = hdiag[qn_mask].ravel()

    def hop(x):
        cstruct = vec2tensor(x, qn_mask)
        ret = expr(asxp(cstruct))[qn_mask].ravel()
        return asnumpy(ret)

    if distributed_execution is not None:
        from renormalizer.tn.distributed import run_ttns_davidson

        node_index = ttns.node_idx[snode]
        energy, center, _ = run_ttns_davidson(
            expr,
            distributed_execution=distributed_execution,
            qn_mask=qn_mask,
            initial_guess=cguess,
            diagonal=hdiag,
            node_index=node_index,
            parent_index=ttns.node_idx[snode.parent],
            child_index=snode.parent.children.index(snode),
            degree=len(snode.children) + 1,
            center_kind="two_site",
            solver_config={
                "tol": 1e-7,
                "max_cycle": 100,
                "max_space": 24,
                "lindep": 1e-14,
                "require_convergence": False,
            },
            decision_selectors={
                "algo": ttns.optimize_config.algo,
                "nroots": ttns.optimize_config.nroots,
                "ground_state_topology": "existing_two_site",
            },
        )
        return energy, center

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


def vec2tensor(c, qn_mask):
    cstruct = np.zeros(qn_mask.shape, dtype=c.dtype)
    np.place(cstruct, qn_mask, c)
    return cstruct
