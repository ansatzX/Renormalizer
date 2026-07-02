# -*- coding: utf-8 -*-
"""

This module contains functions to optimize mps for a single ground state or
several lowest excited states with state-averaged algorithm.

"""

from typing import Tuple, List, Union
from functools import partial
from itertools import product
from collections import deque
import logging
import time

import numpy as np
import scipy

from renormalizer.lib import davidson
from renormalizer.model.h_qc import qc_model, int_to_h, generate_ladder_operator, simplify_op
from renormalizer.model import Model, Op
from renormalizer.mps.backend import backend, xp, primme, IMPORT_PRIMME_EXCEPTION
from renormalizer.mps.matrix import multi_tensor_contract, tensordot, asnumpy, asxp
from renormalizer.mps.hop_expr import batched_hop_expr, hop_expr
from renormalizer.mps.svd_qn import get_qn_mask
from renormalizer.mps import Mpo, Mps, StackedMpo
from renormalizer.mps.lib import Environ, cvec2cmat
from renormalizer.mps.oe_contract_wrap import oe_contract
from renormalizer.utils import Quantity, CompressConfig, CompressCriteria, profiling
from renormalizer.backend import PackedVectorSpec


logger = logging.getLogger(__name__)


def construct_mps_mpo(model, mmax, nexciton, offset=Quantity(0)):
    """
    MPO/MPS structure 2
    e1,ph11,ph12,..e2,ph21,ph22,...en,phn1,phn2...
    """

    """
    initialize MPO
    """
    mpo = Mpo(model, offset=offset)

    """
    initialize MPS according to quantum number
    """
    mps = Mps.random(model, nexciton, mmax, percent=1)
    # print("initialize left-canonical:", mps.check_left_canonical())

    return mps, mpo


def optimize_mps(mps: Mps, mpo: Union[Mpo, StackedMpo], omega: float = None) -> Tuple[List, Mps]:
    r"""DMRG ground state algorithm and state-averaged excited states algorithm

    Parameters
    ----------
    mps : renormalizer.mps.Mps
        initial guess of mps. The MPS is overwritten during the optimization.
    mpo : Union[renormalizer.mps.Mpo, renormalizer.mps.StackedMpo]
        mpo of Hamiltonian
    omega: float, optional
        target the eigenpair near omega with special variational function
        :math:(\hat{H}-\omega)^2. Default is `None`.

    Returns
    -------
    energy : list
        list of energy of each marco sweep.
        :math:`[e_0, e_0, \cdots, e_0]` if ``nroots=1``.
        :math:`[[e_0, \cdots, e_n], \dots, [e_0, \cdots, e_n]]` if ``nroots=n``.
    mps : renormalizer.mps.Mps
        optimized ground state MPS.
            Note it's not the same with the overwritten input MPS.

    See Also
    --------
    renormalizer.utils.configs.OptimizeConfig : The optimization configuration.

    Note
    ----
    When On-the-fly swapping algorithm is used, the site ordering of the returned
    MPS is changed and the original MPO will not correspond to it and should be
    updated with returned mps.model.
    """

    assert mps.optimize_config.method in ["2site", "1site"]
    logger.info(f"optimization method: {mps.optimize_config.method}")
    logger.info(f"e_rtol: {mps.optimize_config.e_rtol}")
    logger.info(f"e_atol: {mps.optimize_config.e_atol}")
    logger.info(f"procedure: {mps.optimize_config.procedure}")

    # ensure that mps is left or right-canonical
    # TODO: start from a mix-canonical MPS
    if mps.is_left_canonical:
        mps.ensure_right_canonical()
        env = "R"
    else:
        mps.ensure_left_canonical()
        env = "L"

    compress_config_bk = mps.compress_config

    # construct the environment matrix
    if omega is not None:
        if isinstance(mpo, StackedMpo):
            raise NotImplementedError("StackedMPO + omega is not implemented yet")
        identity = Mpo.identity(mpo.model)
        mpo = mpo.add(identity.scale(-omega))
        environ = Environ(mps, [mpo, mpo], env)
    else:
        if isinstance(mpo, StackedMpo):
            environ = [Environ(mps, item, env) for item in mpo.mpos]
        else:
            environ = Environ(mps, mpo, env)

    macro_iteration_result = []
    # Idx of the active site with lowest energy for each sweep
    # determines the index of active site of the returned mps
    opt_e_idx: int = None
    res_mps: Union[Mps, List[Mps]] = None
    for isweep, (compress_config, percent) in enumerate(mps.optimize_config.procedure):
        logger.debug(f"isweep: {isweep}")

        if isinstance(compress_config, CompressConfig):
            mps.compress_config = compress_config
        elif isinstance(compress_config, int):
            mps.compress_config = CompressConfig(criteria=CompressCriteria.fixed,
                    max_bonddim = compress_config)
        else:
            assert False
        logger.debug(f"compress config in current loop: {compress_config}, percent: {percent}")

        logger.debug(f"{mps}")

        micro_iteration_result, res_mps, mpo = single_sweep(mps, mpo, environ, omega, percent, opt_e_idx)

        opt_e = min(micro_iteration_result)
        macro_iteration_result.append(opt_e[0])
        opt_e_idx = opt_e[1]

        logger.debug(
            f"{isweep+1} sweeps are finished, lowest energy = {min(macro_iteration_result)}"
        )
        # check if convergence
        if isweep > 0 and percent == 0:
            v1, v2 = sorted(macro_iteration_result)[:2]
            if np.allclose(
                v1, v2, rtol=mps.optimize_config.e_rtol, atol=mps.optimize_config.e_atol
            ):
                logger.info("DMRG has converged!")
                break
    else:
        logger.warning("DMRG did not converge! Please increase the procedure!")
        logger.info(f"The lowest two energies: {sorted(macro_iteration_result)[:2]}.")

    assert res_mps is not None
    # remove the redundant basis near the edge
    # and restore the original compress_config of the input mps
    if mps.optimize_config.nroots == 1:
        res_mps = res_mps.normalize("mps_only").ensure_left_canonical().canonicalise()
        res_mps.compress_config = compress_config_bk
        logger.info(f"{res_mps}")
    else:
        res_mps = [mp.normalize("mps_only").ensure_left_canonical().canonicalise() for mp in res_mps]
        for res in res_mps:
            res.compress_config = compress_config_bk
        logger.info(f"{res_mps[0]}")

    return macro_iteration_result, res_mps


def single_sweep(
    mps: Mps,
    mpo: Union[Mpo, StackedMpo],
    environ: Environ,
    omega: float,
    percent: float,
    last_opt_e_idx: int
):

    method = mps.optimize_config.method
    nroots = mps.optimize_config.nroots

    # in state-averaged calculation, contains C of each state for better initial guess
    averaged_ms = []
    # optmized mps
    res_mps: Union[Mps, List[Mps]] = None
    # energies after optimizing each site
    micro_iteration_result = []
    for imps in mps.iter_idx_list(full=True):
        if method == "2site" and (
            (mps.to_right and imps == mps.site_num - 1)
            or ((not mps.to_right) and imps == 0)
        ):
            break

        if mps.to_right:
            lmethod, rmethod = "System", "Enviro"
        else:
            lmethod, rmethod = "Enviro", "System"

        if method == "1site":
            lidx = imps - 1
            cidx = [imps]
            ridx = imps + 1
        elif method == "2site":
            if mps.to_right:
                lidx = imps - 1
                cidx = [imps, imps + 1]
                ridx = imps + 2
            else:
                lidx = imps - 2
                cidx = [imps - 1, imps]  # center site
                ridx = imps + 1
        else:
            assert False
        logger.debug(f"optimize site: {cidx}")

        if omega is None:
            operator = mpo
        else:
            assert isinstance(mpo, Mpo)
            operator = [mpo, mpo]

        if isinstance(mpo, StackedMpo):
            ltensor = [environ_item.GetLR("L", lidx, mps, operator_item, itensor=None, method=lmethod) for environ_item, operator_item in zip(environ, operator.mpos)]
            rtensor = [environ_item.GetLR("R", ridx, mps, operator_item, itensor=None, method=rmethod) for environ_item, operator_item in zip(environ, operator.mpos)]
        else:
            ltensor = environ.GetLR("L", lidx, mps, operator, itensor=None, method=lmethod)
            rtensor = environ.GetLR("R", ridx, mps, operator, itensor=None, method=rmethod)

        # get the quantum number pattern
        qnbigl, qnbigr, qnmat = mps._get_big_qn(cidx)
        qn_mask = get_qn_mask(qnmat, mps.qntot)
        cshape = qn_mask.shape

        # center mo
        if isinstance(mpo, StackedMpo):
            cmo = [[asxp(mpo_item[idx]) for idx in cidx] for mpo_item in mpo.mpos]
        else:
            cmo = [asxp(mpo[idx]) for idx in cidx]

        use_direct_eigh = np.prod(cshape) < 1000 or mps.optimize_config.algo == "direct"
        if use_direct_eigh:
            e, c = eigh_direct(mps, qn_mask, ltensor, rtensor, cmo, omega)
        else:
            # the iterative approach
            # generate initial guess
            if nroots == 1:
                if method == "1site":
                    # initial guess   b-S-c
                    #                   a
                    raw_cguess = mps[cidx[0]]
                else:
                    # initial guess b-S-c-S-e
                    #                 a   d
                    raw_cguess = tensordot(mps[cidx[0]], mps[cidx[1]], axes=1)
                cguess = [asnumpy(raw_cguess)[qn_mask]]
            else:
                cguess = []
                for ms in averaged_ms:
                    if method == "1site":
                        raw_cguess = asnumpy(ms)
                    else:
                        if mps.to_right:
                            raw_cguess = tensordot(ms, mps[cidx[1]], axes=1)
                        else:
                            raw_cguess = tensordot(mps[cidx[0]], ms, axes=1)
                    cguess.append(asnumpy(raw_cguess)[qn_mask])

            guess_dim = np.sum(qn_mask)
            cguess.extend(
                [np.random.rand(guess_dim) - 0.5 for i in range(len(cguess), nroots)]
            )
            e, c = eigh_iterative(mps, qn_mask, ltensor, rtensor, cmo, omega, cguess)

        # if multi roots, both davidson and primme return np.ndarray
        if nroots > 1:
            e = e.tolist()
        logger.debug(f"energy: {e}")
        micro_iteration_result.append((e, cidx))

        cstruct = cvec2cmat(c, qn_mask, nroots=nroots)

        # store the "optimal" mps (usually in the middle of each sweep)
        if cidx == last_opt_e_idx:
            if nroots == 1:
                res_mps = mps.copy()
                res_mps._update_mps(cstruct, cidx, qnbigl, qnbigr, percent)
            else:
                res_mps = [mps.copy() for i in range(len(cstruct))]
                for iroot in range(len(cstruct)):
                    res_mps[iroot]._update_mps(
                        cstruct[iroot], cidx, qnbigl, qnbigr, percent
                    )

        averaged_ms = mps._update_mps(cstruct, cidx, qnbigl, qnbigr, percent)
        if mps.compress_config.ofs is not None:
            mpo.try_swap_site(mps.model, mps.compress_config.ofs_swap_jw)

    mps._switch_direction()
    return micro_iteration_result, res_mps, mpo


def get_ham_direct(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor: Union[xp.ndarray, List[xp.ndarray]],
    rtensor: Union[xp.ndarray, List[xp.ndarray]],
    cmo: List[xp.ndarray],
    omega: float,
):
    logger.debug("use direct eigensolver")

    # direct algorithm
    if omega is None:
        if mps.optimize_config.method == "1site":
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            ham = oe_contract(
                "abc,bdef,lfk->adlcek",
                ltensor, cmo[0], rtensor,
                backend=backend.opt_einsum_name
            )
            ham = ham[:, :, :, qn_mask][qn_mask, :]
        else:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            ham = oe_contract(
                "abc,bdef,fghj,ljk->adglcehk",
                ltensor, cmo[0], cmo[1], rtensor,
                backend=backend.opt_einsum_name,
            )
            ham = ham[:, :, :, :, qn_mask][qn_mask, :]
    else:
        if mps.optimize_config.method == "1site":
            #   S-a e j-S
            #   O-b-O-g-O
            #   |   f   |
            #   O-c-O-i-O
            #   S-d h k-S
            ham = oe_contract(
                "abcd, befg, cfhi, jgik -> aejdhk",
                ltensor, cmo[0], cmo[0], rtensor,
                backend=backend.opt_einsum_name,
            )
            ham = ham[:, :, :, qn_mask][qn_mask, :]
        else:
            #   S-a e   j o-S
            #   O-b-O-g-O-l-O
            #   |   f   k   |
            #   O-c-O-i-O-n-O
            #   S-d h   m p-S
            ham = oe_contract(
                "abcd, befg, cfhi, gjkl, ikmn, olnp -> aejodhmp",
                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor,
                backend=backend.opt_einsum_name,
            )
            ham = ham[:, :, :, :, qn_mask][qn_mask, :]

    return ham


def sign_fix(c, nroots):
    if nroots > 1:
        if isinstance(c, list):
            return [ci / np.sign(ci[np.abs(ci).argmax()]) for ci in c]
        else:
            idx = np.abs(c).argmax(axis=0)
            return c / np.sign(c[idx, range(c.shape[1])])
    else:
        return c / np.sign(c[np.abs(c).argmax()])


def eigh_direct(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor: Union[xp.ndarray, List[xp.ndarray]],
    rtensor: Union[xp.ndarray, List[xp.ndarray]],
    cmo: List[xp.ndarray],
    omega: float,
):
    if isinstance(ltensor, list):
        assert isinstance(rtensor, list)
        assert len(ltensor) == len(rtensor)
        ham = sum([get_ham_direct(mps, qn_mask, ltensor_item, rtensor_item, cmo_item, omega) for ltensor_item, rtensor_item, cmo_item in zip(ltensor, rtensor, cmo)])
    else:
        ham = get_ham_direct(mps, qn_mask, ltensor, rtensor, cmo, omega)
    inverse = mps.optimize_config.inverse
    w, v = scipy.linalg.eigh(asnumpy(ham) * inverse)

    nroots = mps.optimize_config.nroots
    if nroots == 1:
        e = w[0]
        c = v[:, 0]
    else:
        e = w[:nroots]
        c = [v[:, iroot] for iroot in range(min(nroots, v.shape[1]))]
    return e, sign_fix(c, nroots)


def get_ham_iterative(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor: Union[xp.ndarray, List[xp.ndarray]],
    rtensor: Union[xp.ndarray, List[xp.ndarray]],
    cmo: List[xp.ndarray],
    omega: float,
):
    # iterative algorithm
    method = mps.optimize_config.method
    inverse = mps.optimize_config.inverse

    # diagonal elements of H for preconditioning
    if omega is None:
        tmp_ltensor = xp.einsum("aba -> ba", ltensor)
        tmp_cmo0 = xp.einsum("abbc -> abc", cmo[0])
        tmp_rtensor = xp.einsum("aba -> ba", rtensor)
        if method == "1site":
            #   S-a c f-S
            #   O-b-O-g-O
            #   S-a c f-S
            path = [([0, 1], "ba, bcg -> acg"), ([1, 0], "acg, gf -> acf")]
            hdiag = multi_tensor_contract(path, tmp_ltensor, tmp_cmo0, tmp_rtensor)
        else:
            #   S-a c   d f-S
            #   O-b-O-e-O-g-O
            #   S-a c   d f-S
            tmp_cmo1 = xp.einsum("abbc -> abc", cmo[1])
            path = [
                ([0, 1], "ba, bce -> ace"),
                ([0, 1], "edg, gf -> edf"),
                ([0, 1], "ace, edf -> acdf"),
            ]
            hdiag = multi_tensor_contract(
                path, tmp_ltensor, tmp_cmo0, tmp_cmo1, tmp_rtensor
            )
    else:
        if method == "1site":
            #   S-a d h-S
            #   O-b-O-f-O
            #   |   e   |
            #   O-c-O-g-O
            #   S-a d h-S
            hdiag = oe_contract(
                "abca, bdef, cedg, hfgh -> adh",
                ltensor, cmo[0], cmo[0], rtensor,
                backend=backend.opt_einsum_name,
            )
        else:
            #   S-a d   h l-S
            #   O-b-O-f-O-j-O
            #   |   e   i   |
            #   O-c-O-g-O-k-O
            #   S-a d   h l-S
            hdiag = oe_contract(
                "abca, bdef, cedg, fhij, gihk, ljkl -> adhl",
                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor,
                backend=backend.opt_einsum_name,
            )

    hdiag = asnumpy(hdiag[qn_mask] * inverse)

    # Define the H operator

    # contraction expression
    cshape = qn_mask.shape
    expr = hop_expr(ltensor, rtensor, cmo, cshape, omega is not None)
    expr_batched_factory = lambda nrhs: batched_hop_expr(ltensor, rtensor, cmo, cshape, nrhs, omega is not None)
    return hdiag, expr, expr_batched_factory


def func_sum(funcs):
    def new_func(*args, **kwargs):
        return sum([func(*args, **kwargs) for func in funcs])
    return new_func


def _rhs_batch_mode_from_equation(equation):
    if not equation or "->" not in equation:
        return None
    output_modes = equation.split("->", 1)[1].replace(" ", "")
    return output_modes[-1] if output_modes else None


def _batched_rhs_path_profile(expr):
    summary = getattr(expr, "path_summary", None) or {}
    if not summary:
        return {
            "flops": 0,
            "num_gemm": 0,
            "num_batched_gemm": 1,
            "fallback_reason": None,
            "path_fields": {},
        }

    path_fields = {}
    for key in ("path", "contraction_count", "contraction_types", "contraction_steps"):
        if key in summary:
            path_fields[key] = summary[key]
    if summary.get("largest_intermediate") is not None:
        path_fields["expr_largest_intermediate"] = summary["largest_intermediate"]

    rhs_mode = _rhs_batch_mode_from_equation(getattr(expr, "equation", None))
    if rhs_mode is not None:
        path_fields["rhs_batch_mode"] = rhs_mode

    steps = summary.get("contraction_steps") or []
    total_gemm = sum(1 for item in summary.get("contraction_types") or [] if str(item).upper() == "GEMM")
    batched_gemm = 0
    if rhs_mode is not None:
        for step in steps:
            if str(step.get("contraction_type", "")).upper() != "GEMM":
                continue
            input_modes = step.get("input_modes") or []
            if rhs_mode in str(step.get("output_modes", "")) and any(rhs_mode in str(modes) for modes in input_modes):
                batched_gemm += 1

    fallback_reason = None
    if total_gemm and batched_gemm == 0:
        fallback_reason = "batched RHS expression path has no GEMM carrying RHS mode"

    return {
        "flops": summary.get("flop_count") or 0,
        "num_gemm": max(total_gemm - batched_gemm, 0),
        "num_batched_gemm": batched_gemm,
        "fallback_reason": fallback_reason,
        "path_fields": path_fields,
    }


def _apply_hop_to_packed_vectors(x, qn_mask, expr, batched_expr, inverse):
    nrhs = 1 if x.ndim == 1 else x.shape[1]
    profile_enabled = profiling.should_record_op() and x.ndim == 2
    started = time.perf_counter() if profile_enabled else None
    spec = PackedVectorSpec(
        qn_mask=qn_mask,
        center_shape=qn_mask.shape,
        packed_dim=int(np.sum(qn_mask)),
        nrhs=int(nrhs),
    )
    packed = asxp(x)
    active_expr = expr if x.ndim == 1 else batched_expr
    if x.ndim == 1:
        cstruct = backend.unpack_masked_vectors(packed, spec)
        cout = expr(cstruct) * inverse
    elif x.ndim == 2:
        cstruct = backend.unpack_masked_vectors(packed, spec)
        cout = batched_expr(cstruct) * inverse
    else:
        raise ValueError("packed RHS input must be 1D or 2D, got shape {0}".format(x.shape))
    result = asnumpy(backend.pack_masked_vectors(cout, spec))
    if profile_enabled:
        input_nbytes = int(getattr(x, "nbytes", 0))
        output_nbytes = int(getattr(result, "nbytes", 0))
        cstruct_nbytes = int(getattr(cstruct, "nbytes", 0))
        cout_nbytes = int(getattr(cout, "nbytes", 0))
        path_profile = _batched_rhs_path_profile(active_expr)

        def device_payload(device):
            return {
                "kind": getattr(device, "kind", None),
                "index": getattr(device, "index", None),
                "local_rank": getattr(device, "local_rank", None),
                "global_rank": getattr(device, "global_rank", None),
                "visible_id": getattr(device, "visible_id", None),
            }

        def operand_payload(name, array, modes):
            info = backend.array_info(array)
            return {
                "name": name,
                "modes": list(modes),
                "shape": info.shape,
                "dtype": str(info.dtype),
                "itemsize": info.itemsize,
                "size": info.size,
                "nbytes": info.nbytes,
                "ndim": info.ndim,
                "strides": info.strides,
                "order": info.order,
                "contiguous": info.contiguous,
                "writeable": info.writeable,
                "owns_data": info.owns_data,
                "backend": info.backend_name,
                "device": str(info.device),
                "device_kind": info.device.kind,
                "device_index": info.device.index,
                "is_host": info.is_host,
                "is_device": info.is_device,
                "is_distributed": info.is_distributed,
            }

        input_modes = ("packed",) if x.ndim == 1 else ("packed", "rhs")
        profiling.record(
            "contraction_execute",
            backend=backend.name,
            equation=getattr(active_expr, "equation", None),
            lowering="batched_rhs_hop",
            input_shapes=[tuple(x.shape)],
            input_dtypes=[str(getattr(packed, "dtype", None))],
            operands=[
                operand_payload("packed_rhs", packed, input_modes),
            ],
            output_shape=tuple(result.shape),
            dtype=str(getattr(result, "dtype", None)),
            device=str(backend.current_device()),
            device_info=device_payload(backend.current_device()),
            center_shape=tuple(spec.center_shape),
            packed_dim=int(spec.packed_dim),
            qn_mask_true_count=int(np.sum(qn_mask)),
            center_tensor_shape=tuple(getattr(cstruct, "shape", ())),
            output_center_shape=tuple(getattr(cout, "shape", ())),
            flops=int(path_profile["flops"]),
            read_bytes=input_nbytes,
            write_bytes=output_nbytes,
            copy_bytes=0,
            workspace_bytes=max(cstruct_nbytes, cout_nbytes),
            largest_intermediate=max(input_nbytes, output_nbytes, cstruct_nbytes, cout_nbytes),
            num_gemm=int(path_profile["num_gemm"]),
            num_batched_gemm=int(path_profile["num_batched_gemm"]),
            num_grouped_tasks=0,
            num_blocks=0,
            num_shape_buckets=0,
            num_rhs=int(nrhs),
            fallback_reason=path_profile["fallback_reason"],
            **path_profile["path_fields"],
            wall_s=time.perf_counter() - started,
        )
    return result


def eigh_iterative(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor: Union[xp.ndarray, List[xp.ndarray]],
    rtensor: Union[xp.ndarray, List[xp.ndarray]],
    cmo: List[xp.ndarray],
    omega: float,
    cguess: List[np.ndarray],
):
    # iterative algorithm
    inverse = mps.optimize_config.inverse
    if isinstance(ltensor, list):
        assert isinstance(rtensor, list)
        assert len(ltensor) == len(rtensor)
        ham = [get_ham_iterative(mps, qn_mask, ltensor_item, rtensor_item, cmo_item, omega) for ltensor_item, rtensor_item, cmo_item in zip(ltensor, rtensor, cmo)]
        hdiag = sum([hdiag_item for hdiag_item, expr_item, expr_batched_factory_item in ham])
        expr = func_sum([expr_item for hdiag_item, expr_item, expr_batched_factory_item in ham])
        expr_batched_factory = lambda nrhs: func_sum([
            expr_batched_factory_item(nrhs)
            for hdiag_item, expr_item, expr_batched_factory_item in ham
        ])
    else:
        hdiag, expr, expr_batched_factory = get_ham_iterative(mps, qn_mask, ltensor, rtensor, cmo, omega)

    count = 0

    def hop(x):
        nonlocal count
        count += 1
        expr_batched = expr_batched_factory(x.shape[1]) if x.ndim == 2 else None
        return _apply_hop_to_packed_vectors(x, qn_mask, expr, expr_batched, inverse)

    # Find the eigenvectors
    algo = mps.optimize_config.algo
    nroots = mps.optimize_config.nroots
    if algo == "davidson":
        precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

        e, c = davidson(
            hop, cguess, precond, max_cycle=100, nroots=nroots, max_memory=64000
        )
        # if one root, davidson return e as np.float

    # elif algo == "arpack":
    #    # scipy arpack solver : much slower than pyscf/davidson
    #    A = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros), matvec=hop)
    #    e, c = scipy.sparse.linalg.eigsh(A, k=nroots, which="SA", v0=cguess)
    #    # scipy return numpy.array
    #    if nroots == 1:
    #        e = e[0]
    # elif algo == "lobpcg":
    #    precond = lambda x: scipy.sparse.diags(1/(hdiag+1e-4)) @ x
    #    A = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros),
    #            matvec=hop, matmat=hop)
    #    M = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros),
    #            matvec=precond, matmat=hop)
    #    e, c = scipy.sparse.linalg.lobpcg(A, np.array(cguess).T,
    #            M=M, largest=False)
    elif algo == "primme":
        if primme is None:
            logger.error("can not import primme")
            raise IMPORT_PRIMME_EXCEPTION
        h_dim = np.sum(qn_mask)
        precond = lambda x: scipy.sparse.diags(1 / (hdiag + 1e-4)) @ x
        A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop, matmat=hop)
        M = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=precond, matmat=hop)
        e, c = primme.eigsh(
            A,
            k=min(nroots, h_dim),
            which="SA",
            v0=np.array(cguess).T,
            OPinv=M,
            method="PRIMME_DYNAMIC",
            tol=1e-6,
        )
    else:
        assert False
    logger.debug(f"use {algo}, HC hops: {count}")
    return e, sign_fix(c, nroots)


class DmrgFCISolver:
    """
    DMRG interface for PySCF FCI/CASCI/CASSCF
    """
    def __init__(self):
        self.mps: Mps = None
        self.nsorb: int = None
        self.bond_dimension: int = 32
        self.procedure = None
        self.rdm1_mpos = []
        self.rdm2_mpos = []

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        if self.nsorb is None:
            self.nsorb = norb * 2
        else:
            assert norb * 2 == self.nsorb

        import pyscf
        h2 = pyscf.ao2mo.restore(1, h2, norb)
        # spatial orbital to spin orbital
        h1, h2 = int_to_h(h1, h2)

        basis, ham_terms = qc_model(h1, h2)


        model = Model(basis, ham_terms)
        mpo = Mpo(model)
        logger.info(f"mpo_bond_dims:{mpo.bond_dims}")

        if isinstance(nelec, (int, np.integer)):
            nelec = [nelec - nelec//2, nelec//2]

        M = self.bond_dimension
        mps = Mps.random(model, nelec, M, percent=1.0)

        if self.procedure is None:
            mps.optimize_config.procedure = [[M, 0.4], [M, 0.2], [M, 0.1], [M, 0], [M, 0], [M, 0], [M, 0]]
        else:
            mps.optimize_config.procedure = self.procedure
        mps.optimize_config.method = "2site"
        energies, mps = optimize_mps(mps.copy(), mpo)
        gs_e = min(energies) + ecore

        self.mps = mps
        return gs_e, mps

    def _make_rdm1_mpos(self, model: Model, norb: int):
        assert norb == self.nsorb // 2
        assert not self.rdm1_mpos
        a_ops, a_dag_ops = generate_ladder_operator(self.nsorb)
        process_op = partial(simplify_op, norbs=self.nsorb, conserve_qn=True)
        for i in range(norb):
            for j in range(i + 1):
                opaa = process_op(a_dag_ops[2*i] * a_ops[2*j])
                opbb = process_op(a_dag_ops[2*i+1] * a_ops[2*j+1])
                mpo = Mpo(model, terms=[opaa, opbb])
                self.rdm1_mpos.append(mpo)

    def make_rdm1(self, params, norb, nelec):
        r"""
        Evaluate the spin-traced one-body reduced density matrix (1RDM).

        .. math::

            \textrm{1RDM}[p,q] = \langle p_{\alpha}^\dagger q_{\alpha} \rangle
                + \langle p_{\beta}^\dagger q_{\beta} \rangle

        Returns
        -------
        rdm1: np.ndarray
            The spin-traced one-body RDM.

        See Also
        --------
        make_rdm2: Evaluate the spin-traced two-body reduced density matrix (2RDM).
        """
        if params is None:
            mps = self.mps
        else:
            mps = params

        if not self.rdm1_mpos:
            self._make_rdm1_mpos(self.mps.model, norb)

        expectations = deque(mps.expectations(self.rdm1_mpos))
        rdm1 = np.zeros([norb] * 2)
        for i in range(norb):
            for j in range(i + 1):
                rdm1[i, j] = rdm1[j, i] = expectations.popleft()
        return rdm1

    def _make_rdm2_mpos(self, model: Model, norb: int):
        assert norb == self.nsorb // 2
        assert not self.rdm2_mpos
        a_ops, a_dag_ops = generate_ladder_operator(self.nsorb)
        process_op = partial(simplify_op, norbs=self.nsorb, conserve_qn=True)
        calculated_indices = set()
        # a^\dagger_p a^\dagger_q a_r a_s
        # possible spins: aaaa, abba, baab, bbbb
        for p, q, r, s in product(range(norb), repeat=4):
            if (p, q, r, s) in calculated_indices:
                continue
            opaaaa = process_op(a_dag_ops[2 * p] * a_dag_ops[2 * q] * a_ops[2 * r] * a_ops[2 * s])
            opabba = process_op(a_dag_ops[2 * p] * a_dag_ops[2 * q + 1] * a_ops[2 * r + 1] * a_ops[2 * s])
            opbaab = process_op(a_dag_ops[2 * p + 1] * a_dag_ops[2 * q] * a_ops[2 * r] * a_ops[2 * s + 1])
            opbbbb = process_op(a_dag_ops[2 * p + 1] * a_dag_ops[2 * q + 1] * a_ops[2 * r + 1] * a_ops[2 * s + 1])
            mpo = Mpo(model, terms=[opaaaa, opabba, opbaab, opbbbb])
            self.rdm2_mpos.append(mpo)
            indices = [(p, q, r, s), (s, r, q, p), (q, p, s, r), (r, s, p, q)]
            for idx in indices:
                calculated_indices.add(idx)

    def make_rdm2(self, params, norb, nelec):
        r"""
        Evaluate the spin-traced two-body reduced density matrix (2RDM).

        .. math::

            \begin{aligned}
                \textrm{2RDM}[p,q,r,s] & = \langle p_{\alpha}^\dagger r_{\alpha}^\dagger
                s_{\alpha}  q_{\alpha} \rangle
                   + \langle p_{\beta}^\dagger r_{\alpha}^\dagger s_{\alpha}  q_{\beta} \rangle \\
                   & \quad + \langle p_{\alpha}^\dagger r_{\beta}^\dagger s_{\beta}  q_{\alpha} \rangle
                   + \langle p_{\beta}^\dagger r_{\beta}^\dagger s_{\beta}  q_{\beta} \rangle
            \end{aligned}

        Returns
        -------
        rdm2: np.ndarray
            The spin-traced two-body RDM.

        See Also
        --------
        make_rdm1: Evaluate the spin-traced one-body reduced density matrix (1RDM).
        """
        if params is None:
            mps = self.mps
        else:
            mps = params

        if not self.rdm2_mpos:
            self._make_rdm2_mpos(self.mps.model, norb)

        expectations = deque(mps.expectations(self.rdm2_mpos))
        rdm2 = np.zeros([norb] * 4)
        calculated_indices = set()
        for p, q, r, s in product(range(norb), repeat=4):
            if (p, q, r, s) in calculated_indices:
                continue
            v = expectations.popleft()
            indices = [(p, q, r, s), (s, r, q, p), (q, p, s, r), (r, s, p, q)]
            for idx in indices:
                calculated_indices.add(idx)
                rdm2[idx] = v
        # transpose to PySCF notation: rdm2[p,q,r,s] = <p^+ r^+ s q>
        rdm2 = rdm2.transpose(0, 3, 1, 2)
        return rdm2

    def make_rdm12(self, params, norb, nelec):
        rdm1 = self.make_rdm1(params, norb, nelec)
        rdm2 = self.make_rdm2(params, norb, nelec)
        return rdm1, rdm2

    def spin_square(self, params, norb, nelec):
        # S^2 = (S+ * S- + S- * S+)/2 + Sz * Sz
        raise NotImplementedError
