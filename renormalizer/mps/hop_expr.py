# -*- coding: utf-8 -*-

from renormalizer.backend._gemm.experimental_einsum import (
    UnsupportedRuntimeDtypeError,
)
from renormalizer.backend._gemm.mps_lowering import build_mps_ir_hop
from renormalizer.mps.backend import backend
from renormalizer.mps.matrix import asxp
from renormalizer.mps.oe_contract_wrap import oe_contract_expression


def _legacy_hop(
    equation,
    constants,
    cshape,
    center_kind,
    *,
    requested_policy="legacy_oe",
    fallback_reason=None,
    resolved_oe_path=None,
):
    options = {}
    if resolved_oe_path is not None:
        options["optimize"] = resolved_oe_path
        options["_resolved_oe_path"] = resolved_oe_path
    return oe_contract_expression(
        equation,
        *constants,
        cshape,
        constants=list(range(len(constants))),
        _profile_network="mps",
        _profile_center_kind=center_kind,
        _requested_policy=requested_policy,
        _fallback_reason=fallback_reason,
        _skip_experimental_oe_ir=requested_policy == "execution_ir",
        **options,
    )


def _build_hop(equation, constants, cshape, center_kind):
    config = backend.config
    if config.execution_policy == "execution_ir":
        try:
            ir_hop = build_mps_ir_hop(equation, constants, cshape, center_kind)
        except NotImplementedError as error:
            if config.fallback_policy == "error":
                raise
            fallback_reason = "{}: {}".format(type(error).__name__, error)
            return _legacy_hop(
                equation,
                constants,
                cshape,
                center_kind,
                requested_policy="execution_ir",
                fallback_reason=fallback_reason,
            )
        if config.fallback_policy == "error":
            return ir_hop
        runtime_fallback = _legacy_hop(
            equation,
            constants,
            cshape,
            center_kind,
            requested_policy="execution_ir",
            resolved_oe_path=ir_hop.resolved_oe_path,
        )

        def policy_hop(matrix, *args, **kwargs):
            try:
                return ir_hop(matrix, *args, **kwargs)
            except UnsupportedRuntimeDtypeError as error:
                reason = "{}: {}".format(type(error).__name__, error)
                return runtime_fallback._call_with_fallback_reason(
                    reason, matrix, *args, **kwargs
                )

        for attribute in (
            "execution_plans",
            "execution_plan_selector",
            "resolved_oe_path",
            "execution_plan",
        ):
            if hasattr(ir_hop, attribute):
                setattr(policy_hop, attribute, getattr(ir_hop, attribute))
        return policy_hop
    return _legacy_hop(equation, constants, cshape, center_kind)


def hop_expr(ltensor, rtensor, cmo, cshape, twolayer:bool=False):

    nsite = len(cmo)
    # whether have the ancilla
    ancilla = 2 * nsite + 2 == len(cshape)
    if not ancilla:
        assert nsite + 2 == len(cshape)

    ltensor = asxp(ltensor)
    rtensor = asxp(rtensor)
    for i in range(len(cmo)):
        cmo[i] = asxp(cmo[i])

    if nsite == 0:
        # ancilla not defined
        del ancilla

    if twolayer:
        assert nsite in [1, 2]
        # Only used in ground state algorithm
        # Hopefully generalize to CV in the future
        assert not ancilla
        if nsite == 1:
            #   S-a e j-S
            #   O-b-O-g-O
            #   |   f   |
            #   O-c-O-i-O
            #   S-d h k-S
            expr = _build_hop(
                "abcd, befg, cfhi, jgik, aej -> dhk",
                (ltensor, cmo[0], cmo[0], rtensor), cshape, "one_site",
            )
        else:
            #   S-a e   j o-S
            #   O-b-O-g-O-l-O
            #   |   f   k   |
            #   O-c-O-i-O-n-O
            #   S-d h   m p-S
            expr = _build_hop(
                "abcd, befg, cfhi, gjkl, ikmn, olnp, aejo -> dhmp",
                (ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor),
                cshape,
                "two_site",
            )
        # early return
        return expr

    # Single layer, the most common case
    # Could be written in an automatic way
    # But for now probably an overkill
    if nsite == 0:
        # S-a   l-S
        #
        # O-b - b-O
        #
        # S-c   k-S
        expr = _build_hop(
            "abc, lbk, ck -> al",
            (ltensor, rtensor), cshape, "zero_site",
        )
    elif nsite == 1:
        if not ancilla:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            expr = _build_hop(
                "abc, bdef, lfk, cek -> adl",
                (ltensor, cmo[0], rtensor), cshape, "one_site",
            )
        else:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            #     g
            expr = _build_hop(
                "abc, bdef, lfk, cegk -> adgl",
                (ltensor, cmo[0], rtensor), cshape, "one_site",
            )
    else:
        if not ancilla:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            expr = _build_hop(
                "abc, bdef, fghj, ljk, cehk -> adgl",
                (ltensor, cmo[0], cmo[1], rtensor), cshape, "two_site",
            )
        else:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            #     m   n
            expr = _build_hop(
                "abc, bdef, fghj, ljk, cemhnk -> admgnl",
                (ltensor, cmo[0], cmo[1], rtensor), cshape, "two_site",
            )

    return expr
