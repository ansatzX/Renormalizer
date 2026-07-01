# -*- coding: utf-8 -*-

import time

from renormalizer.mps.matrix import asxp
from renormalizer.mps.oe_contract_wrap import oe_contract_expression
from renormalizer.utils import profiling


def _record_hop_expr(expr, nsite, ancilla, twolayer, ltensor, rtensor, cmo, cshape, started):
    profiling.record(
        "hop_expr",
        nsite=nsite,
        ancilla=ancilla if nsite != 0 else None,
        twolayer=twolayer,
        l_shape=tuple(ltensor.shape),
        r_shape=tuple(rtensor.shape),
        mpo_shapes=[tuple(item.shape) for item in cmo],
        cshape=tuple(cshape),
        wall_s=time.perf_counter() - started,
    )
    return expr


def hop_expr(ltensor, rtensor, cmo, cshape, twolayer:bool=False):

    profile_enabled = profiling.should_record_op()
    started = time.perf_counter() if profile_enabled else None
    nsite = len(cmo)
    # whether have the ancilla
    ancilla = 2 * nsite + 2 == len(cshape)
    ancilla_for_profile = ancilla if nsite != 0 else None
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
            expr = oe_contract_expression(
                "abcd, befg, cfhi, jgik, aej -> dhk",
                ltensor, cmo[0], cmo[0], rtensor, cshape,
                constants=[0, 1, 2, 3]
            )
        else:
            #   S-a e   j o-S
            #   O-b-O-g-O-l-O
            #   |   f   k   |
            #   O-c-O-i-O-n-O
            #   S-d h   m p-S
            expr = oe_contract_expression(
                "abcd, befg, cfhi, gjkl, ikmn, olnp, aejo -> dhmp",
                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3, 4, 5],
            )
        # early return
        if profile_enabled:
            return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, cshape, started)
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
        expr = oe_contract_expression(
            "abc, lbk, ck -> al",
            ltensor, rtensor, cshape,
            constants=[0, 1],
        )
    elif nsite == 1:
        if not ancilla:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            expr = oe_contract_expression(
                "abc, bdef, lfk, cek -> adl",
                ltensor, cmo[0], rtensor, cshape,
                constants=[0, 1, 2],
            )
        else:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            #     g
            expr = oe_contract_expression(
                "abc, bdef, lfk, cegk -> adgl",
                ltensor, cmo[0], rtensor, cshape,
                constants=[0, 1, 2],
            )
    else:
        if not ancilla:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            expr = oe_contract_expression(
                "abc, bdef, fghj, ljk, cehk -> adgl",
                ltensor, cmo[0], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3],
            )
        else:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            #     m   n
            expr = oe_contract_expression(
                "abc, bdef, fghj, ljk, cemhnk -> admgnl",
                ltensor, cmo[0], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3],
            )

    if profile_enabled:
        return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, cshape, started)
    return expr


def batched_hop_expr(ltensor, rtensor, cmo, cshape, nrhs, twolayer: bool = False):
    profile_enabled = profiling.should_record_op()
    started = time.perf_counter() if profile_enabled else None
    nsite = len(cmo)
    ancilla = 2 * nsite + 2 == len(cshape)
    ancilla_for_profile = ancilla if nsite != 0 else None
    if not ancilla:
        assert nsite + 2 == len(cshape)

    ltensor = asxp(ltensor)
    rtensor = asxp(rtensor)
    for i in range(len(cmo)):
        cmo[i] = asxp(cmo[i])

    batched_shape = tuple(cshape) + (int(nrhs),)

    if twolayer:
        assert nsite in [1, 2]
        assert not ancilla
        if nsite == 1:
            expr = oe_contract_expression(
                "abcd, befg, cfhi, jgik, aejr -> dhkr",
                ltensor, cmo[0], cmo[0], rtensor, batched_shape,
                constants=[0, 1, 2, 3],
            )
        else:
            expr = oe_contract_expression(
                "abcd, befg, cfhi, gjkl, ikmn, olnp, aejor -> dhmpr",
                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor, batched_shape,
                constants=[0, 1, 2, 3, 4, 5],
            )
        if profile_enabled:
            return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, batched_shape, started)
        return expr

    if nsite == 0:
        expr = oe_contract_expression(
            "abc, lbk, ckr -> alr",
            ltensor, rtensor, batched_shape,
            constants=[0, 1],
        )
    elif nsite == 1:
        if not ancilla:
            expr = oe_contract_expression(
                "abc, bdef, lfk, cekr -> adlr",
                ltensor, cmo[0], rtensor, batched_shape,
                constants=[0, 1, 2],
            )
        else:
            expr = oe_contract_expression(
                "abc, bdef, lfk, cegkr -> adglr",
                ltensor, cmo[0], rtensor, batched_shape,
                constants=[0, 1, 2],
            )
    else:
        if not ancilla:
            expr = oe_contract_expression(
                "abc, bdef, fghj, ljk, cehkr -> adglr",
                ltensor, cmo[0], cmo[1], rtensor, batched_shape,
                constants=[0, 1, 2, 3],
            )
        else:
            expr = oe_contract_expression(
                "abc, bdef, fghj, ljk, cemhnkr -> admgnlr",
                ltensor, cmo[0], cmo[1], rtensor, batched_shape,
                constants=[0, 1, 2, 3],
            )

    if profile_enabled:
        return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, batched_shape, started)
    return expr
