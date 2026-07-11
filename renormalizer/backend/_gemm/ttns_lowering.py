"""Specialized construction-time lowering for TTNS local H-v callables."""

from renormalizer.backend._execution.planner import lower_einsum_path
from renormalizer.backend._gemm.experimental_einsum import _build_ir_expression


def build_ttns_ir_hop(equation, constants, center_shape, center_kind):
    constants = tuple(constants)
    return _build_ir_expression(
        equation,
        constants + (tuple(center_shape),),
        tuple(range(len(constants))),
        network="ttns",
        center_kind=center_kind,
        optimize="optimal",
        lowerer=lower_einsum_path,
    )


__all__ = ["build_ttns_ir_hop"]
