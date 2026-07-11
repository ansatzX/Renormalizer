# wraps opt_einsum contraction to show memory errors
import logging

import opt_einsum as oe

from renormalizer.mps.backend import backend, xp
from renormalizer.utils import profiling


logger = logging.getLogger(__name__)


def _classify_oe_lowering(blas_flag):
    if not blas_flag or "EINSUM" in blas_flag:
        return "einsum"
    if blas_flag == "GEMM":
        return "gemm"
    return "tensordot"


def active_memory_errors():
    return backend.memory_errors


def active_array_types():
    array_types = backend.ndarray
    return array_types if isinstance(array_types, tuple) else (array_types,)


def log_error(e, args, kwargs):
    logger.exception(e)
    logger.fatal("The arguments are:")
    for i, arg in enumerate(args):
        if isinstance(arg, active_array_types()):
            logger.fatal(f"{i} Array type: {type(arg)}, shape:{arg.shape}")
        else:
            logger.fatal(f"{i} Non-array argument: {arg}")
    for k, v in kwargs.items():
        logger.fatal(f"{k}: {v}")


def update_kwargs(args, kwargs):
    # in Reno, the expressions are usually simple (at most 10 tensors involved)
    # Yet the performance requirement is critical - we frequently hit the memory limit
    if "optimize" not in kwargs:
        if len(args) <= 15:
            algo = "optimal"
        elif len(args) <= 25:
            algo = "dp"
        else:
            algo = "auto-hq"
        # modify in-place
        kwargs["optimize"] = algo

def oe_contract(*args, **kwargs):
    update_kwargs(args, kwargs)
    try:
        return oe.contract(*args, **kwargs)
    except active_memory_errors() as e:
        logger.fatal("Out of memory error calling oe.contract")
        log_error(e, args, kwargs)
        raise e


def oe_contract_expression(*args, **kwargs):
    profile_network = kwargs.pop("_profile_network", None)
    profile_center_kind = kwargs.pop("_profile_center_kind", None)
    update_kwargs(args, kwargs)
    expr = oe.contract_expression(*args, **kwargs)
    def expr_wrapped(matrix: xp.ndarray, *args2, **kwargs2):
        if not profiling.enabled() or profile_network is None:
            try:
                return expr(matrix, *args2, **kwargs2)
            except active_memory_errors() as e:
                logger.fatal("Out of memory error calling oe contract expression")
                log_error(e, args, kwargs)
                logger.fatal(f"Input matrix type: {type(matrix)}, shape: {matrix.shape}")
                raise e

        from time import perf_counter

        start = perf_counter()
        try:
            result = expr(matrix, *args2, **kwargs2)
        except active_memory_errors() as e:
            logger.fatal("Out of memory error calling oe contract expression")
            log_error(e, args, kwargs)
            logger.fatal(f"Input matrix type: {type(matrix)}, shape: {matrix.shape}")
            raise e
        wall_s = perf_counter() - start

        from renormalizer.backend._execution.profiling import (
            completion_timing_metadata,
            local_hv_payload,
            oe_path_identity,
        )

        operand_shapes = []
        for operand in args[1:-1]:
            operand_shapes.append(tuple(int(dimension) for dimension in operand.shape))
        operand_shapes.append(tuple(int(dimension) for dimension in matrix.shape))
        selected_path = tuple(tuple(int(operand) for operand in step[0]) for step in expr.contraction_list)
        actual_steps = tuple(_classify_oe_lowering(step[4]) for step in expr.contraction_list)
        device = backend.current_device()
        timing = completion_timing_metadata(backend.name, device)
        payload = local_hv_payload(
            network=profile_network,
            center_kind=profile_center_kind,
            input_shapes=tuple(operand_shapes),
            output_shape=tuple(int(dimension) for dimension in result.shape),
            requested_policy="legacy_oe",
            actual_policy="legacy_oe",
            planner_source="opt_einsum",
            oe_path_hash=oe_path_identity(selected_path),
            actual_steps=actual_steps,
            wall_s=wall_s,
            **timing,
            fallback_reason=None,
            backend=backend.name,
            device=device,
        )
        profiling.record("local_hv_execute", **payload)
        return result
    return expr_wrapped
