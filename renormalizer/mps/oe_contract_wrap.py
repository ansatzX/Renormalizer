# wraps opt_einsum contraction to show memory errors
import logging
import time

from renormalizer.mps.backend import backend, xp
from renormalizer.utils import profiling


logger = logging.getLogger(__name__)


def active_memory_errors():
    return backend.memory_errors


def active_array_types():
    return backend.ndarray


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
    profile_enabled = profiling.should_record_op()
    started = time.perf_counter() if profile_enabled else None
    try:
        result = backend.contract(*args, **kwargs)
    except active_memory_errors() as e:
        logger.fatal("Out of memory error calling oe.contract")
        log_error(e, args, kwargs)
        raise e
    if profile_enabled:
        contract_path = getattr(backend, "contract_path", None)
        path_summary = (
            profiling.contract_path_summary(contract_path, args, kwargs)
            if callable(contract_path)
            else {}
        )
        profiling.record(
            "oe_contract",
            backend=backend.name,
            equation=profiling.first_string(args),
            input_shapes=profiling.array_shapes(args),
            operand_array_types=profiling.array_type_names(args),
            operand_array_backends=profiling.array_backend_names(args),
            output_shape=profiling.array_shape(result),
            optimize=kwargs.get("optimize"),
            **path_summary,
            **profiling.oe_compute_payload("oe_contract", args[1:], result, path_summary),
            wall_s=time.perf_counter() - started,
        )
    return result


def oe_contract_expression(*args, **kwargs):
    update_kwargs(args, kwargs)
    expr = backend.contract_expression(*args, **kwargs)
    path_summary = None
    equation = profiling.first_string(args)

    def expr_wrapped(matrix: xp.ndarray, *args2, **kwargs2):
        nonlocal path_summary
        profile_enabled = profiling.should_record_op()
        started = time.perf_counter() if profile_enabled else None
        try:
            result = expr(matrix, *args2, **kwargs2)
        except active_memory_errors() as e:
            logger.fatal("Out of memory error calling oe contract expression")
            log_error(e, args, kwargs)
            logger.fatal(f"Input matrix type: {type(matrix)}, shape: {matrix.shape}")
            raise e
        if profile_enabled:
            if path_summary is None:
                path_summary = profiling.contract_expression_path_summary(backend.contract_path, args, kwargs, expr)
                expr_wrapped.path_summary = path_summary
            profile_operands = (matrix, *args2, *args[1:])
            profiling.record(
                "oe_contract_expression",
                backend=backend.name,
                equation=profiling.first_string(args),
                input_shapes=[tuple(matrix.shape)] + profiling.array_shapes(args2),
                operand_array_types=profiling.array_type_names(profile_operands),
                operand_array_backends=profiling.array_backend_names(profile_operands),
                output_shape=profiling.array_shape(result),
                optimize=kwargs.get("optimize"),
                **path_summary,
                **profiling.oe_compute_payload("oe_expression_execute", profile_operands, result, path_summary),
                wall_s=time.perf_counter() - started,
            )
        return result
    expr_wrapped.equation = equation
    expr_wrapped.path_summary = None
    return expr_wrapped
