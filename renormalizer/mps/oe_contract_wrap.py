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


def build_experimental_einsum(*args, **kwargs):
    from renormalizer.backend._gemm.experimental_einsum import (
        build_experimental_einsum as build,
    )

    return build(*args, **kwargs)


def oe_contract_expression(*args, **kwargs):
    profile_network = kwargs.pop("_profile_network", None)
    profile_center_kind = kwargs.pop("_profile_center_kind", None)
    requested_policy = kwargs.pop("_requested_policy", "legacy_oe")
    fallback_reason = kwargs.pop("_fallback_reason", None)
    skip_experimental_oe_ir = kwargs.pop("_skip_experimental_oe_ir", False)
    resolved_oe_path = kwargs.pop("_resolved_oe_path", None)
    update_kwargs(args, kwargs)
    config = backend.config
    ir_expression = None
    if (
        not skip_experimental_oe_ir
        and fallback_reason is None
        and config.execution_policy == "execution_ir"
        and config.experimental_oe_ir
    ):
        try:
            ir_expression = build_experimental_einsum(
                args,
                kwargs,
                profile_network=profile_network,
                profile_center_kind=profile_center_kind,
            )
        except NotImplementedError as error:
            if config.fallback_policy == "error":
                raise
            requested_policy = "execution_ir"
            fallback_reason = "{}: {}".format(type(error).__name__, error)
        else:
            if config.fallback_policy == "error":
                return ir_expression
            requested_policy = "execution_ir"
            resolved_oe_path = ir_expression.resolved_oe_path
            kwargs["optimize"] = resolved_oe_path
    expr = oe.contract_expression(*args, **kwargs)
    declared_operands = tuple(args[1:])

    def ordered_operand_shapes(variable_operands):
        try:
            constant_indices = tuple(
                int(index) for index in kwargs.get("constants", ())
            )
        except (TypeError, ValueError) as error:
            raise TypeError("profiled OE constants must be integer indices") from error
        if len(set(constant_indices)) != len(constant_indices) or any(
            index < 0 or index >= len(declared_operands)
            for index in constant_indices
        ):
            raise ValueError("profiled OE expression has invalid constant indices")
        constant_set = set(constant_indices)
        expected_variables = len(declared_operands) - len(constant_set)
        if len(variable_operands) != expected_variables:
            raise TypeError(
                "profiled OE expression expected {} variable operands, got {}".format(
                    expected_variables, len(variable_operands)
                )
            )
        variable_index = 0
        shapes = []
        for operand_index, declared in enumerate(declared_operands):
            if operand_index in constant_set:
                operand = declared
            else:
                operand = variable_operands[variable_index]
                variable_index += 1
            try:
                shape = tuple(int(dimension) for dimension in operand.shape)
            except AttributeError as error:
                raise TypeError(
                    "profiled OE operand {} is missing shape metadata".format(
                        operand_index
                    )
                ) from error
            shapes.append(shape)
        return tuple(shapes)

    def execute_expression(matrix, args2, kwargs2, call_fallback_reason):
        if not profiling.enabled() or profile_network is None:
            try:
                return expr(matrix, *args2, **kwargs2)
            except active_memory_errors() as e:
                logger.fatal("Out of memory error calling oe contract expression")
                log_error(e, args, kwargs)
                logger.fatal(f"Input matrix type: {type(matrix)}, shape: {matrix.shape}")
                raise e

        from time import perf_counter

        operand_shapes = ordered_operand_shapes((matrix,) + tuple(args2))
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

        selected_path = resolved_oe_path
        if selected_path is None:
            selected_path = tuple(
                tuple(int(operand) for operand in step[0])
                for step in expr.contraction_list
            )
        actual_steps = tuple(_classify_oe_lowering(step[4]) for step in expr.contraction_list)
        device = backend.current_device()
        timing = completion_timing_metadata(backend.name, device)
        payload = local_hv_payload(
            network=profile_network,
            center_kind=profile_center_kind,
            input_shapes=operand_shapes,
            output_shape=tuple(int(dimension) for dimension in result.shape),
            requested_policy=requested_policy,
            actual_policy="legacy_oe",
            planner_source="opt_einsum",
            oe_path_hash=oe_path_identity(selected_path),
            actual_steps=actual_steps,
            wall_s=wall_s,
            **timing,
            fallback_reason=call_fallback_reason,
            backend=backend.name,
            device=device,
        )
        profiling.record("local_hv_execute", **payload)
        return result

    def expr_wrapped(matrix: xp.ndarray, *args2, **kwargs2):
        return execute_expression(matrix, args2, kwargs2, fallback_reason)

    def call_with_fallback_reason(reason, matrix, *args2, **kwargs2):
        return execute_expression(matrix, args2, kwargs2, reason)

    expr_wrapped._call_with_fallback_reason = call_with_fallback_reason
    if ir_expression is not None:
        from renormalizer.backend._gemm.experimental_einsum import (
            UnsupportedRuntimeDtypeError,
        )

        def policy_expression(*variable_operands, **variable_kwargs):
            try:
                return ir_expression(*variable_operands, **variable_kwargs)
            except UnsupportedRuntimeDtypeError as error:
                reason = "{}: {}".format(type(error).__name__, error)
                return call_with_fallback_reason(
                    reason, *variable_operands, **variable_kwargs
                )

        for attribute in (
            "execution_plans",
            "execution_plan_selector",
            "resolved_oe_path",
            "execution_plan",
        ):
            if hasattr(ir_expression, attribute):
                setattr(policy_expression, attribute, getattr(ir_expression, attribute))
        return policy_expression
    return expr_wrapped
