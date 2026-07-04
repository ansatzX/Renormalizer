import json
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_profiling_runtime():
    from renormalizer.utils import profiling

    profiling.close_event_output()
    profiling.flush_summaries()
    yield
    profiling.close_event_output()
    profiling.flush_summaries()


def test_profiling_log_level_is_registered():
    from renormalizer.utils.log import PROFILING, parse_log_level

    assert PROFILING < logging.DEBUG
    assert logging.getLevelName(PROFILING) == "PROFILING"
    assert parse_log_level("PROFILING") == PROFILING
    assert parse_log_level("5") == PROFILING
    assert hasattr(logging.getLogger("renormalizer"), "profiling")


def test_init_log_allows_profiling_records_through_default_stream_handler():
    from renormalizer.utils.log import DEBUG, PROFILING, default_stream_handler, init_log

    old_level = logging.getLogger("renormalizer").level
    old_handler_level = default_stream_handler.level
    try:
        init_log(PROFILING)
        assert logging.getLogger("renormalizer").level == PROFILING
        assert default_stream_handler.level <= PROFILING
    finally:
        init_log(old_level)
        default_stream_handler.setLevel(old_handler_level or DEBUG)


def test_register_file_output_allows_profiling_records_when_package_level_is_profiling(tmp_path):
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger, register_file_output
    from renormalizer.utils import profiling

    old_level = package_logger.level
    handler = None
    log_path = tmp_path / "reno.log"
    try:
        init_log(PROFILING)
        handler = register_file_output(log_path)
        profiling.record("file_event", value=7)
    finally:
        if handler is not None:
            package_logger.removeHandler(handler)
            handler.close()
        init_log(old_level or DEBUG)

    text = log_path.read_text()
    assert "[PROFILING]" in text
    assert "RENORMALIZER_PROFILING" in text
    assert '"event":"file_event"' in text


def test_profiling_event_is_suppressed_at_debug_level(caplog):
    from renormalizer.utils import profiling

    caplog.set_level(logging.DEBUG, logger="renormalizer")

    profiling.record("unit_test", value=1)

    assert not [
        record for record in caplog.records
        if record.getMessage().startswith(profiling.LOG_PREFIX)
    ]


def test_profiling_boundary_does_not_initialize_runtime_when_disabled(monkeypatch):

    from renormalizer.utils import profiling

    monkeypatch.setattr(profiling, "_runtime", None)
    assert profiling.enabled() is False
    assert profiling.should_record_op() is False
    profiling.record("suppressed")
    profiling.flush_summaries()
    assert profiling._runtime is None


def test_profiling_boundary_records_when_enabled(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record("boundary_event", value=2)

    messages = [
        record.getMessage() for record in caplog.records
        if record.levelno == PROFILING and record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    assert len(messages) == 1
    payload = json.loads(messages[0][len(profiling.LOG_PREFIX):])
    assert payload["event"] == "boundary_event"
    assert payload["value"] == 2


def test_mps_import_does_not_eagerly_initialize_profiling_runtime():
    code = """
import renormalizer.mps.matrix
from renormalizer.utils import profiling
assert profiling._runtime is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr


def test_profiling_event_logs_structured_json_at_profiling_level(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record("unit_test", shape=(2, 3), value=1)

    messages = [
        record.getMessage() for record in caplog.records
        if record.levelno == PROFILING and record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    assert len(messages) == 1
    payload = json.loads(messages[0][len(profiling.LOG_PREFIX):])
    assert payload["event"] == "unit_test"
    assert payload["shape"] == [2, 3]
    assert payload["value"] == 1
    assert "wall_time" not in payload


def test_profiling_scope_context_is_added_to_events(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")

    with profiling.scope(stage="tdvp_ps", site=4, direction="L"):
        profiling.record("unit_test_scope", input_shape=(8, 2, 8))

    messages = [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    assert len(messages) == 1
    payload = json.loads(messages[0][len(profiling.LOG_PREFIX):])
    assert payload["event"] == "unit_test_scope"
    assert payload["stage"] == "tdvp_ps"
    assert payload["site"] == 4
    assert payload["direction"] == "L"
    assert payload["input_shape"] == [8, 2, 8]


def test_push_scope_and_pop_scope_add_context_without_context_manager(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")

    token = profiling.push_scope(stage="tdvp_ps", site=3)
    try:
        profiling.record("unit_test_push", value=9)
    finally:
        profiling.pop_scope(token)
    profiling.record("unit_test_after_pop", value=10)

    payloads = _profiling_payloads(caplog, profiling)
    first = next(payload for payload in payloads if payload["event"] == "unit_test_push")
    second = next(payload for payload in payloads if payload["event"] == "unit_test_after_pop")
    assert first["stage"] == "tdvp_ps"
    assert first["site"] == 3
    assert "site" not in second


def test_profiling_event_serializes_complex_values(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record("complex_dt", evolve_dt=1.5 + 2.5j)

    payloads = _profiling_payloads(caplog, profiling)
    assert payloads[0]["evolve_dt"] == {"real": 1.5, "imag": 2.5}


def test_contraction_execute_compute_payload_classifies_backend_lowerings():
    from renormalizer.utils import profiling

    grouped = profiling.contraction_execute_compute_payload("grouped_gemm")
    assert grouped["compute_class"] == "contraction_plan"
    assert grouped["compute_subclass"] == "backend_execute"
    assert grouped["compute_role"] == "kernel"

    legacy_default = profiling.contraction_execute_compute_payload()
    assert legacy_default["compute_class"] == "tensordot"

    rhs_hop = profiling.contraction_execute_compute_payload("batched_rhs_hop")
    assert rhs_hop["compute_class"] == "contraction_plan"

    rhs_loop = profiling.contraction_execute_compute_payload("fallback_rhs_loop")
    assert rhs_loop["compute_class"] == "contraction_plan"


def test_contraction_execute_standardization_preserves_explicit_legacy_compute_class():
    from renormalizer.utils import profiling

    payload = profiling.standardize_event_payload(
        "contraction_execute",
        **profiling.contraction_execute_compute_payload(),
        backend="numpy",
        lowering="fallback_rhs_loop",
        num_rhs=3,
        num_rhs_loop_calls=3,
    )

    assert payload["compute_class"] == "tensordot"
    assert payload["practical_profile"]["workload_class"] == "tensordot"


def _profiling_payloads(caplog, profiling):
    payloads = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(profiling.LOG_PREFIX):
            payloads.append(json.loads(message[len(profiling.LOG_PREFIX):]))
    return payloads


def _jsonl_payloads(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_profiling_helpers_summarize_contract_expression_path():
    import numpy as np
    import opt_einsum as oe

    from renormalizer.utils import profiling

    a = np.ones((2, 3))
    args = ("ab,bc,cd->ad", a, (3, 4), (4, 5))
    kwargs = {"constants": [0], "optimize": "greedy"}
    expr = oe.contract_expression(*args, **kwargs)

    summary = profiling.contract_expression_path_summary(oe.contract_path, args, kwargs, expr)

    assert summary["path"] == [[1, 2], [0, 1]]
    assert summary["contraction_count"] == 2
    assert summary["flop_count"] >= 1
    assert summary["largest_intermediate"] >= 1
    assert summary["contraction_types"] == ["GEMM", "GEMM"]
    assert summary["contraction_steps"] == [
        {
            "step": 0,
            "path": [1, 2],
            "operand_positions": [2, 1],
            "input_modes": ["cd", "bc"],
            "input_shapes": [[4, 5], [3, 4]],
            "output_modes": "db",
            "output_shape": [5, 3],
            "remaining_modes": ["ab", "db"],
            "remaining_shapes": [[2, 3], [5, 3]],
            "contracted_modes": ["c"],
            "contraction_type": "GEMM",
            "scaling": 3,
            "size": 15,
        },
        {
            "step": 1,
            "path": [0, 1],
            "operand_positions": [1, 0],
            "input_modes": ["db", "ab"],
            "input_shapes": [[5, 3], [2, 3]],
            "output_modes": "ad",
            "output_shape": [2, 5],
            "remaining_modes": ["ad"],
            "remaining_shapes": [[2, 5]],
            "contracted_modes": ["b"],
            "contraction_type": "GEMM",
            "scaling": 3,
            "size": 10,
        },
    ]


def test_profiling_device_payload_normalizes_backend_device_spec():
    from renormalizer.backend import DeviceSpec
    from renormalizer.utils import profiling

    payload = profiling.device_payload(
        DeviceSpec(
            kind="cuda",
            index=1,
            local_rank=2,
            global_rank=10,
            visible_id="GPU-1",
        )
    )

    assert payload == {
        "kind": "cuda",
        "index": 1,
        "local_rank": 2,
        "global_rank": 10,
        "visible_id": "GPU-1",
    }


def test_profiling_array_operand_payload_uses_backend_array_info():
    import numpy as np

    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling

    backend = NumpyBackend()
    array = np.arange(6, dtype=np.float64).reshape(3, 2)
    payload = profiling.array_operand_payload(
        backend,
        "packed_rhs",
        array,
        ("packed", "rhs"),
    )

    assert payload["name"] == "packed_rhs"
    assert payload["modes"] == ["packed", "rhs"]
    assert payload["shape"] == (3, 2)
    assert payload["dtype"] == "float64"
    assert payload["itemsize"] == array.itemsize
    assert payload["size"] == array.size
    assert payload["nbytes"] == array.nbytes
    assert payload["ndim"] == 2
    assert payload["backend"] == "numpy"
    assert payload["device_kind"] == "cpu"
    assert payload["is_host"] is True
    assert payload["is_device"] is False
    assert payload["is_distributed"] is False


def test_profiling_helpers_build_svd_qn_block_payload():
    import numpy as np

    from renormalizer.utils import profiling

    payload = profiling.svd_qn_block_payload(
        nl=(0,),
        nr=np.array([1]),
        lset=np.array([0, 2]),
        rset=np.array([1, 3, 5]),
        block=np.ones((2, 3)),
        rank=2,
    )

    assert payload == {
        "left_qn": [0],
        "right_qn": [1],
        "left_size": 2,
        "right_size": 3,
        "block_shape": (2, 3),
        "rank": 2,
    }


def test_tensordot_payload_classifies_outer_product_without_gemm():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3))
    right = np.ones((4, 5))
    result = np.tensordot(left, right, axes=0)

    payload = profiling.tensordot_compute_payload(left, right, axes=0, result=result)

    assert payload["contraction_kind"] == "outer"
    assert payload["kernel_kind"] == "outer"
    assert payload["contracted_ndim"] == 0
    assert payload["contracted_size"] == 1
    assert payload["left_free_size"] == 6
    assert payload["right_free_size"] == 20
    assert payload["output_elements"] == 120
    assert payload["equivalent_gemm"] is False
    assert payload["num_gemm"] == 0
    assert payload["multiply_count"] == 120
    assert payload["add_count"] == 0
    assert payload["flops_estimate"] == 120


def test_tensordot_outer_product_core_profile_uses_outer_kernel():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3))
    right = np.ones((4, 5))
    result = np.tensordot(left, right, axes=0)

    payload = profiling.tensordot_compute_payload(left, right, axes=0, result=result)
    core = payload["practical_profile"]["core_compute_profile"]
    mix = {item["kind"]: item for item in core["kernel_mix"]}

    assert core["kernel_groups"]["primary"] == ["outer_product"]
    assert mix["outer_product"]["count"] == 1
    assert mix["outer_product"]["flops_estimate"] == payload["flops_estimate"]
    assert "gemm_like" not in mix


def test_tensordot_payload_reports_layout_and_problem_tags():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    result = np.tensordot(left, right, axes=([2], [1]))

    payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=result)

    assert payload["left_free_axes"] == (0, 1)
    assert payload["right_free_axes"] == (0, 2)
    assert payload["left_contract_axes"] == (2,)
    assert payload["right_contract_axes"] == (1,)
    assert payload["left_free_ndim"] == 2
    assert payload["right_free_ndim"] == 2
    assert payload["output_rank"] == 4
    assert payload["layout_hint"] == "right_axis_permutation"
    assert set(payload["profile_tags"]) >= {
        "gemm_like",
        "tiny_gemm",
        "skinny_gemm",
        "axis_permutation",
        "high_rank_free",
    }


def test_tensordot_payload_reports_practical_problem_fields():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    result = np.tensordot(left, right, axes=([2], [1]))

    payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=result)

    assert payload["problem_kind"] == "tensordot:gemm"
    assert payload["problem_signature"] == "gemm:m=6,n=30,k=4,layout=right_axis_permutation"
    assert payload["problem_size_bin"] == "tiny"
    assert payload["operation_family"] == "tensordot"
    assert payload["algorithmic_kernel"] == "gemm"
    assert payload["requires_axis_permutation"] is True
    assert payload["memory_bytes_estimate"] == payload["read_bytes"] + payload["write_bytes"]
    assert payload["arithmetic_intensity_bin"] == "low"


def test_tensordot_payload_reports_axis_permutation_copy_pressure():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    result = np.tensordot(left, right, axes=([2], [1]))

    payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=result)

    assert payload["left_permutation_bytes"] == 0
    assert payload["right_permutation_bytes"] == right.nbytes
    assert payload["axis_permutation_copy_bytes"] == right.nbytes
    assert payload["copy_bytes"] == right.nbytes
    assert payload["working_set_bytes_estimate"] == (
        payload["read_bytes"] + payload["write_bytes"] + right.nbytes
    )
    assert set(payload["bottleneck_hints"]) >= {"axis_permutation_copy", "memory_bandwidth"}


def test_tensordot_payload_reports_actual_operand_layout_metadata():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.arange(2 * 3 * 4).reshape(2, 3, 4)[:, ::-1, :]
    right = np.asfortranarray(np.ones((4, 5)))
    result = np.tensordot(left, right, axes=([2], [0]))

    payload = profiling.tensordot_compute_payload(left, right, axes=([2], [0]), result=result)

    assert payload["input_shapes"] == [left.shape, right.shape]
    assert payload["input_strides"] == [left.strides, right.strides]
    assert payload["input_contiguous"] == [False, True]
    assert payload["input_orders"] == ["unknown", "F"]
    assert payload["input_backends"] == ["numpy", "numpy"]
    assert payload["input_device_kinds"] == ["cpu", "cpu"]
    assert payload["input_locations"] == ["host", "host"]
    assert payload["input_is_host"] == [True, True]
    assert payload["input_is_device"] == [False, False]
    assert payload["input_is_distributed"] == [False, False]
    assert payload["output_shape"] == result.shape
    assert payload["output_strides"] == result.strides
    assert payload["output_contiguous"] is True
    assert payload["output_backend"] == "numpy"
    assert payload["output_device_kind"] == "cpu"
    assert payload["output_location"] == "host"
    assert payload["output_is_host"] is True
    assert payload["output_is_device"] is False
    assert payload["output_is_distributed"] is False
    assert payload["compute_profile"]["layout_profile"]["input_layouts"] == [
        {"shape": left.shape, "strides": left.strides, "order": "unknown", "contiguous": False},
        {"shape": right.shape, "strides": right.strides, "order": "F", "contiguous": True},
    ]


def test_tensordot_payload_includes_compact_event_practical_profile():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    result = np.tensordot(left, right, axes=([2], [1]))

    payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=result)
    profile = payload["practical_profile"]

    assert profile["workload_class"] == "tensordot"
    assert profile["measurement_scope"] == "single_tensordot"
    assert profile["precision_level"] == "mnk_layout"
    assert profile["dominant_cost_kind"] == "axis_permutation_copy"
    assert profile["dominant_operation"] == {
        "kind": "gemm",
        "m": 6,
        "n": 30,
        "k": 4,
        "layout_hint": "right_axis_permutation",
    }
    assert profile["primary_kernel"] == "gemm"
    assert profile["primary_issue"] == "axis_permutation_copy"
    assert profile["parallelism_hint"] == "tiny_or_skinny_gemm"
    assert profile["key_metrics"] == {
        "m": 6,
        "n": 30,
        "k": 4,
        "axis_permutation_copy_bytes": right.nbytes,
        "copy_fraction_of_working_set": pytest.approx(
            right.nbytes / payload["working_set_bytes_estimate"]
        ),
        "arithmetic_intensity_flops_per_byte": pytest.approx(
            payload["flops_estimate"] / payload["memory_bytes_estimate"]
        ),
    }
    assert profile["optimization_targets"] == [
        "avoid_axis_permutation_copies",
        "batch_or_fuse_tiny_gemm",
        "optimize_skinny_gemm",
    ]


def test_tensordot_practical_profile_exposes_core_kernel_summary():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    result = np.tensordot(left, right, axes=([2], [1]))

    payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=result)
    core = payload["practical_profile"]["core_compute_profile"]
    mix = {item["kind"]: item for item in core["kernel_mix"]}

    assert core["compute_class"] == "tensordot"
    assert core["execution_granularity"] == "single_backend_contraction"
    assert core["kernel_groups"] == {
        "primary": ["gemm_like"],
        "overhead": ["layout_copy"],
        "subset": [],
    }
    assert core["core_kernel_summary"]["practical_view"]["headline"] == "single backend kernel"
    assert mix["gemm_like"]["accounting"] == "primary"
    assert mix["gemm_like"]["count"] == 1
    assert mix["gemm_like"]["flops_estimate"] == payload["flops_estimate"]
    assert mix["layout_copy"]["accounting"] == "overhead"
    assert mix["layout_copy"]["bytes"] == right.nbytes


def test_tensordot_payload_includes_actionable_event_diagnosis():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    result = np.tensordot(left, right, axes=([2], [1]))

    payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=result)
    diagnosis = payload["practical_profile"]["diagnosis"]

    assert diagnosis == {
        "measurement_focus": "layout_copy",
        "primary_issue": "axis_permutation_copy",
        "parallelism_hint": "tiny_or_skinny_gemm",
        "batching_candidate": True,
        "copy_bound_candidate": True,
        "precision_evidence": {
            "m": 6,
            "n": 30,
            "k": 4,
            "flops_estimate": payload["flops_estimate"],
            "working_set_bytes_estimate": payload["working_set_bytes_estimate"],
            "copy_fraction": pytest.approx(
                payload["axis_permutation_copy_bytes"] / payload["working_set_bytes_estimate"]
            ),
        },
        "recommended_action": "avoid_axis_permutation_copies",
    }


def test_oe_payload_classifies_path_kind_from_contraction_types():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 5))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5))),
        result,
        {
            "contraction_count": 3,
            "contraction_types": ["GEMM", "TDOT", "GEMM"],
            "flop_count": 512,
            "largest_intermediate": 64,
        },
    )

    assert payload["path_kind"] == "mixed"
    assert payload["kernel_kind"] == "oe_mixed_path"
    assert payload["contraction_type_counts"] == {"GEMM": 2, "TDOT": 1}
    assert payload["dominant_contraction_type"] == "GEMM"
    assert payload["gemm_step_count"] == 2
    assert payload["non_gemm_step_count"] == 1


def test_oe_payload_reports_gemm_coverage_and_intermediate_pressure():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 3))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 4)), np.ones((4, 5)), np.ones((5, 3))),
        result,
        {
            "contraction_count": 3,
            "contraction_types": ["GEMM", "TDOT", "OUTER/EINSUM"],
            "flop_count": 4096,
            "largest_intermediate": 240,
        },
    )

    assert payload["problem_kind"] == "oe:mixed"
    assert payload["problem_signature"] == "oe_mixed_path:steps=3,gemm=1,non_gemm=2,largest=240"
    assert payload["problem_size_bin"] == "tiny"
    assert payload["operation_family"] == "oe_contract"
    assert payload["algorithmic_kernel"] == "oe_mixed_path"
    assert payload["gemm_step_fraction"] == pytest.approx(1 / 3)
    assert payload["non_gemm_step_fraction"] == pytest.approx(2 / 3)
    assert payload["largest_intermediate_to_output_ratio"] == pytest.approx(40.0)
    assert set(payload["bottleneck_hints"]) >= {"oe_path", "oe_mixed_path", "oe_non_gemm_steps"}


def test_oe_payload_reports_flop_weighted_path_costs():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 7))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        result,
        {
            "contraction_count": 3,
            "contraction_types": ["GEMM", "TDOT", "GEMM"],
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )

    assert payload["gemm_flops_estimate"] == 300
    assert payload["non_gemm_flops_estimate"] == 800
    assert payload["gemm_flop_fraction"] == pytest.approx(300 / 1100)
    assert payload["non_gemm_flop_fraction"] == pytest.approx(800 / 1100)
    assert payload["dominant_step_type"] == "TDOT"
    assert payload["dominant_step_flops_estimate"] == 800
    assert payload["path_cost_model"] == "non_gemm_flop_dominant"
    assert set(payload["bottleneck_hints"]) >= {"oe_non_gemm_flops", "oe_non_gemm_steps"}


def test_oe_payload_includes_compact_event_practical_profile():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 7))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        result,
        {
            "contraction_count": 3,
            "contraction_types": ["GEMM", "TDOT", "GEMM"],
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )
    profile = payload["practical_profile"]

    assert profile["workload_class"] == "oe"
    assert profile["measurement_scope"] == "oe_path"
    assert profile["precision_level"] == "path_step_costs"
    assert profile["dominant_cost_kind"] == "non_gemm_path"
    assert profile["dominant_operation"] == {
        "step": 1,
        "contraction_type": "TDOT",
        "flops_estimate": 800,
        "output_elements": 60,
    }
    assert profile["top_step_costs"][:2] == [
        {
            "step": 1,
            "contraction_type": "TDOT",
            "flops_estimate": 800,
            "output_elements": 60,
        },
        {
            "step": 0,
            "contraction_type": "GEMM",
            "flops_estimate": 200,
            "output_elements": 8,
        },
    ]
    assert profile["primary_kernel"] == "oe_mixed_path"
    assert profile["primary_issue"] == "non_gemm_path"
    assert profile["path_regime"] == "non_gemm_flop_dominant"
    assert profile["key_metrics"] == {
        "contraction_count": 3,
        "gemm_step_fraction": pytest.approx(2 / 3),
        "tensordot_step_fraction": pytest.approx(1 / 3),
        "generic_einsum_step_fraction": pytest.approx(0.0),
        "non_gemm_step_fraction": pytest.approx(1 / 3),
        "gemm_flop_fraction": pytest.approx(300 / 1100),
        "tensordot_flop_fraction": pytest.approx(800 / 1100),
        "generic_einsum_flop_fraction": pytest.approx(0.0),
        "non_gemm_flop_fraction": pytest.approx(800 / 1100),
        "gemm_flops_estimate": 300,
        "tensordot_flops_estimate": 800,
        "generic_einsum_flops_estimate": 0,
        "non_gemm_flops_estimate": 800,
        "dominant_step_type": "TDOT",
        "dominant_step_flops_estimate": 800,
        "dominant_step_flop_fraction": pytest.approx(800 / 1100),
        "max_step_output_elements": 60,
        "total_step_output_elements": 82,
        "unique_step_output_shape_count": 3,
        "largest_intermediate_to_output_ratio": pytest.approx(120 / result.size),
    }
    assert profile["optimization_targets"] == [
        "reduce_non_gemm_path_cost",
        "limit_largest_intermediate",
    ]


def test_oe_practical_profile_exposes_serial_path_core_kernel_mix():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 7))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )
    core = payload["practical_profile"]["core_compute_profile"]
    mix = {item["kind"]: item for item in core["kernel_mix"]}

    assert core["compute_class"] == "oe"
    assert core["execution_granularity"] == "sequential_oe_contraction_path"
    assert core["serial_depth"] == 3
    assert core["kernel_groups"]["primary"] == ["oe_gemm_step", "oe_tensordot_step"]
    assert core["kernel_groups"]["overhead"] == ["oe_intermediate"]
    assert core["core_kernel_summary"]["practical_view"]["headline"] == "serial contraction path"
    assert mix["oe_gemm_step"]["count"] == 2
    assert mix["oe_gemm_step"]["flops_estimate"] == 300
    assert mix["oe_tensordot_step"]["count"] == 1
    assert mix["oe_tensordot_step"]["flops_estimate"] == 800


def test_oe_payload_infers_path_profile_from_steps_without_redundant_type_list():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 7))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 7))),
        result,
        {
            "contraction_count": 2,
            "flop_count": 1000,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 7)],
                    "output_shape": (2, 7),
                    "flops_estimate": 800,
                },
            ],
        },
    )
    profile = payload["practical_profile"]

    assert payload["path_kind"] == "mixed"
    assert payload["kernel_kind"] == "oe_mixed_path"
    assert payload["contraction_type_counts"] == {"GEMM": 1, "TDOT": 1}
    assert payload["gemm_step_count"] == 1
    assert payload["non_gemm_step_count"] == 1
    assert profile["path_regime"] == "non_gemm_flop_dominant"
    assert profile["diagnosis"]["measurement_focus"] == "path_kernel_mix"
    assert profile["diagnosis"]["precision_evidence"]["kernel_mix"] == {
        "gemm_steps": 1,
        "tensordot_steps": 1,
        "generic_einsum_steps": 0,
        "non_gemm_steps": 1,
        "gemm_flops_estimate": 200,
        "tensordot_flops_estimate": 800,
        "generic_einsum_flops_estimate": 0,
        "non_gemm_flops_estimate": 800,
    }


def test_oe_payload_does_not_fake_flop_split_from_type_counts_only():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 7))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 7))),
        result,
        {
            "contraction_count": 2,
            "contraction_types": ["GEMM", "TDOT"],
            "flop_count": 1000,
            "largest_intermediate": 14,
        },
    )
    profile = payload["practical_profile"]
    breakdown = payload["compute_profile"]["cost_factor_breakdown"]

    assert payload["gemm_flop_fraction"] is None
    assert payload["tensordot_flop_fraction"] is None
    assert payload["non_gemm_flop_fraction"] is None
    assert profile["precision_level"] == "path_type_counts"
    assert profile["path_regime"] == "mixed_type_count_path"
    assert profile["cost_model"]["source"] == "oe_path_type_count_estimate"
    assert profile["dominant_work"] == {
        "unit": "oe_path_type_mix",
        "contraction_count": 2,
        "gemm_step_count": 1,
        "tensordot_step_count": 1,
        "generic_einsum_step_count": 0,
        "non_gemm_step_count": 1,
    }
    assert profile["diagnosis"]["precision_evidence"]["step_costs_available"] is False
    assert profile["diagnosis"]["precision_evidence"]["flop_split_available"] is False
    assert breakdown["factors"][0]["name"] == "non_gemm_path_steps"
    assert breakdown["factors"][0]["accounting"] == "primary"
    assert breakdown["limits"]["factor_wall_time"] == "not_measured"


def test_oe_payload_reports_non_gemm_step_details_and_shape_diversity():
    import numpy as np

    from renormalizer.utils import profiling

    result = np.ones((2, 7))
    payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        result,
        {
            "contraction_count": 3,
            "contraction_types": ["GEMM", "TDOT", "OUTER/EINSUM"],
            "flop_count": 1024,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "size": 8,
                    "scaling": 3,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "size": 60,
                    "scaling": 4,
                },
                {
                    "step": 2,
                    "contraction_type": "OUTER/EINSUM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "size": 14,
                    "scaling": 2,
                },
            ],
        },
    )

    assert payload["step_type_counts"] == {"GEMM": 1, "OUTER/EINSUM": 1, "TDOT": 1}
    assert payload["unique_step_output_shape_count"] == 3
    assert payload["max_step_output_elements"] == 60
    assert payload["total_step_output_elements"] == 82
    assert payload["non_gemm_steps"] == [
        {
            "step": 1,
            "contraction_type": "TDOT",
            "input_shapes": [(2, 4), (4, 5, 6)],
            "output_shape": (2, 5, 6),
            "output_elements": 60,
            "size": 60,
            "scaling": 4,
        },
        {
            "step": 2,
            "contraction_type": "OUTER/EINSUM",
            "input_shapes": [(2, 5, 6), (7,)],
            "output_shape": (2, 7),
            "output_elements": 14,
            "size": 14,
            "scaling": 2,
        },
    ]
    assert set(payload["profile_tags"]) >= {"mixed_path", "non_gemm_steps", "shape_diverse"}


def test_oe_payload_reports_operand_and_output_layout_metadata():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)[:, :, ::-1]
    right = np.asfortranarray(np.ones((4, 5), dtype=np.float64))
    result = np.ones((2, 3, 5), dtype=np.float64)

    payload = profiling.oe_compute_payload(
        "oe_contract",
        (left, right),
        result,
        {
            "contraction_count": 1,
            "contraction_types": ["TDOT"],
            "flop_count": 240,
            "largest_intermediate": result.size,
        },
    )

    assert payload["input_shapes"] == [left.shape, right.shape]
    assert payload["input_strides"] == [left.strides, right.strides]
    assert payload["input_orders"] == ["unknown", "F"]
    assert payload["input_contiguous"] == [False, True]
    assert payload["input_backends"] == ["numpy", "numpy"]
    assert payload["input_device_kinds"] == ["cpu", "cpu"]
    assert payload["input_locations"] == ["host", "host"]
    assert payload["input_is_host"] == [True, True]
    assert payload["input_is_device"] == [False, False]
    assert payload["input_is_distributed"] == [False, False]
    assert payload["output_shape"] == result.shape
    assert payload["output_strides"] == result.strides
    assert payload["output_order"] == "C"
    assert payload["output_contiguous"] is True
    assert payload["output_backend"] == "numpy"
    assert payload["output_device_kind"] == "cpu"
    assert payload["output_location"] == "host"
    assert payload["output_is_host"] is True
    assert payload["output_is_device"] is False
    assert payload["output_is_distributed"] is False


def test_svd_payload_reports_block_density_and_shape_distribution():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.ones((10, 10))
    coef_matrix = coef_array.reshape(10, 10)
    blocks = [
        {"block_shape": (2, 3)},
        {"block_shape": (2, 3)},
        {"block_shape": (4, 1)},
    ]

    payload = profiling.svd_qn_compute_payload("SVD", coef_array, coef_matrix, blocks, (coef_array,))

    assert payload["matrix_elements"] == 100
    assert payload["total_block_elements"] == 16
    assert payload["block_element_fraction"] == pytest.approx(0.16)
    assert payload["largest_block_fraction"] == pytest.approx(6 / 16)
    assert payload["block_shape_counts"] == {"2x3": 2, "4x1": 1}
    assert payload["max_block_aspect_ratio"] == pytest.approx(4.0)


def test_svd_payload_reports_source_matrix_and_output_layout_metadata():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    coef_matrix = np.asfortranarray(coef_array.reshape(6, 4))
    out_u = np.ones((6, 3), dtype=np.float64)[:, ::-1]
    out_s = np.ones((3,), dtype=np.float64)
    outputs = (out_u, out_s)

    payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        outputs,
    )

    assert payload["input_shapes"] == [coef_array.shape, coef_matrix.shape]
    assert payload["input_strides"] == [coef_array.strides, coef_matrix.strides]
    assert payload["input_orders"] == ["C", "F"]
    assert payload["input_contiguous"] == [True, True]
    assert payload["input_backends"] == ["numpy", "numpy"]
    assert payload["input_device_kinds"] == ["cpu", "cpu"]
    assert payload["input_locations"] == ["host", "host"]
    assert payload["input_is_host"] == [True, True]
    assert payload["input_is_device"] == [False, False]
    assert payload["input_is_distributed"] == [False, False]
    assert payload["matrix_shape"] == coef_matrix.shape
    assert payload["matrix_strides"] == coef_matrix.strides
    assert payload["matrix_order"] == "F"
    assert payload["matrix_contiguous"] is True
    assert payload["matrix_backend"] == "numpy"
    assert payload["matrix_device_kind"] == "cpu"
    assert payload["matrix_location"] == "host"
    assert payload["output_shapes"] == [out_u.shape, out_s.shape]
    assert payload["output_strides"] == [out_u.strides, out_s.strides]
    assert payload["output_orders"] == ["unknown", "C"]
    assert payload["output_contiguous"] == [False, True]
    assert payload["output_backends"] == ["numpy", "numpy"]
    assert payload["output_device_kinds"] == ["cpu", "cpu"]
    assert payload["output_locations"] == ["host", "host"]
    assert payload["output_is_host"] == [True, True]
    assert payload["output_is_device"] == [False, False]
    assert payload["output_is_distributed"] == [False, False]


def test_svd_payload_reports_block_batching_and_fragmentation_metrics():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.ones((20, 20))
    blocks = [
        {"block_shape": (2, 3), "rank": 2},
        {"block_shape": (2, 3), "rank": 2},
        {"block_shape": (2, 3), "rank": 2},
        {"block_shape": (4, 1), "rank": 1},
        {"block_shape": (1, 8), "rank": 1},
    ]

    payload = profiling.svd_qn_compute_payload("SVD", coef_array, coef_array, blocks, (coef_array,))

    assert payload["problem_kind"] == "svd:qn_blocks"
    assert payload["problem_signature"] == "svd_qn:blocks=5,unique=3,dominant=2x3,rank=8"
    assert payload["problem_size_bin"] == "tiny"
    assert payload["operation_family"] == "decomposition"
    assert payload["algorithmic_kernel"] == "svd_qn"
    assert payload["block_count"] == 5
    assert payload["batchable_block_count"] == 3
    assert payload["batchable_block_fraction"] == pytest.approx(3 / 5)
    assert payload["block_shape_fragmentation"] == pytest.approx(3 / 5)
    assert payload["qn_block_density"] == pytest.approx(30 / 400)
    assert payload["qn_sparse_fraction"] == pytest.approx(1 - 30 / 400)
    assert "batchable_svd_blocks" in payload["profile_tags"]


def test_svd_payload_reports_flop_weighted_block_costs():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.ones((20, 20))
    blocks = [
        {"block_shape": (10, 10), "rank": 10},
        {"block_shape": (10, 10), "rank": 10},
        {"block_shape": (2, 2), "rank": 2},
    ]

    payload = profiling.svd_qn_compute_payload("SVD", coef_array, coef_array, blocks, (coef_array,))
    block_10 = profiling.decomposition_flops_estimate((10, 10), "SVD")
    block_2 = profiling.decomposition_flops_estimate((2, 2), "SVD")
    dense = profiling.decomposition_flops_estimate((20, 20), "SVD")
    total = block_10 * 2 + block_2

    assert payload["block_flops_estimate"] == total
    assert payload["dense_flops_estimate"] == dense
    assert payload["block_flop_fraction"] == pytest.approx(total / dense)
    assert payload["largest_block_flop_fraction"] == pytest.approx(block_10 / total)
    assert payload["batchable_flops_estimate"] == block_10 * 2
    assert payload["batchable_flop_fraction"] == pytest.approx((block_10 * 2) / total)
    assert set(payload["bottleneck_hints"]) >= {"qn_sparse_blocks", "batchable_svd_blocks"}


def test_svd_payload_reports_block_reuse_and_batching_potential():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.ones((20, 20))
    blocks = [
        {"block_shape": (2, 3), "rank": 2},
        {"block_shape": (2, 3), "rank": 2},
        {"block_shape": (2, 3), "rank": 2},
        {"block_shape": (4, 1), "rank": 1},
        {"block_shape": (1, 8), "rank": 1},
    ]

    payload = profiling.svd_qn_compute_payload("SVD", coef_array, coef_array, blocks, (coef_array,))

    assert payload["unique_block_shape_count"] == 3
    assert payload["dominant_block_shape"] == "2x3"
    assert payload["dominant_block_shape_count"] == 3
    assert payload["block_shape_reuse_fraction"] == pytest.approx(3 / 5)
    assert payload["tiny_block_count"] == 5
    assert payload["skinny_block_count"] == 2
    assert payload["rank_sum"] == 8
    assert payload["max_block_rank"] == 2
    assert set(payload["profile_tags"]) >= {
        "tiny_svd_blocks",
        "skinny_svd_blocks",
        "reused_block_shapes",
        "shape_diverse",
    }


def test_svd_payload_includes_compact_event_practical_profile():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    blocks = [
        {"block_shape": (2, 2), "rank": 2},
        {"block_shape": (2, 2), "rank": 2},
        {"block_shape": (1, 4), "rank": 1},
    ]
    outputs = (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,)))

    payload = profiling.svd_qn_compute_payload("SVD", coef_array, coef_matrix, blocks, outputs)
    profile = payload["practical_profile"]

    assert profile["workload_class"] == "svd"
    assert profile["measurement_scope"] == "qn_decomposition"
    assert profile["precision_level"] == "qn_block_groups"
    assert profile["dominant_cost_kind"] == "tiny_qn_blocks"
    assert profile["dominant_operation"] == {
        "shape": "2x2",
        "count": 2,
        "rank_sum": 4,
        "elements": 4,
        "flops_per_block_estimate": profiling.decomposition_flops_estimate((2, 2), "SVD"),
        "total_flops_estimate": profiling.decomposition_flops_estimate((2, 2), "SVD") * 2,
        "flop_fraction": pytest.approx(
            profile["top_block_shape_groups"][0]["total_flops_estimate"]
            / payload["block_flops_estimate"]
        ),
        "batchable": True,
    }
    assert profile["primary_kernel"] == "svd_qn"
    assert profile["primary_issue"] == "tiny_qn_blocks"
    assert profile["block_regime"] == "sparse_and_batchable"
    assert profile["key_metrics"] == {
        "block_count": 3,
        "unique_block_shape_count": 2,
        "qn_block_density": pytest.approx(12 / 24),
        "sparse_saved_flop_fraction": pytest.approx(
            1 - payload["block_flops_estimate"] / payload["dense_flops_estimate"]
        ),
        "batchable_flop_fraction": pytest.approx(
            payload["batchable_flops_estimate"] / payload["block_flops_estimate"]
        ),
        "block_shape_reuse_fraction": pytest.approx(2 / 3),
        "tiny_block_count": 3,
        "skinny_block_count": 1,
        "dominant_block_shape": "2x2",
        "dominant_block_shape_count": 2,
        "rank_sum": 5,
        "max_block_rank": 2,
    }
    assert profile["optimization_targets"] == [
        "batch_reused_qn_blocks",
        "reduce_tiny_block_overhead",
        "preserve_qn_sparsity",
        "manage_block_shape_fragmentation",
    ]


def test_svd_practical_profile_exposes_independent_block_core_kernel_mix():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    blocks = [
        {"block_shape": (2, 2), "rank": 2},
        {"block_shape": (2, 2), "rank": 2},
        {"block_shape": (1, 4), "rank": 1},
    ]
    outputs = (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,)))

    payload = profiling.svd_qn_compute_payload("SVD", coef_array, coef_matrix, blocks, outputs)
    core = payload["practical_profile"]["core_compute_profile"]
    mix = {item["kind"]: item for item in core["kernel_mix"]}

    assert core["compute_class"] == "svd"
    assert core["execution_granularity"] == "independent_qn_block_decompositions"
    assert core["independent_units"] == 3
    assert core["batchable_units"] == 2
    assert core["kernel_groups"] == {
        "primary": ["svd_qn_block"],
        "overhead": [],
        "subset": ["batchable_qn_block", "tiny_qn_block"],
    }
    assert core["core_kernel_summary"]["practical_view"]["headline"] == "independent QN block decompositions"
    assert mix["svd_qn_block"]["count"] == 3
    assert mix["batchable_qn_block"]["accounting"] == "subset"
    assert mix["tiny_qn_block"]["accounting"] == "subset"


def test_svd_payload_reports_actionable_block_shape_groups():
    import numpy as np

    from renormalizer.utils import profiling

    coef_array = np.ones((20, 20))
    blocks = [
        {"block_shape": (4, 4), "rank": 4},
        {"block_shape": (4, 4), "rank": 4},
        {"block_shape": (4, 4), "rank": 4},
        {"block_shape": (2, 8), "rank": 2},
        {"block_shape": (2, 8), "rank": 2},
        {"block_shape": (7, 1), "rank": 1},
    ]

    payload = profiling.svd_qn_compute_payload("SVD", coef_array, coef_array, blocks, (coef_array,))
    block_4x4 = profiling.decomposition_flops_estimate((4, 4), "SVD")
    block_2x8 = profiling.decomposition_flops_estimate((2, 8), "SVD")
    block_7x1 = profiling.decomposition_flops_estimate((7, 1), "SVD")
    total = block_4x4 * 3 + block_2x8 * 2 + block_7x1
    profile = payload["practical_profile"]

    assert payload["batchable_block_group_count"] == 2
    assert payload["top_block_shape_groups"][:2] == [
        {
            "shape": "4x4",
            "count": 3,
            "rank_sum": 12,
            "elements": 16,
            "flops_per_block_estimate": block_4x4,
            "total_flops_estimate": block_4x4 * 3,
            "flop_fraction": pytest.approx((block_4x4 * 3) / total),
            "batchable": True,
        },
        {
            "shape": "2x8",
            "count": 2,
            "rank_sum": 4,
            "elements": 16,
            "flops_per_block_estimate": block_2x8,
            "total_flops_estimate": block_2x8 * 2,
            "flop_fraction": pytest.approx((block_2x8 * 2) / total),
            "batchable": True,
        },
    ]
    assert profile["batchable_block_group_count"] == 2
    assert profile["dominant_block_shape_flop_fraction"] == pytest.approx((block_4x4 * 3) / total)
    assert profile["diagnosis"]["measurement_focus"] == "qn_block_batching"
    assert profile["diagnosis"]["precision_evidence"]["batchable_block_group_count"] == 2


def test_core_event_practical_profiles_expose_operational_schema():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    tensordot_profile = tensordot_payload["practical_profile"]

    assert tensordot_profile["cost_model"] == {
        "source": "shape_estimate",
        "flops_estimate": tensordot_payload["flops_estimate"],
        "read_bytes": tensordot_payload["read_bytes"],
        "write_bytes": tensordot_payload["write_bytes"],
        "copy_bytes": tensordot_payload["copy_bytes"],
        "communication_bytes": 0,
        "working_set_bytes_estimate": tensordot_payload["working_set_bytes_estimate"],
        "arithmetic_intensity_flops_per_byte": pytest.approx(
            tensordot_payload["flops_estimate"] / tensordot_payload["memory_bytes_estimate"]
        ),
        "effective_arithmetic_intensity_flops_per_byte": pytest.approx(
            tensordot_payload["flops_estimate"] / tensordot_payload["working_set_bytes_estimate"]
        ),
    }
    assert tensordot_profile["dominant_work"] == {
        "unit": "single_tensordot_kernel",
        "kernel": "gemm",
        "m": 6,
        "n": 30,
        "k": 4,
        "layout_hint": "right_axis_permutation",
    }
    assert tensordot_profile["parallelization"] == {
        "unit": "backend_kernel",
        "backend_kernel": "gemm",
        "batching_candidate": True,
        "rhs_batching_candidate": False,
        "independent_work_items": 1,
    }
    assert tensordot_profile["measurement_limits"]["memory_scope"] == "array_nbytes_estimate"
    assert "rss_pss_time_series" in tensordot_profile["measurement_limits"]["not_measured"]

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )
    oe_profile = oe_payload["practical_profile"]

    assert oe_profile["dominant_work"] == {
        "unit": "oe_path_step",
        "step": 1,
        "contraction_type": "TDOT",
        "flops_estimate": 800,
        "output_elements": 60,
        "flop_fraction": pytest.approx(800 / 1100),
    }
    assert oe_profile["parallelization"] == {
        "unit": "sequential_path_steps_with_backend_parallel_kernels",
        "independent_path_steps": False,
        "gemm_step_count": 2,
        "tensordot_step_count": 1,
        "generic_einsum_step_count": 0,
        "non_gemm_step_count": 1,
        "shape_diverse": True,
    }
    assert oe_profile["cost_model"]["source"] == "oe_path_step_estimate"
    assert oe_profile["cost_model"]["gemm_flop_fraction"] == pytest.approx(300 / 1100)
    assert oe_profile["cost_model"]["non_gemm_flop_fraction"] == pytest.approx(800 / 1100)
    assert oe_profile["measurement_limits"]["runtime_scope"] == "python_wall_time_if_event_timed"

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )
    svd_profile = svd_payload["practical_profile"]

    assert svd_profile["dominant_work"]["unit"] == "qn_block_shape_group"
    assert svd_profile["dominant_work"]["shape"] == "2x2"
    assert svd_profile["parallelization"] == {
        "unit": "independent_qn_block_groups",
        "batchable_block_group_count": 1,
        "batchable_block_count": 2,
        "block_count": 3,
        "unique_block_shape_count": 2,
        "shape_fragmentation": pytest.approx(2 / 3),
    }
    assert svd_profile["cost_model"]["source"] == "qn_block_flop_estimate"
    assert svd_profile["cost_model"]["sparse_saved_flop_fraction"] == pytest.approx(
        1 - svd_payload["block_flops_estimate"] / svd_payload["dense_flops_estimate"]
    )
    assert "per_backend_kernel_time" in svd_profile["measurement_limits"]["not_measured"]


def test_core_event_payloads_include_concise_compute_profile():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    tensordot_profile = tensordot_payload["compute_profile"]

    assert tensordot_profile["schema"] == "renormalizer.compute_profile.v1"
    assert tensordot_profile["event_scope"] == "single_event"
    assert tensordot_profile["compute_class"] == "tensordot"
    assert tensordot_profile["measurement_scope"] == "single_tensordot"
    assert tensordot_profile["precision_level"] == "mnk_layout"
    assert tensordot_profile["primary_kernel"] == "gemm"
    assert tensordot_profile["estimated_cost"]["flops"] == tensordot_payload["flops_estimate"]
    assert tensordot_profile["estimated_cost"]["copy_bytes"] == tensordot_payload["copy_bytes"]
    assert tensordot_profile["work"] == {
        "unit": "single_tensordot_kernel",
        "kernel": "gemm",
        "m": 6,
        "n": 30,
        "k": 4,
        "layout_hint": "right_axis_permutation",
    }
    assert tensordot_profile["parallelism"]["unit"] == "backend_kernel"
    assert tensordot_profile["top_work_items"] == [tensordot_profile["work"]]
    assert tensordot_profile["execution_route"] == {
        "schema": "renormalizer.execution_route.v1",
        "compute_class": "tensordot",
        "measurement_scope": "single_tensordot",
        "granularity": "single_backend_contraction",
        "primary_primitive": "gemm",
        "parallel_unit": "backend_kernel",
        "serial_units": 1,
        "independent_units": 1,
        "batchable_units": 0,
        "copy_bytes": right.nbytes,
        "workspace_bytes": 0,
        "communication_bytes": 0,
        "working_set_bytes": tensordot_payload["working_set_bytes_estimate"],
        "fallback": False,
        "fallback_reasons": [],
    }

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
            ],
        },
    )
    oe_profile = oe_payload["compute_profile"]

    assert oe_profile["compute_class"] == "oe"
    assert oe_profile["measurement_scope"] == "oe_path"
    assert oe_profile["precision_level"] == "path_step_costs"
    assert oe_profile["primary_kernel"] == "oe_mixed_path"
    assert oe_profile["work"]["unit"] == "oe_path_step"
    assert oe_profile["parallelism"]["unit"] == "sequential_path_steps_with_backend_parallel_kernels"
    assert oe_profile["top_work_items"][0] == {
        "step": 1,
        "contraction_type": "TDOT",
        "flops_estimate": 800,
        "output_elements": 60,
        "flop_fraction": pytest.approx(800 / 1100),
    }
    assert oe_profile["execution_route"] == {
        "schema": "renormalizer.execution_route.v1",
        "compute_class": "oe",
        "measurement_scope": "oe_path",
        "granularity": "sequential_oe_contraction_path",
        "primary_primitive": "oe_mixed_path",
        "parallel_unit": "sequential_path_steps_with_backend_parallel_kernels",
        "serial_units": 3,
        "independent_units": 0,
        "batchable_units": 0,
        "copy_bytes": 0,
        "workspace_bytes": 120 * oe_result.dtype.itemsize,
        "communication_bytes": 0,
        "working_set_bytes": oe_profile["estimated_cost"]["working_set_bytes"],
        "fallback": False,
        "fallback_reasons": [],
    }

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )
    svd_profile = svd_payload["compute_profile"]

    assert svd_profile["compute_class"] == "svd"
    assert svd_profile["measurement_scope"] == "qn_decomposition"
    assert svd_profile["precision_level"] == "qn_block_groups"
    assert svd_profile["primary_kernel"] == "svd_qn"
    assert svd_profile["parallelism"]["unit"] == "independent_qn_block_groups"
    assert svd_profile["top_work_items"][0]["shape"] == "2x2"
    assert svd_profile["top_work_items"][0]["batchable"] is True
    assert svd_profile["execution_route"] == {
        "schema": "renormalizer.execution_route.v1",
        "compute_class": "svd",
        "measurement_scope": "qn_decomposition",
        "granularity": "independent_qn_block_decompositions",
        "primary_primitive": "svd_qn",
        "parallel_unit": "independent_qn_block_groups",
        "serial_units": 0,
        "independent_units": 3,
        "batchable_units": 2,
        "copy_bytes": 0,
        "workspace_bytes": 0,
        "communication_bytes": 0,
        "working_set_bytes": svd_profile["estimated_cost"]["working_set_bytes"],
        "fallback": False,
        "fallback_reasons": [],
    }


def test_core_event_payloads_include_concise_core_operation_index():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    tensordot_op = tensordot_payload["core_operation"]

    assert tensordot_op == {
        "schema": "renormalizer.core_operation.v1",
        "compute_class": "tensordot",
        "operation_family": "tensordot",
        "kernel": "gemm",
        "dominant_kernel": "gemm_like",
        "execution_model": "single_backend_kernel",
        "parallelism_unit": "backend_kernel",
        "practical_parallel_unit": "single_gemm",
        "shape_signature": "m=6,n=30,k=4,layout=right_axis_permutation",
        "practical_bucket_key": tensordot_payload["practical_bucket"]["aggregation_key"],
        "primary_work_count": 1,
        "serial_units": 1,
        "independent_units": 1,
        "batchable_units": 0,
        "flops": tensordot_payload["flops_estimate"],
        "read_bytes": tensordot_payload["read_bytes"],
        "write_bytes": tensordot_payload["write_bytes"],
        "copy_bytes": tensordot_payload["copy_bytes"],
        "workspace_bytes": 0,
        "communication_bytes": 0,
        "dominant_cost_kind": "axis_permutation_copy",
        "recommended_action": "avoid_axis_permutation_copies",
    }

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )
    oe_op = oe_payload["core_operation"]

    assert oe_op["schema"] == "renormalizer.core_operation.v1"
    assert oe_op["compute_class"] == "oe"
    assert oe_op["operation_family"] == "oe_contract"
    assert oe_op["kernel"] == "oe_mixed_path"
    assert oe_op["dominant_kernel"] == "oe_tensordot_step"
    assert oe_op["execution_model"] == "sequential_path_with_backend_kernels"
    assert oe_op["parallelism_unit"] == "path_steps"
    assert oe_op["practical_parallel_unit"] == "sequential_path_step"
    assert oe_op["primary_work_count"] == 3
    assert oe_op["serial_units"] == 3
    assert oe_op["batchable_units"] == 0
    assert oe_op["flops"] == 1100
    assert oe_op["workspace_bytes"] == oe_payload["workspace_bytes"]
    assert oe_op["dominant_cost_kind"] == "non_gemm_path"
    assert oe_op["recommended_action"] == "reduce_non_gemm_path_cost"

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )
    svd_op = svd_payload["core_operation"]

    assert svd_op["schema"] == "renormalizer.core_operation.v1"
    assert svd_op["compute_class"] == "svd"
    assert svd_op["operation_family"] == "decomposition"
    assert svd_op["kernel"] == "svd_qn"
    assert svd_op["dominant_kernel"] == "svd_qn_block"
    assert svd_op["execution_model"] == "independent_qn_block_groups"
    assert svd_op["parallelism_unit"] == "qn_block_groups"
    assert svd_op["practical_parallel_unit"] == "qn_block_shape_group"
    assert svd_op["shape_signature"] == "blocks=3,unique=2,dominant=2x2,batchable_groups=1"
    assert svd_op["primary_work_count"] == 3
    assert svd_op["independent_units"] == 3
    assert svd_op["batchable_units"] == 2
    assert svd_op["flops"] == svd_payload["block_flops_estimate"]
    assert svd_op["dominant_cost_kind"] == "tiny_qn_blocks"
    assert svd_op["recommended_action"] == "batch_reused_qn_blocks"


def test_core_compute_profiles_roll_up_cost_factors_without_double_counting():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    tensordot_rollup = tensordot_payload["compute_profile"]["cost_factor_rollup"]

    assert tensordot_rollup["schema"] == "renormalizer.cost_factor_rollup.v1"
    assert tensordot_rollup["rank_basis"] == "exclusive_primary_cost_factors"
    assert tensordot_rollup["primary"]["top_factor"]["name"] == "layout_copy"
    assert tensordot_rollup["primary"]["totals_by_resource"]["data_movement"] == {
        "bytes": right.nbytes,
    }
    assert tensordot_rollup["diagnostic"]["factor_names"] == []

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )
    oe_rollup = oe_payload["compute_profile"]["cost_factor_rollup"]

    assert oe_rollup["primary"]["top_factor"]["name"] == "tensordot_path_flops"
    assert oe_rollup["primary"]["factor_names"][:2] == [
        "tensordot_path_flops",
        "largest_intermediate",
    ]
    assert oe_rollup["primary"]["totals_by_resource"]["tensordot_kernel"] == {
        "flops": 800,
    }
    assert oe_rollup["diagnostic"]["factor_names"] == ["non_gemm_path_flops"]
    assert oe_rollup["diagnostic"]["by_accounting"]["diagnostic_aggregate"]["factor_names"] == [
        "non_gemm_path_flops",
    ]
    assert oe_rollup["diagnostic"]["by_accounting"]["diagnostic_aggregate"]["totals_by_unit"] == {
        "flops": 800,
    }

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )
    svd_rollup = svd_payload["compute_profile"]["cost_factor_rollup"]

    assert svd_rollup["primary"]["factor_names"] == ["qn_block_decomposition_flops"]
    assert svd_rollup["primary"]["totals_by_resource"]["decomposition"] == {
        "flops": svd_payload["flops_estimate"],
    }
    assert svd_rollup["diagnostic"]["by_accounting"]["diagnostic_subset"]["factor_names"] == [
        "tiny_qn_blocks",
        "batchable_qn_block_flops",
    ]
    assert svd_rollup["diagnostic"]["by_accounting"]["diagnostic_signal"]["factor_names"] == [
        "shape_fragmentation",
    ]


def test_core_compute_profiles_expose_practical_measurement_plan():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    tensordot_profile = tensordot_payload["compute_profile"]

    assert tensordot_profile["practical_kernel_view"]["core_operation"] == "tensordot"
    assert tensordot_profile["practical_kernel_view"]["dominant_kernel"] == "gemm_like"
    assert tensordot_profile["practical_measurement_plan"] == {
        "schema": "renormalizer.practical_measurement_plan.v1",
        "compute_class": "tensordot",
        "measurement_scope": "single_tensordot",
        "precision_level": "mnk_layout",
        "cost_source": "shape_estimate",
        "current_cost_proxy": "layout_copy_bytes",
        "comparison_key": "m,n,k,layout_hint",
        "parallel_unit": "backend_kernel",
        "batching_key": "m,n,k,layout_hint",
        "wall_time_status": {
            "event_wall_time": "not_measured",
            "per_kernel_wall_time": "not_attributed",
        },
        "critical_event_fields": [
            "m",
            "n",
            "k",
            "layout_hint",
            "flops_estimate",
            "axis_permutation_copy_bytes",
            "wall_s",
        ],
        "next_measurements": [
            "backend_kernel_wall_time_by_mnk_layout",
            "axis_permutation_copy_wall_time",
            "shape_bucket_call_count",
        ],
        "precision_gaps": [
            "total_event_wall_time",
            "per_kernel_wall_time",
            "layout_copy_wall_time",
        ],
    }

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
            ],
        },
    )
    oe_profile = oe_payload["compute_profile"]
    oe_plan = oe_profile["practical_measurement_plan"]

    assert oe_profile["practical_kernel_view"]["core_operation"] == "oe_contract"
    assert oe_profile["practical_kernel_view"]["dominant_kernel"] == "oe_tensordot_step"
    assert oe_plan["current_cost_proxy"] == "path_step_flops"
    assert oe_plan["comparison_key"] == "path_step_kernel_mix"
    assert oe_plan["parallel_unit"] == "serial_path_steps"
    assert oe_plan["batching_key"] == "step_contraction_type,input_shape,output_shape"
    assert oe_plan["critical_event_fields"] == [
        "contraction_count",
        "step_costs",
        "top_step_costs",
        "largest_intermediate_elements",
        "wall_s",
    ]
    assert oe_plan["next_measurements"] == [
        "per_path_step_wall_time",
        "actual_step_kernel_type",
        "intermediate_allocation_bytes",
    ]
    assert oe_plan["precision_gaps"] == [
        "total_event_wall_time",
        "per_kernel_wall_time",
        "per_path_step_wall_time",
    ]

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )
    svd_profile = svd_payload["compute_profile"]
    svd_plan = svd_profile["practical_measurement_plan"]

    assert svd_profile["practical_kernel_view"]["core_operation"] == "svd"
    assert svd_profile["practical_kernel_view"]["dominant_kernel"] == "svd_qn_block"
    assert svd_plan["current_cost_proxy"] == "block_shape_flops"
    assert svd_plan["comparison_key"] == "qn_block_shape_groups"
    assert svd_plan["parallel_unit"] == "independent_qn_blocks"
    assert svd_plan["batching_key"] == "block_shape,dtype,mode"
    assert svd_plan["critical_event_fields"] == [
        "block_count",
        "top_block_shape_groups",
        "batchable_block_group_count",
        "block_flops_estimate",
        "wall_s",
    ]
    assert svd_plan["next_measurements"] == [
        "per_block_shape_wall_time",
        "block_shape_group_counts",
        "decomposition_kernel_wall_time",
    ]
    assert svd_plan["precision_gaps"] == [
        "total_event_wall_time",
        "per_kernel_wall_time",
        "per_block_shape_wall_time",
    ]


def test_core_compute_profiles_include_practical_execution_summary():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)

    assert tensordot_payload["compute_profile"]["execution_summary"] == {
        "compute_class": "tensordot",
        "execution_model": "single_backend_kernel",
        "primary_kernel": "gemm",
        "dominant_cost_kind": "axis_permutation_copy",
        "recommended_action": "avoid_axis_permutation_copies",
        "measurement_focus": "layout_copy",
        "parallelism": "backend_kernel",
        "precision_level": "mnk_layout",
        "next_measurements": [
            "wall_time_by_shape",
            "axis_permutation_copy_bytes",
            "backend_kernel_time",
        ],
    }

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
            ],
        },
    )

    assert oe_payload["compute_profile"]["execution_summary"] == {
        "compute_class": "oe",
        "execution_model": "sequential_path_with_backend_kernels",
        "primary_kernel": "oe_mixed_path",
        "dominant_cost_kind": "non_gemm_path",
        "recommended_action": "reduce_non_gemm_path_cost",
        "measurement_focus": "path_kernel_mix",
        "parallelism": "serial_path_steps",
        "precision_level": "path_step_costs",
        "next_measurements": [
            "path_step_wall_time",
            "path_step_flops",
            "largest_intermediate_bytes",
        ],
    }

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )

    assert svd_payload["compute_profile"]["execution_summary"] == {
        "compute_class": "svd",
        "execution_model": "independent_qn_block_groups",
        "primary_kernel": "svd_qn",
        "dominant_cost_kind": "tiny_qn_blocks",
        "recommended_action": "batch_reused_qn_blocks",
        "measurement_focus": "qn_block_batching",
        "parallelism": "independent_qn_block_groups",
        "precision_level": "qn_block_groups",
        "next_measurements": [
            "block_shape_wall_time",
            "block_shape_groups",
            "decomposition_kernel_time",
        ],
    }


def test_core_compute_profiles_include_practical_workload_signature():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    tensordot_signature = tensordot_payload["compute_profile"]["workload_signature"]

    assert tensordot_signature == {
        "schema": "renormalizer.workload_signature.v1",
        "compute_class": "tensordot",
        "measurement_scope": "single_tensordot",
        "call_granularity": "single_kernel",
        "kernel_family": "gemm",
        "parallelism_unit": "backend_kernel",
        "dominant_issue": "axis_permutation_copy",
        "recommended_action": "avoid_axis_permutation_copies",
        "precision_level": "mnk_layout",
        "shape_signature": "m=6,n=30,k=4,layout=right_axis_permutation",
        "cost_source": "shape_estimate",
        "estimated_flops": tensordot_payload["flops_estimate"],
        "data_movement": {
            "read_bytes": tensordot_payload["read_bytes"],
            "write_bytes": tensordot_payload["write_bytes"],
            "copy_bytes": tensordot_payload["copy_bytes"],
            "communication_bytes": 0,
        },
        "practical_bucket_key": (
            "tensordot|kernel=gemm|dtype=float64,float64->float64|m=6|n=30|k=4"
            "|layout=right_axis_permutation"
        ),
        "practical_comparison_key": "gemm:m=6,n=30,k=4,layout=right_axis_permutation",
        "practical_parallel_unit": "single_gemm",
    }

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )
    oe_signature = oe_payload["compute_profile"]["workload_signature"]

    assert oe_signature == {
        "schema": "renormalizer.workload_signature.v1",
        "compute_class": "oe",
        "measurement_scope": "oe_path",
        "call_granularity": "serial_path",
        "kernel_family": "oe_mixed_path",
        "parallelism_unit": "path_steps",
        "dominant_issue": "non_gemm_path",
        "recommended_action": "reduce_non_gemm_path_cost",
        "precision_level": "path_step_costs",
        "shape_signature": "steps=3,gemm=2,non_gemm=1,dominant_step=1:TDOT",
        "cost_source": "oe_path_step_estimate",
        "estimated_flops": 1100,
        "data_movement": {
            "read_bytes": oe_payload["read_bytes"],
            "write_bytes": oe_payload["write_bytes"],
            "copy_bytes": 0,
            "communication_bytes": 0,
        },
        "practical_bucket_key": (
            "oe|kernel=oe_mixed_path|types=GEMM>TDOT>GEMM|steps=3"
            "|dominant=TDOT:2x4+4x5x6->2x5x6"
        ),
        "practical_comparison_key": "oe_path:types=GEMM>TDOT>GEMM;steps=3;largest=120",
        "practical_parallel_unit": "sequential_path_step",
    }

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )
    svd_signature = svd_payload["compute_profile"]["workload_signature"]

    assert svd_signature == {
        "schema": "renormalizer.workload_signature.v1",
        "compute_class": "svd",
        "measurement_scope": "qn_decomposition",
        "call_granularity": "qn_block_groups",
        "kernel_family": "svd_qn",
        "parallelism_unit": "qn_block_groups",
        "dominant_issue": "tiny_qn_blocks",
        "recommended_action": "batch_reused_qn_blocks",
        "precision_level": "qn_block_groups",
        "shape_signature": "blocks=3,unique=2,dominant=2x2,batchable_groups=1",
        "cost_source": "qn_block_flop_estimate",
        "estimated_flops": svd_payload["flops_estimate"],
        "data_movement": {
            "read_bytes": svd_payload["read_bytes"],
            "write_bytes": svd_payload["write_bytes"],
            "copy_bytes": 0,
            "communication_bytes": 0,
        },
        "practical_bucket_key": "svd|kernel=svd_qn|matrix=6x4|groups=2x2:2,1x4:1",
        "practical_comparison_key": "svd_qn:kernel=svd_qn;matrix=6x4;groups=2x2:2,1x4:1",
        "practical_parallel_unit": "qn_block_shape_group",
    }


def test_contraction_execute_workload_signature_keeps_backend_lowering_separate(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="fallback_rhs_loop",
            input_shapes=[(12, 3)],
            output_shape=(12, 3),
            output_dtype="float64",
            flops=2048,
            read_bytes=1024,
            write_bytes=512,
            num_rhs=3,
            num_rhs_loop_calls=3,
            fallback_reason="batched RHS is not implemented for this expression",
            wall_s=0.25,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    signature = event["compute_profile"]["workload_signature"]

    assert signature == {
        "schema": "renormalizer.workload_signature.v1",
        "compute_class": "tensordot",
        "measurement_scope": "backend_contraction_execute",
        "call_granularity": "backend_lowering",
        "kernel_family": "fallback_rhs_loop",
        "parallelism_unit": "backend_lowering",
        "dominant_issue": "python_rhs_loop",
        "recommended_action": "batch_rhs_hop",
        "precision_level": "lowering_counters",
        "shape_signature": "lowering=fallback_rhs_loop,rhs=3,gemm=0,batched=0,grouped=0",
        "cost_source": "backend_lowering_estimate",
        "estimated_flops": 2048,
        "data_movement": {
            "read_bytes": 1024,
            "write_bytes": 512,
            "copy_bytes": 0,
            "communication_bytes": 0,
        },
    }


def test_core_compute_profiles_include_actionable_optimization_profile():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    tensordot_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    tensordot_profile = tensordot_payload["compute_profile"]["optimization_profile"]

    assert tensordot_profile == {
        "schema": "renormalizer.optimization_profile.v1",
        "family": "tensor_contraction",
        "execution_model": "single_backend_kernel",
        "bottleneck": "axis_permutation_copy",
        "recommended_action": "avoid_axis_permutation_copies",
        "work_unit": {
            "kind": "gemm",
            "m": 6,
            "n": 30,
            "k": 4,
            "layout_hint": "right_axis_permutation",
        },
        "parallelism_model": {
            "unit": "backend_kernel",
            "kernel_parallelism": "gemm",
            "independent_work_items": 1,
            "batching_candidate": True,
            "rhs_batching_candidate": False,
        },
        "key_metrics": {
            "flops": tensordot_payload["flops_estimate"],
            "working_set_bytes": tensordot_payload["working_set_bytes_estimate"],
            "effective_arithmetic_intensity_flops_per_byte": pytest.approx(
                tensordot_payload["flops_estimate"] / tensordot_payload["working_set_bytes_estimate"]
            ),
            "copy_bytes": tensordot_payload["copy_bytes"],
            "copy_fraction_of_working_set": pytest.approx(
                tensordot_payload["copy_bytes"] / tensordot_payload["working_set_bytes_estimate"]
            ),
            "communication_bytes": 0,
        },
        "actionability": {
            "priority_score": 95,
            "dominant_resource": "layout_copy",
            "optimization_scope": "layout",
            "evidence": {
                "cost_source": "shape_estimate",
                "precision_level": "mnk_layout",
                "runtime_scope": "python_wall_time_if_event_timed",
            },
        },
    }

    oe_result = np.ones((2, 7))
    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        oe_result,
        {
            "contraction_count": 3,
            "flop_count": 1100,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 200,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 800,
                },
                {
                    "step": 2,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 100,
                },
            ],
        },
    )
    oe_profile = oe_payload["compute_profile"]["optimization_profile"]

    assert oe_profile["family"] == "contraction_path"
    assert oe_profile["execution_model"] == "sequential_path_with_backend_kernels"
    assert oe_profile["bottleneck"] == "non_gemm_path"
    assert oe_profile["recommended_action"] == "reduce_non_gemm_path_cost"
    assert oe_profile["work_unit"] == {
        "kind": "path_step",
        "step": 1,
        "contraction_type": "TDOT",
        "flops_estimate": 800,
        "flop_fraction": pytest.approx(800 / 1100),
    }
    assert oe_profile["parallelism_model"] == {
        "unit": "path_steps",
        "path_steps_are_serial": True,
        "gemm_step_count": 2,
        "tensordot_step_count": 1,
        "generic_einsum_step_count": 0,
        "non_gemm_step_count": 1,
        "shape_diverse": True,
    }
    assert oe_profile["key_metrics"] == {
        "flops": 1100,
        "gemm_flop_fraction": pytest.approx(300 / 1100),
        "tensordot_flop_fraction": pytest.approx(800 / 1100),
        "generic_einsum_flop_fraction": pytest.approx(0.0),
        "non_gemm_flop_fraction": pytest.approx(800 / 1100),
        "dominant_step_flop_fraction": pytest.approx(800 / 1100),
        "largest_intermediate_to_output_ratio": pytest.approx(120 / 14),
        "communication_bytes": 0,
    }
    assert oe_profile["actionability"] == {
        "priority_score": 85,
        "dominant_resource": "serial_path",
        "optimization_scope": "contraction_path",
        "evidence": {
            "cost_source": "oe_path_step_estimate",
            "precision_level": "path_step_costs",
            "runtime_scope": "python_wall_time_if_event_timed",
        },
    }

    coef_array = np.ones((2, 3, 4))
    coef_matrix = coef_array.reshape(6, 4)
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_matrix,
        [
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (2, 2), "rank": 2},
            {"block_shape": (1, 4), "rank": 1},
        ],
        (np.ones((6, 3)), np.ones((3,)), np.ones((4, 3)), np.ones((3,))),
    )
    svd_profile = svd_payload["compute_profile"]["optimization_profile"]

    assert svd_profile["family"] == "blocked_decomposition"
    assert svd_profile["execution_model"] == "independent_qn_block_groups"
    assert svd_profile["bottleneck"] == "tiny_qn_blocks"
    assert svd_profile["recommended_action"] == "batch_reused_qn_blocks"
    assert svd_profile["work_unit"] == {
        "kind": "qn_block_shape_group",
        "dominant_shape": "2x2",
        "block_count": 3,
        "unique_block_shape_count": 2,
        "batchable_block_group_count": 1,
    }
    assert svd_profile["parallelism_model"] == {
        "unit": "qn_block_groups",
        "independent_block_groups": True,
        "block_count": 3,
        "unique_block_shape_count": 2,
        "batchable_block_group_count": 1,
        "shape_fragmentation": pytest.approx(2 / 3),
    }
    assert svd_profile["key_metrics"] == {
        "flops": svd_payload["flops_estimate"],
        "qn_block_density": pytest.approx(svd_payload["qn_block_density"]),
        "sparse_saved_flop_fraction": pytest.approx(1.0 - svd_payload["block_flop_fraction"]),
        "batchable_flop_fraction": pytest.approx(
            svd_payload["batchable_flops_estimate"] / svd_payload["block_flops_estimate"]
        ),
        "tiny_block_count": 3,
        "communication_bytes": 0,
    }
    assert svd_profile["actionability"] == {
        "priority_score": 70,
        "dominant_resource": "tiny_block_overhead",
        "optimization_scope": "qn_block_batching",
        "evidence": {
            "cost_source": "qn_block_flop_estimate",
            "precision_level": "qn_block_groups",
            "runtime_scope": "python_wall_time_if_event_timed",
        },
    }


def test_distributed_contraction_plan_profile_derives_communication_metrics(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="numpy",
            lowering="distributed",
            plan_hash="plan-1",
            flops=4096,
            read_bytes=128,
            write_bytes=64,
            distributed_modes=("i",),
            communication=[
                {"kind": "allreduce", "bytes": 96, "num_messages": 3},
                {"collective": "gather", "bytes": 160, "num_messages": 2, "wall_s": 0.25},
            ],
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]

    assert event["communication"] == [
        {
            "kind": "allreduce",
            "primitive": "allreduce",
            "collective": "allreduce",
            "is_collective": True,
            "is_point_to_point": False,
            "bytes": 96,
            "num_messages": 3,
            "block_size": 32,
            "wall_s": 0,
        },
        {
            "kind": "gather",
            "primitive": "gather",
            "collective": "gather",
            "is_collective": True,
            "is_point_to_point": False,
            "bytes": 160,
            "num_messages": 2,
            "block_size": 80,
            "wall_s": 0.25,
        },
    ]
    assert event["comm_bytes"] == 256
    assert event["communication_profile"] == {
        "communication_required": True,
        "distributed": True,
        "distributed_modes": ["i"],
        "rank": None,
        "world_size": None,
        "local_shape": None,
        "global_shape": None,
        "bytes": 256,
        "num_messages": 5,
        "num_collectives": 2,
        "max_block_size": 80,
        "bytes_by_collective": {"allreduce": 96, "gather": 160},
        "messages_by_collective": {"allreduce": 3, "gather": 2},
        "wall_s_by_collective": {"allreduce": 0, "gather": pytest.approx(0.25)},
        "wall_s": pytest.approx(0.25),
        "dominant_collective": "gather",
        "bytes_by_primitive": {"allreduce": 96, "gather": 160},
        "messages_by_primitive": {"allreduce": 3, "gather": 2},
        "wall_s_by_primitive": {"allreduce": 0, "gather": pytest.approx(0.25)},
        "dominant_primitive": "gather",
        "point_to_point_bytes": 0,
        "num_point_to_point_messages": 0,
        "working_set_bytes": 448,
        "communication_fraction_of_working_set": pytest.approx(256 / 448),
        "recommended_action": "reduce_communication",
    }

    profile = event["compute_profile"]
    assert profile["event_scope"] == "plan"
    assert profile["compute_class"] == "contraction_plan"
    assert profile["primary_kernel"] == "distributed"
    assert profile["dominant_cost_kind"] == "communication"
    assert profile["estimated_cost"]["communication_bytes"] == 256
    assert profile["estimated_cost"]["working_set_bytes"] == 128 + 64 + 256
    assert profile["parallelism"] == {
        "unit": "planned_backend_lowering",
        "backend_kernel": "distributed",
        "batched_kernel": False,
        "grouped_kernel": False,
        "distributed": True,
        "communication_num_messages": 5,
        "dominant_communication_collective": "gather",
    }
    assert profile["optimization_targets"] == ["reduce_communication"]
    assert profile["diagnosis"]["primary_issue"] == "communication"
    assert profile["diagnosis"]["precision_evidence"]["communication_bytes"] == 256
    assert profile["workload_signature"]["plan_hash"] == "plan-1"
    assert profile["workload_signature"]["communication"] == {
        "num_messages": 5,
        "dominant_collective": "gather",
    }

    optimization = profile["optimization_profile"]
    assert optimization["family"] == "contraction_plan"
    assert optimization["execution_model"] == "distributed_planner"
    assert optimization["bottleneck"] == "communication"
    assert optimization["recommended_action"] == "reduce_communication"
    assert optimization["work_unit"]["lowering"] == "distributed"
    assert optimization["parallelism_model"] == {
        "unit": "planned_backend_lowering",
        "distributed": True,
        "communication_num_messages": 5,
        "dominant_communication_collective": "gather",
    }
    assert optimization["key_metrics"] == {
        "flops": 4096,
        "working_set_bytes": 448,
        "copy_bytes": 0,
        "communication_bytes": 256,
        "communication_fraction_of_working_set": pytest.approx(256 / 448),
    }
    assert optimization["actionability"] == {
        "priority_score": 45,
        "dominant_resource": "communication",
        "optimization_scope": "distributed_layout",
        "evidence": {
            "cost_source": "contraction_plan_estimate",
            "precision_level": "planned_lowering_counters",
            "runtime_scope": "plan_estimate_only",
        },
    }


def test_contraction_plan_jsonl_standardizes_cost_profile(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="numpy",
            lowering="grouped_gemm",
            plan_hash="plan-1",
            flops=4096,
            read_bytes=1024,
            write_bytes=512,
            copy_bytes=256,
            workspace_bytes=128,
            largest_intermediate=64,
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["cost_profile"] == {
        "source": "profile_fields",
        "flops": 4096,
        "read_bytes": 1024,
        "write_bytes": 512,
        "copy_bytes": 256,
        "comm_bytes": 0,
        "workspace_bytes": 128,
        "peak_bytes": 896,
        "memory_bytes": 1536,
        "working_set_bytes": 1792,
        "largest_intermediate": 64,
        "arithmetic_intensity_flops_per_byte": pytest.approx(4096 / 1536),
        "effective_arithmetic_intensity_flops_per_byte": pytest.approx(4096 / 1792),
        "copy_fraction_of_working_set": pytest.approx(256 / 1792),
        "communication_fraction_of_working_set": pytest.approx(0.0),
        "workspace_fraction_of_peak": pytest.approx(128 / 896),
        "compute_s": None,
        "memory_s": None,
        "copy_s": None,
        "comm_s": None,
        "total_s": None,
        "estimated_time_s": None,
    }


def test_contraction_plan_jsonl_classifies_copy_profile(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="numpy",
            lowering="grouped_gemm",
            plan_hash="plan-1",
            flops=4096,
            read_bytes=1024,
            write_bytes=512,
            copy_bytes=256,
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["copy_profile"] == {
        "copy_required": True,
        "copy_bytes": 256,
        "copy_kind": "packing_or_bucketed_execution",
        "copy_source": "copy_bytes",
    }


def test_contraction_plan_jsonl_standardizes_copy_movement_profile_for_host_to_device(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="cupy",
            lowering="gemm",
            device_kind="cuda",
            device_index=0,
            copy_bytes=2048,
            copy_source_location="host",
            copy_destination_location="device",
            copy_source_device_kind="cpu",
            copy_destination_device_kind="cuda",
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["copy_movement_profile"] == {
        "copy_required": True,
        "copy_bytes": 2048,
        "copy_kind": "unspecified",
        "copy_source": "copy_bytes",
        "copy_domain": "host_to_device",
        "source_location": "host",
        "destination_location": "device",
        "source_device_kind": "cpu",
        "destination_device_kind": "cuda",
        "route": ["host", "device"],
        "bandwidth_scope": "host_device",
    }


def test_contraction_plan_jsonl_standardizes_workspace_profile(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="numpy",
            lowering="grouped_gemm",
            plan_hash="plan-1",
            flops=4096,
            read_bytes=1024,
            write_bytes=512,
            workspace_bytes=2048,
            largest_intermediate=256,
            workspace_released=False,
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["workspace_profile"] == {
        "workspace_required": True,
        "workspace_required_bytes": 2048,
        "workspace_provided": False,
        "workspace_provided_bytes": None,
        "workspace_sufficient": None,
        "workspace_slack_bytes": None,
        "workspace_device_kind": None,
        "workspace_device_index": None,
        "workspace_released": False,
        "largest_intermediate": 256,
    }


def test_contraction_plan_jsonl_standardizes_dtype_and_device_profiles(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="torch",
            lowering="distributed",
            output_dtype="complex128",
            input_dtypes=["float64", "complex128"],
            device_kind="cuda",
            device_index=3,
            rank=2,
            world_size=8,
            local_shape=(4, 16),
            global_shape=(32, 16),
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["dtype_profile"] == {
        "dtype": "complex128",
        "output_dtype": "complex128",
        "input_dtypes": ["float64", "complex128"],
        "input_dtype_set": ["complex128", "float64"],
        "numeric_kind": "complex",
        "precision_bits": 128,
        "component_bits": 64,
        "mixed_input_dtypes": True,
    }
    assert event["device_profile"] == {
        "backend": "torch",
        "device": None,
        "device_kind": "cuda",
        "device_index": 3,
        "array_location": "distributed",
        "distributed": True,
        "rank": 2,
        "world_size": 8,
        "local_shape": [4, 16],
        "global_shape": [32, 16],
    }


def test_contraction_plan_jsonl_normalizes_backend_prefixed_dtype_names(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="torch",
            lowering="gemm",
            output_dtype="torch.complex128",
            input_dtypes=["torch.complex128", "torch.complex128"],
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["dtype_profile"] == {
        "dtype": "complex128",
        "output_dtype": "complex128",
        "input_dtypes": ["complex128", "complex128"],
        "input_dtype_set": ["complex128"],
        "numeric_kind": "complex",
        "precision_bits": 128,
        "component_bits": 64,
        "mixed_input_dtypes": False,
    }


def test_contraction_plan_jsonl_standardizes_layout_profile(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="numpy",
            lowering="gemm",
            input_shapes=[(2, 3, 4), (4, 5)],
            output_shape=(2, 3, 5),
            layout_hint="left_axis_permutation",
            axis_permutation_copy_bytes=1024,
            copy_bytes=1024,
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["layout_profile"] == {
        "input_layouts": [
            {"shape": [2, 3, 4], "strides": None, "order": "unknown", "contiguous": None},
            {"shape": [4, 5], "strides": None, "order": "unknown", "contiguous": None},
        ],
        "output_layout": {"shape": [2, 3, 5], "strides": None, "order": "unknown", "contiguous": None},
        "layout_hint": "left_axis_permutation",
        "layout_transform_required": True,
        "layout_transform_kind": "axis_permutation",
        "estimated_layout_copy_bytes": 1024,
    }


def test_contraction_plan_jsonl_standardizes_lowering_profile(tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils.log import DEBUG, init_log, package_logger
    from renormalizer.utils import profiling

    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        profiling.record(
            "contraction_plan",
            backend="torch",
            lowering="fallback_einsum",
            fallback_from="grouped_gemm",
            fallback_to="einsum",
            fallback_policy="record",
            fallback_reason="backend lacks grouped_gemm",
            num_grouped_tasks=12,
            num_shape_buckets=3,
        )
    finally:
        profiling.close_event_output()
        init_log(old_level or DEBUG)

    event = _jsonl_payloads(event_path)[0]
    assert event["lowering_profile"] == {
        "lowering": "fallback_einsum",
        "lowering_family": "fallback",
        "fallback_used": True,
        "fallback_from": "grouped_gemm",
        "fallback_to": "einsum",
        "fallback_policy": "record",
        "fallback_reason": "backend lacks grouped_gemm",
        "silent_fallback": False,
        "lowering_route": ["grouped_gemm", "einsum"],
        "kernel_counts": {
            "num_gemm": 0,
            "num_batched_gemm": 0,
            "num_grouped_tasks": 12,
            "num_blocks": 0,
            "num_shape_buckets": 3,
            "num_rhs_loop_calls": 0,
        },
        "recommended_action": "remove_backend_fallback",
    }


def test_compute_summary_practical_profiles_expose_operational_schema(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        kernel_kind="gemm",
        m=16,
        n=8,
        k=4,
        flops=1024,
        read_bytes=768,
        write_bytes=1024,
        copy_bytes=512,
        axis_permutation_copy_bytes=512,
        profile_tags=("axis_permutation", "tiny_gemm"),
        wall_s=0.4,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=3,
        contraction_type_counts={"GEMM": 1, "TDOT": 2},
        gemm_flops_estimate=400,
        non_gemm_flops_estimate=600,
        dominant_step_flops_estimate=600,
        flops_estimate=1000,
        read_bytes=300,
        write_bytes=100,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=4,
        unique_block_shape_count=2,
        block_shape_reuse_fraction=0.5,
        tiny_block_count=2,
        block_flops_estimate=1000,
        dense_flops_estimate=4000,
        batchable_flops_estimate=750,
        total_block_elements=200,
        matrix_elements=1000,
        flops_estimate=1000,
        read_bytes=120,
        write_bytes=80,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summaries = {
        payload["compute_class"]: payload["practical_profile"]
        for payload in payloads
        if payload["event"] == "profile_compute_summary"
    }

    assert summaries["tensordot"]["cost_model"]["source"] == "aggregate_shape_estimate"
    assert summaries["tensordot"]["dominant_work"] == {
        "unit": "tensordot_problem_shape",
        "kernel": "gemm",
        "max_m": 16,
        "max_n": 8,
        "max_k": 4,
    }
    assert summaries["tensordot"]["parallelization"]["unit"] == "backend_kernel"
    assert summaries["oe"]["cost_model"]["source"] == "aggregate_oe_path_estimate"
    assert summaries["oe"]["dominant_work"]["unit"] == "oe_path_step"
    assert summaries["oe"]["parallelization"]["independent_path_steps"] is False
    assert summaries["svd"]["cost_model"]["source"] == "aggregate_qn_block_estimate"
    assert summaries["svd"]["dominant_work"]["unit"] == "qn_block_shape_groups"
    assert summaries["svd"]["parallelization"]["unit"] == "independent_qn_block_groups"


def test_compute_summaries_include_concise_compute_profile(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        kernel_kind="gemm",
        m=16,
        n=8,
        k=4,
        flops=1024,
        read_bytes=768,
        write_bytes=1024,
        wall_s=0.4,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=2,
        contraction_type_counts={"GEMM": 1, "TDOT": 1},
        gemm_flops_estimate=200,
        non_gemm_flops_estimate=800,
        dominant_step_flops_estimate=800,
        flops_estimate=1000,
        read_bytes=300,
        write_bytes=100,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=3,
        unique_block_shape_count=2,
        block_flops_estimate=1000,
        dense_flops_estimate=4000,
        batchable_flops_estimate=750,
        total_block_elements=200,
        matrix_elements=1000,
        flops_estimate=1000,
        read_bytes=120,
        write_bytes=80,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summaries = {
        payload["compute_class"]: payload
        for payload in payloads
        if payload["event"] == "profile_compute_summary"
    }
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    overview_rows = {row["compute_class"]: row for row in overview["classes"]}

    for compute_class, summary in summaries.items():
        profile = summary["compute_profile"]
        assert profile["schema"] == "renormalizer.compute_profile.v1"
        assert profile["event_scope"] == "aggregate"
        assert profile["compute_class"] == compute_class
        assert profile["call_count"] == 1
        assert profile["total_wall_s"] == pytest.approx(summary["total_wall_s"])
        assert profile["estimated_cost"]["flops"] == summary["total_flops_estimate"]
        assert profile["estimated_cost"]["working_set_bytes"] == summary["working_set_bytes_estimate"]
        assert profile["top_work_items"]

        row_profile = overview_rows[compute_class]["compute_profile"]
        assert row_profile["schema"] == "renormalizer.compute_profile.v1"
        assert row_profile["event_scope"] == "aggregate"
        assert row_profile["compute_class"] == compute_class
        assert row_profile["wall_fraction"] == pytest.approx(overview_rows[compute_class]["wall_fraction"])

    assert summaries["tensordot"]["compute_profile"]["work"]["unit"] == "tensordot_problem_shape"
    assert summaries["oe"]["compute_profile"]["work"]["unit"] == "oe_path_step"
    assert summaries["svd"]["compute_profile"]["work"]["unit"] == "qn_block_shape_groups"


def test_compute_summary_phase_timing_includes_grouped_gemm_pack_kernel_scatter(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "grouped_gemm_execute",
        compute_class="contraction_plan",
        compute_subclass="grouped_gemm_execute",
        compute_role="kernel",
        compute_accounting="inclusive",
        backend="numpy",
        lowering="grouped_gemm",
        flops=1024,
        read_bytes=512,
        write_bytes=256,
        compute_s=0.40,
        pointer_setup_s=0.01,
        pack_s=0.12,
        kernel_s=0.20,
        scatter_s=0.05,
        loop_s=0.0,
        wall_s=0.43,
    )

    profiling.flush_summaries()

    summary = next(
        payload
        for payload in _profiling_payloads(caplog, profiling)
        if payload["event"] == "profile_compute_summary"
        and payload["compute_subclass"] == "grouped_gemm_execute"
    )
    assert summary["phase_timing"] == {
        "compute_s": pytest.approx(0.40),
        "pointer_setup_s": pytest.approx(0.01),
        "pack_s": pytest.approx(0.12),
        "kernel_s": pytest.approx(0.20),
        "scatter_s": pytest.approx(0.05),
        "loop_s": pytest.approx(0.0),
    }
    assert summary["total_pack_s"] == pytest.approx(0.12)
    assert summary["total_kernel_s"] == pytest.approx(0.20)
    assert summary["total_scatter_s"] == pytest.approx(0.05)


def test_compute_summary_accepts_late_phase_timing_metrics(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "gemv_batch_execute",
        compute_class="contraction_plan",
        compute_subclass="gemv_batch_execute",
        compute_role="kernel",
        backend="numpy",
        lowering="gemv",
        flops=12,
        read_bytes=72,
        write_bytes=16,
        wall_s=0.01,
    )
    profiling.record(
        "contraction_execute",
        compute_class="contraction_plan",
        compute_subclass="backend_execute",
        compute_role="kernel",
        backend="numpy",
        lowering="grouped_gemm",
        flops=8,
        read_bytes=32,
        write_bytes=32,
        pack_s=0.02,
        kernel_s=0.03,
        scatter_s=0.04,
        wall_s=0.09,
    )

    profiling.flush_summaries()

    overview = next(
        payload
        for payload in _profiling_payloads(caplog, profiling)
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "contraction_plan"
        and payload.get("phase_timing")
    )
    assert overview["phase_timing"]["pack_s"] == pytest.approx(0.02)
    assert overview["phase_timing"]["kernel_s"] == pytest.approx(0.03)
    assert overview["phase_timing"]["scatter_s"] == pytest.approx(0.04)


def test_grouped_gemm_execute_policy_without_fallback_is_not_silent_fallback(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "grouped_gemm_execute",
            backend="cupy",
            device_kind="cuda",
            lowering="grouped_gemm",
            policy="cublas_grouped",
            grouped_gemm_implementation="backend_cublas_grouped",
            num_tasks=8,
            num_groups=2,
            group_sizes=[4, 4],
            total_flops=4096,
            compute_s=0.20,
            wall_s=0.25,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "grouped_gemm_execute")
    assert event["fallback_to"] is None
    assert event["lowering_profile"]["fallback_used"] is False
    assert event["lowering_profile"]["silent_fallback"] is False
    assert event["lowering_profile"]["fallback_to"] is None
    assert event["practical_profile"]["parallelism_hint"] == "grouped_backend_kernel"
    assert event["compute_profile"]["workload_signature"]["parallelism_unit"] == "grouped_backend_kernel"


def test_compute_summaries_aggregate_execution_resources(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    for stream_provided, workspace_required, workspace_nbytes, wall_s in (
        (True, 65536, 262144, 0.02),
        (False, 131072, 196608, 0.03),
    ):
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="cupy",
            device_kind="cuda",
            lowering="batched_gemm",
            input_shapes=[(8, 16, 32), (8, 32, 16)],
            output_shape=(8, 16, 16),
            output_dtype="complex128",
            flops=131072,
            read_bytes=65536,
            write_bytes=32768,
            workspace_bytes=workspace_required,
            largest_intermediate=4096,
            num_batched_gemm=1,
            stream_provided=stream_provided,
            stream_type="ExternalCudaStream" if stream_provided else None,
            workspace_provided=True,
            workspace_nbytes=workspace_nbytes,
            workspace_device_kind="cuda",
            workspace_device_index=0,
            wall_s=wall_s,
        )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    class_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "tensordot"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "tensordot")

    expected_resources = {
        "call_count": 2,
        "stream_call_count": 1,
        "workspace_call_count": 2,
        "stream_fraction": pytest.approx(0.5),
        "workspace_fraction": pytest.approx(1.0),
        "max_workspace_required_bytes": 131072,
        "max_workspace_provided_bytes": 262144,
        "max_workspace_slack_bytes": 196608,
    }
    expected_temporary = {
        "max_workspace_required_bytes": 131072,
        "max_workspace_provided_bytes": 262144,
        "max_workspace_slack_bytes": 196608,
        "max_largest_intermediate_elements": 4096,
        "max_largest_intermediate_bytes": 0,
    }

    for payload in (summary, class_summary, row):
        assert payload["total_stream_provided_calls"] == 1
        assert payload["total_workspace_provided_calls"] == 2
        assert payload["max_workspace_bytes"] == 131072
        assert payload["max_workspace_provided_bytes"] == 262144
        assert payload["max_workspace_slack_bytes"] == 196608
        assert payload["practical_profile"]["execution_resources"] == expected_resources
        assert payload["practical_profile"]["temporary_memory"] == expected_temporary
        assert payload["compute_profile"]["execution_resources"] == expected_resources
        assert payload["compute_profile"]["temporary_memory"] == expected_temporary


def test_compute_summary_workload_signatures_preserve_aggregate_shape_context(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        kernel_kind="gemm",
        m=16,
        n=8,
        k=4,
        flops=1024,
        read_bytes=768,
        write_bytes=1024,
        wall_s=0.4,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=3,
        contraction_type_counts={"GEMM": 1, "TDOT": 2},
        gemm_flops_estimate=400,
        non_gemm_flops_estimate=600,
        dominant_step_flops_estimate=600,
        max_step_output_elements=128,
        flops_estimate=1000,
        read_bytes=300,
        write_bytes=100,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=4,
        unique_block_shape_count=2,
        block_flops_estimate=1000,
        dense_flops_estimate=4000,
        batchable_flops_estimate=750,
        batchable_block_group_count=1,
        total_block_elements=200,
        matrix_elements=1000,
        flops_estimate=1000,
        read_bytes=120,
        write_bytes=80,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summaries = {
        payload["compute_class"]: payload
        for payload in payloads
        if payload["event"] == "profile_compute_summary"
    }

    assert summaries["tensordot"]["compute_profile"]["workload_signature"]["call_granularity"] == "aggregate"
    assert summaries["tensordot"]["compute_profile"]["workload_signature"]["shape_signature"] == (
        "max_m=16,max_n=8,max_k=4"
    )
    assert summaries["oe"]["compute_profile"]["workload_signature"]["call_granularity"] == "aggregate_path"
    assert summaries["oe"]["compute_profile"]["workload_signature"]["shape_signature"] == (
        "steps=3,gemm=1,non_gemm=2,max_step_output=128"
    )
    assert summaries["svd"]["compute_profile"]["workload_signature"]["call_granularity"] == (
        "aggregate_qn_block_groups"
    )
    assert summaries["svd"]["compute_profile"]["workload_signature"]["shape_signature"] == (
        "blocks=4,unique=2,batchable_groups=1"
    )


def test_profiling_helpers_summarize_mp_and_tree_payloads():
    import numpy as np

    from renormalizer.utils import profiling

    class FakeMps(list):
        is_mps = True
        is_mpo = False
        is_mpdm = False

    mp = FakeMps([np.ones((2, 3)), None, np.zeros((4,))])
    assert profiling.mp_event_name(mp, "copy") == "mps_copy"
    assert profiling.mp_tensor_shapes(mp) == [(2, 3), (4,)]
    assert profiling.array_total_bytes(mp) == 2 * 3 * 8 + 4 * 8

    leaf = SimpleNamespace(children=[], shape=(2,), tensor=np.ones((2,)))
    root = SimpleNamespace(children=[leaf], shape=(1, 2), tensor=np.ones((1, 2)))
    class FakeTree(list):
        pass
    tree = FakeTree([root, leaf])
    tree.node_list = tree

    assert profiling.tree_edge_count(tree) == 1
    assert profiling.tree_node_shapes(tree) == [(1, 2), (2,)]
    assert profiling.tree_total_bytes(tree) == 1 * 2 * 8 + 2 * 8


def test_tensordot_writes_full_event_to_jsonl_in_trace_mode(caplog, monkeypatch, tmp_path):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        result = tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert result.shape == (2, 4)
    log_payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in log_payloads if payload["event"] == "tensordot"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "tensordot")
    assert event["input_shapes"] == [[2, 3], [3, 4]]
    assert event["operand_array_types"] == ["numpy.ndarray", "numpy.ndarray"]
    assert event["operand_array_backends"] == ["numpy", "numpy"]
    assert event["axes"] == [[1], [0]]
    assert event["output_shape"] == [2, 4]
    assert event["compute_class"] == "tensordot"
    assert event["compute_subclass"] == "api_tensordot"
    assert event["compute_role"] == "composite"
    assert event["compute_accounting"] == "inclusive"
    assert event["m"] == 2
    assert event["n"] == 4
    assert event["k"] == 3
    assert event["left_free_shape"] == [2]
    assert event["right_free_shape"] == [4]
    assert event["contracted_shape"] == [3]
    assert event["equivalent_gemm"] is True
    assert event["flops_estimate"] == 48
    assert event["read_bytes"] == 2 * 3 * 8 + 3 * 4 * 8
    assert event["write_bytes"] == 2 * 4 * 8
    assert event["backend"] == "numpy"
    assert event["wall_s"] >= 0


def test_direct_tensordot_counts_as_primary_compute_class(caplog):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    result = tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
    profiling.flush_summaries()

    assert result.shape == (2, 4)
    payloads = _profiling_payloads(caplog, profiling)
    compute_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    class_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "contraction_plan"
    )
    assert compute_summary["source_events"] == ["tensordot"]
    assert class_summary["source_events"] == ["contraction_execute"]
    assert class_summary["source_subclasses"] == ["backend_execute"]
    assert class_summary["kernel_kinds"] == ["gemm"]
    assert class_summary["total_gemm"] == 1
    assert class_summary["max_m"] == 2
    assert class_summary["max_n"] == 4
    assert class_summary["max_k"] == 3


def test_contraction_execute_jsonl_has_standard_spec_fields(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="gemm",
            input_shapes=[(2, 3), (3, 4)],
            output_shape=(2, 4),
            output_dtype="float64",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["equation"] is None
    assert event["dtype"] == "float64"
    assert event["device"] is None
    assert event["flops"] == 0
    assert event["read_bytes"] == 0
    assert event["write_bytes"] == 0
    assert event["copy_bytes"] == 0
    assert event["copy_profile"] == {
        "copy_required": False,
        "copy_bytes": 0,
        "copy_kind": "none",
        "copy_source": None,
    }
    assert event["workspace_bytes"] == 0
    assert event["largest_intermediate"] == 0
    assert event["num_gemm"] == 1
    assert event["num_batched_gemm"] == 0
    assert event["num_grouped_tasks"] == 0
    assert event["num_blocks"] == 0
    assert event["num_shape_buckets"] == 0
    assert event["fallback_from"] is None
    assert event["fallback_policy"] is None
    assert event["fallback_reason"] is None
    assert event["wall_s"] == 0


def test_contraction_execute_jsonl_standardizes_cost_profile(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="batched_gemm",
            input_shapes=[(5, 2, 3), (5, 3, 4)],
            output_shape=(5, 2, 4),
            output_dtype="float64",
            flops=240,
            read_bytes=520,
            write_bytes=320,
            copy_bytes=128,
            workspace_bytes=64,
            largest_intermediate=40,
            num_batched_gemm=1,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["cost_profile"] == {
        "source": "profile_fields",
        "flops": 240,
        "read_bytes": 520,
        "write_bytes": 320,
        "copy_bytes": 128,
        "comm_bytes": 0,
        "workspace_bytes": 64,
        "peak_bytes": 512,
        "memory_bytes": 840,
        "working_set_bytes": 968,
        "largest_intermediate": 40,
        "arithmetic_intensity_flops_per_byte": pytest.approx(240 / 840),
        "effective_arithmetic_intensity_flops_per_byte": pytest.approx(240 / 968),
        "copy_fraction_of_working_set": pytest.approx(128 / 968),
        "communication_fraction_of_working_set": pytest.approx(0.0),
        "workspace_fraction_of_peak": pytest.approx(64 / 512),
        "compute_s": None,
        "memory_s": None,
        "copy_s": None,
        "comm_s": None,
        "total_s": None,
        "estimated_time_s": None,
    }


def test_contraction_execute_jsonl_classifies_copy_source(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="gemm",
            input_shapes=[(2, 3), (3, 4)],
            output_shape=(2, 4),
            copy_bytes=128,
            axis_permutation_copy_bytes=128,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["copy_profile"] == {
        "copy_required": True,
        "copy_bytes": 128,
        "copy_kind": "layout_transform",
        "copy_source": "axis_permutation_copy_bytes",
    }


def test_contraction_execute_jsonl_standardizes_copy_movement_profile_for_layout_copy(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="cupy",
            device_kind="cuda",
            device_index=0,
            lowering="gemm",
            input_shapes=[(2, 3, 4), (4, 5)],
            output_shape=(2, 3, 5),
            layout_hint="left_axis_permutation",
            copy_bytes=128,
            axis_permutation_copy_bytes=128,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["copy_movement_profile"] == {
        "copy_required": True,
        "copy_bytes": 128,
        "copy_kind": "layout_transform",
        "copy_source": "axis_permutation_copy_bytes",
        "copy_domain": "layout_transform",
        "source_location": "device",
        "destination_location": "device",
        "source_device_kind": "cuda",
        "destination_device_kind": "cuda",
        "route": ["device", "device"],
        "bandwidth_scope": "device_layout",
    }


def test_contraction_execute_jsonl_classifies_rhs_loop_fallback(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="fallback_rhs_loop",
            input_shapes=[(10, 4)],
            output_shape=(10, 4),
            num_rhs=4,
            num_rhs_loop_calls=4,
            fallback_from="batched_rhs_hop",
            fallback_to="rhs_loop",
            fallback_reason="example algorithm rebuilds environments per RHS",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["rhs_profile"] == {
        "rhs_count": 4,
        "rhs_loop_calls": 4,
        "rhs_loop_required": True,
        "batching_candidate": True,
        "batching_status": "fallback_to_loop",
        "fallback_from": "batched_rhs_hop",
        "fallback_to": "rhs_loop",
        "fallback_reason": "example algorithm rebuilds environments per RHS",
        "recommended_action": "batch_rhs_hop",
    }


def test_contraction_execute_jsonl_includes_practical_profile(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="batched_gemm",
            input_shapes=[(5, 2, 3), (5, 3, 4)],
            output_shape=(5, 2, 4),
            output_dtype="float64",
            flops=240,
            read_bytes=520,
            write_bytes=320,
            copy_bytes=128,
            workspace_bytes=64,
            num_batched_gemm=1,
            wall_s=0.05,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    profile = event["practical_profile"]
    assert profile["workload_class"] == "tensordot"
    assert profile["primary_kernel"] == "batched_gemm"
    assert profile["primary_issue"] == "copy_overhead"
    assert profile["parallelism_hint"] == "batched_backend_kernel"
    assert profile["key_metrics"]["lowering"] == "batched_gemm"
    assert profile["key_metrics"]["num_gemm"] == 0
    assert profile["key_metrics"]["num_batched_gemm"] == 1
    assert profile["key_metrics"]["num_grouped_tasks"] == 0
    assert profile["key_metrics"]["num_blocks"] == 0
    assert profile["key_metrics"]["num_shape_buckets"] == 0
    assert profile["key_metrics"]["copy_fraction_of_working_set"] == pytest.approx(128 / (520 + 320 + 128))
    assert profile["key_metrics"]["arithmetic_intensity_flops_per_byte"] == pytest.approx(240 / (520 + 320))
    assert profile["key_metrics"]["communication_bytes"] == 0
    assert profile["key_metrics"]["communication_num_messages"] == 0
    assert profile["key_metrics"]["max_communication_block_size"] == 0
    assert profile["key_metrics"]["communication_fraction_of_working_set"] == pytest.approx(0.0)
    assert profile["optimization_targets"] == ["reduce_layout_or_device_copies"]


def test_contraction_execute_jsonl_includes_core_operation_index(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="batched_gemm",
            input_shapes=[(5, 2, 3), (5, 3, 4)],
            output_shape=(5, 2, 4),
            output_dtype="float64",
            flops=240,
            read_bytes=520,
            write_bytes=320,
            copy_bytes=128,
            workspace_bytes=64,
            num_batched_gemm=1,
            wall_s=0.05,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    core = event["core_operation"]

    assert core["schema"] == "renormalizer.core_operation.v1"
    assert core["event"] == "contraction_execute"
    assert core["compute_class"] == "tensordot"
    assert core["operation_family"] == "tensordot"
    assert core["kernel"] == "batched_gemm"
    assert core["dominant_kernel"] == "gemm_like"
    assert core["parallelism_unit"] == "batched_backend_kernel"
    assert core["primary_work_count"] == 1
    assert core["flops"] == 240
    assert core["read_bytes"] == 520
    assert core["write_bytes"] == 320
    assert core["copy_bytes"] == 128
    assert core["workspace_bytes"] == 64
    assert core["communication_bytes"] == 0
    assert core["dominant_cost_kind"] == "copy_overhead"
    assert core["recommended_action"] == "reduce_layout_or_device_copies"


def test_contraction_execute_practical_profile_includes_actionable_diagnosis(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="batched_gemm",
            input_shapes=[(5, 2, 3), (5, 3, 4)],
            output_shape=(5, 2, 4),
            output_dtype="float64",
            flops=240,
            read_bytes=520,
            write_bytes=320,
            copy_bytes=128,
            workspace_bytes=64,
            num_batched_gemm=1,
            wall_s=0.05,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    profile = event["practical_profile"]
    resources = {
        "stream_provided": False,
        "stream_type": None,
        "workspace_provided": False,
        "workspace_required_bytes": 64,
        "workspace_provided_bytes": None,
        "workspace_slack_bytes": None,
        "workspace_device_kind": None,
        "workspace_device_index": None,
    }

    assert profile["diagnosis"] == {
        "measurement_focus": "copy_overhead",
        "primary_issue": "copy_overhead",
        "parallelism_hint": "batched_backend_kernel",
        "fallback_candidate": False,
        "communication_bound_candidate": False,
        "copy_bound_candidate": True,
        "precision_evidence": {
            "lowering": "batched_gemm",
            "flops": 240,
            "working_set_bytes_estimate": 968,
            "copy_fraction": pytest.approx(128 / 968),
            "communication_fraction": pytest.approx(0.0),
            "num_gemm": 0,
            "num_batched_gemm": 1,
            "num_grouped_tasks": 0,
            "num_rhs_loop_calls": 0,
            "execution_resources": resources,
        },
        "recommended_action": "reduce_layout_or_device_copies",
    }


def test_contraction_execute_profiles_expose_stream_and_workspace_resources(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="cupy",
            device_kind="cuda",
            device_index=0,
            lowering="batched_gemm",
            input_shapes=[(8, 16, 32), (8, 32, 16)],
            output_shape=(8, 16, 16),
            output_dtype="complex128",
            flops=131072,
            read_bytes=65536,
            write_bytes=32768,
            workspace_bytes=65536,
            largest_intermediate=4096,
            num_batched_gemm=1,
            stream_provided=True,
            stream_type="ExternalCudaStream",
            workspace_provided=True,
            workspace_nbytes=262144,
            workspace_device_kind="cuda",
            workspace_device_index=0,
            wall_s=0.02,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    resources = {
        "stream_provided": True,
        "stream_type": "ExternalCudaStream",
        "workspace_provided": True,
        "workspace_required_bytes": 65536,
        "workspace_provided_bytes": 262144,
        "workspace_slack_bytes": 196608,
        "workspace_device_kind": "cuda",
        "workspace_device_index": 0,
    }

    profile = event["practical_profile"]
    assert profile["execution_resources"] == resources
    assert profile["key_metrics"]["workspace_required_bytes"] == 65536
    assert profile["key_metrics"]["workspace_provided_bytes"] == 262144
    assert profile["key_metrics"]["workspace_slack_bytes"] == 196608
    assert profile["key_metrics"]["largest_intermediate"] == 4096
    assert profile["diagnosis"]["precision_evidence"]["execution_resources"] == resources

    compute_profile = event["compute_profile"]
    assert compute_profile["execution_resources"] == resources
    assert compute_profile["core_compute_profile"]["execution_resources"] == resources
    assert compute_profile["core_compute_profile"]["temporary_memory"] == {
        "workspace_required_bytes": 65536,
        "workspace_provided_bytes": 262144,
        "workspace_slack_bytes": 196608,
        "largest_intermediate": 4096,
    }


def test_contraction_execute_jsonl_standardizes_workspace_and_async_profiles(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="cupy",
            device_kind="cuda",
            device_index=0,
            lowering="batched_gemm",
            input_shapes=[(8, 16, 32), (8, 32, 16)],
            output_shape=(8, 16, 16),
            output_dtype="complex128",
            workspace_bytes=65536,
            largest_intermediate=4096,
            stream_provided=True,
            stream_type="ExternalCudaStream",
            workspace_provided=True,
            workspace_nbytes=262144,
            workspace_device_kind="cuda",
            workspace_device_index=0,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["workspace_profile"] == {
        "workspace_required": True,
        "workspace_required_bytes": 65536,
        "workspace_provided": True,
        "workspace_provided_bytes": 262144,
        "workspace_sufficient": True,
        "workspace_slack_bytes": 196608,
        "workspace_device_kind": "cuda",
        "workspace_device_index": 0,
        "largest_intermediate": 4096,
    }
    assert event["async_profile"] == {
        "stream_provided": True,
        "stream_type": "ExternalCudaStream",
        "async_requested": True,
        "execution_mode": "external_stream",
    }


def test_contraction_execute_jsonl_standardizes_dtype_and_device_profiles(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="cupy",
            device_kind="cuda",
            device_index=1,
            lowering="batched_gemm",
            input_shapes=[(8, 16, 32), (8, 32, 16)],
            input_dtypes=["float64", "complex128"],
            output_shape=(8, 16, 16),
            output_dtype="complex128",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["dtype_profile"] == {
        "dtype": "complex128",
        "output_dtype": "complex128",
        "input_dtypes": ["float64", "complex128"],
        "input_dtype_set": ["complex128", "float64"],
        "numeric_kind": "complex",
        "precision_bits": 128,
        "component_bits": 64,
        "mixed_input_dtypes": True,
    }
    assert event["device_profile"] == {
        "backend": "cupy",
        "device": None,
        "device_kind": "cuda",
        "device_index": 1,
        "array_location": "device",
        "distributed": False,
        "rank": None,
        "world_size": None,
        "local_shape": None,
        "global_shape": None,
    }


def test_contraction_execute_jsonl_standardizes_layout_profile(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="gemm",
            input_shapes=[(2, 3, 4), (4, 5)],
            input_strides=[(96, 32, 8), (40, 8)],
            input_orders=["C", "C"],
            input_contiguous=[True, True],
            output_shape=(2, 3, 5),
            output_strides=(120, 40, 8),
            output_order="C",
            output_contiguous=True,
            layout_hint="reshape_only",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["layout_profile"] == {
        "input_layouts": [
            {"shape": [2, 3, 4], "strides": [96, 32, 8], "order": "C", "contiguous": True},
            {"shape": [4, 5], "strides": [40, 8], "order": "C", "contiguous": True},
        ],
        "output_layout": {"shape": [2, 3, 5], "strides": [120, 40, 8], "order": "C", "contiguous": True},
        "layout_hint": "reshape_only",
        "layout_transform_required": False,
        "layout_transform_kind": "none",
        "estimated_layout_copy_bytes": 0,
    }


def test_contraction_execute_layout_profile_preserves_input_output_copy_split(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="gemm",
            input_shapes=[(2, 3), (3, 4)],
            output_shape=(2, 4),
            copy_bytes=112,
            layout_transform_copy_bytes=112,
            input_layout_transform_copy_bytes=48,
            output_layout_transform_copy_bytes=64,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["layout_profile"]["layout_transform_required"] is True
    assert event["layout_profile"]["layout_transform_kind"] == "layout_transform"
    assert event["layout_profile"]["estimated_layout_copy_bytes"] == 112
    assert event["layout_profile"]["input_layout_transform_copy_bytes"] == 48
    assert event["layout_profile"]["output_layout_transform_copy_bytes"] == 64
    assert event["copy_profile"]["copy_breakdown"] == {
        "layout_transform_bytes": 112,
        "input_layout_transform_bytes": 48,
        "output_layout_transform_bytes": 64,
    }


def test_contraction_execute_layout_profile_uses_operand_layouts_when_available(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="batched_rhs_hop",
            input_shapes=[(3, 2)],
            operands=[
                {
                    "name": "packed_rhs",
                    "shape": (3, 2),
                    "strides": (16, 8),
                    "order": "C",
                    "contiguous": True,
                }
            ],
            output_shape=(3, 2),
            layout_hint="reshape_only",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["layout_profile"]["input_layouts"] == [
        {"shape": [3, 2], "strides": [16, 8], "order": "C", "contiguous": True}
    ]


def test_contraction_execute_jsonl_reclassifies_backend_execute_from_lowering(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="batched_rhs_hop",
            input_shapes=[(3, 2)],
            output_shape=(3, 2),
            output_dtype="float64",
            num_rhs=2,
            num_rhs_loop_calls=0,
            num_batched_gemm=1,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["compute_class"] == "contraction_plan"
    assert event["practical_profile"]["workload_class"] == "contraction_plan"
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["rhs_profile"]["batching_status"] == "batched"
    assert event["rhs_profile"]["recommended_action"] is None
    assert event["lowering_profile"]["lowering_family"] == "batched_gemm"
    assert event["practical_profile"]["parallelism_hint"] == "batched_backend_kernel"
    assert event["practical_profile"]["parallelization"]["batched_kernel"] is True
    assert event["practical_profile"]["parallelization"]["rhs_batching_candidate"] is True
    assert event["compute_profile"]["workload_signature"]["parallelism_unit"] == "batched_backend_kernel"


def test_contraction_execute_jsonl_standardizes_lowering_profile_for_rhs_fallback(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="fallback_rhs_loop",
            input_shapes=[(10, 4)],
            output_shape=(10, 4),
            output_dtype="float64",
            num_rhs=4,
            num_rhs_loop_calls=4,
            fallback_from="batched_rhs_hop",
            fallback_to="rhs_loop",
            fallback_policy="record",
            fallback_reason="example algorithm rebuilds environments per RHS",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["lowering_profile"] == {
        "lowering": "fallback_rhs_loop",
        "lowering_family": "fallback",
        "fallback_used": True,
        "fallback_from": "batched_rhs_hop",
        "fallback_to": "rhs_loop",
        "fallback_policy": "record",
        "fallback_reason": "example algorithm rebuilds environments per RHS",
        "silent_fallback": False,
        "lowering_route": ["batched_rhs_hop", "rhs_loop"],
        "kernel_counts": {
            "num_gemm": 0,
            "num_batched_gemm": 0,
            "num_grouped_tasks": 0,
            "num_blocks": 0,
            "num_shape_buckets": 0,
            "num_rhs_loop_calls": 4,
        },
        "recommended_action": "batch_rhs_hop",
    }


def test_contraction_execute_practical_profile_reports_fallback_details(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="fallback_tensordot",
            input_shapes=[(5, 2, 3), (5, 3, 4)],
            output_shape=(5, 2, 4),
            output_dtype="float64",
            fallback_from="batched_gemm",
            fallback_to="loop_matmul",
            fallback_policy="record",
            fallback_reason="backend lacks batched_matmul for batch shape (5,)",
            wall_s=0.05,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    profile = event["practical_profile"]
    assert profile["primary_issue"] == "backend_fallback"
    assert profile["parallelism_hint"] == "missing_backend_kernel"
    assert profile["key_metrics"]["fallback_from"] == "batched_gemm"
    assert profile["key_metrics"]["fallback_to"] == "loop_matmul"
    assert profile["key_metrics"]["fallback_policy"] == "record"
    assert profile["key_metrics"]["fallback_reason"] == "backend lacks batched_matmul for batch shape (5,)"
    assert profile["diagnosis"]["precision_evidence"]["fallback_to"] == "loop_matmul"
    assert profile["optimization_targets"] == ["remove_backend_fallback"]
    assert event["core_operation"]["fallback"] == {
        "used": True,
        "from": "batched_gemm",
        "to": "loop_matmul",
        "policy": "record",
        "reason": "backend lacks batched_matmul for batch shape (5,)",
        "silent": False,
        "route": ["batched_gemm", "loop_matmul"],
    }
    assert event["compute_profile"]["execution_summary"] == {
        "compute_class": "tensordot",
        "execution_model": "backend_fallback_loop",
        "primary_kernel": "fallback_tensordot",
        "dominant_cost_kind": "backend_fallback",
        "recommended_action": "remove_backend_fallback",
        "measurement_focus": "backend_fallback",
        "parallelism": "missing_backend_kernel",
        "precision_level": "lowering_counters",
        "next_measurements": [
            "fallback_count_by_lowering",
            "backend_kernel_coverage",
            "wall_time_by_lowering",
        ],
    }
    assert event["compute_profile"]["optimization_profile"]["execution_model"] == "backend_fallback_loop"
    assert event["compute_profile"]["optimization_profile"]["parallelism_model"]["fallback_from"] == "batched_gemm"
    assert event["compute_profile"]["optimization_profile"]["parallelism_model"]["fallback_to"] == "loop_matmul"


def test_contraction_execute_measurement_plan_is_lowering_aware(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="cupy",
            device_kind="cuda",
            lowering="grouped_gemm",
            input_shapes=[(4, 8, 16), (4, 16, 8)],
            output_shape=(4, 8, 8),
            output_dtype="complex128",
            flops=32768,
            read_bytes=32768,
            write_bytes=8192,
            num_gemm=0,
            num_grouped_tasks=12,
            num_blocks=12,
            num_shape_buckets=3,
            workspace_bytes=4096,
            copy_bytes=1024,
            wall_s=0.03,
        )
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="fallback_tensordot",
            input_shapes=[(5, 2, 3), (5, 3, 4)],
            output_shape=(5, 2, 4),
            output_dtype="float64",
            fallback_from="batched_gemm",
            fallback_to="loop_matmul",
            fallback_policy="record",
            fallback_reason="backend lacks batched_matmul for batch shape (5,)",
            wall_s=0.05,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    events = [
        payload for payload in _jsonl_payloads(event_path)
        if payload["event"] == "contraction_execute"
    ]
    grouped = next(payload for payload in events if payload["lowering"] == "grouped_gemm")
    fallback = next(payload for payload in events if payload["lowering"] == "fallback_tensordot")

    assert grouped["compute_profile"]["practical_measurement_plan"] == {
        "schema": "renormalizer.practical_measurement_plan.v1",
        "compute_class": "tensordot",
        "measurement_scope": "backend_contraction_execute",
        "precision_level": "lowering_counters",
        "cost_source": "backend_lowering_estimate",
        "current_cost_proxy": "grouped_gemm_tasks",
        "comparison_key": "lowering,num_blocks,num_shape_buckets",
        "parallel_unit": "grouped_backend_kernel",
        "batching_key": "shape_buckets",
        "wall_time_status": {
            "event_wall_time": "measured",
            "per_kernel_wall_time": "estimated_from_primary_flops_estimate",
        },
        "critical_event_fields": [
            "lowering",
            "input_shapes",
            "output_shape",
            "num_gemm",
            "num_batched_gemm",
            "num_grouped_tasks",
            "num_blocks",
            "num_shape_buckets",
            "flops",
            "copy_bytes",
            "workspace_bytes",
            "wall_s",
        ],
        "next_measurements": [
            "grouped_gemm_wall_time_by_shape_bucket",
            "bucket_occupancy",
            "packing_copy_wall_time",
        ],
        "precision_gaps": [
            "per_kernel_wall_time",
            "grouped_gemm_wall_time_by_shape_bucket",
            "packing_copy_wall_time",
        ],
    }
    assert grouped["compute_profile"]["practical_kernel_view"]["dominant_kernel"] == "gemm_like"

    assert fallback["compute_profile"]["practical_measurement_plan"] == {
        "schema": "renormalizer.practical_measurement_plan.v1",
        "compute_class": "tensordot",
        "measurement_scope": "backend_contraction_execute",
        "precision_level": "lowering_counters",
        "cost_source": "backend_lowering_estimate",
        "current_cost_proxy": "fallback_route",
        "comparison_key": "fallback_from,fallback_to,lowering",
        "parallel_unit": "missing_backend_kernel",
        "batching_key": "fallback_from",
        "wall_time_status": {
            "event_wall_time": "measured",
            "per_kernel_wall_time": "not_attributed",
        },
        "critical_event_fields": [
            "lowering",
            "fallback_from",
            "fallback_to",
            "fallback_policy",
            "fallback_reason",
            "input_shapes",
            "output_shape",
            "wall_s",
        ],
        "next_measurements": [
            "fallback_count_by_route",
            "fallback_wall_time_by_route",
            "missing_backend_kernel_coverage",
        ],
        "precision_gaps": [
            "per_kernel_wall_time",
            "fallback_kernel_wall_time",
        ],
    }


def test_contraction_execute_practical_profile_prioritizes_rhs_loop_over_generic_fallback(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="fallback_rhs_loop",
            input_shapes=[(3, 4)],
            output_shape=(3, 4),
            output_dtype="float64",
            read_bytes=96,
            write_bytes=96,
            num_rhs=4,
            num_rhs_loop_calls=4,
            fallback_reason="unit test rhs fallback",
            wall_s=0.05,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    profile = event["practical_profile"]
    compute_profile = event["compute_profile"]
    optimization = compute_profile["optimization_profile"]

    assert profile["primary_issue"] == "python_rhs_loop"
    assert profile["parallelism_hint"] == "rhs_batching_needed"
    assert profile["diagnosis"]["measurement_focus"] == "rhs_loop"
    assert profile["diagnosis"]["fallback_candidate"] is True
    assert profile["diagnosis"]["recommended_action"] == "batch_rhs_hop"
    assert profile["optimization_targets"][:2] == ["batch_rhs_hop", "remove_backend_fallback"]
    assert compute_profile["execution_summary"] == {
        "compute_class": "tensordot",
        "execution_model": "python_rhs_loop",
        "primary_kernel": "fallback_rhs_loop",
        "dominant_cost_kind": "python_rhs_loop",
        "recommended_action": "batch_rhs_hop",
        "measurement_focus": "rhs_loop",
        "parallelism": "rhs_batching_needed",
        "precision_level": "lowering_counters",
        "next_measurements": [
            "rhs_loop_wall_time",
            "num_rhs_loop_calls",
            "batched_rhs_kernel_coverage",
        ],
    }
    assert optimization["bottleneck"] == "python_rhs_loop"
    assert optimization["recommended_action"] == "batch_rhs_hop"
    assert optimization["execution_model"] == "python_rhs_loop"
    assert optimization["actionability"]["priority_score"] == 100
    assert optimization["parallelism_model"]["num_rhs_loop_calls"] == 4


def test_contraction_execute_infers_fallback_target_for_legacy_events(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="fallback_tensordot",
            input_shapes=[(5, 2, 3), (5, 3, 4)],
            output_shape=(5, 2, 4),
            output_dtype="float64",
            fallback_from="batched_gemm",
            fallback_policy="record",
            fallback_reason="backend lacks batched_matmul for batch shape (5,)",
            wall_s=0.05,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")

    assert event["fallback_to"] == "loop_matmul"
    assert event["practical_profile"]["key_metrics"]["fallback_to"] == "loop_matmul"
    assert event["practical_profile"]["diagnosis"]["precision_evidence"]["fallback_to"] == "loop_matmul"


def test_distributed_contraction_execute_jsonl_has_standard_spec_fields(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="distributed",
            input_shapes=[(2, 3), (3, 4)],
            output_shape=(2, 4),
            output_dtype="float64",
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert event["practical_profile"]["workload_class"] == "contraction_plan"
    assert event["distributed_modes"] == []
    assert event["rank"] is None
    assert event["world_size"] is None
    assert event["local_shape"] is None
    assert event["global_shape"] == [2, 4]
    assert event["communication"] == []


def test_distributed_contraction_execute_practical_profile_reports_communication(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="distributed",
            input_shapes=[(4, 8), (8, 6)],
            output_shape=(4, 6),
            output_dtype="float64",
            flops=384,
            read_bytes=320,
            write_bytes=192,
            communication=[
                {"kind": "alltoall", "bytes": 256, "num_messages": 4, "wall_s": 0.02},
                {"kind": "allreduce", "bytes": 96, "num_messages": 1, "block_size": 96, "wall_s": 0.01},
            ],
            distributed_modes=("a",),
            rank=0,
            world_size=4,
            local_shape=(2, 6),
            wall_s=0.08,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    profile = event["practical_profile"]

    assert profile["primary_issue"] == "communication"
    assert profile["parallelism_hint"] == "distributed_contraction"
    assert profile["key_metrics"]["communication_bytes"] == 352
    assert profile["key_metrics"]["communication_num_messages"] == 5
    assert profile["key_metrics"]["max_communication_block_size"] == 96
    assert profile["key_metrics"]["communication_bytes_by_collective"] == {
        "allreduce": 96,
        "alltoall": 256,
    }
    assert profile["key_metrics"]["communication_messages_by_collective"] == {
        "allreduce": 1,
        "alltoall": 4,
    }
    assert profile["key_metrics"]["communication_wall_s_by_collective"] == {
        "allreduce": pytest.approx(0.01),
        "alltoall": pytest.approx(0.02),
    }
    assert profile["key_metrics"]["dominant_communication_collective"] == "alltoall"
    assert profile["key_metrics"]["communication_fraction_of_working_set"] == pytest.approx(352 / (320 + 192 + 352))
    assert profile["optimization_targets"] == ["reduce_communication"]
    assert event["communication_profile"] == {
        "communication_required": True,
        "distributed": True,
        "distributed_modes": ["a"],
        "rank": 0,
        "world_size": 4,
        "local_shape": [2, 6],
        "global_shape": [4, 6],
        "bytes": 352,
        "num_messages": 5,
        "num_collectives": 2,
        "max_block_size": 96,
        "bytes_by_collective": {"allreduce": 96, "alltoall": 256},
        "messages_by_collective": {"allreduce": 1, "alltoall": 4},
        "wall_s_by_collective": {"allreduce": pytest.approx(0.01), "alltoall": pytest.approx(0.02)},
        "wall_s": pytest.approx(0.03),
        "dominant_collective": "alltoall",
        "bytes_by_primitive": {"allreduce": 96, "alltoall": 256},
        "messages_by_primitive": {"allreduce": 1, "alltoall": 4},
        "wall_s_by_primitive": {"allreduce": pytest.approx(0.01), "alltoall": pytest.approx(0.02)},
        "dominant_primitive": "alltoall",
        "point_to_point_bytes": 0,
        "num_point_to_point_messages": 0,
        "working_set_bytes": 864,
        "communication_fraction_of_working_set": pytest.approx(352 / 864),
        "recommended_action": "reduce_communication",
    }
    assert event["core_operation"]["distributed"] == {
        "distributed": True,
        "distributed_modes": ["a"],
        "rank": 0,
        "world_size": 4,
        "local_shape": [2, 6],
        "global_shape": [4, 6],
        "communication": {
            "bytes": 352,
            "num_messages": 5,
            "num_collectives": 2,
            "max_block_size": 96,
            "dominant_collective": "alltoall",
            "bytes_by_collective": {"allreduce": 96, "alltoall": 256},
            "messages_by_collective": {"allreduce": 1, "alltoall": 4},
            "wall_s_by_collective": {"allreduce": pytest.approx(0.01), "alltoall": pytest.approx(0.02)},
            "wall_s": pytest.approx(0.03),
        },
    }


def test_contraction_execute_jsonl_standardizes_communication_entries(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            **profiling.contraction_execute_compute_payload(),
            backend="numpy",
            lowering="distributed",
            input_shapes=[(2, 3), (3, 4)],
            output_shape=(2, 4),
            communication={"kind": "allreduce", "bytes": 32},
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["communication"] == [
        {
            "kind": "allreduce",
            "primitive": "allreduce",
            "collective": "allreduce",
            "is_collective": True,
            "is_point_to_point": False,
            "bytes": 32,
            "num_messages": 1,
            "block_size": 32,
            "wall_s": 0,
        }
    ]


def test_operation_jsonl_events_include_active_stage_scope_in_trace_mode(caplog, monkeypatch, tmp_path):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        with profiling.scope(stage="mps_evolve", method="tdvp_ps"):
            tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    log_payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in log_payloads if payload["event"] == "tensordot"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "tensordot")
    assert event["stage"] == "mps_evolve"
    assert event["method"] == "tdvp_ps"


def test_jsonl_events_include_timeline_and_span_context(caplog, tmp_path):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        with profiling.span("outer", stage="outer") as outer_span:
            with profiling.span("inner", stage="inner") as inner_span:
                tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "tensordot")
    assert isinstance(event["event_id"], int)
    assert event["event_id"] > 0
    assert isinstance(event["timestamp_ns"], int)
    assert event["timestamp_ns"] > 0
    assert event["span_id"] == inner_span
    assert event["parent_span_id"] == outer_span
    assert event["span_name"] == "inner"
    assert event["stage"] == "inner"


def test_trace_mode_keeps_log_concise_with_summary(caplog, monkeypatch, tmp_path):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
        profiling.flush_summaries()
    finally:
        profiling.close_event_output()

    log_payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in log_payloads if payload["event"] == "tensordot"]
    summary = next(
        payload for payload in log_payloads
        if payload["event"] == "profile_summary" and payload["source_event"] == "tensordot"
    )
    assert summary["source_event"] == "tensordot"
    assert summary["call_count"] == 1
    overhead = next(payload for payload in log_payloads if payload["event"] == "profile_overhead")
    jsonl_payloads = _jsonl_payloads(event_path)
    assert overhead["events_written"] == len(jsonl_payloads)
    event = next(payload for payload in jsonl_payloads if payload["event"] == "tensordot")
    assert event["input_shapes"] == [[2, 3], [3, 4]]


def test_trace_summary_excludes_large_execution_metadata(caplog, tmp_path):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        profiling.record(
            "contraction_execute",
            lowering="grouped_gemm",
            task_operands=[
                [
                    {"name": "task0.A", "shape": (2, 3), "modes": ("m", "k")},
                    {"name": "task0.B", "shape": (3, 4), "modes": ("k", "n")},
                ],
            ],
            task_specs=[
                {
                    "index": 0,
                    "m": 2,
                    "n": 4,
                    "k": 3,
                },
            ],
            operands=[
                {"name": "operand0", "shape": (2, 3), "dtype": "float64"},
                {"name": "operand1", "shape": (3, 4), "dtype": "float64"},
            ],
            communication=[
                {
                    "collective": "alltoall",
                    "bytes": 96,
                    "num_messages": 2,
                    "block_size": 48,
                    "wall_s": 0.01,
                },
            ],
            input_states=[
                {"tensor_id": 0, "local_shape": (1, 3), "local_nbytes": 24},
            ],
            output_state={"tensor_id": 2, "local_shape": (1, 4), "local_nbytes": 32},
            output_block_keys=[{"qn_left": (0,), "qn_right": (1,), "extra": ()}],
            unique_output_block_keys=[{"qn_left": (0,), "qn_right": (1,), "extra": ()}],
            result_block_shapes=[{"qn_left": (0,), "qn_right": (1,), "extra": (), "shape": (2, 4)}],
            wall_s=0.25,
        )
        profiling.flush_summaries()
    finally:
        profiling.close_event_output()

    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "contraction_execute")
    assert event["task_operands"][0][0]["name"] == "task0.A"
    assert event["task_specs"][0]["m"] == 2
    assert event["operands"][0]["name"] == "operand0"
    assert event["communication"][0]["collective"] == "alltoall"
    assert event["output_state"]["tensor_id"] == 2

    log_payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in log_payloads
        if payload["event"] == "profile_summary" and payload["source_event"] == "contraction_execute"
    )
    for field in (
        "task_operands",
        "task_specs",
        "operands",
        "communication",
        "input_states",
        "output_state",
        "output_block_keys",
        "unique_output_block_keys",
        "result_block_shapes",
    ):
        assert field not in summary["signature"]


def test_compute_events_emit_class_level_summary_rows(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "tensordot",
        compute_class="tensordot",
        compute_subclass="direct_tensordot",
        backend="numpy",
        input_shapes=[(2, 3), (3, 4)],
        output_shape=(2, 4),
        flops_estimate=48,
        read_bytes=144,
        write_bytes=64,
        wall_s=0.1,
    )
    profiling.record(
        "tensordot",
        compute_class="tensordot",
        compute_subclass="direct_tensordot",
        backend="numpy",
        input_shapes=[(4, 5), (5, 6)],
        output_shape=(4, 6),
        flops_estimate=240,
        read_bytes=320,
        write_bytes=192,
        wall_s=0.2,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        backend="numpy",
        flops_estimate=512,
        read_bytes=256,
        write_bytes=384,
        wall_s=0.4,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    compute_summaries = [
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
    ]
    td_summary = next(
        payload for payload in compute_summaries
        if payload["compute_class"] == "tensordot"
    )
    svd_summary = next(
        payload for payload in compute_summaries
        if payload["compute_class"] == "svd"
    )
    assert td_summary["compute_subclass"] == "direct_tensordot"
    assert td_summary["backend"] == "numpy"
    assert td_summary["call_count"] == 2
    assert td_summary["source_events"] == ["tensordot"]
    assert td_summary["total_wall_s"] == pytest.approx(0.3)
    assert td_summary["total_flops_estimate"] == 288
    assert td_summary["total_read_bytes"] == 464
    assert td_summary["total_write_bytes"] == 256
    assert "input_shapes" not in td_summary
    assert svd_summary["call_count"] == 1
    assert svd_summary["total_flops_estimate"] == 512


def test_compute_summary_separates_roles_and_reports_derived_rates(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "tensordot",
        compute_class="tensordot",
        compute_subclass="api_tensordot",
        compute_role="composite",
        backend="numpy",
        flops_estimate=100,
        read_bytes=30,
        write_bytes=20,
        wall_s=0.1,
    )
    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        backend="numpy",
        lowering="gemm",
        flops=200,
        read_bytes=40,
        write_bytes=10,
        copy_bytes=5,
        comm_bytes=0,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summaries = [
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    ]
    assert {(payload["compute_role"], payload["compute_subclass"]) for payload in summaries} == {
        ("composite", "api_tensordot"),
        ("kernel", "backend_execute"),
    }

    composite = next(payload for payload in summaries if payload["compute_role"] == "composite")
    assert composite["total_wall_s"] == pytest.approx(0.1)
    assert composite["total_flops_estimate"] == 100
    assert composite["total_read_bytes"] == 30
    assert composite["total_write_bytes"] == 20
    assert composite["total_memory_bytes"] == 50
    assert composite["flops_per_s_estimate"] == pytest.approx(1000.0)
    assert composite["read_bandwidth_Bps"] == pytest.approx(300.0)
    assert composite["write_bandwidth_Bps"] == pytest.approx(200.0)
    assert composite["memory_bandwidth_Bps"] == pytest.approx(500.0)
    assert composite["arithmetic_intensity_flops_per_byte"] == pytest.approx(2.0)

    kernel = next(payload for payload in summaries if payload["compute_role"] == "kernel")
    assert kernel["lowering"] == "gemm"
    assert kernel["total_flops_estimate"] == 200
    assert kernel["total_copy_bytes"] == 5
    assert kernel["copy_bandwidth_Bps"] == pytest.approx(25.0)


def test_compute_summaries_derive_comm_bytes_from_communication_entries(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="distributed",
        flops=200,
        read_bytes=40,
        write_bytes=10,
        communication=[
            {"collective": "alltoall", "bytes": 96, "num_messages": 2, "block_size": 48, "wall_s": 0.05},
            {"collective": "allreduce", "bytes": 32, "num_messages": 1, "block_size": 32, "wall_s": 0.01},
        ],
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    compute_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    class_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "tensordot"
    )
    assert compute_summary["total_comm_bytes"] == 128
    assert compute_summary["comm_bandwidth_Bps"] == pytest.approx(640.0)
    assert compute_summary["comm_bytes_by_collective"] == {"allreduce": 32, "alltoall": 96}
    assert compute_summary["comm_wall_s_by_collective"] == {"allreduce": pytest.approx(0.01), "alltoall": pytest.approx(0.05)}
    assert compute_summary["comm_messages_by_collective"] == {"allreduce": 1, "alltoall": 2}
    assert compute_summary["max_comm_block_size_by_collective"] == {"allreduce": 32, "alltoall": 48}
    assert class_summary["total_comm_bytes"] == 128
    assert class_summary["comm_bandwidth_Bps"] == pytest.approx(640.0)
    assert class_summary["comm_bytes_by_collective"] == {"allreduce": 32, "alltoall": 96}
    assert class_summary["comm_wall_s_by_collective"] == {"allreduce": pytest.approx(0.01), "alltoall": pytest.approx(0.05)}
    assert class_summary["comm_messages_by_collective"] == {"allreduce": 1, "alltoall": 2}
    assert class_summary["max_comm_block_size_by_collective"] == {"allreduce": 32, "alltoall": 48}
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    tensordot_row = next(row for row in overview["classes"] if row["compute_class"] == "tensordot")
    assert tensordot_row["comm_bytes_by_collective"] == {"allreduce": 32, "alltoall": 96}
    assert tensordot_row["comm_messages_by_collective"] == {"allreduce": 1, "alltoall": 2}
    assert tensordot_row["max_comm_block_size_by_collective"] == {"allreduce": 32, "alltoall": 48}


def test_compute_class_summary_counts_only_primary_compute(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "tensordot",
        compute_class="tensordot",
        compute_subclass="api_tensordot",
        compute_role="composite",
        compute_accounting="inclusive",
        backend="numpy",
        flops_estimate=100,
        read_bytes=30,
        write_bytes=20,
        wall_s=0.1,
    )
    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        flops=200,
        read_bytes=40,
        write_bytes=10,
        wall_s=0.2,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_type_counts={"GEMM": 2, "TDOT": 1},
        flops_estimate=300,
        gemm_flops_estimate=300,
        read_bytes=50,
        write_bytes=20,
        wall_s=0.3,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    class_summaries = [
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
    ]
    tensordot = next(payload for payload in class_summaries if payload["compute_class"] == "tensordot")
    oe = next(payload for payload in class_summaries if payload["compute_class"] == "oe")
    assert tensordot["call_count"] == 1
    assert tensordot["source_events"] == ["contraction_execute"]
    assert tensordot["source_subclasses"] == ["backend_execute"]
    assert tensordot["total_wall_s"] == pytest.approx(0.2)
    assert tensordot["total_flops_estimate"] == 200
    assert oe["call_count"] == 1
    assert oe["source_events"] == ["oe_contract"]
    assert oe["total_wall_s"] == pytest.approx(0.3)


def test_flush_summaries_emits_primary_compute_class_overview(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        flops=200,
        read_bytes=40,
        write_bytes=10,
        wall_s=0.2,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_type_counts={"GEMM": 2, "TDOT": 1},
        flops_estimate=300,
        gemm_flops_estimate=300,
        read_bytes=50,
        write_bytes=20,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        flops_estimate=500,
        read_bytes=60,
        write_bytes=30,
        max_block_m=12,
        max_block_n=10,
        max_block_elements=120,
        total_block_elements=180,
        output_rank=8,
        singular_value_count=14,
        wall_s=0.5,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    assert overview["accounting"] == "primary"
    assert overview["class_count"] == 3
    assert overview["total_wall_s"] == pytest.approx(1.0)
    assert [row["compute_class"] for row in overview["classes"]] == ["svd", "oe", "tensordot"]
    assert [row["wall_fraction"] for row in overview["classes"]] == pytest.approx([0.5, 0.3, 0.2])
    assert [row["core_operation"] for row in overview["core_operation_ranking"]] == [
        "svd",
        "oe_contract",
        "tensordot",
    ]
    assert [row["total_wall_s"] for row in overview["core_operation_ranking"]] == pytest.approx([0.5, 0.3, 0.2])
    assert overview["core_operation_ranking"][0]["dominant_kernel"] == "svd_qn_block"
    assert overview["core_operation_ranking"][0]["primary_kernel_count"] == 1
    assert overview["core_operation_ranking"][0]["diagnostic_signal_count"] >= 0
    assert [
        (item["core_operation"], item["recommended_focus"], item["parallelization_hint"])
        for item in overview["core_operation_action_plan"]
    ] == [
        ("svd", "inspect_qn_decomposition", "single_decomposition"),
        ("oe_contract", "optimize_gemm_path", "sequential_path_steps"),
        ("tensordot", "optimize_or_batch_gemm_like_tensordot", "single_backend_kernel"),
    ]
    assert overview["core_operation_action_plan"][0]["rank"] == 1
    assert overview["core_operation_action_plan"][0]["primary_bottleneck"] == "svd_qn_block"
    assert [
        (item["core_operation"], item["top_factor"]["name"], item["top_factor"]["unit"])
        for item in overview["core_operation_factor_ranking"]
    ] == [
        ("svd", "qn_block_decomposition_flops", "flops"),
        ("oe_contract", "gemm_path_flops", "flops"),
        ("tensordot", "compute_kernel_flops", "flops"),
    ]
    assert overview["core_operation_factor_ranking"][0]["rank"] == 1
    svd_row, oe_row, tensordot_row = overview["classes"]
    assert svd_row["top_cost_factor"]["name"] == "qn_block_decomposition_flops"
    assert oe_row["top_cost_factor"]["name"] == "gemm_path_flops"
    assert tensordot_row["top_cost_factor"]["name"] == "compute_kernel_flops"
    assert svd_row["core_operation_action"] == {
        "compute_class": "svd",
        "core_operation": "svd",
        "total_wall_s": pytest.approx(0.5),
        "wall_fraction": pytest.approx(0.5),
        "primary_bottleneck": "svd_qn_block",
        "recommended_focus": "inspect_qn_decomposition",
        "diagnostic_focuses": [],
        "parallelization_hint": "single_decomposition",
        "diagnostic_signal_count": 0,
        "evidence_kernel": "svd_qn_block",
        "rank_basis": "wall_s_estimate",
    }
    assert svd_row["total_flops_estimate"] == 500
    assert svd_row["max_svd_block_m"] == 12
    assert svd_row["max_svd_block_n"] == 10
    assert svd_row["max_svd_block_elements"] == 120
    assert svd_row["total_svd_block_elements"] == 180
    assert svd_row["max_output_rank"] == 8
    assert svd_row["total_singular_values"] == 14
    assert svd_row["flops_per_s_estimate"] == pytest.approx(1000.0)
    assert svd_row["memory_bandwidth_Bps"] == pytest.approx(180.0)
    assert svd_row["arithmetic_intensity_flops_per_byte"] == pytest.approx(500 / 90)
    assert oe_row["total_oe_gemm_steps"] == 2
    assert oe_row["flops_per_s_estimate"] == pytest.approx(1000.0)
    assert oe_row["memory_bandwidth_Bps"] == pytest.approx(70 / 0.3)
    assert oe_row["arithmetic_intensity_flops_per_byte"] == pytest.approx(300 / 70)
    assert tensordot_row["total_gemm"] == 1
    assert tensordot_row["flops_per_s_estimate"] == pytest.approx(1000.0)
    assert tensordot_row["memory_bandwidth_Bps"] == pytest.approx(250.0)
    assert tensordot_row["arithmetic_intensity_flops_per_byte"] == pytest.approx(4.0)


def test_compute_class_overview_reports_practical_kernel_shape_and_bottleneck_breakdown(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        kernel_kind="gemm",
        m=2,
        n=4,
        k=3,
        flops=48,
        read_bytes=64,
        write_bytes=32,
        wall_s=0.2,
    )
    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="fallback_rhs_loop",
        kernel_kind="fallback_rhs_loop",
        read_bytes=96,
        write_bytes=96,
        num_rhs=3,
        num_rhs_loop_calls=3,
        fallback_reason="unit test rhs fallback",
        wall_s=0.4,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=3,
        contraction_type_counts={"GEMM": 2, "TDOT": 1},
        largest_intermediate_elements=128,
        flops_estimate=512,
        read_bytes=128,
        write_bytes=64,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=2,
        max_block_m=12,
        max_block_n=10,
        max_block_elements=120,
        total_block_elements=180,
        matrix_elements=200,
        output_rank=8,
        singular_value_count=14,
        flops_estimate=500,
        read_bytes=60,
        write_bytes=30,
        wall_s=0.1,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    tensordot_row = next(row for row in overview["classes"] if row["compute_class"] == "tensordot")
    oe_row = next(row for row in overview["classes"] if row["compute_class"] == "oe")
    svd_row = next(row for row in overview["classes"] if row["compute_class"] == "svd")

    assert tensordot_row["kernel_kind_counts"] == {"fallback_rhs_loop": 1, "gemm": 1}
    assert tensordot_row["kernel_kind_wall_s"]["gemm"] == pytest.approx(0.2)
    assert tensordot_row["kernel_kind_wall_s"]["fallback_rhs_loop"] == pytest.approx(0.4)
    assert tensordot_row["dominant_kernel_kind"] == "fallback_rhs_loop"
    assert tensordot_row["problem_size_bins"] == {"tiny": 2}
    assert {item["key"] for item in tensordot_row["top_problem_shapes"]} == {
        "gemm:m=2,n=4,k=3",
        "fallback_rhs_loop:rhs=3",
    }
    assert {"fallback", "python_rhs_loop"} <= set(tensordot_row["bottleneck_hints"])

    assert oe_row["dominant_kernel_kind"] == "oe_mixed_path"
    assert oe_row["problem_size_bins"] == {"tiny": 1}
    assert oe_row["top_problem_shapes"][0]["key"] == "oe_mixed_path:steps=3,types=GEMM:2|TDOT:1,largest=128"
    assert oe_row["total_oe_gemm_steps"] == 2
    assert oe_row["total_oe_non_gemm_steps"] == 1
    assert "oe_path" in oe_row["bottleneck_hints"]
    assert "oe_mixed_path" in oe_row["bottleneck_hints"]
    assert "oe_non_gemm_steps" in oe_row["bottleneck_hints"]

    assert svd_row["dominant_kernel_kind"] == "svd_qn"
    assert svd_row["top_problem_shapes"][0]["key"] == "svd_qn:blocks=2,max=12x10,rank=8"
    assert svd_row["total_decomposition_matrix_elements"] == 200
    assert svd_row["svd_block_density"] == pytest.approx(0.9)
    assert "decomposition" in svd_row["bottleneck_hints"]

    class_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "tensordot"
    )
    assert class_summary["kernel_kind_counts"] == {"fallback_rhs_loop": 1, "gemm": 1}
    assert class_summary["dominant_kernel_kind"] == "fallback_rhs_loop"


def test_compute_summaries_use_explicit_practical_problem_fields(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        problem_kind="oe:mixed",
        problem_signature="oe_mixed_path:steps=3,gemm=1,non_gemm=2,largest=240",
        problem_size_bin="small",
        algorithmic_kernel="oe_mixed_path",
        bottleneck_hints=("oe_path", "oe_non_gemm_steps"),
        flops_estimate=4096,
        read_bytes=256,
        write_bytes=48,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "oe"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "oe")

    assert summary["kernel_kind_counts"] == {"oe_mixed_path": 1}
    assert summary["problem_size_bins"] == {"small": 1}
    assert summary["top_problem_shapes"] == [
        {
            "key": "oe_mixed_path:steps=3,gemm=1,non_gemm=2,largest=240",
            "count": 1,
            "wall_s": pytest.approx(0.2),
        }
    ]
    assert summary["bottleneck_hints"] == ["oe_non_gemm_steps", "oe_path"]
    assert row["problem_size_bins"] == {"small": 1}
    assert row["top_problem_shapes"][0]["key"] == "oe_mixed_path:steps=3,gemm=1,non_gemm=2,largest=240"


def test_compute_class_overview_aggregates_practical_profile_tags(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        profile_tags=("mixed_path", "non_gemm_steps", "shape_diverse"),
        non_gemm_step_count=2,
        max_step_output_elements=60,
        total_step_output_elements=82,
        unique_step_output_shape_count=3,
        flops_estimate=100,
        read_bytes=40,
        write_bytes=10,
        wall_s=0.2,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        profile_tags=("tiny_svd_blocks", "skinny_svd_blocks", "reused_block_shapes"),
        unique_block_shape_count=3,
        block_shape_reuse_fraction=0.6,
        tiny_block_count=5,
        skinny_block_count=2,
        flops_estimate=200,
        read_bytes=80,
        write_bytes=20,
        wall_s=0.3,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    oe_row = next(row for row in overview["classes"] if row["compute_class"] == "oe")
    svd_row = next(row for row in overview["classes"] if row["compute_class"] == "svd")

    assert set(oe_row["profile_tags"]) == {"mixed_path", "non_gemm_steps", "shape_diverse"}
    assert oe_row["profile_tag_counts"] == {"mixed_path": 1, "non_gemm_steps": 1, "shape_diverse": 1}
    assert oe_row["max_oe_step_output_elements"] == 60
    assert oe_row["total_oe_step_output_elements"] == 82
    assert oe_row["max_oe_unique_step_output_shapes"] == 3

    assert set(svd_row["profile_tags"]) == {"reused_block_shapes", "skinny_svd_blocks", "tiny_svd_blocks"}
    assert svd_row["max_svd_unique_block_shapes"] == 3
    assert svd_row["max_svd_block_shape_reuse_fraction"] == pytest.approx(0.6)
    assert svd_row["total_svd_tiny_blocks"] == 5
    assert svd_row["total_svd_skinny_blocks"] == 2


def test_compute_summaries_aggregate_practical_cost_breakdown(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "tensordot",
        compute_class="tensordot",
        compute_subclass="api_tensordot",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        flops_estimate=120,
        read_bytes=80,
        write_bytes=40,
        copy_bytes=32,
        workspace_bytes=16,
        peak_bytes=88,
        axis_permutation_copy_bytes=32,
        profile_tags=("axis_permutation",),
        wall_s=0.1,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        gemm_flops_estimate=300,
        non_gemm_flops_estimate=800,
        dominant_step_flops_estimate=800,
        flops_estimate=1100,
        read_bytes=100,
        write_bytes=20,
        wall_s=0.2,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_flops_estimate=1000,
        dense_flops_estimate=4000,
        batchable_flops_estimate=750,
        batchable_flop_fraction=0.75,
        flops_estimate=1000,
        read_bytes=200,
        write_bytes=40,
        wall_s=0.3,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    tensordot = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    oe = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "oe"
    )
    svd = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "svd"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    tensordot_row = next(row for row in overview["classes"] if row["compute_class"] == "tensordot")
    oe_row = next(row for row in overview["classes"] if row["compute_class"] == "oe")
    svd_row = next(row for row in overview["classes"] if row["compute_class"] == "svd")

    assert tensordot["total_tensordot_axis_permutation_copy_bytes"] == 32
    assert oe["total_oe_gemm_flops_estimate"] == 300
    assert oe["total_oe_non_gemm_flops_estimate"] == 800
    assert oe["oe_gemm_flop_fraction"] == pytest.approx(300 / 1100)
    assert oe["oe_non_gemm_flop_fraction"] == pytest.approx(800 / 1100)
    assert oe["max_oe_dominant_step_flops_estimate"] == 800
    oe_core_mix = {
        item["kind"]: item
        for item in oe["practical_profile"]["core_compute_profile"]["kernel_mix"]
    }
    assert oe["practical_profile"]["core_compute_profile"]["dominant_time_kind"] == "oe_non_gemm_step"
    assert oe_core_mix["oe_gemm_step"]["flops_estimate"] == 300
    assert oe_core_mix["oe_gemm_step"]["wall_fraction"] == pytest.approx(300 / 1100)
    assert oe_core_mix["oe_non_gemm_step"]["flops_estimate"] == 800
    assert oe_core_mix["oe_non_gemm_step"]["wall_fraction"] == pytest.approx(800 / 1100)
    assert svd["total_svd_block_flops_estimate"] == 1000
    assert svd["total_svd_dense_flops_estimate"] == 4000
    assert svd["total_svd_batchable_flops_estimate"] == 750
    assert svd["svd_block_flop_fraction"] == pytest.approx(0.25)
    assert svd["svd_sparse_saved_flop_fraction"] == pytest.approx(0.75)
    assert svd["svd_batchable_flop_fraction"] == pytest.approx(0.75)
    assert svd["max_svd_batchable_flop_fraction"] == pytest.approx(0.75)
    assert tensordot["working_set_bytes_estimate"] == 152
    assert tensordot["effective_arithmetic_intensity_flops_per_byte"] == pytest.approx(120 / 152)
    assert tensordot["copy_fraction_of_working_set"] == pytest.approx(32 / 152)
    assert tensordot["tensordot_axis_permutation_copy_fraction_of_working_set"] == pytest.approx(32 / 152)
    assert tensordot["cost_profile"] == {
        "source": "aggregate_profile_fields",
        "flops": 120,
        "read_bytes": 80,
        "write_bytes": 40,
        "copy_bytes": 32,
        "comm_bytes": 0,
        "workspace_bytes": 16,
        "peak_bytes": 88,
        "memory_bytes": 120,
        "working_set_bytes": 152,
        "largest_intermediate": 0,
        "arithmetic_intensity_flops_per_byte": pytest.approx(1.0),
        "effective_arithmetic_intensity_flops_per_byte": pytest.approx(120 / 152),
        "copy_fraction_of_working_set": pytest.approx(32 / 152),
        "communication_fraction_of_working_set": pytest.approx(0.0),
        "workspace_fraction_of_peak": pytest.approx(16 / 88),
        "compute_s": None,
        "memory_s": None,
        "copy_s": None,
        "comm_s": None,
        "total_s": None,
        "estimated_time_s": None,
    }
    assert tensordot_row["total_tensordot_axis_permutation_copy_bytes"] == 32
    assert tensordot_row["cost_profile"] == tensordot["cost_profile"]
    assert oe_row["total_oe_gemm_flops_estimate"] == 300
    assert oe_row["total_oe_non_gemm_flops_estimate"] == 800
    assert oe_row["oe_non_gemm_flop_fraction"] == pytest.approx(800 / 1100)
    assert svd_row["total_svd_block_flops_estimate"] == 1000
    assert svd_row["total_svd_dense_flops_estimate"] == 4000
    assert svd_row["svd_sparse_saved_flop_fraction"] == pytest.approx(0.75)
    assert svd_row["svd_batchable_flop_fraction"] == pytest.approx(0.75)


def test_oe_cost_factors_rank_exclusive_path_components(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=4,
        contraction_type_counts={"GEMM": 1, "TDOT": 2, "EINSUM": 1},
        gemm_flops_estimate=100,
        tensordot_flops_estimate=300,
        generic_einsum_flops_estimate=600,
        non_gemm_flops_estimate=900,
        flops_estimate=1000,
        read_bytes=320,
        write_bytes=80,
        wall_s=0.3,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "oe"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "oe")

    breakdown = summary["practical_profile"]["cost_factor_breakdown"]
    assert [factor["name"] for factor in breakdown["factors"][:3]] == [
        "generic_einsum_path_flops",
        "tensordot_path_flops",
        "gemm_path_flops",
    ]
    assert all(
        factor["name"] != "non_gemm_path_flops"
        for factor in breakdown["factors"]
    )
    assert breakdown["diagnostic_factors"] == [
        {
            "name": "non_gemm_path_flops",
            "value": 900,
            "unit": "flops",
            "source": "non_gemm_flops_estimate",
            "resource": "serial_path",
            "accounting": "diagnostic_aggregate",
        }
    ]
    assert breakdown["accounting_model"] == {
        "factors": "exclusive_primary_cost_factors",
        "diagnostic_factors": "non_additive_aggregate_or_subset_signals",
    }
    assert row["top_cost_factor"]["name"] == "generic_einsum_path_flops"


def test_oe_aggregate_summary_keeps_type_count_precision_when_step_flops_missing(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=2,
        contraction_type_counts={"GEMM": 1, "TDOT": 1},
        flops_estimate=1000,
        read_bytes=320,
        write_bytes=80,
        wall_s=0.3,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "oe"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "oe")
    profile = summary["practical_profile"]
    breakdown = profile["cost_factor_breakdown"]

    assert profile["precision_level"] == "aggregate_path_type_counts"
    assert profile["cost_model"]["source"] == "aggregate_oe_path_type_count_estimate"
    assert profile["gemm_flop_fraction"] is None
    assert profile["tensordot_flop_fraction"] is None
    assert profile["non_gemm_flop_fraction"] is None
    assert profile["dominant_step_flop_fraction"] is None
    assert profile["diagnosis"]["precision_evidence"]["flop_split_available"] is False
    assert breakdown["factors"][0]["name"] == "non_gemm_path_steps"
    assert breakdown["factors"][0]["accounting"] == "primary"
    assert summary["compute_profile"]["optimization_profile"]["work_unit"] == {
        "kind": "path_type_mix",
        "contraction_count": 2,
        "gemm_step_count": 1,
        "tensordot_step_count": 1,
        "generic_einsum_step_count": 0,
        "non_gemm_step_count": 1,
    }
    practical_view = (
        summary["compute_profile"]["core_compute_profile"]
        ["core_kernel_summary"]["practical_view"]
    )
    assert practical_view["optimization_summary"]["recommended_focus"] == "reduce_non_gemm_path_cost"
    assert row["top_cost_factor"]["name"] == "non_gemm_path_steps"


def test_svd_cost_factors_keep_block_diagnostics_out_of_primary_ranking(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=5,
        batchable_block_count=4,
        batchable_block_group_count=1,
        unique_block_shape_count=3,
        tiny_block_count=2,
        block_flops_estimate=1000,
        dense_flops_estimate=4000,
        batchable_flops_estimate=750,
        flops_estimate=1000,
        read_bytes=200,
        write_bytes=40,
        wall_s=0.3,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "svd"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "svd")

    breakdown = summary["practical_profile"]["cost_factor_breakdown"]
    assert [factor["name"] for factor in breakdown["factors"]] == [
        "qn_block_decomposition_flops",
    ]
    assert [factor["name"] for factor in breakdown["diagnostic_factors"]] == [
        "tiny_qn_blocks",
        "batchable_qn_block_flops",
        "shape_fragmentation",
    ]
    assert row["top_cost_factor"]["name"] == "qn_block_decomposition_flops"


def test_compute_summaries_aggregate_svd_block_group_metrics(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=6,
        batchable_block_group_count=2,
        dominant_block_shape_flop_fraction=0.68,
        block_flops_estimate=1000,
        dense_flops_estimate=4000,
        batchable_flops_estimate=750,
        flops_estimate=1000,
        read_bytes=200,
        write_bytes=40,
        wall_s=0.3,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "svd"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "svd")

    assert summary["max_svd_batchable_block_group_count"] == 2
    assert summary["max_svd_dominant_block_shape_flop_fraction"] == pytest.approx(0.68)
    assert summary["practical_profile"]["batchable_block_group_count"] == 2
    assert summary["practical_profile"]["dominant_block_shape_flop_fraction"] == pytest.approx(0.68)
    assert row["max_svd_batchable_block_group_count"] == 2
    assert row["practical_profile"]["batchable_block_group_count"] == 2


def test_profile_summary_signature_omits_large_step_cost_details(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        step_costs=(
            {"step": 0, "contraction_type": "GEMM", "flops_estimate": 100},
            {"step": 1, "contraction_type": "TDOT", "flops_estimate": 200},
        ),
        top_block_shape_groups=(
            {"shape": "16x16", "count": 4, "total_flops_estimate": 4096},
        ),
        wall_s=0.1,
    )
    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(payload for payload in payloads if payload["event"] == "profile_summary")

    assert "step_costs" not in summary["signature"]
    assert "top_block_shape_groups" not in summary["signature"]


def test_compute_summaries_emit_compact_practical_profile_for_core_compute_classes(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        kernel_kind="gemm",
        m=16,
        n=8,
        k=4,
        flops=1024,
        read_bytes=768,
        write_bytes=1024,
        copy_bytes=512,
        axis_permutation_copy_bytes=512,
        profile_tags=("axis_permutation", "tiny_gemm"),
        wall_s=0.4,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=3,
        contraction_type_counts={"GEMM": 1, "TDOT": 2},
        gemm_flops_estimate=400,
        non_gemm_flops_estimate=600,
        dominant_step_flops_estimate=600,
        flops_estimate=1000,
        read_bytes=300,
        write_bytes=100,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=4,
        unique_block_shape_count=2,
        block_shape_reuse_fraction=0.5,
        tiny_block_count=2,
        block_flops_estimate=1000,
        dense_flops_estimate=4000,
        batchable_flops_estimate=750,
        total_block_elements=200,
        matrix_elements=1000,
        flops_estimate=1000,
        read_bytes=120,
        write_bytes=80,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summaries = {
        payload["compute_class"]: payload
        for payload in payloads
        if payload["event"] == "profile_compute_summary"
    }
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    overview_rows = {row["compute_class"]: row for row in overview["classes"]}

    tensordot_profile = summaries["tensordot"]["practical_profile"]
    oe_profile = summaries["oe"]["practical_profile"]
    svd_profile = summaries["svd"]["practical_profile"]

    assert tensordot_profile["workload_class"] == "tensordot"
    assert tensordot_profile["primary_kernel"] == "gemm"
    assert tensordot_profile["top_kernels"] == [
        {"key": "gemm", "count": 1, "wall_s": pytest.approx(0.4)}
    ]
    assert tensordot_profile["top_problem_shapes"] == [
        {"key": "gemm:m=16,n=8,k=4", "count": 1, "wall_s": pytest.approx(0.4)}
    ]
    assert tensordot_profile["top_bottlenecks"][0] == {
        "key": "axis_permutation_copy",
        "count": 1,
        "wall_s": pytest.approx(0.4),
    }
    assert {item["key"] for item in tensordot_profile["top_signals"]} == {
        "axis_permutation",
        "tiny_gemm",
    }
    assert tensordot_profile["axis_permutation_call_fraction"] == pytest.approx(1.0)
    assert tensordot_profile["axis_permutation_copy_fraction"] == pytest.approx(512 / (768 + 1024 + 512))
    assert tensordot_profile["diagnosis"] == {
        "primary_issue": "axis_permutation_copy",
        "parallelism_hint": "tiny_or_skinny_gemm",
        "precision_evidence": {
            "max_m": 16,
            "max_n": 8,
            "max_k": 4,
            "copy_fraction": pytest.approx(512 / (768 + 1024 + 512)),
        },
        "recommended_action": "avoid_axis_permutation_copies",
    }
    assert "avoid_axis_permutation_copies" in tensordot_profile["optimization_targets"]
    assert "batch_or_fuse_tiny_gemm" in tensordot_profile["optimization_targets"]

    assert oe_profile["workload_class"] == "oe"
    assert oe_profile["primary_kernel"] == "oe_mixed_path"
    assert oe_profile["top_kernels"] == [
        {"key": "oe_mixed_path", "count": 1, "wall_s": pytest.approx(0.3)}
    ]
    assert oe_profile["top_problem_shapes"] == [
        {
            "key": "oe_mixed_path:steps=3,types=GEMM:1|TDOT:2,largest=0",
            "count": 1,
            "wall_s": pytest.approx(0.3),
        }
    ]
    assert oe_profile["top_bottlenecks"][0] == {
        "key": "oe_non_gemm_flops",
        "count": 1,
        "wall_s": pytest.approx(0.3),
    }
    assert summaries["oe"]["dominant_bottleneck_hint"] == "oe_non_gemm_flops"
    assert overview_rows["oe"]["dominant_bottleneck_hint"] == "oe_non_gemm_flops"
    assert oe_profile["gemm_flop_fraction"] == pytest.approx(0.4)
    assert oe_profile["non_gemm_flop_fraction"] == pytest.approx(0.6)
    assert oe_profile["dominant_step_flop_fraction"] == pytest.approx(0.6)
    assert oe_profile["diagnosis"] == {
        "primary_issue": "non_gemm_path",
        "path_regime": "non_gemm_flop_dominant",
        "precision_evidence": {
            "precision_level": "aggregate_path_costs",
            "flop_split_available": True,
            "gemm_flop_fraction": pytest.approx(0.4),
            "non_gemm_flop_fraction": pytest.approx(0.6),
            "dominant_step_flop_fraction": pytest.approx(0.6),
        },
        "recommended_action": "reduce_non_gemm_path_cost",
    }
    assert "reduce_non_gemm_path_cost" in oe_profile["optimization_targets"]

    assert svd_profile["workload_class"] == "svd"
    assert svd_profile["primary_kernel"] == "svd_qn"
    assert svd_profile["top_kernels"] == [
        {"key": "svd_qn", "count": 1, "wall_s": pytest.approx(0.2)}
    ]
    assert svd_profile["top_problem_shapes"] == [
        {"key": "svd_qn:blocks=4,max=0x0,rank=0", "count": 1, "wall_s": pytest.approx(0.2)}
    ]
    assert svd_profile["top_bottlenecks"][0] == {
        "key": "decomposition",
        "count": 1,
        "wall_s": pytest.approx(0.2),
    }
    assert svd_profile["qn_block_density"] == pytest.approx(0.2)
    assert svd_profile["sparse_saved_flop_fraction"] == pytest.approx(0.75)
    assert svd_profile["batchable_flop_fraction"] == pytest.approx(0.75)
    assert svd_profile["diagnosis"] == {
        "primary_issue": "tiny_qn_blocks",
        "block_regime": "sparse_and_batchable",
        "precision_evidence": {
            "qn_block_density": pytest.approx(0.2),
            "sparse_saved_flop_fraction": pytest.approx(0.75),
            "batchable_flop_fraction": pytest.approx(0.75),
        },
        "recommended_action": "batch_reused_qn_blocks",
    }
    assert "batch_reused_qn_blocks" in svd_profile["optimization_targets"]
    assert "reduce_tiny_block_overhead" in svd_profile["optimization_targets"]

    assert overview_rows["tensordot"]["practical_profile"]["axis_permutation_call_fraction"] == pytest.approx(1.0)
    assert overview_rows["oe"]["practical_profile"]["non_gemm_flop_fraction"] == pytest.approx(0.6)
    assert overview_rows["svd"]["practical_profile"]["sparse_saved_flop_fraction"] == pytest.approx(0.75)


def test_practical_profiles_expose_actionable_execution_breakdown_for_core_compute_classes(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="fallback_rhs_loop",
        kernel_kind="fallback_rhs_loop",
        num_rhs=4,
        num_rhs_loop_calls=4,
        flops=2048,
        read_bytes=512,
        write_bytes=256,
        copy_bytes=128,
        axis_permutation_copy_bytes=128,
        profile_tags=("axis_permutation", "tiny_gemm"),
        wall_s=0.4,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=3,
        contraction_type_counts={"GEMM": 1, "TDOT": 2},
        gemm_flops_estimate=400,
        non_gemm_flops_estimate=1200,
        dominant_step_flops_estimate=1000,
        max_step_output_elements=96,
        total_step_output_elements=160,
        flops_estimate=1600,
        read_bytes=320,
        write_bytes=80,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=5,
        unique_block_shape_count=2,
        batchable_block_group_count=1,
        batchable_block_count=4,
        tiny_block_count=3,
        block_flops_estimate=1000,
        dense_flops_estimate=5000,
        batchable_flops_estimate=800,
        total_block_elements=200,
        matrix_elements=1000,
        flops_estimate=1000,
        read_bytes=200,
        write_bytes=40,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summaries = {
        payload["compute_class"]: payload
        for payload in payloads
        if payload["event"] == "profile_compute_summary"
    }

    tensordot = summaries["tensordot"]["practical_profile"]
    assert tensordot["execution_breakdown"] == {
        "schema": "renormalizer.execution_breakdown.v1",
        "measurement_scope": "tensordot_summary",
        "work_unit": "tensordot_problem_shape",
        "estimated_flops": 2048,
        "working_set_bytes": 896,
        "copy_bytes": 128,
        "serial_dependency_units": 4,
        "independent_work_units": 4,
        "gemm_like_work_units": 0,
        "non_gemm_work_units": 0,
        "batchable_work_units": 4,
        "batchable_group_count": 0,
    }
    assert tensordot["parallelism_opportunity"] == {
        "kind": "rhs_batching",
        "candidate": True,
        "work_units": 4,
        "reason": "python_rhs_loop",
    }
    assert tensordot["cost_factor_breakdown"]["rank_source"] == "bottleneck_priority_then_proxy_magnitude"
    assert tensordot["cost_factor_breakdown"]["factors"][0]["name"] == "rhs_loop"
    assert {
        factor["name"]: (factor["value"], factor["unit"], factor["source"])
        for factor in tensordot["cost_factor_breakdown"]["factors"]
    }["layout_copy"] == (128, "bytes", "copy_bytes")

    oe = summaries["oe"]["practical_profile"]
    assert oe["execution_breakdown"]["serial_dependency_units"] == 3
    assert oe["execution_breakdown"]["gemm_like_work_units"] == 1
    assert oe["execution_breakdown"]["non_gemm_work_units"] == 2
    assert oe["parallelism_opportunity"] == {
        "kind": "path_kernel_specialization",
        "candidate": False,
        "work_units": 3,
        "reason": "sequential_oe_path",
    }
    assert [
        factor["name"]
        for factor in oe["cost_factor_breakdown"]["factors"][:2]
    ] == ["non_gemm_path_flops", "gemm_path_flops"]

    svd = summaries["svd"]["practical_profile"]
    assert svd["execution_breakdown"]["serial_dependency_units"] == 0
    assert svd["execution_breakdown"]["independent_work_units"] == 5
    assert svd["execution_breakdown"]["batchable_work_units"] == 4
    assert svd["execution_breakdown"]["batchable_group_count"] == 1
    assert svd["parallelism_opportunity"] == {
        "kind": "qn_block_grouping",
        "candidate": True,
        "work_units": 4,
        "reason": "reused_qn_block_shapes",
    }
    assert [
        factor["name"]
        for factor in svd["cost_factor_breakdown"]["factors"]
    ] == ["qn_block_decomposition_flops"]
    assert [
        factor["name"]
        for factor in svd["cost_factor_breakdown"]["diagnostic_factors"][:2]
    ] == ["tiny_qn_blocks", "batchable_qn_block_flops"]


def test_practical_profiles_expose_precise_core_compute_kernel_mix(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="gemm",
        num_gemm=1,
        flops=2048,
        read_bytes=512,
        write_bytes=256,
        copy_bytes=128,
        wall_s=0.4,
    )
    profiling.record(
        "oe_contract",
        compute_class="oe",
        compute_subclass="oe_contract",
        compute_role="composite",
        compute_accounting="primary",
        backend="numpy",
        contraction_count=4,
        contraction_type_counts={"GEMM": 1, "TDOT": 2, "EINSUM": 1},
        gemm_flops_estimate=100,
        tensordot_flops_estimate=300,
        generic_einsum_flops_estimate=600,
        non_gemm_flops_estimate=900,
        dominant_step_flops_estimate=600,
        largest_intermediate_elements=50,
        flops_estimate=1000,
        read_bytes=320,
        write_bytes=80,
        wall_s=0.3,
    )
    profiling.record(
        "svd_qn",
        compute_class="svd",
        compute_subclass="svd_qn",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        block_count=5,
        batchable_block_count=4,
        batchable_block_group_count=1,
        unique_block_shape_count=2,
        tiny_block_count=3,
        block_flops_estimate=1000,
        dense_flops_estimate=5000,
        batchable_flops_estimate=800,
        flops_estimate=1000,
        read_bytes=200,
        write_bytes=40,
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summaries = {
        payload["compute_class"]: payload["practical_profile"]
        for payload in payloads
        if payload["event"] == "profile_compute_summary"
    }

    tensordot_core = summaries["tensordot"]["core_compute_profile"]
    tensordot_mix = {item["kind"]: item for item in tensordot_core["kernel_mix"]}
    assert tensordot_core["execution_granularity"] == "single_backend_contraction"
    assert tensordot_core["primary_kernel_mix_is_additive"] is True
    assert tensordot_core["time_attribution"][0]["kind"] == "gemm_like"
    assert tensordot_core["time_attribution"][0]["wall_s_estimate"] == pytest.approx(0.4)
    assert tensordot_core["dominant_time_kind"] == "gemm_like"
    assert tensordot_core["kernel_groups"] == {
        "primary": ["gemm_like"],
        "overhead": ["layout_copy"],
        "subset": [],
    }
    assert tensordot_core["measurement_basis"] == {
        "total_wall_time": "measured",
        "per_kernel_wall_time": "estimated_from_primary_flops_estimate",
        "primary_wall_time_is_additive": True,
        "non_additive_signal_kinds": ["layout_copy"],
        "non_additive_signals_are_wall_attributed": False,
    }
    tensordot_summary = tensordot_core["core_kernel_summary"]
    assert tensordot_summary["compute_family"] == "tensordot"
    assert tensordot_summary["rank_basis"] == "wall_s_estimate"
    assert [item["kind"] for item in tensordot_summary["primary_kernels"]] == ["gemm_like"]
    assert tensordot_summary["primary_kernels"][0]["wall_fraction"] == pytest.approx(1.0)
    assert [item["kind"] for item in tensordot_summary["overhead_signals"]] == ["layout_copy"]
    assert tensordot_summary["overhead_signals"][0]["included_in_wall_ranking"] is False
    assert tensordot_summary["subset_signals"] == []
    assert tensordot_mix["gemm_like"]["count"] == 1
    assert tensordot_mix["gemm_like"]["flops_estimate"] == 2048
    assert tensordot_mix["gemm_like"]["accounting"] == "primary"
    assert tensordot_mix["gemm_like"]["wall_attribution_basis"] == "flops_estimate"
    assert tensordot_mix["layout_copy"]["bytes"] == 128
    assert tensordot_mix["layout_copy"]["accounting"] == "overhead"
    assert tensordot_mix["layout_copy"]["wall_attribution_basis"] == "not_attributed"

    oe = summaries["oe"]
    oe_core = oe["core_compute_profile"]
    oe_mix = {item["kind"]: item for item in oe_core["kernel_mix"]}
    assert oe["tensordot_step_fraction"] == pytest.approx(0.5)
    assert oe["generic_einsum_step_fraction"] == pytest.approx(0.25)
    assert oe_core["execution_granularity"] == "sequential_oe_contraction_path"
    assert oe_core["serial_depth"] == 4
    assert oe_core["primary_kernel_mix_is_additive"] is True
    assert oe_core["dominant_time_kind"] == "oe_generic_einsum_step"
    assert oe_core["kernel_groups"] == {
        "primary": ["oe_gemm_step", "oe_tensordot_step", "oe_generic_einsum_step"],
        "overhead": ["oe_intermediate"],
        "subset": [],
    }
    assert oe_core["measurement_basis"] == {
        "total_wall_time": "measured",
        "per_kernel_wall_time": "estimated_from_primary_flops_estimate",
        "primary_wall_time_is_additive": True,
        "non_additive_signal_kinds": ["oe_intermediate"],
        "non_additive_signals_are_wall_attributed": False,
    }
    oe_summary = oe_core["core_kernel_summary"]
    assert oe_summary["compute_family"] == "oe_contract"
    assert [item["kind"] for item in oe_summary["primary_kernels"]] == [
        "oe_generic_einsum_step",
        "oe_tensordot_step",
        "oe_gemm_step",
    ]
    assert oe_summary["primary_kernels"][0]["wall_s_estimate"] == pytest.approx(0.18)
    assert [item["kind"] for item in oe_summary["overhead_signals"]] == ["oe_intermediate"]
    assert oe_summary["overhead_signals"][0]["included_in_wall_ranking"] is False
    assert oe_summary["subset_signals"] == []
    assert oe_mix["oe_gemm_step"]["count"] == 1
    assert oe_mix["oe_gemm_step"]["accounting"] == "primary"
    assert oe_mix["oe_gemm_step"]["wall_s_estimate"] == pytest.approx(0.03)
    assert oe_mix["oe_tensordot_step"]["count"] == 2
    assert oe_mix["oe_tensordot_step"]["flops_estimate"] == 300
    assert oe_mix["oe_tensordot_step"]["wall_s_estimate"] == pytest.approx(0.09)
    assert oe_mix["oe_generic_einsum_step"]["count"] == 1
    assert oe_mix["oe_generic_einsum_step"]["flops_estimate"] == 600
    assert oe_mix["oe_generic_einsum_step"]["wall_s_estimate"] == pytest.approx(0.18)
    assert oe_mix["oe_intermediate"]["accounting"] == "overhead"

    svd_core = summaries["svd"]["core_compute_profile"]
    svd_mix = {item["kind"]: item for item in svd_core["kernel_mix"]}
    assert svd_core["execution_granularity"] == "independent_qn_block_decompositions"
    assert svd_core["primary_kernel_mix_is_additive"] is True
    assert svd_core["contains_non_additive_kernel_signals"] is True
    assert svd_core["time_attribution"][0]["kind"] == "svd_qn_block"
    assert svd_core["time_attribution"][0]["wall_s_estimate"] == pytest.approx(0.2)
    assert svd_core["kernel_groups"] == {
        "primary": ["svd_qn_block"],
        "overhead": [],
        "subset": ["batchable_qn_block", "tiny_qn_block"],
    }
    assert svd_core["measurement_basis"] == {
        "total_wall_time": "measured",
        "per_kernel_wall_time": "estimated_from_primary_flops_estimate",
        "primary_wall_time_is_additive": True,
        "non_additive_signal_kinds": ["batchable_qn_block", "tiny_qn_block"],
        "non_additive_signals_are_wall_attributed": False,
    }
    svd_summary = svd_core["core_kernel_summary"]
    assert svd_summary["compute_family"] == "svd"
    assert [item["kind"] for item in svd_summary["primary_kernels"]] == ["svd_qn_block"]
    assert svd_summary["primary_kernels"][0]["wall_s_estimate"] == pytest.approx(0.2)
    assert svd_summary["overhead_signals"] == []
    assert [item["kind"] for item in svd_summary["subset_signals"]] == [
        "batchable_qn_block",
        "tiny_qn_block",
    ]
    assert all(item["included_in_wall_ranking"] is False for item in svd_summary["subset_signals"])
    assert svd_mix["svd_qn_block"]["count"] == 5
    assert svd_mix["svd_qn_block"]["accounting"] == "primary"
    assert svd_mix["batchable_qn_block"]["count"] == 4
    assert svd_mix["batchable_qn_block"]["accounting"] == "subset"
    assert svd_mix["batchable_qn_block"]["wall_attribution_basis"] == "not_attributed"


def test_core_kernel_summary_exposes_practical_analysis_view():
    from renormalizer.utils import profiling

    profile = profiling._core_compute_profile(
        "tensordot",
        {
            "measurement_scope": "tensordot_summary",
            "precision_level": "aggregate_mnk_layout",
            "primary_kernel": "gemm",
            "parallelization": {
                "call_count": 2,
                "total_gemm": 2,
            },
            "key_metrics": {},
        },
        {
            "flops": 2048,
            "read_bytes": 1024,
            "write_bytes": 512,
            "copy_bytes": 256,
            "communication_bytes": 0,
            "working_set_bytes": 1792,
        },
        total_wall_s=0.5,
    )

    view = profile["core_kernel_summary"]["practical_view"]
    assert view["schema"] == "renormalizer.practical_kernel_view.v1"
    assert view["core_operation"] == "tensordot"
    assert view["dominant_kernel"] == "gemm_like"
    assert view["wall_time_ranking"] == [
        {
            "kind": "gemm_like",
            "count": 2,
            "wall_s_estimate": pytest.approx(0.5),
            "wall_fraction": pytest.approx(1.0),
            "flops_estimate": 2048,
        }
    ]
    assert view["diagnostic_signals"] == [
        {
            "kind": "layout_copy",
            "signal_type": "overhead",
            "value": 256,
            "unit": "bytes",
            "included_in_wall_ranking": False,
        }
    ]
    assert view["accounting_model"] == {
        "primary_kernel_wall_time": "exclusive_estimate",
        "diagnostic_signal_wall_time": "not_attributed",
        "safe_to_sum_wall_time": ["primary_kernels"],
    }
    assert view["optimization_summary"] == {
        "primary_bottleneck": "gemm_like",
        "recommended_focus": "optimize_or_batch_gemm_like_tensordot",
        "diagnostic_focuses": ["avoid_axis_permutation_copies"],
        "parallelization_hint": "single_backend_kernel",
        "diagnostic_signal_count": 1,
    }


def test_practical_view_keeps_wall_bottleneck_and_diagnostic_focuses_separate():
    from renormalizer.utils import profiling

    oe_profile = profiling._core_compute_profile(
        "oe",
        {
            "measurement_scope": "oe_path",
            "precision_level": "path_step_costs",
            "parallelization": {
                "gemm_step_count": 1,
                "tensordot_step_count": 1,
                "generic_einsum_step_count": 2,
                "non_gemm_step_count": 3,
            },
            "key_metrics": {
                "contraction_count": 4,
                "gemm_flops_estimate": 100,
                "tensordot_flops_estimate": 200,
                "generic_einsum_flops_estimate": 700,
            },
            "max_largest_intermediate_elements": 256,
        },
        {"flops": 1000, "copy_bytes": 0, "communication_bytes": 0, "working_set_bytes": 4096},
        total_wall_s=1.0,
    )
    oe_view = oe_profile["core_kernel_summary"]["practical_view"]
    assert oe_view["optimization_summary"] == {
        "primary_bottleneck": "oe_generic_einsum_step",
        "recommended_focus": "reduce_non_gemm_path_cost",
        "diagnostic_focuses": ["limit_largest_intermediate"],
        "parallelization_hint": "sequential_path_steps",
        "diagnostic_signal_count": 1,
    }

    svd_profile = profiling._core_compute_profile(
        "svd",
        {
            "measurement_scope": "qn_decomposition",
            "precision_level": "qn_block_groups",
            "parallelization": {
                "block_count": 5,
                "batchable_block_count": 4,
                "batchable_block_group_count": 2,
            },
            "key_metrics": {
                "block_count": 5,
                "batchable_block_count": 4,
                "tiny_block_count": 2,
            },
            "batchable_flop_fraction": 0.8,
        },
        {"flops": 500, "copy_bytes": 0, "communication_bytes": 0, "working_set_bytes": 2048},
        total_wall_s=0.5,
    )
    svd_view = svd_profile["core_kernel_summary"]["practical_view"]
    assert svd_view["optimization_summary"] == {
        "primary_bottleneck": "svd_qn_block",
        "recommended_focus": "reduce_tiny_block_overhead",
        "diagnostic_focuses": ["batch_reused_qn_blocks", "reduce_tiny_block_overhead"],
        "parallelization_hint": "batch_qn_block_groups",
        "diagnostic_signal_count": 2,
    }


def test_jsonl_event_core_compute_profiles_use_event_wall_time(caplog, tmp_path):
    import numpy as np

    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        left = np.ones((2, 3, 4))
        right = np.ones((5, 4, 6))
        td_result = np.tensordot(left, right, axes=([2], [1]))
        profiling.record(
            "tensordot",
            backend="numpy",
            **profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result),
            wall_s=0.4,
        )

        oe_result = np.ones((2, 7))
        profiling.record(
            "oe_contract",
            backend="numpy",
            **profiling.oe_compute_payload(
                "oe_contract",
                (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
                oe_result,
                {
                    "contraction_count": 3,
                    "flop_count": 1000,
                    "largest_intermediate": 120,
                    "contraction_steps": [
                        {
                            "step": 0,
                            "contraction_type": "GEMM",
                            "input_shapes": [(2, 3), (3, 4)],
                            "output_shape": (2, 4),
                            "flops_estimate": 100,
                        },
                        {
                            "step": 1,
                            "contraction_type": "TDOT",
                            "input_shapes": [(2, 4), (4, 5, 6)],
                            "output_shape": (2, 5, 6),
                            "flops_estimate": 300,
                        },
                        {
                            "step": 2,
                            "contraction_type": "OUTER/EINSUM",
                            "input_shapes": [(2, 5, 6), (7,)],
                            "output_shape": (2, 7),
                            "flops_estimate": 600,
                        },
                    ],
                },
            ),
            wall_s=0.3,
        )

        coef_array = np.ones((20, 20))
        profiling.record(
            "svd_qn",
            backend="numpy",
            **profiling.svd_qn_compute_payload(
                "SVD",
                coef_array,
                coef_array,
                [
                    {"block_shape": (10, 10), "rank": 10},
                    {"block_shape": (10, 10), "rank": 10},
                    {"block_shape": (2, 2), "rank": 2},
                ],
                (coef_array,),
            ),
            wall_s=0.2,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    events = {payload["event"]: payload for payload in _jsonl_payloads(event_path)}

    tensordot_core = events["tensordot"]["compute_profile"]["core_compute_profile"]
    assert tensordot_core["total_wall_s"] == pytest.approx(0.4)
    assert tensordot_core["dominant_time_kind"] == "gemm_like"
    assert tensordot_core["time_attribution"][0]["wall_s_estimate"] == pytest.approx(0.4)
    assert events["tensordot"]["practical_profile"]["core_compute_profile"]["total_wall_s"] == pytest.approx(0.4)

    oe_core = events["oe_contract"]["compute_profile"]["core_compute_profile"]
    assert oe_core["total_wall_s"] == pytest.approx(0.3)
    assert oe_core["dominant_time_kind"] == "oe_generic_einsum_step"
    assert {
        item["kind"]: item["wall_s_estimate"]
        for item in oe_core["time_attribution"]
    } == {
        "oe_gemm_step": pytest.approx(0.03),
        "oe_tensordot_step": pytest.approx(0.09),
        "oe_generic_einsum_step": pytest.approx(0.18),
    }
    assert events["oe_contract"]["practical_profile"]["core_compute_profile"]["total_wall_s"] == pytest.approx(0.3)

    svd_core = events["svd_qn"]["compute_profile"]["core_compute_profile"]
    assert svd_core["total_wall_s"] == pytest.approx(0.2)
    assert svd_core["dominant_time_kind"] == "svd_qn_block"
    assert svd_core["time_attribution"][0]["wall_s_estimate"] == pytest.approx(0.2)
    assert events["svd_qn"]["practical_profile"]["core_compute_profile"]["total_wall_s"] == pytest.approx(0.2)


def test_core_compute_events_expose_practical_bucket_keys():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    td_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    td_bucket = td_payload["practical_bucket"]

    assert td_bucket["schema"] == "renormalizer.practical_bucket.v1"
    assert td_bucket["aggregation_key"] == (
        "tensordot|kernel=gemm|dtype=float64,float64->float64|m=6|n=30|k=4"
        "|layout=right_axis_permutation"
    )
    assert td_bucket["comparison_key"] == "gemm:m=6,n=30,k=4,layout=right_axis_permutation"
    assert td_bucket["parallel_unit"] == "single_gemm"
    assert td_bucket["comparison_axes"] == [
        "wall_s",
        "flops_estimate",
        "axis_permutation_copy_bytes",
        "effective_arithmetic_intensity_estimate",
    ]
    assert td_payload["practical_profile"]["practical_bucket"] == td_bucket
    assert (
        td_payload["compute_profile"]["workload_signature"]["practical_bucket_key"]
        == td_bucket["aggregation_key"]
    )

    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        np.ones((2, 7)),
        {
            "contraction_count": 3,
            "flop_count": 1000,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 100,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 300,
                },
                {
                    "step": 2,
                    "contraction_type": "OUTER/EINSUM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 600,
                },
            ],
        },
    )
    oe_bucket = oe_payload["practical_bucket"]

    assert oe_bucket["aggregation_key"] == (
        "oe|kernel=oe_mixed_path|types=GEMM>TDOT>OUTER/EINSUM|steps=3"
        "|dominant=OUTER/EINSUM:2x5x6+7->2x7"
    )
    assert oe_bucket["parallel_unit"] == "sequential_path_step"
    assert oe_bucket["comparison_axes"] == [
        "wall_s",
        "path_step_wall_time",
        "path_step_flops",
        "largest_intermediate_elements",
    ]
    assert oe_bucket["step_buckets"][0] == {
        "signature": "OUTER/EINSUM:2x5x6+7->2x7",
        "contraction_type": "OUTER/EINSUM",
        "input_shape_key": "2x5x6+7",
        "output_shape": "2x7",
        "count": 1,
        "total_flops_estimate": 600,
        "total_output_elements": 14,
        "step_indices": [2],
    }
    assert (
        oe_payload["compute_profile"]["workload_signature"]["practical_bucket_key"]
        == oe_bucket["aggregation_key"]
    )

    coef_array = np.ones((20, 20))
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_array,
        [
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (2, 2), "rank": 2},
        ],
        (coef_array,),
    )
    svd_bucket = svd_payload["practical_bucket"]

    assert svd_bucket["aggregation_key"] == (
        "svd|kernel=svd_qn|matrix=20x20|groups=10x10:2,2x2:1"
    )
    assert svd_bucket["parallel_unit"] == "qn_block_shape_group"
    assert svd_bucket["comparison_axes"] == [
        "wall_s",
        "block_shape_wall_time",
        "block_shape_flops",
        "decomposition_kernel_time",
    ]
    assert svd_bucket["block_group_signature"] == "10x10:2,2x2:1"
    assert (
        svd_payload["compute_profile"]["workload_signature"]["practical_bucket_key"]
        == svd_bucket["aggregation_key"]
    )


def test_core_compute_events_expose_compact_kernel_observation():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    td_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    td_observation = td_payload["kernel_observation"]

    assert td_observation["schema"] == "renormalizer.kernel_observation.v1"
    assert td_observation["family"] == "tensordot"
    assert td_observation["scope"] == "single_call"
    assert td_observation["accounting"] == "inclusive"
    assert td_observation["kernel"] == "gemm"
    assert td_observation["aggregation_key"] == td_payload["practical_bucket"]["aggregation_key"]
    assert td_observation["shape"] == {
        "m": 6,
        "n": 30,
        "k": 4,
        "layout": "right_axis_permutation",
        "input_shapes": ["2x3x4", "5x4x6"],
        "output_shape": "2x3x5x6",
        "dtype": "float64,float64->float64",
    }
    assert td_observation["cost"] == {
        "flops_estimate": td_payload["flops_estimate"],
        "read_bytes": td_payload["read_bytes"],
        "write_bytes": td_payload["write_bytes"],
        "copy_bytes": td_payload["copy_bytes"],
        "working_set_bytes_estimate": td_payload["working_set_bytes_estimate"],
    }
    assert td_observation["measurement"] == {
        "cost_basis": "single_gemm",
        "wall_time_basis": "caller_wall_s_if_recorded",
        "memory_basis": "array_nbytes_and_shape_estimate",
        "resource_timeseries": False,
        "external_telemetry_required": [
            "rss_pss_timeseries",
            "cpu_thread_context_switch_timeseries",
            "gpu_kernel_timeline",
            "host_device_transfer_timeline",
        ],
    }
    assert td_observation["parallelism"] == {
        "unit": "single_gemm",
        "candidate": "aggregate_repeated_small_gemm",
        "batchable": False,
        "requires_aggregation": True,
        "independent_units": 1,
    }

    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        np.ones((2, 7)),
        {
            "contraction_count": 3,
            "flop_count": 1000,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 100,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 300,
                },
                {
                    "step": 2,
                    "contraction_type": "OUTER/EINSUM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 600,
                },
            ],
        },
    )
    oe_observation = oe_payload["kernel_observation"]

    assert oe_observation["family"] == "oe_contract"
    assert oe_observation["scope"] == "path_call"
    assert oe_observation["accounting"] == "primary"
    assert oe_observation["kernel"] == "oe_mixed_path"
    assert oe_observation["shape"] == {
        "path_type_signature": "GEMM>TDOT>OUTER/EINSUM",
        "contraction_count": 3,
        "dominant_step_signature": "OUTER/EINSUM:2x5x6+7->2x7",
        "largest_intermediate_elements": 120,
        "unique_step_output_shape_count": 3,
    }
    assert oe_observation["parallelism"] == {
        "unit": "sequential_path_step",
        "candidate": "path_kernel_mix",
        "batchable": False,
        "independent_units": 0,
        "serial_units": 3,
        "backend_kernel_units": 3,
        "batchable_groups": 0,
    }
    assert oe_observation["measurement"]["cost_basis"] == "oe_path_step_estimate"
    assert oe_observation["measurement"]["wall_time_basis"] == "caller_wall_s_if_recorded"
    assert oe_observation["diagnosis"] == {
        "primary_issue": "non_gemm_path",
        "recommended_action": "reduce_non_gemm_path_cost",
        "evidence": {
            "gemm_step_count": 1,
            "tensordot_step_count": 1,
            "generic_einsum_step_count": 1,
            "non_gemm_step_count": 2,
            "largest_intermediate_to_output_ratio": 60 / 7,
        },
    }
    assert "step_costs" not in oe_observation
    assert "contraction_steps" not in oe_observation

    coef_array = np.ones((20, 20))
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_array,
        [
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (2, 2), "rank": 2},
        ],
        (coef_array,),
    )
    svd_observation = svd_payload["kernel_observation"]

    assert svd_observation["family"] == "svd"
    assert svd_observation["scope"] == "decomposition_call"
    assert svd_observation["kernel"] == "svd_qn"
    assert svd_observation["shape"] == {
        "matrix_shape": "20x20",
        "block_group_signature": "10x10:2,2x2:1",
        "block_count": 3,
        "unique_block_shape_count": 2,
        "dominant_block_shape": "10x10",
    }
    assert svd_observation["parallelism"] == {
        "unit": "qn_block_shape_group",
        "candidate": "block_shape_batching",
        "batchable": True,
        "independent_units": 3,
        "shape_groups": 2,
        "batchable_groups": 1,
    }
    assert svd_observation["measurement"]["cost_basis"] == "qn_sparse_blocked_decomposition"
    assert svd_observation["diagnosis"] == {
        "primary_issue": "tiny_qn_blocks",
        "recommended_action": "batch_reused_qn_blocks",
        "evidence": {
            "qn_block_density": 0.51,
            "batchable_block_fraction": 2 / 3,
            "tiny_block_count": 3,
            "block_shape_fragmentation": 2 / 3,
        },
    }
    assert "block_shape_counts" not in svd_observation


def test_kernel_observation_exposes_practical_execution_model():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    td_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    td_model = td_payload["kernel_observation"]["execution_model"]

    assert td_model == {
        "schema": "renormalizer.execution_model.v1",
        "model": "single_backend_contraction",
        "primary_unit": "single_gemm",
        "serial_dependency_units": 1,
        "independent_work_units": 1,
        "backend_kernel_units": 1,
        "batchable_work_units": 0,
        "batching": {
            "available_now": False,
            "scope": "cross_call_shape_aggregation_required",
            "grouping_basis": "m,n,k,dtype,layout",
            "bucket_key": td_payload["practical_bucket"]["batching_key"],
            "reason": "tiny_or_skinny_gemm_needs_repeated_shape_bucket",
        },
        "data_motion": {
            "working_set_bytes_estimate": td_payload["working_set_bytes_estimate"],
            "copy_bytes": td_payload["copy_bytes"],
            "copy_pressure": "axis_permutation",
            "peak_workspace_bytes_estimate": 0,
        },
    }

    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        np.ones((2, 7)),
        {
            "contraction_count": 3,
            "flop_count": 1000,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 100,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 300,
                },
                {
                    "step": 2,
                    "contraction_type": "OUTER/EINSUM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 600,
                },
            ],
        },
    )
    oe_model = oe_payload["kernel_observation"]["execution_model"]
    oe_working_set_bytes = (
        oe_payload["read_bytes"]
        + oe_payload["write_bytes"]
        + oe_payload["workspace_bytes"]
    )

    assert oe_model == {
        "schema": "renormalizer.execution_model.v1",
        "model": "serial_oe_contraction_path",
        "primary_unit": "path_step",
        "serial_dependency_units": 3,
        "independent_work_units": 0,
        "backend_kernel_units": 3,
        "batchable_work_units": 0,
        "kernel_units": {
            "gemm_steps": 1,
            "tensordot_steps": 1,
            "generic_einsum_steps": 1,
            "non_gemm_steps": 2,
        },
        "batching": {
            "available_now": False,
            "scope": "none",
            "grouping_basis": "path_step_bucket_signature",
            "batchable_groups": 0,
        },
        "data_motion": {
            "working_set_bytes_estimate": oe_working_set_bytes,
            "copy_bytes": 0,
            "copy_pressure": "large_intermediate",
            "largest_intermediate_elements": 120,
            "largest_intermediate_to_output_ratio": 60 / 7,
            "peak_workspace_bytes_estimate": oe_payload["workspace_bytes"],
        },
    }

    coef_array = np.ones((20, 20))
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_array,
        [
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (2, 2), "rank": 2},
        ],
        (coef_array,),
    )
    svd_model = svd_payload["kernel_observation"]["execution_model"]
    svd_working_set_bytes = svd_payload["read_bytes"] + svd_payload["write_bytes"]

    assert svd_model == {
        "schema": "renormalizer.execution_model.v1",
        "model": "independent_qn_block_decomposition",
        "primary_unit": "qn_block",
        "serial_dependency_units": 0,
        "independent_work_units": 3,
        "backend_kernel_units": 3,
        "batchable_work_units": 2,
        "kernel_units": {
            "block_count": 3,
            "shape_groups": 2,
            "tiny_blocks": 3,
            "skinny_blocks": 0,
        },
        "batching": {
            "available_now": True,
            "scope": "same_shape_qn_blocks",
            "grouping_basis": "block_shape",
            "batchable_groups": 1,
            "batchable_flop_fraction": svd_payload["batchable_flop_fraction"],
        },
        "data_motion": {
            "working_set_bytes_estimate": svd_working_set_bytes,
            "copy_bytes": 0,
            "copy_pressure": "qn_sparse_block_density",
            "matrix_elements": 400,
            "total_block_elements": 204,
            "qn_block_density": 0.51,
            "peak_workspace_bytes_estimate": 0,
        },
    }


def test_kernel_observation_exposes_practical_execution_units_and_limits():
    import numpy as np

    from renormalizer.utils import profiling

    left = np.ones((2, 3, 4))
    right = np.ones((5, 4, 6))
    td_result = np.tensordot(left, right, axes=([2], [1]))
    td_payload = profiling.tensordot_compute_payload(left, right, axes=([2], [1]), result=td_result)
    td_observation = td_payload["kernel_observation"]

    assert td_observation["execution_units"] == [
        {
            "kind": "gemm_like",
            "role": "primary",
            "count": 1,
            "source": "mnk_shape_model",
            "flops_estimate": td_payload["flops_estimate"],
            "serial_dependency": False,
            "batchable": False,
        },
        {
            "kind": "layout_copy",
            "role": "overhead",
            "count": 1,
            "source": "axis_permutation_copy_bytes",
            "bytes": td_payload["copy_bytes"],
            "serial_dependency": False,
            "batchable": False,
        },
    ]
    assert [limit["kind"] for limit in td_observation["optimization_limits"]] == [
        "cross_call_aggregation_required",
        "layout_copy_estimate",
    ]
    assert td_observation["optimization_limits"][0]["evidence"]["bucket_key"] == (
        td_payload["practical_bucket"]["batching_key"]
    )

    oe_payload = profiling.oe_compute_payload(
        "oe_contract",
        (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5, 6))),
        np.ones((2, 7)),
        {
            "contraction_count": 3,
            "flop_count": 1000,
            "largest_intermediate": 120,
            "contraction_steps": [
                {
                    "step": 0,
                    "contraction_type": "GEMM",
                    "input_shapes": [(2, 3), (3, 4)],
                    "output_shape": (2, 4),
                    "flops_estimate": 100,
                },
                {
                    "step": 1,
                    "contraction_type": "TDOT",
                    "input_shapes": [(2, 4), (4, 5, 6)],
                    "output_shape": (2, 5, 6),
                    "flops_estimate": 300,
                },
                {
                    "step": 2,
                    "contraction_type": "OUTER/EINSUM",
                    "input_shapes": [(2, 5, 6), (7,)],
                    "output_shape": (2, 7),
                    "flops_estimate": 600,
                },
            ],
        },
    )
    oe_observation = oe_payload["kernel_observation"]

    assert oe_observation["execution_units"] == [
        {
            "kind": "oe_gemm_step",
            "role": "primary",
            "count": 1,
            "source": "path_step_profile",
            "flops_estimate": 100,
            "serial_dependency": True,
            "batchable": False,
        },
        {
            "kind": "oe_tensordot_step",
            "role": "primary",
            "count": 1,
            "source": "path_step_profile",
            "flops_estimate": 300,
            "serial_dependency": True,
            "batchable": False,
        },
        {
            "kind": "oe_generic_einsum_step",
            "role": "primary",
            "count": 1,
            "source": "path_step_profile",
            "flops_estimate": 600,
            "serial_dependency": True,
            "batchable": False,
        },
        {
            "kind": "oe_intermediate",
            "role": "overhead",
            "count": 1,
            "source": "largest_intermediate_elements",
            "elements": 120,
            "bytes": oe_payload["largest_intermediate_bytes"],
            "serial_dependency": False,
            "batchable": False,
        },
    ]
    assert [limit["kind"] for limit in oe_observation["optimization_limits"]] == [
        "serial_path_dependency",
        "generic_einsum_not_gemm_lowered",
        "large_intermediate_pressure",
    ]

    coef_array = np.ones((20, 20))
    svd_payload = profiling.svd_qn_compute_payload(
        "SVD",
        coef_array,
        coef_array,
        [
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (10, 10), "rank": 10},
            {"block_shape": (2, 2), "rank": 2},
        ],
        (coef_array,),
    )
    svd_observation = svd_payload["kernel_observation"]

    assert svd_observation["execution_units"] == [
        {
            "kind": "svd_qn_block",
            "role": "primary",
            "count": 3,
            "source": "block_count",
            "flops_estimate": svd_payload["block_flops_estimate"],
            "serial_dependency": False,
            "batchable": False,
        },
        {
            "kind": "batchable_qn_block",
            "role": "subset",
            "count": 2,
            "source": "batchable_block_count",
            "flops_estimate": svd_payload["batchable_flops_estimate"],
            "serial_dependency": False,
            "batchable": True,
        },
        {
            "kind": "tiny_qn_block",
            "role": "subset",
            "count": 3,
            "source": "tiny_block_count",
            "serial_dependency": False,
            "batchable": False,
        },
    ]
    assert [limit["kind"] for limit in svd_observation["optimization_limits"]] == [
        "batching_requires_same_shape_driver",
        "tiny_blocks_may_be_overhead_bound",
        "shape_fragmentation",
    ]


def test_compute_summaries_report_rhs_fallback_metrics(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="fallback_rhs_loop",
        read_bytes=96,
        write_bytes=96,
        num_rhs=3,
        num_rhs_loop_calls=3,
        fallback_reason="unit test rhs fallback",
        wall_s=0.12,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    compute_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    class_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "tensordot"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    tensordot_row = next(row for row in overview["classes"] if row["compute_class"] == "tensordot")

    assert compute_summary["kernel_kinds"] == ["fallback_rhs_loop"]
    assert compute_summary["total_rhs_vectors"] == 3
    assert compute_summary["max_rhs"] == 3
    assert compute_summary["total_rhs_loop_calls"] == 3
    assert class_summary["total_rhs_loop_calls"] == 3
    assert tensordot_row["kernel_kinds"] == ["fallback_rhs_loop"]
    assert tensordot_row["total_rhs_vectors"] == 3
    assert tensordot_row["total_rhs_loop_calls"] == 3
    assert tensordot_row["fallback_calls"] == 1


def test_compute_summaries_preserve_explicit_zero_gemm_count(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="block_grouped_gemm",
        input_shapes=[],
        output_shape=(2, 4),
        num_gemm=0,
        num_batched_gemm=0,
        num_grouped_tasks=0,
        num_blocks=0,
        num_shape_buckets=0,
        flops=0,
        read_bytes=0,
        write_bytes=0,
        wall_s=0.01,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "tensordot")

    assert summary["total_gemm"] == 0
    assert row["total_gemm"] == 0


def test_compute_summaries_report_block_grouped_reduction_pressure(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="block_grouped_gemm",
        input_shapes=[],
        output_shape=(2, 4),
        num_gemm=2,
        num_batched_gemm=0,
        num_grouped_tasks=5,
        num_blocks=2,
        num_shape_buckets=1,
        scatter_add_required=True,
        reduction_mode="scatter_add",
        num_output_reduction_groups=2,
        num_scatter_add_tasks=5,
        max_output_contributions=3,
        flops=96,
        read_bytes=288,
        write_bytes=128,
        wall_s=0.01,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    class_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "tensordot"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    row = next(item for item in overview["classes"] if item["compute_class"] == "tensordot")

    for payload in (summary, class_summary, row):
        assert payload["total_output_reduction_groups"] == 2
        assert payload["total_scatter_add_tasks"] == 5
        assert payload["max_output_contributions"] == 3
        assert payload["scatter_add_calls"] == 1


def test_compute_summaries_report_fallback_sources(caplog):
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    profiling.record(
        "contraction_execute",
        compute_class="tensordot",
        compute_subclass="backend_execute",
        compute_role="kernel",
        compute_accounting="primary",
        backend="numpy",
        lowering="fallback_tensordot",
        fallback_from="batched_gemm",
        fallback_reason="backend lacks batched_matmul for batch shape (5,)",
        fallback_policy="record",
        wall_s=0.2,
    )

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    compute_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_summary"
        and payload["compute_class"] == "tensordot"
    )
    class_summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_compute_class_summary"
        and payload["compute_class"] == "tensordot"
    )
    overview = next(payload for payload in payloads if payload["event"] == "profile_compute_class_overview")
    tensordot_row = next(row for row in overview["classes"] if row["compute_class"] == "tensordot")

    assert compute_summary["fallback_sources"] == ["batched_gemm"]
    assert compute_summary["fallback_reasons"] == ["backend lacks batched_matmul for batch shape (5,)"]
    assert compute_summary["fallback_policies"] == ["record"]
    assert compute_summary["fallback_calls"] == 1
    assert class_summary["fallback_sources"] == ["batched_gemm"]
    assert class_summary["fallback_reasons"] == ["backend lacks batched_matmul for batch shape (5,)"]
    assert class_summary["fallback_policies"] == ["record"]
    assert class_summary["fallback_calls"] == 1
    assert tensordot_row["fallback_sources"] == ["batched_gemm"]
    assert tensordot_row["fallback_reasons"] == ["backend lacks batched_matmul for batch shape (5,)"]
    assert tensordot_row["fallback_policies"] == ["record"]
    assert tensordot_row["fallback_calls"] == 1


def test_oe_contract_writes_full_event_to_jsonl_in_trace_mode(caplog, monkeypatch, tmp_path):
    import numpy as np

    from renormalizer.mps.oe_contract_wrap import oe_contract
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        result = oe_contract("ab,bc->ac", np.ones((2, 3)), np.ones((3, 4)), optimize="greedy")
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert result.shape == (2, 4)
    log_payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in log_payloads if payload["event"] == "oe_contract"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "oe_contract")
    assert event["equation"] == "ab,bc->ac"
    assert event["input_shapes"] == [[2, 3], [3, 4]]
    assert event["output_shape"] == [2, 4]
    assert event["compute_class"] == "oe"
    assert event["compute_subclass"] == "oe_contract"
    assert event["compute_role"] == "composite"
    assert event["compute_accounting"] == "primary"
    assert event["input_dtypes"] == ["float64", "float64"]
    assert event["output_dtype"] == "float64"
    assert event["flops_estimate"] >= 1
    assert event["read_bytes"] == 2 * 3 * 8 + 3 * 4 * 8
    assert event["write_bytes"] == 2 * 4 * 8
    assert event["largest_intermediate_elements"] >= 1
    assert event["largest_intermediate_bytes"] == event["largest_intermediate_elements"] * 8
    assert event["peak_bytes"] == event["largest_intermediate_bytes"]
    assert event["optimize"] == "greedy"
    assert event["wall_s"] >= 0


def test_oe_contract_expression_records_path_summary_in_jsonl(caplog, tmp_path):
    import numpy as np

    from renormalizer.mps.oe_contract_wrap import oe_contract_expression
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    a = np.ones((2, 3))
    try:
        expr = oe_contract_expression("ab,bc,cd->ad", a, (3, 4), (4, 5), constants=[0], optimize="greedy")
        result = expr(np.ones((3, 4)), np.ones((4, 5)))
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert result.shape == (2, 5)
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "oe_contract_expression")
    assert event["path"] == [[1, 2], [0, 1]]
    assert event["compute_class"] == "oe"
    assert event["compute_subclass"] == "oe_expression_execute"
    assert event["compute_role"] == "composite"
    assert event["compute_accounting"] == "primary"
    assert event["input_dtypes"] == ["float64", "float64", "float64"]
    assert event["output_dtype"] == "float64"
    assert event["flops_estimate"] >= 1
    assert event["read_bytes"] == 2 * 3 * 8 + 3 * 4 * 8 + 4 * 5 * 8
    assert event["write_bytes"] == 2 * 5 * 8
    assert event["contraction_count"] == 2
    assert event["flop_count"] >= 1
    assert event["largest_intermediate"] >= 1
    assert event["contraction_types"] == ["GEMM", "GEMM"]
    assert event["contraction_steps"][0]["input_modes"] == ["cd", "bc"]
    assert event["contraction_steps"][0]["input_shapes"] == [[4, 5], [3, 4]]
    assert event["contraction_steps"][0]["output_modes"] == "db"
    assert event["contraction_steps"][0]["output_shape"] == [5, 3]
    assert event["contraction_steps"][1]["input_modes"] == ["db", "ab"]
    assert event["contraction_steps"][1]["output_modes"] == "ad"
    assert event["contraction_steps"][1]["remaining_shapes"] == [[2, 5]]


def test_hop_expr_writes_full_event_to_jsonl_in_trace_mode(caplog, monkeypatch, tmp_path):
    import numpy as np

    from renormalizer.mps.hop_expr import hop_expr
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        expr = hop_expr(np.ones((2, 3, 4)), np.ones((5, 3, 7)), [], (4, 7))
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert callable(expr)
    log_payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in log_payloads if payload["event"] == "hop_expr"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "hop_expr")
    assert event["nsite"] == 0
    assert event["l_shape"] == [2, 3, 4]
    assert event["r_shape"] == [5, 3, 7]
    assert event["cshape"] == [4, 7]
    assert event["wall_s"] >= 0


def test_svd_qn_writes_full_event_to_jsonl_in_trace_mode(caplog, monkeypatch, tmp_path):
    import numpy as np

    from renormalizer.mps.svd_qn import svd_qn
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    qn = np.zeros((2, 1), dtype=int)
    try:
        u, qnl, v, qnr = svd_qn(np.eye(2), qn, qn, np.array([0]), QR=True, system="L", full_matrices=False)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert u.shape == (2, 2)
    assert v.shape == (2, 2)
    assert len(qnl) == 2
    assert len(qnr) == 2
    log_payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in log_payloads if payload["event"] == "svd_qn"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "svd_qn")
    assert event["mode"] == "QR"
    assert event["compute_class"] == "svd"
    assert event["compute_subclass"] == "qr_qn"
    assert event["compute_role"] == "kernel"
    assert event["input_dtype"] == "float64"
    assert event["output_dtype"] == "float64"
    assert event["read_bytes"] == 2 * 2 * 8
    assert event["write_bytes"] >= 2 * 2 * 8
    assert event["flops_estimate"] > 0
    assert event["system"] == "L"
    assert event["coef_shape"] == [2, 2]
    assert event["matrix_shape"] == [2, 2]
    assert event["block_count"] == 1
    assert event["blocks"] == [
        {
            "left_qn": [0],
            "right_qn": [0],
            "left_size": 2,
            "right_size": 2,
            "block_shape": [2, 2],
            "rank": 2,
        }
    ]
    assert event["output_rank"] == 2
    assert event["wall_s"] >= 0


def test_eigh_qn_writes_full_event_to_jsonl_in_trace_mode(caplog, monkeypatch, tmp_path):
    import numpy as np

    from renormalizer.mps.svd_qn import eigh_qn
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    qn = np.zeros((2, 1), dtype=int)
    try:
        u, s, new_qn = eigh_qn(np.eye(2), qn, qn, np.array([0]), system="L")
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert u.shape == (2, 2)
    assert s.shape == (2,)
    assert len(new_qn) == 2
    log_payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in log_payloads if payload["event"] == "eigh_qn"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "eigh_qn")
    assert event["compute_class"] == "svd"
    assert event["compute_subclass"] == "eigh_qn"
    assert event["compute_role"] == "kernel"
    assert event["system"] == "L"
    assert event["dm_shape"] == [2, 2]
    assert event["matrix_shape"] == [2, 2]
    assert event["input_dtype"] == "float64"
    assert event["output_dtype"] == "float64"
    assert event["read_bytes"] == 2 * 2 * 8
    assert event["write_bytes"] >= 2 * 2 * 8
    assert event["flops_estimate"] > 0
    assert event["block_count"] == 1
    assert event["blocks"] == [
        {
            "left_qn": [0],
            "right_qn": [0],
            "left_size": 2,
            "right_size": 2,
            "block_shape": [2, 2],
            "rank": 2,
        }
    ]
    assert event["output_rank"] == 2
    assert event["singular_value_count"] == 2
    assert event["wall_s"] >= 0


def test_mps_copy_to_complex_and_environ_events_write_jsonl(caplog, tmp_path):
    from renormalizer import BasisHalfSpin, Model, Mpo, Mps, Op
    from renormalizer.mps.lib import Environ
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    model = Model([BasisHalfSpin(0), BasisHalfSpin(1)], [])
    mps = Mps.hartree_product_state(model, condition={})
    mpo = Mpo(model, Op("X", 0))

    try:
        mps.copy()
        mps.to_complex()
        environ = Environ(mps, mpo)
        environ.GetLR("L", 0, mps, mpo, method="System")
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    payloads = _jsonl_payloads(event_path)
    events = [payload["event"] for payload in payloads]
    assert "mps_copy" in events
    assert "mps_to_complex" in events
    assert "environ_build" in events
    assert "environ_contract_site" in events
    assert "environ_getlr" in events
    environ_event = next(payload for payload in payloads if payload["event"] == "environ_contract_site")
    assert environ_event["mp_type"] == "Mps"
    assert environ_event["domain"] in ["L", "R"]
    assert environ_event["output_shape"]
    assert environ_event["lowering"] == "multi_tensor_contract"
    assert environ_event["num_contraction_steps"] == 3
    assert environ_event["contraction_plan_summary"]["contraction_count"] == 3
    assert environ_event["contraction_plan_summary"]["lowerings"]
    assert len(environ_event["contraction_plan_summary"]["steps"]) == 3
    assert environ_event["num_gemm"] + environ_event["num_batched_gemm"] + environ_event["num_grouped_tasks"] >= 1
    assert environ_event["wall_s"] >= 0


def test_multi_mpo_environ_event_reports_execution_plan_summary(caplog, tmp_path):
    from renormalizer import BasisHalfSpin, Model, Mpo, Mps, Op
    from renormalizer.mps.lib import Environ
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    model = Model([BasisHalfSpin(0), BasisHalfSpin(1)], [])
    mps = Mps.hartree_product_state(model, condition={})
    mpos = [Mpo(model, Op("X", 0)), Mpo(model, Op("X", 1))]

    try:
        Environ(mps, mpos)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    payloads = _jsonl_payloads(event_path)
    environ_event = next(
        payload for payload in payloads
        if payload["event"] == "environ_contract_site" and payload["multi_mpo"] is True
    )
    assert environ_event["lowering"] == "multi_tensor_contract"
    assert environ_event["num_contraction_steps"] == 4
    assert environ_event["contraction_plan_summary"]["contraction_count"] == 4
    assert environ_event["contraction_plan_summary"]["lowerings"]
    assert len(environ_event["contraction_plan_summary"]["steps"]) == 4
    assert environ_event["num_gemm"] + environ_event["num_batched_gemm"] + environ_event["num_grouped_tasks"] >= 1
    assert environ_event["wall_s"] >= 0


def test_op_events_are_summarized_by_default(caplog, monkeypatch):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling
    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)

    payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in payloads if payload["event"] == "tensordot"]

    profiling.flush_summaries()
    payloads = _profiling_payloads(caplog, profiling)
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_summary" and payload["source_event"] == "tensordot"
    )
    assert summary["source_event"] == "tensordot"
    assert summary["call_count"] == 1
    assert summary["total_wall_s"] >= 0
    assert summary["signature"]["input_shapes"] == [[2, 3], [3, 4]]
    assert summary["signature"]["output_shape"] == [2, 4]


def test_operation_events_are_not_written_to_jsonl_without_registered_output(caplog):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling

    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in payloads if payload["event"] == "tensordot"]
    summary = next(
        payload for payload in payloads
        if payload["event"] == "profile_summary" and payload["source_event"] == "tensordot"
    )
    assert summary["source_event"] == "tensordot"
    overhead = next(payload for payload in payloads if payload["event"] == "profile_overhead")
    assert overhead["events_written"] == 0


def test_register_event_output_uses_default_timestamped_jsonl_name(tmp_path, monkeypatch):
    from renormalizer.utils import profiling

    monkeypatch.chdir(tmp_path)

    path = profiling.register_event_output()
    try:
        assert path.parent == tmp_path
        assert path.name.startswith("profiling-")
        assert path.suffix == ".jsonl"
        assert path.exists()
    finally:
        profiling.close_event_output()


def test_flush_summaries_emits_profile_overhead_event(caplog, monkeypatch):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling
    profiling.flush_summaries()
    caplog.set_level(PROFILING, logger="renormalizer")

    tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    overhead = next(payload for payload in payloads if payload["event"] == "profile_overhead")
    assert overhead["events_seen"] >= 1
    assert overhead["events_summarized"] >= 1
    assert overhead["record_overhead_s"] >= 0
    assert overhead["flush_overhead_s"] >= 0


def test_empty_flush_does_not_emit_profile_overhead_event(caplog, monkeypatch):
    import numpy as np

    from renormalizer.mps.matrix import tensordot
    from renormalizer.utils.log import PROFILING
    from renormalizer.utils import profiling
    caplog.set_level(PROFILING, logger="renormalizer")

    tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)
    profiling.flush_summaries()
    caplog.clear()

    profiling.flush_summaries()

    payloads = _profiling_payloads(caplog, profiling)
    assert not [payload for payload in payloads if payload["event"] == "profile_overhead"]
