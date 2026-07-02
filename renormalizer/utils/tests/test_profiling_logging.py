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
    assert event["input_dtypes"] == ["float64", "float64"]
    assert event["output_dtype"] == "float64"
    assert event["flops_estimate"] >= 1
    assert event["read_bytes"] == 2 * 3 * 8 + 3 * 4 * 8
    assert event["write_bytes"] == 2 * 4 * 8
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
