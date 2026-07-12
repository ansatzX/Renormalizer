import importlib
import importlib.util
import json
import sys

import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend._execution.profiling import local_hv_payload, phase_summary_payload
from renormalizer.backend._gemm.profiling import grouped_gemm_payload, summarize_shape_buckets
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.matrix import asnumpy
from renormalizer.utils import profiling
from renormalizer.utils._profiling.events import validate_source_event
from renormalizer.utils.log import DEBUG, PROFILING, init_log


@pytest.fixture(autouse=True)
def reset_profiling_runtime(monkeypatch):
    profiling.close_event_output()
    monkeypatch.setattr(profiling, "_runtime", None)
    init_log(DEBUG)
    yield
    profiling.close_event_output()
    init_log(DEBUG)


def test_shape_bucket_summary_is_bounded_sorted_and_deterministic():
    buckets = {(i, i + 1, i + 2): (i % 7) + 1 for i in reversed(range(100))}

    summary = summarize_shape_buckets(buckets, limit=32)

    assert len(summary) == 32
    assert summary == sorted(summary, key=lambda item: (-item["count"], item["shape"]))
    assert summary == summarize_shape_buckets(dict(reversed(list(buckets.items()))), limit=32)


def test_shape_bucket_summary_rejects_oversized_gemm_shape():
    with pytest.raises(ValueError, match="exactly three"):
        summarize_shape_buckets({(2, 3, 4, 5): 1})


def test_grouped_gemm_payload_does_not_claim_unexecuted_grouping():
    payload = grouped_gemm_payload(
        operation="gemm",
        task_count=3,
        shape_buckets={(2, 3, 4): 3},
        executed_grouped=False,
    )

    assert payload["grouped_execution"] is False
    assert payload["shape_buckets"] == [{"shape": [2, 3, 4], "count": 3}]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("operation", np.str_("gemm"), TypeError),
        ("operation", np.array("gemm"), TypeError),
        ("task_count", np.int64(2), TypeError),
        ("task_count", -1, ValueError),
        ("executed_grouped", np.bool_(False), TypeError),
        ("executed_grouped", 1, TypeError),
    ],
)
def test_grouped_gemm_payload_rejects_non_python_metadata(field, value, error):
    kwargs = {
        "operation": "gemm",
        "task_count": 2,
        "shape_buckets": {(2, 3, 4): 2},
        "executed_grouped": False,
    }
    kwargs[field] = value

    with pytest.raises(error):
        grouped_gemm_payload(**kwargs)


def test_local_payload_records_planner_fallback_timing_and_bounded_shapes():
    payload = local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=tuple(tuple(range(40)) for _ in range(100)),
        output_shape=tuple(range(40)),
        requested_policy="legacy_oe",
        actual_policy="legacy_oe",
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=("gemm", "tensordot"),
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
        fallback_reason=None,
    )

    assert payload["planner_source"] == "opt_einsum"
    assert payload["actual_steps"] == ["gemm", "tensordot"]
    assert payload["fallback"] is None
    assert payload["fallback_reason"] is None
    assert payload["wall_s"] == 0.25
    assert payload["timing_semantics"] == "host_elapsed_synchronous"
    assert payload["device_synchronized"] is True
    assert payload["actual_step_count"] == 2
    assert payload["actual_steps_truncated"] is False
    assert len(payload["input_shapes"]) == 32
    assert all(len(shape) == 16 for shape in payload["input_shapes"])
    assert payload["input_shape_ranks"] == [40] * 32
    assert payload["input_shape_count"] == 100
    assert payload["input_shapes_truncated"] is True
    assert len(payload["output_shape"]) == 16
    assert payload["output_shape_rank"] == 40
    assert payload["output_shape_truncated"] is True


def test_local_payload_truncation_flags_are_exact_at_bounds():
    payload = local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=tuple([(1,) * 16] * 32),
        output_shape=(1,) * 16,
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=tuple(["gemm"] * 32),
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
    )

    assert payload["input_shape_ranks"] == [16] * 32
    assert payload["input_shapes_truncated"] is False
    assert payload["output_shape_rank"] == 16
    assert payload["output_shape_truncated"] is False
    assert payload["actual_steps_truncated"] is False


def test_local_payload_rank_only_truncation_is_canonical():
    payload = local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=((1,) * 17,),
        output_shape=(1,),
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=("gemm",),
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
    )

    assert payload["input_shapes"] == [[1] * 16]
    assert payload["input_shape_ranks"] == [17]
    validated = validate_source_event({"event": "local_hv_execute", **payload})
    assert validated["input_shapes_truncated"] is False


def test_local_payload_combined_sequence_and_rank_truncation_is_canonical():
    input_shapes = ((1,) * 17,) + tuple([(1,)] * 32)
    payload = local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=input_shapes,
        output_shape=(1,),
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=("gemm",),
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
    )

    assert payload["input_shape_count"] == 33
    assert len(payload["input_shapes"]) == 32
    assert payload["input_shapes"][0] == [1] * 16
    assert payload["input_shape_ranks"] == [17] + [1] * 31
    validated = validate_source_event({"event": "local_hv_execute", **payload})
    assert validated["input_shapes_truncated"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_shapes", (tuple([1] * 16 + [np.int64(1)]),)),
        ("input_shapes", tuple([(1,)] * 32 + [(np.int64(1),)])),
        ("output_shape", tuple([1] * 16 + [np.int64(1)])),
    ],
)
def test_local_payload_validates_all_shape_metadata_before_truncating(field, value):
    kwargs = {
        "network": "mps",
        "center_kind": "one_site",
        "input_shapes": ((2, 2),),
        "output_shape": (2, 2),
        "planner_source": "opt_einsum",
        "oe_path_hash": "abc123",
        "actual_steps": ("gemm",),
        "wall_s": 0.25,
        "timing_semantics": "host_elapsed_synchronous",
        "device_synchronized": True,
    }
    kwargs[field] = value

    with pytest.raises(TypeError, match="shape"):
        local_hv_payload(**kwargs)


def test_local_payload_identifies_actual_policy_when_fallback_occurs():
    payload = local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=((2, 2),),
        output_shape=(2, 2),
        requested_policy="execution_ir",
        actual_policy="legacy_oe",
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=("gemm",),
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
        fallback_reason="execution IR is unavailable for this backend",
    )

    assert payload["fallback"] == "legacy_oe"
    assert payload["fallback_reason"] == "execution IR is unavailable for this backend"


def test_local_payload_bounds_actual_steps_after_validating_all_metadata():
    steps = tuple(f"step-{index}" for index in range(100))
    payload = local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=((2, 2),),
        output_shape=(2, 2),
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=steps,
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
    )

    assert payload["actual_steps"] == list(steps[:32])
    assert payload["actual_step_count"] == 100
    assert payload["actual_steps_truncated"] is True

    invalid_steps = list(steps)
    invalid_steps[99] = np.array("einsum")
    with pytest.raises(TypeError, match="actual steps"):
        local_hv_payload(
            network="mps",
            center_kind="one_site",
            input_shapes=((2, 2),),
            output_shape=(2, 2),
            planner_source="opt_einsum",
            oe_path_hash="abc123",
            actual_steps=invalid_steps,
            wall_s=0.25,
            timing_semantics="host_elapsed_synchronous",
            device_synchronized=True,
        )


@pytest.mark.parametrize(
    "blas_flag,expected",
    [
        ("GEMM", "gemm"),
        ("DOT", "tensordot"),
        ("TDOT", "tensordot"),
        ("TDOT/EINSUM", "einsum"),
        (False, "einsum"),
    ],
)
def test_oe_lowering_classification_matches_opt_einsum_dispatch(blas_flag, expected):
    from renormalizer.mps.oe_contract_wrap import _classify_oe_lowering

    assert _classify_oe_lowering(blas_flag) == expected


def test_phase_summary_payload_is_aggregate_metadata_only():
    payload = phase_summary_payload(
        phase="qn_decomposition",
        network="mps",
        operation="svd",
        operation_count=7,
        wall_s=0.125,
    )

    assert payload == {
        "phase": "qn_decomposition",
        "network": "mps",
        "operation": "svd",
        "operation_count": 7,
        "wall_s": 0.125,
    }


def test_payload_helpers_reject_live_array_metadata():
    with pytest.raises(TypeError, match="Python metadata"):
        phase_summary_payload(
            phase="qn_decomposition",
            network="mps",
            operation="svd",
            operation_count=1,
            wall_s=np.array([0.125]),
        )


def _single_site_operands():
    rng = np.random.default_rng(7)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    center = rng.normal(size=(4, 4, 8))
    return left, mpo, right, center


def test_enabled_mps_hv_jsonl_records_order_and_bounded_path_evidence(tmp_path):
    backend_module = importlib.import_module("renormalizer.mps.backend")

    init_log(PROFILING)
    path = tmp_path / "local-events.jsonl"
    profiling.register_event_output(path)
    left, mpo, right, center = _single_site_operands()
    expression = hop_expr(left, right, [mpo], center.shape)

    first = asnumpy(expression(backend_module.xp.asarray(center)))
    second = asnumpy(expression(backend_module.xp.asarray(center * 2)))
    profiling.close_event_output()

    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
    np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(second, expected * 2, rtol=1e-12, atol=1e-12)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["local_hv_execute", "local_hv_execute"]
    assert events[0]["oe_path_hash"] == events[1]["oe_path_hash"]
    for event in events:
        assert event["network"] == "mps"
        assert event["center_kind"] == "one_site"
        assert event["input_shapes"] == [[2, 3, 4], [3, 5, 4, 6], [7, 6, 8], [4, 4, 8]]
        assert event["input_shape_ranks"] == [3, 4, 3, 3]
        assert event["output_shape"] == [2, 5, 7]
        assert event["output_shape_rank"] == 3
        assert event["requested_policy"] == "legacy_oe"
        assert event["actual_policy"] == "legacy_oe"
        assert event["planner_source"] == "opt_einsum"
        assert len(event["oe_path_hash"]) == 64
        assert event["actual_steps"] == ["gemm", "tensordot", "tensordot"]
        assert event["actual_step_count"] == 3
        assert event["actual_steps_truncated"] is False
        assert event["wall_s"] >= 0
        assert event["timing_semantics"] == "host_elapsed_synchronous"
        assert event["device_synchronized"] is True
        assert event["fallback"] is None
        assert event["fallback_reason"] is None
        assert len(event["input_shapes"]) <= 32


def _truncation_payload(*, truncated):
    width = 17 if truncated else 2
    count = 33 if truncated else 1
    step_count = 33 if truncated else 1
    return local_hv_payload(
        network="mps",
        center_kind="one_site",
        input_shapes=tuple([(1,) * width] * count),
        output_shape=(1,) * width,
        planner_source="opt_einsum",
        oe_path_hash="abc123",
        actual_steps=tuple(["gemm"] * step_count),
        wall_s=0.25,
        timing_semantics="host_elapsed_synchronous",
        device_synchronized=True,
    )


@pytest.mark.parametrize(
    "field,truncated,claimed",
    [
        ("input_shapes_truncated", False, True),
        ("input_shapes_truncated", True, False),
        ("output_shape_truncated", False, True),
        ("output_shape_truncated", True, False),
        ("actual_steps_truncated", False, True),
        ("actual_steps_truncated", True, False),
    ],
)
def test_public_local_record_rejects_false_truncation_claims(
    tmp_path, field, truncated, claimed
):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    payload = _truncation_payload(truncated=truncated)
    payload[field] = claimed

    with pytest.raises(ValueError, match="truncat"):
        profiling.record("local_hv_execute", **payload)

    profiling.close_event_output()
    assert path.read_text() == ""


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_shape_ranks", []),
        ("input_shape_ranks", [3]),
        ("output_shape_rank", 3),
    ],
)
def test_public_local_record_rejects_mismatched_original_rank_evidence(tmp_path, field, value):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    payload = _truncation_payload(truncated=False)
    payload[field] = value

    with pytest.raises(ValueError, match="rank"):
        profiling.record("local_hv_execute", **payload)

    profiling.close_event_output()
    assert path.read_text() == ""


@pytest.mark.parametrize("bucket_count,claimed", [(2, True), (33, False)])
def test_public_grouped_record_rejects_false_bucket_truncation(
    tmp_path, bucket_count, claimed
):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    payload = grouped_gemm_payload(
        operation="gemm",
        task_count=max(2, bucket_count),
        shape_buckets={(index + 1, 2, 3): 1 for index in range(bucket_count)},
        executed_grouped=True,
    )
    payload["shape_buckets_truncated"] = claimed

    with pytest.raises(ValueError, match="truncat"):
        profiling.record("grouped_gemm_execute", **payload)

    profiling.close_event_output()
    assert path.read_text() == ""


def _bounded_sequence_payload(kind, original_count):
    if kind in {"input_shapes", "actual_steps"}:
        payload = local_hv_payload(
            network="mps",
            center_kind="one_site",
            input_shapes=tuple([(1, 1)] * (original_count if kind == "input_shapes" else 1)),
            output_shape=(1, 1),
            planner_source="opt_einsum",
            oe_path_hash="abc123",
            actual_steps=tuple(["gemm"] * (original_count if kind == "actual_steps" else 1)),
            wall_s=0.25,
            timing_semantics="host_elapsed_synchronous",
            device_synchronized=True,
        )
        return "local_hv_execute", payload
    payload = grouped_gemm_payload(
        operation="gemm",
        task_count=max(2, original_count),
        shape_buckets={(index + 1, 2, 3): 1 for index in range(original_count)},
        executed_grouped=True,
    )
    return "grouped_gemm_execute", payload


@pytest.mark.parametrize("kind", ["input_shapes", "actual_steps", "shape_buckets"])
@pytest.mark.parametrize("original_count,retained_count", [(31, 30), (33, 31)])
def test_public_record_rejects_under_retained_bounded_sequences(
    tmp_path, kind, original_count, retained_count
):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    event, payload = _bounded_sequence_payload(kind, original_count)
    payload[kind] = payload[kind][:retained_count]
    if kind == "input_shapes":
        payload["input_shape_ranks"] = payload["input_shape_ranks"][:retained_count]
        payload["input_shapes_truncated"] = True
    elif kind == "actual_steps":
        payload["actual_steps_truncated"] = True
    else:
        payload["shape_buckets_truncated"] = True

    with pytest.raises(ValueError, match="retained"):
        profiling.record(event, **payload)

    profiling.close_event_output()
    assert path.read_text() == ""


@pytest.mark.parametrize("kind", ["input_shapes", "actual_steps", "shape_buckets"])
@pytest.mark.parametrize("original_count,claimed", [(31, True), (33, False)])
def test_public_record_rejects_false_truncation_at_count_boundaries(
    tmp_path, kind, original_count, claimed
):
    init_log(PROFILING)
    path = tmp_path / "events.jsonl"
    profiling.register_event_output(path)
    event, payload = _bounded_sequence_payload(kind, original_count)
    if kind == "input_shapes":
        payload["input_shapes_truncated"] = claimed
    elif kind == "actual_steps":
        payload["actual_steps_truncated"] = claimed
    else:
        payload["shape_buckets_truncated"] = claimed

    with pytest.raises(ValueError, match="truncat"):
        profiling.record(event, **payload)

    profiling.close_event_output()
    assert path.read_text() == ""


def test_disabled_mps_hv_does_not_import_helpers_or_record(monkeypatch):
    backend_module = importlib.import_module("renormalizer.mps.backend")
    time_module = importlib.import_module("time")

    init_log(DEBUG)
    helper_module = "renormalizer.backend._execution.profiling"
    sys.modules.pop(helper_module, None)

    def fail_record(*args, **kwargs):
        raise AssertionError("disabled H-v attempted to record profiling evidence")

    def fail_disabled_work(*args, **kwargs):
        raise AssertionError("disabled H-v performed profiling-only work")

    monkeypatch.setattr(profiling, "record", fail_record)
    monkeypatch.setattr(time_module, "perf_counter", fail_disabled_work)
    monkeypatch.setattr(backend_module.backend, "current_device", fail_disabled_work)
    left, mpo, right, center = _single_site_operands()
    expression = hop_expr(left, right, [mpo], center.shape)

    actual = asnumpy(expression(backend_module.xp.asarray(center)))

    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert helper_module not in sys.modules


@pytest.mark.parametrize(
    "backend_name,device,timing_semantics,device_synchronized",
    [
        ("numpy", "cpu", "host_elapsed_synchronous", True),
        ("torch", "cpu", "host_elapsed_synchronous", True),
        ("jax", "cpu", "host_elapsed_unsynchronized", False),
        ("jax", "cuda:0", "host_elapsed_unsynchronized", False),
        ("cupy", "cuda:0", "host_elapsed_unsynchronized", False),
        ("torch", "cuda:0", "host_elapsed_unsynchronized", False),
        ("unknown", "cpu", "host_elapsed_unsynchronized", False),
    ],
)
def test_completion_timing_metadata_is_backend_aware(
    backend_name, device, timing_semantics, device_synchronized
):
    from renormalizer.backend._execution.profiling import completion_timing_metadata

    assert completion_timing_metadata(backend_name, device) == {
        "timing_semantics": timing_semantics,
        "device_synchronized": device_synchronized,
    }


def _preflight_cupy_cuda0():
    cupy = importlib.import_module("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count < 1:
            pytest.skip("CuPy CUDA device 0 is unavailable: no visible CUDA devices")
        with cupy.cuda.Device(0):
            assert cupy.cuda.runtime.getDevice() == 0
            cupy.empty(1)
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CuPy CUDA device 0 is unavailable: {error}")


def _select_cupy_backend():
    _preflight_cupy_cuda0()
    return set_backend("cupy", device="cuda:0", precision=64)


def test_cupy_selector_propagates_arbitrary_backend_runtime_error(monkeypatch):
    test_module = sys.modules[__name__]

    def fail_set_backend(*args, **kwargs):
        raise RuntimeError("adapter construction regression")

    def fail_if_skipped(reason):
        raise AssertionError(f"backend RuntimeError was converted to skip: {reason}")

    monkeypatch.setattr(test_module, "_preflight_cupy_cuda0", lambda: None)
    monkeypatch.setattr(test_module, "set_backend", fail_set_backend)
    monkeypatch.setattr(pytest, "skip", fail_if_skipped)

    with pytest.raises(RuntimeError, match="adapter construction regression"):
        _select_cupy_backend()


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_cupy_hv_timing_is_explicitly_unsynchronized(tmp_path, monkeypatch):
    selected = _select_cupy_backend()

    def fail_sync():
        raise AssertionError("profiling synchronized the backend")

    monkeypatch.setattr(selected, "sync", fail_sync)
    try:
        init_log(PROFILING)
        path = tmp_path / "cupy-local-events.jsonl"
        profiling.register_event_output(path)
        left, mpo, right, center = _single_site_operands()
        expression = hop_expr(left, right, [mpo], center.shape)

        result = expression(selected.asarray(center))
        profiling.close_event_output()

        assert isinstance(result, selected.device_array_types)
        event = json.loads(path.read_text().strip())
        assert event["backend"] == "cupy"
        assert event["device"] == "cuda:0"
        assert event["timing_semantics"] == "host_elapsed_unsynchronized"
        assert event["device_synchronized"] is False
    finally:
        profiling.close_event_output()
        set_backend("numpy", precision=64)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_jax_cpu_hv_timing_is_explicitly_unsynchronized(tmp_path):
    selected = set_backend("jax", device="cpu", precision=64)
    try:
        init_log(PROFILING)
        path = tmp_path / "jax-local-events.jsonl"
        profiling.register_event_output(path)
        left, mpo, right, center = _single_site_operands()
        expression = hop_expr(left, right, [mpo], center.shape)

        result = asnumpy(expression(selected.asarray(center)))
        profiling.close_event_output()

        expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        event = json.loads(path.read_text().strip())
        assert event["backend"] == "jax"
        assert event["device"] == "cpu"
        assert event["timing_semantics"] == "host_elapsed_unsynchronized"
        assert event["device_synchronized"] is False
    finally:
        profiling.close_event_output()
        set_backend("numpy", precision=64)


def test_qn_decomposition_emits_one_aggregate_phase_event(tmp_path):
    from renormalizer.mps.svd_qn import svd_qn

    init_log(PROFILING)
    path = tmp_path / "svd-events.jsonl"
    profiling.register_event_output(path)
    coefficient = np.arange(4, dtype=float).reshape(2, 2)
    qn_left = np.zeros((2, 1), dtype=int)
    qn_right = np.zeros((2, 1), dtype=int)

    svd_qn(coefficient, qn_left, qn_right, np.zeros(1, dtype=int), full_matrices=False)
    profiling.close_event_output()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "phase_summary"
    assert events[0]["phase"] == "qn_decomposition"
    assert events[0]["network"] == "tensor_network"
    assert events[0]["operation"] == "svd"
    assert events[0]["operation_count"] == 1
    assert events[0]["wall_s"] >= 0


def test_disabled_qn_decomposition_uses_public_body_without_profiling_work(monkeypatch):
    svd_module = importlib.import_module("renormalizer.mps.svd_qn")
    time_module = importlib.import_module("time")
    helper_module = "renormalizer.backend._execution.profiling"
    sys.modules.pop(helper_module, None)

    def fail_disabled_work(*args, **kwargs):
        raise AssertionError("disabled SVD performed profiling-only or wrapper work")

    monkeypatch.setattr(svd_module, "_svd_qn", fail_disabled_work, raising=False)
    monkeypatch.setattr(time_module, "perf_counter", fail_disabled_work)
    monkeypatch.setattr(profiling, "record", fail_disabled_work)
    coefficient = np.arange(4, dtype=float).reshape(2, 2)
    qn_left = np.zeros((2, 1), dtype=int)
    qn_right = np.zeros((2, 1), dtype=int)

    u, singular_values, _, v, _, _ = svd_module.svd_qn(
        coefficient,
        qn_left,
        qn_right,
        np.zeros(1, dtype=int),
        full_matrices=False,
    )

    np.testing.assert_allclose(u @ np.diag(singular_values) @ v.T, coefficient, atol=1e-12)
    assert helper_module not in sys.modules


def test_ttns_optimization_timer_has_truthful_phase_label(tmp_path, monkeypatch):
    gs_module = importlib.import_module("renormalizer.tn.gs")

    class OptimizeConfig:
        procedure = [(4, 0.1), (4, 0.0)]
        distributed_execution = None

    class FakeTTNS:
        optimize_config = OptimizeConfig()
        root = object()

    init_log(PROFILING)
    path = tmp_path / "ttns-environment-events.jsonl"
    profiling.register_event_output(path)
    monkeypatch.setattr(gs_module, "TTNEnviron", lambda ttns, ttno: object())
    monkeypatch.setattr(gs_module, "optimize_recursion", lambda *args: [1.0])

    assert gs_module.optimize_ttns(FakeTTNS(), object()) == [1.0, 1.0]
    profiling.close_event_output()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["phase"] for event in events] == [
        "environment_construction",
        "optimization_sweep",
        "optimization_sweep",
    ]
    assert [event["operation"] for event in events] == [
        "construct",
        "ttns_optimization_sweep",
        "ttns_optimization_sweep",
    ]
    assert all(event["event"] == "phase_summary" for event in events)
    assert all(event["network"] == "ttns" for event in events)
    assert [event["operation_count"] for event in events] == [1, 1, 1]


def _tiny_ttns_system():
    from renormalizer import BasisHalfSpin
    from renormalizer.model.model import heisenberg_ops
    from renormalizer.tn.node import TreeNodeBasis
    from renormalizer.tn.tree import TTNO, TTNS
    from renormalizer.tn.treebase import BasisTree

    root = TreeNodeBasis([BasisHalfSpin(0)])
    root.add_child(TreeNodeBasis([BasisHalfSpin(1)]))
    basis = BasisTree(root)
    return TTNS.random(basis, qntot=0, m_max=2), TTNO(basis, heisenberg_ops(2))


def test_real_enabled_ttns_hv_records_truthful_path_and_preserves_energy(tmp_path):
    from renormalizer.mps.matrix import asxp
    from renormalizer.tn.hop_expr import hop_expr1
    from renormalizer.tn.tree import TTNEnviron

    ttns, ttno = _tiny_ttns_system()
    expected_energy = ttns.expectation(ttno)
    environment = TTNEnviron(ttns, ttno)
    node = ttns.root
    init_log(PROFILING)
    path = tmp_path / "ttns-hv-events.jsonl"
    profiling.register_event_output(path, events={"local_hv_execute"})
    expression = hop_expr1(node, ttns, ttno, environment)

    result = asnumpy(expression(asxp(node.tensor)))
    profiling.close_event_output()

    np.testing.assert_allclose(np.vdot(node.tensor, result), expected_energy, atol=1e-12)
    event = json.loads(path.read_text().strip())
    assert event["network"] == "ttns"
    assert event["center_kind"] == "one_site"
    assert event["planner_source"] == "opt_einsum"
    assert event["requested_policy"] == event["actual_policy"] == "legacy_oe"
    assert len(event["oe_path_hash"]) == 64
    assert event["actual_steps"] == ["gemm", "tensordot", "tensordot"]
    assert event["timing_semantics"] == "host_elapsed_synchronous"
    assert event["device_synchronized"] is True


def test_real_enabled_ttns_optimization_has_truthful_phases_and_energy(tmp_path):
    from renormalizer.tn.gs import optimize_ttns

    ttns, ttno = _tiny_ttns_system()
    exact_energy = np.linalg.eigh(ttno.todense())[0][0]
    init_log(PROFILING)
    path = tmp_path / "ttns-optimization-events.jsonl"
    profiling.register_event_output(path, events={"phase_summary"})

    energies = optimize_ttns(ttns, ttno, procedure=[(2, 0)])
    profiling.close_event_output()

    np.testing.assert_allclose(energies[-1], exact_energy, atol=1e-12)
    np.testing.assert_allclose(ttns.expectation(ttno), exact_energy, atol=1e-12)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(event["phase"] == "environment_construction" for event in events)
    sweep_events = [event for event in events if event["phase"] == "optimization_sweep"]
    assert [event["operation"] for event in sweep_events] == ["ttns_optimization_sweep"]
    assert not any(event["phase"] == "environment_update" for event in events)
    assert any(
        event["phase"] == "qn_decomposition" and event["network"] == "tensor_network"
        for event in events
    )


def test_real_enabled_mps_optimization_uses_dmrg_sweep_label(tmp_path):
    from renormalizer import BasisHalfSpin, Model, Mpo, Mps
    from renormalizer.model.model import heisenberg_ops
    from renormalizer.mps.gs import optimize_mps

    model = Model([BasisHalfSpin(0), BasisHalfSpin(1)], heisenberg_ops(2))
    mps = Mps.random(model, qntot=0, m_max=2)
    mpo = Mpo(model)
    mps.optimize_config.procedure = [(2, 0), (2, 0)]
    exact_energy = np.linalg.eigh(mpo.todense())[0][0]
    init_log(PROFILING)
    path = tmp_path / "mps-optimization-events.jsonl"
    profiling.register_event_output(path, events={"phase_summary"})

    energies, result = optimize_mps(mps, mpo)
    profiling.close_event_output()

    np.testing.assert_allclose(min(energies), exact_energy, atol=1e-12)
    np.testing.assert_allclose(result.expectation(mpo), exact_energy, atol=1e-12)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    sweep_events = [event for event in events if event["phase"] == "optimization_sweep"]
    assert [event["operation"] for event in sweep_events] == ["dmrg_sweep", "dmrg_sweep"]
    assert not any(event["phase"] == "environment_update" for event in events)


def test_tdvp_whole_sweep_timer_has_truthful_phase_label(tmp_path, monkeypatch):
    evolution = importlib.import_module("renormalizer.tn.time_evolution")

    class FakeTTNS:
        root = object()
        evolve_config = type("EvolveConfig", (), {"distributed_execution": None})()

        def check_canonical(self):
            return None

    fake_ttns = FakeTTNS()
    monkeypatch.setattr(evolution, "TTNEnviron", lambda ttns, ttno: object())
    monkeypatch.setattr(evolution, "_tdvp_ps_forward", lambda *args: [1, 2])
    monkeypatch.setattr(evolution, "_tdvp_ps_backward", lambda *args: [3, 4])
    init_log(PROFILING)
    path = tmp_path / "tdvp-phase-events.jsonl"
    profiling.register_event_output(path, events={"phase_summary"})

    assert evolution.evolve_tdvp_ps(fake_ttns, object(), 1.0, 0.1) is fake_ttns
    profiling.close_event_output()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["phase"] for event in events] == [
        "environment_construction",
        "tdvp_sweep",
        "tdvp_sweep",
    ]
    assert [event["operation"] for event in events] == [
        "construct",
        "tdvp_forward_sweep",
        "tdvp_backward_sweep",
    ]
