import json
import logging


def _jsonl_payloads(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_tree_runtime_events_write_jsonl(caplog, tmp_path):
    from renormalizer import BasisHalfSpin, Op
    from renormalizer.tn import BasisTree, TTNO, TTNS
    from renormalizer.tn.tree import TTNEnviron
    from renormalizer.utils import EvolveConfig, EvolveMethod, profiling
    from renormalizer.utils.log import PROFILING

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    basis = BasisTree.binary([BasisHalfSpin(0), BasisHalfSpin(1)])
    try:
        ttns = TTNS(basis, condition={})
        ttno = TTNO(basis, [Op("X", 0)])
        copied = ttns.copy()
        TTNEnviron(copied, ttno)
        copied.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
        copied.evolve(ttno, 0.01)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    payloads = _jsonl_payloads(event_path)
    events = [payload["event"] for payload in payloads]
    assert "ttno_build_summary" in events
    assert "tree_copy_connection" in events
    assert "ttns_copy" in events
    assert "ttn_environ_build" in events
    assert "ttn_environ_build_children_node" in events
    assert "ttn_environ_build_parent_node" in events
    assert "ttn_evolve_1site" in events
    assert "tn_hop_expr1" in events

    build = next(payload for payload in payloads if payload["event"] == "ttn_environ_build")
    assert build["node_count"] == 2
    assert build["edge_count"] == 1
    assert build["wall_s"] >= 0

    node = next(payload for payload in payloads if payload["event"] == "ttn_evolve_1site")
    assert isinstance(node["node_idx"], int)
    assert node["node_shape"]
    assert node["krylov_steps"] >= 1
    assert node["wall_s"] >= 0
