# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Deterministic Stage 4 TTNS multi-GPU validation job."""

import argparse
import hashlib
import json
import os

import numpy as np

from renormalizer.backend._distributed.job import (
    RANK_SCHEMA,
    SCHEMA,
    _add_common_arguments,
    _aggregate_norm_error,
    _arm_allgather_trap,
    _finalize_profile,
    _host,
    _load_rank_records,
    _max_scalar_spread,
    _numerical_passed,
    _prepare_output,
    _profile_distribution,
    _run_root_reference,
    _summary_distribution,
    _summary_numerical,
    _synchronized_action,
    _synchronized_root_action,
    _trapped_execution_config,
    _validate_job_tolerances,
    _validate_rank_records,
    _write_json,
)


def build_parser():
    parser = _add_common_arguments(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument("--tdvp-sites", choices=("one", "two"), default="one")
    return parser


def _tree_model(args):
    from renormalizer import BasisHalfSpin
    from renormalizer.model.model import heisenberg_ops
    from renormalizer.tn import BasisTree, TTNO, TTNS

    if args.model_size < 4:
        raise ValueError("TTNS model-size must be at least 4")
    np.random.seed(args.seed)
    basis = BasisTree.binary(
        [BasisHalfSpin(index) for index in range(args.model_size)]
    )
    state = TTNS.random(basis, 0, args.bond_dimension, 1)
    return state, TTNO(basis, heisenberg_ops(args.model_size))


def _state_arrays(state):
    tensors = [_host(node.tensor) for node in state]
    qn_arrays = [np.asarray(node.qn) for node in state]
    return tensors, qn_arrays


def _state_hash(state):
    digest = hashlib.sha256()
    tensors, qn_arrays = _state_arrays(state)
    for value in tensors + qn_arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _max_tensor_error(left, right):
    return max(
        float(np.max(np.abs(_host(a.tensor) - _host(b.tensor))))
        for a, b in zip(left, right)
    )


def _run_tdvp(args, initial, operator, execution):
    from renormalizer.cons import backend
    from renormalizer.utils import EvolveConfig, EvolveMethod

    def setup():
        method = (
            EvolveMethod.tdvp_ps
            if args.tdvp_sites == "one"
            else EvolveMethod.tdvp_ps2
        )
        distributed = initial.copy()
        distributed.evolve_config = EvolveConfig(method)
        selected = backend.current
        tensor_count = sum(int(node.tensor.size) for node in initial)
        reference_receive = selected.array_namespace.empty(
            tensor_count + 1, dtype=np.complex128
        )
        return method, distributed, selected, reference_receive

    method, distributed, selected, reference_receive = _synchronized_action(
        execution, setup, "TTNS TDVP workflow setup"
    )

    def reference_operation():
        reference = initial.copy()
        reference.evolve_config = EvolveConfig(method)
        reference = reference.evolve(operator, args.tau, normalize=False)
        flattened = [
            selected.asarray(
                node.tensor, dtype=reference_receive.dtype
            ).reshape(-1)
            for node in reference
        ]
        norm = selected.asarray([reference.ttns_norm], dtype=np.complex128)
        return selected.array_namespace.ascontiguousarray(
            selected.array_namespace.concatenate([*flattened, norm])
        )

    reference_values = _run_root_reference(
        execution, reference_operation, reference_receive
    )
    _synchronized_action(
        execution,
        lambda: setattr(distributed.evolve_config, "distributed_execution", execution),
        "TTNS TDVP distributed config",
    )
    distributed = distributed.evolve(operator, args.tau, normalize=False)

    def compare_results():
        distributed_values = np.concatenate(
            [_host(node.tensor).reshape(-1) for node in distributed]
        )
        reference_host = _host(reference_values)
        return {
            "max_abs_error": float(
                np.max(np.abs(distributed_values - reference_host[:-1]))
            ),
            "tdvp_norm_error": float(
                abs(distributed.ttns_norm - reference_host[-1].real)
            ),
        }

    return distributed, _synchronized_action(
        execution, compare_results, "TTNS TDVP result allocation"
    )


def _run_davidson(args, initial, operator, execution):
    from renormalizer.cons import backend
    from renormalizer.tn.gs import optimize_ttns

    def setup():
        procedure = [
            [args.bond_dimension, 0] for _ in range(args.davidson_sweeps)
        ]
        distributed = initial.copy()
        selected = backend.current
        reference_receive = selected.array_namespace.empty(2, dtype=np.float64)
        return procedure, distributed, selected, reference_receive

    procedure, distributed, selected, reference_receive = _synchronized_action(
        execution, setup, "TTNS Davidson workflow setup"
    )

    def reference_operation():
        reference = initial.copy()
        reference_energies = optimize_ttns(reference, operator, procedure)
        return selected.asarray(
            [min(reference_energies), reference.ttns_norm], dtype=np.float64
        )

    reference_values = _host(
        _run_root_reference(execution, reference_operation, reference_receive)
    )
    reference_energy = float(reference_values[0])
    reference_norm = float(reference_values[1])
    def attach_config():
        distributed.optimize_config.algo = "davidson"
        distributed.optimize_config.nroots = 1
        distributed.optimize_config.distributed_execution = execution

    _synchronized_action(
        execution, attach_config, "TTNS Davidson distributed config"
    )
    distributed_energies = optimize_ttns(distributed, operator, procedure)

    def summarize_result():
        energy = float(min(distributed_energies))
        expectation = float(np.real(distributed.expectation(operator)))
        return {
            "energy": energy,
            "energy_error": abs(energy - reference_energy),
            "residual_norm": abs(expectation - energy),
            "davidson_norm_error": abs(float(distributed.ttns_norm) - 1.0),
            "davidson_reference_norm": reference_norm,
        }

    return distributed, _synchronized_action(
        execution, summarize_result, "TTNS Davidson result allocation"
    )


def _root_summary(args, runtime, output_dir, rank_record):
    records, reasons = _load_rank_records(output_dir)
    rank_reasons = _validate_rank_records(
        records, rank_record, runtime.world_size
    )
    reasons.extend(rank_reasons)
    hashes = {record.get("state_hash") for record in records}
    numerical_summary = _summary_numerical(records, args)
    numerical_passed = (
        numerical_summary["passed"] and len(records) == runtime.world_size
    )
    if not numerical_passed:
        reasons.append("numerical tolerance failure")
    scalar_spread = _max_scalar_spread(records)
    exact_state_hash = (
        len(records) == runtime.world_size and len(hashes) == 1
    )
    rank_consistent = (
        not rank_reasons
        and exact_state_hash
        and np.isfinite(scalar_spread)
        and scalar_spread <= args.rank_atol
    )
    if not np.isfinite(scalar_spread) or scalar_spread > args.rank_atol:
        reasons.append("rank scalar spread exceeds tolerance")
    fallback_count = max(
        (int(record.get("fallback", {}).get("count", 0)) for record in records),
        default=0,
    )
    root_operation_count = sum(
        int(record.get("fallback", {}).get("root_operation_count", 0))
        for record in records
    )
    overall_passed = not reasons and numerical_passed and rank_consistent
    return {
        "schema": SCHEMA,
        "status": "pass" if overall_passed else "fail",
        "network": "ttns",
        "mode": args.mode,
        "world_size": runtime.world_size,
        "git_commit": rank_record["git_commit"],
        "backend": {"name": "cupy", "precision": args.precision, "execution_policy": "execution_ir"},
        "model": {"kind": "heisenberg", "size": args.model_size, "bond_dimension": args.bond_dimension, "seed": args.seed},
        "solver": rank_record["solver"],
        "tolerances": {"atol": args.atol, "rtol": args.rtol, "rank_atol": args.rank_atol},
        "numerical": {**numerical_summary, "passed": numerical_passed},
        "rank_consistency": {"passed": rank_consistent, "exact_state_hash": exact_state_hash, "max_scalar_spread": scalar_spread},
        "distribution": _summary_distribution(
            records, args.shard_solver_vectors
        ),
        "fallback": {"count": fallback_count, "root_operation_count": root_operation_count, "synchronized": True},
        "ranks": records,
        "reasons": reasons,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.expected_world_size not in {1, 2, 4, 8}:
        raise ValueError("expected-world-size must be one of 1, 2, 4, 8")
    if not args.shard_solver_vectors:
        raise ValueError("formal Stage 4 jobs require --shard-solver-vectors")

    from renormalizer import set_backend
    from renormalizer.backend.config import BackendConfig
    from renormalizer.backend.distributed_runtime import cupy_distributed_runtime
    from renormalizer.cons import get_git_commit_hash
    from renormalizer.tn.distributed import synchronize_ttns_update
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING, init_log

    local_rank = int(os.environ["LOCAL_RANK"])
    with cupy_distributed_runtime(
        precision=args.precision,
        expected_world_size=args.expected_world_size,
    ) as runtime:
        _validate_job_tolerances(runtime, args)
        _synchronized_action(
            runtime,
            lambda: set_backend(
                "cupy",
                config=BackendConfig(
                    device="cuda:{}".format(local_rank),
                    precision=args.precision,
                    execution_policy="execution_ir",
                    fallback_policy=args.fallback_policy,
                ),
            ),
            "backend configuration",
        )
        output_dir = _prepare_output(args.output_dir, runtime)
        profile_path = output_dir / "profile-rank-{:05d}.jsonl".format(runtime.rank)
        def setup_profile():
            init_log(PROFILING)
            profiling.register_event_output(profile_path)
            profiling.record(
                "run_start",
                backend="cupy",
                rank=runtime.rank,
                world_size=runtime.world_size,
                network="ttns",
                mode=args.mode,
                model_seed=args.seed,
                requested_policy="execution_ir",
            )

        _synchronized_action(
            runtime, setup_profile, "profile setup"
        )
        execution = _synchronized_action(
            runtime,
            lambda: _trapped_execution_config(
                runtime.execution_config(),
                solver_options={
                    "krylov": {"block_size": args.krylov_block_size},
                    "davidson": {
                        "tol": args.davidson_tol,
                        "max_cycle": args.davidson_max_cycle,
                        "max_space": args.davidson_max_space,
                        "lindep": args.davidson_lindep,
                    },
                },
            ),
            "distributed execution setup",
        )
        initial, operator = _synchronized_action(
            runtime, lambda: _tree_model(args), "model setup"
        )
        final_state = initial
        numerical = {}
        if args.mode in {"tdvp", "both"}:
            final_state, result = _run_tdvp(args, initial, operator, execution)
            numerical.update(result)
        if args.mode in {"davidson", "both"}:
            final_state, result = _run_davidson(args, initial, operator, execution)
            numerical.update(result)
        tensors, qn_arrays, state_hash = _synchronized_action(
            runtime,
            lambda: (*_state_arrays(final_state), _state_hash(final_state)),
            "final state evidence allocation",
        )
        synchronize_ttns_update(
            execution,
            tensors,
            qn_arrays=qn_arrays,
            metadata=(args.mode, len(tensors)),
        )
        distribution = _finalize_profile(runtime, profile_path)
        distribution["allgather_trap_armed"] = True
        if args.mode in {"davidson", "both"}:
            numerical["residual_norm"] = distribution["max_solver_residual_norm"]
        numerical["norm_error"] = _synchronized_action(
            runtime,
            lambda: _aggregate_norm_error(numerical, args.mode),
            "numerical norm aggregation",
        )
        numerical["passed"] = _numerical_passed(numerical, args)
        def build_rank_record():
            commit = get_git_commit_hash()
            return {
                "schema": RANK_SCHEMA,
                "run_id": "{}:{}:{}:{}".format(
                    commit, args.mode, args.seed, runtime.world_size
                ),
                "rank": runtime.rank,
                "local_rank": runtime.local_rank,
                "world_size": runtime.world_size,
                "device": "cuda:{}".format(runtime.local_rank),
                "git_commit": commit,
                "state_hash": state_hash,
                "numerical": numerical,
                "distribution": distribution,
                "fallback": {
                    "count": distribution["fallback_count"],
                    "root_operation_count": distribution["root_operation_count"],
                },
                "solver": {
                    "krylov_block_size": args.krylov_block_size,
                    "davidson_tol": args.davidson_tol,
                    "davidson_max_cycle": args.davidson_max_cycle,
                    "davidson_max_space": args.davidson_max_space,
                    "davidson_lindep": args.davidson_lindep,
                    "davidson_sweeps": args.davidson_sweeps,
                    "davidson_sites": "two",
                },
            }

        rank_record = _synchronized_action(
            runtime, build_rank_record, "rank evidence allocation"
        )
        _synchronized_action(
            runtime,
            lambda: _write_json(
                output_dir / "rank-{:05d}.json".format(runtime.rank),
                rank_record,
            ),
            "rank evidence write",
        )

        def write_summary():
            _write_json(
                output_dir / "summary.json",
                _root_summary(args, runtime, output_dir, rank_record),
            )

        _synchronized_root_action(runtime, write_summary, "summary aggregation")
        status = _synchronized_action(
            runtime,
            lambda: json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )["status"],
            "summary read",
        )
        _synchronized_action(
            runtime, profiling.close_event_output, "profile close"
        )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
