# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
import subprocess
import sys


def test_default_example_cases_match_example_run_sh_order():
    from renormalizer.backend.example_benchmark import default_example_cases

    cases = default_example_cases()

    assert [case.name for case in cases] == [
        "fmo",
        "sbm",
        "h2o_qc",
        "dynamics_std",
        "transport_kubo_std",
        "ttns_junction_zt",
        "ttns_junction_ft",
        "ttns_sbm_zt",
        "ttns_sbm_ft",
        "ssh",
    ]
    assert cases[0].script.as_posix() == "example/fmo.py"
    assert cases[3].args == ("std.yaml",)
    assert cases[4].args == ("std.yaml",)
    assert cases[6].args == ("32", "1", "100")
    assert cases[7].args == ("050", "001", "050")
    assert cases[0].assets == ("fmo_sdf.json",)
    assert cases[2].assets == ("h2o_fcidump.txt",)
    assert cases[3].assets == ("std.yaml",)
    assert cases[4].assets == ("std.yaml",)
    assert cases[0].expected_steps == 251
    assert cases[1].expected_steps == 201
    assert cases[3].expected_steps == 301
    assert cases[8].expected_steps == 400


def test_select_cases_accepts_non_default_sbm_probe():
    from renormalizer.backend.example_benchmark import _select_cases, default_example_cases

    assert "sbm_probe" not in [case.name for case in default_example_cases()]

    cases = _select_cases("sbm_probe")

    assert [case.name for case in cases] == ["sbm_probe"]
    assert cases[0].script.as_posix() == "renormalizer/backend/probes/sbm_probe.py"
    assert cases[0].expected_steps == 6
    assert cases[0].notes.startswith("Short SBM profiling probe")


def test_select_cases_accepts_non_default_holstein_multistate_probe():
    from renormalizer.backend.example_benchmark import _select_cases, default_example_cases, repo_root

    assert "holstein_multistate_probe" not in [case.name for case in default_example_cases()]

    cases = _select_cases("holstein_multistate_probe")

    assert [case.name for case in cases] == ["holstein_multistate_probe"]
    assert cases[0].script.as_posix() == "renormalizer/backend/probes/holstein_multistate_probe.py"
    assert (repo_root() / cases[0].script).exists()
    assert cases[0].notes.startswith("Multi-state Holstein DMRG probe")


def test_make_cpu_jobs_expands_all_cases_for_1_and_48_cores(tmp_path):
    from renormalizer.backend.example_benchmark import default_example_cases, make_cpu_jobs

    jobs = make_cpu_jobs(default_example_cases(), tmp_path, threads=(1, 48))

    assert len(jobs) == 20
    assert {(job.case.name, job.threads) for job in jobs} == {
        (case.name, threads)
        for case in default_example_cases()
        for threads in (1, 48)
    }
    assert all(job.backend == "numpy" for job in jobs)
    assert all(job.device == "cpu" for job in jobs)
    assert all(job.suite == "cpu" for job in jobs)
    assert jobs[0].run_dir == tmp_path / "cpu" / "numpy" / "threads-1" / "fmo"
    assert jobs[1].run_dir == tmp_path / "cpu" / "numpy" / "threads-48" / "fmo"


def test_make_cupy_jobs_uses_single_card_backend_and_thread_count(tmp_path):
    from renormalizer.backend.example_benchmark import default_example_cases, make_cupy_jobs

    jobs = make_cupy_jobs(
        default_example_cases()[:2],
        tmp_path,
        backend="cupy",
        device="cuda:0",
        threads=48,
    )

    assert [(job.case.name, job.backend, job.device, job.threads) for job in jobs] == [
        ("fmo", "cupy", "cuda:0", 48),
        ("sbm", "cupy", "cuda:0", 48),
    ]
    assert jobs[0].suite == "cupy"
    assert jobs[0].run_dir == tmp_path / "cupy" / "cupy" / "cuda-0" / "threads-48" / "fmo"


def test_thread_env_sets_all_relevant_cpu_thread_controls():
    from renormalizer.backend.example_benchmark import build_thread_env

    env = build_thread_env(48, base_env={"PATH": "/bin", "OMP_NUM_THREADS": "1"})

    assert env["PATH"] == "/bin"
    for name in (
        "RENO_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert env[name] == "48"


def test_default_child_command_does_not_enable_full_profile_events(tmp_path):
    from renormalizer.backend.example_benchmark import (
        default_example_cases,
        make_cpu_jobs,
        build_child_command,
        build_job_env,
    )

    job = make_cpu_jobs(default_example_cases()[:1], tmp_path, threads=(1,))[0]
    command = build_child_command(job)
    env = build_job_env(job, base_env={"PYTHONPATH": "/x"})

    assert command[:3] == [sys.executable, "-m", "renormalizer.backend.example_runner"]
    assert "--backend" in command
    assert command[command.index("--backend") + 1] == "numpy"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--cwd") + 1] == str(job.run_dir)
    assert command[command.index("--script") + 1].endswith("example/fmo.py")
    assert "--profile-events" not in command
    assert env.get("RENO_LOG_LEVEL") != "PROFILING"
    assert command[-1:] == ["--"]


def test_full_profile_child_command_passes_event_path_and_profile_log_level(tmp_path):
    from renormalizer.backend.example_benchmark import (
        default_example_cases,
        make_cpu_jobs,
        build_child_command,
        build_job_env,
    )

    job = make_cpu_jobs(default_example_cases()[:1], tmp_path, threads=(1,), profile_mode="full")[0]
    command = build_child_command(job)
    env = build_job_env(job, base_env={})

    assert command[command.index("--profile-events") + 1] == str(job.profile_events_path)
    assert env["RENO_LOG_LEVEL"] == "PROFILING"
    assert command[-1:] == ["--"]


def test_parse_proc_status_extracts_resource_fields():
    from renormalizer.backend.example_benchmark import parse_proc_status

    sample = """
Name:\tpython
VmRSS:\t  123456 kB
VmHWM:\t  234567 kB
Threads:\t17
voluntary_ctxt_switches:\t91
nonvoluntary_ctxt_switches:\t7
"""

    parsed = parse_proc_status(sample)

    assert parsed == {
        "rss_kb": 123456,
        "hwm_kb": 234567,
        "threads": 17,
        "voluntary_ctxt_switches": 91,
        "nonvoluntary_ctxt_switches": 7,
    }


def test_collect_profile_summary_counts_events_and_wall_time(tmp_path):
    from renormalizer.backend.example_benchmark import collect_profile_summary

    profile_path = tmp_path / "profile-events.jsonl"
    profile_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "tensordot", "wall_s": 1.25}),
                json.dumps({"event": "svd_qn", "wall_s": 2.0}),
                json.dumps({"event": "tensordot", "wall_s": 0.75}),
            ]
        ),
        encoding="utf-8",
    )

    summary = collect_profile_summary(profile_path)

    assert summary["total_events"] == 3
    assert summary["event_counts"] == {"tensordot": 2, "svd_qn": 1}
    assert summary["wall_s_by_event"] == {"tensordot": 2.0, "svd_qn": 2.0}


def test_collect_profile_summary_extracts_root_cause_profile(tmp_path):
    from renormalizer.backend.example_benchmark import collect_profile_summary

    profile_path = tmp_path / "profile-events.jsonl"
    profile_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "tensordot",
                        "wall_s": 0.01,
                        "algorithmic_kernel": "gemm",
                        "m": 12,
                        "n": 48,
                        "k": 12,
                        "arithmetic_intensity_bin": "low",
                        "bottleneck_hints": ["small_gemm", "skinny_gemm", "memory_bandwidth"],
                    }
                ),
                json.dumps(
                    {
                        "event": "contraction_execute",
                        "wall_s": 0.02,
                        "lowering": "gemm",
                    }
                ),
                json.dumps(
                    {
                        "event": "grouped_gemm_execute",
                        "wall_s": 0.03,
                        "pack_strategy": "stack_per_call",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    summary = collect_profile_summary(profile_path)
    analysis = summary["profile_analysis"]

    assert analysis["tensordot_total"] == 1
    assert analysis["tensordot_kernel_counts"] == {"gemm": 1}
    assert analysis["small_gemm_count"] == 1
    assert analysis["small_gemm_pct"] == 1.0
    assert analysis["skinny_gemm_count"] == 1
    assert analysis["low_arithmetic_intensity_count"] == 1
    assert analysis["contraction_lowerings"] == {"gemm": 1}
    assert analysis["grouped_gemm_execute_count"] == 1
    assert analysis["grouped_pack_strategies"] == {"stack_per_call": 1}
    assert analysis["top_tensordot_shapes"] == [{"shape_mnk": "12,48,12", "count": 1}]


def test_grouped_gemm_diagnostic_replays_profile_shapes(tmp_path):
    from renormalizer.backend.example_benchmark import (
        extract_tensordot_mnk_shapes,
        run_grouped_gemm_diagnostic_from_shapes,
    )

    profile_path = tmp_path / "profile-events.jsonl"
    profile_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "tensordot", "m": 2, "n": 2, "k": 2}),
                json.dumps({"event": "tensordot", "m": 2, "n": 2, "k": 2}),
                json.dumps({"event": "tensordot", "m": 3, "n": 1, "k": 2}),
                json.dumps({"event": "svd_qn", "shape": [2, 2]}),
            ]
        ),
        encoding="utf-8",
    )

    shapes = extract_tensordot_mnk_shapes(profile_path)
    diagnostic = run_grouped_gemm_diagnostic_from_shapes(
        shapes,
        repeats=1,
        seed=1,
        pack_threshold=2,
    )

    assert shapes == [(2, 2, 2), (2, 2, 2), (3, 1, 2)]
    assert diagnostic["task_count"] == 3
    assert diagnostic["unique_shape_count"] == 2
    assert set(diagnostic["timings"]) == {
        "loop",
        "default_grouped",
        "forced_stack_grouped",
        "forced_prepacked_grouped",
    }
    assert diagnostic["timings"]["loop"]["median_s"] >= 0.0
    assert diagnostic["timings"]["default_grouped"]["profile"]["kernel_calls"] >= 1
    assert diagnostic["timings"]["forced_stack_grouped"]["profile"]["pack_strategy"] in {
        "none",
        "stack_per_call",
    }
    assert diagnostic["timings"]["forced_prepacked_grouped"]["profile"]["prepacked"] is True


def test_compare_cpu_results_computes_speedup_efficiency_and_labels():
    from renormalizer.backend.example_benchmark import compare_cpu_results

    rows = compare_cpu_results(
        [
            {"case": "fmo", "backend": "numpy", "suite": "cpu", "threads": 1, "status": "passed", "wall_s": 480.0},
            {"case": "fmo", "backend": "numpy", "suite": "cpu", "threads": 48, "status": "passed", "wall_s": 20.0},
            {"case": "sbm", "backend": "numpy", "suite": "cpu", "threads": 1, "status": "passed", "wall_s": 48.0},
            {"case": "sbm", "backend": "numpy", "suite": "cpu", "threads": 48, "status": "passed", "wall_s": 8.0},
            {"case": "ssh", "backend": "numpy", "suite": "cpu", "threads": 1, "status": "failed", "wall_s": 1.0},
        ]
    )

    by_case = {row["case"]: row for row in rows}
    assert by_case["fmo"]["speedup_48_vs_1"] == 24.0
    assert by_case["fmo"]["efficiency_48"] == 0.5
    assert by_case["fmo"]["status"] == "good"
    assert by_case["sbm"]["speedup_48_vs_1"] == 6.0
    assert by_case["sbm"]["efficiency_48"] == 0.125
    assert by_case["sbm"]["status"] == "poor"
    assert by_case["ssh"]["status"] == "incomplete"


def test_summarize_step_progress_extracts_completed_steps_and_throughput(tmp_path):
    from renormalizer.backend.example_benchmark import summarize_step_progress

    stderr = tmp_path / "stderr.log"
    stderr.write_text(
        "\n".join(
            [
                "2026-07-05[INFO] step 1 complete, time cost 0:00:12.500000.",
                "2026-07-05[INFO] step 2 complete, time cost 0:02:00.250000.",
                "2026-07-05[INFO] step 3/200, at time 0.2/20.0 begin.",
            ]
        ),
        encoding="utf-8",
    )

    progress = summarize_step_progress(stderr, expected_steps=200, wall_s=300.0)

    assert progress["completed_steps"] == 2
    assert progress["last_step"] == 2
    assert progress["expected_steps"] == 200
    assert progress["progress_fraction"] == 0.01
    assert progress["mean_logged_step_s"] == 66.375
    assert progress["step_throughput_per_s"] == 2 / 300.0


def test_compare_cpu_results_uses_sampled_progress_throughput():
    from renormalizer.backend.example_benchmark import compare_cpu_results

    rows = compare_cpu_results(
        [
            {
                "case": "sbm",
                "backend": "numpy",
                "suite": "cpu",
                "threads": 1,
                "status": "sampled",
                "wall_s": 3600.0,
                "progress": {"completed_steps": 30, "step_throughput_per_s": 30 / 3600.0},
            },
            {
                "case": "sbm",
                "backend": "numpy",
                "suite": "cpu",
                "threads": 48,
                "status": "sampled",
                "wall_s": 1800.0,
                "progress": {"completed_steps": 90, "step_throughput_per_s": 90 / 1800.0},
            },
        ]
    )

    assert rows[0]["metric_basis"] == "progress_throughput"
    assert rows[0]["speedup_48_vs_1"] == 6.0
    assert rows[0]["efficiency_48"] == 0.125
    assert rows[0]["status"] == "poor"


def test_timeout_can_be_recorded_as_sampled_progress(tmp_path):
    from renormalizer.backend.example_benchmark import (
        ExampleCase,
        BenchmarkJob,
        run_job,
    )

    root = Path.cwd()
    script = tmp_path / "slow.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "\n".join(
            [
                "import sys, time",
                "print('step 1 complete, time cost 0:00:00.010000.', file=sys.stderr, flush=True)",
                "time.sleep(10)",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    job = BenchmarkJob(
        ExampleCase("slow", script, expected_steps=5),
        "cpu",
        "numpy",
        "cpu",
        1,
        run_dir,
        root,
    )

    record = run_job(
        job,
        sample_interval=0.05,
        timeout=2.0,
        timeout_status="sampled",
        metadata={"git_commit": "test"},
    )

    assert record["timed_out"] is True
    assert record["status"] == "sampled"
    assert record["progress"]["completed_steps"] == 1
    assert record["progress"]["last_step"] == 1


def test_run_parser_accepts_sampled_timeout_status():
    from renormalizer.backend.example_benchmark import _make_parser

    parser = _make_parser()
    args = parser.parse_args(
        [
            "run",
            "--suite",
            "cpu",
            "--timeout",
            "1800",
            "--timeout-status",
            "sampled",
        ]
    )

    assert args.timeout == 1800
    assert args.timeout_status == "sampled"


def test_generate_html_report_writes_environment_results_and_profile_summary(tmp_path):
    from renormalizer.backend.example_benchmark import generate_html_report

    output = tmp_path / "report.html"
    records = [
        {
            "case": "fmo",
            "suite": "cpu",
            "backend": "numpy",
            "device": "cpu",
            "threads": 1,
            "status": "passed",
            "wall_s": 480.0,
            "returncode": 0,
            "max_rss_kb": 1024,
            "max_threads": 1,
            "profile_summary": {
                "total_events": 10,
                "event_counts": {"tensordot": 10},
                "profile_analysis": {
                    "tensordot_total": 10,
                    "small_gemm_pct": 1.0,
                    "low_arithmetic_intensity_pct": 0.9,
                    "tensordot_kernel_counts": {"gemm": 8, "gemv": 2},
                    "contraction_lowerings": {"gemm": 10},
                    "grouped_gemm_execute_count": 0,
                    "grouped_pack_strategies": {},
                    "top_tensordot_shapes": [{"shape_mnk": "12,48,12", "count": 4}],
                },
            },
            "progress": {"completed_steps": 200, "step_throughput_per_s": 200 / 480.0},
        },
        {
            "case": "fmo",
            "suite": "cpu",
            "backend": "numpy",
            "device": "cpu",
            "threads": 48,
            "status": "passed",
            "wall_s": 20.0,
            "returncode": 0,
            "max_rss_kb": 2048,
            "max_threads": 48,
            "profile_summary": {"event_counts": {"tensordot": 12}},
            "progress": {"completed_steps": 200, "step_throughput_per_s": 10.0},
        },
    ]

    generate_html_report(
        records,
        output,
        metadata={
            "git_commit": "abc123",
            "python": "3.11.x",
            "dependencies": {"numpy": "2.0.0"},
        },
        title="NumPy2 CPU Example Benchmark",
    )

    html = output.read_text(encoding="utf-8")
    assert "NumPy2 CPU Example Benchmark" in html
    assert "abc123" in html
    assert "numpy" in html
    assert "fmo" in html
    assert "24.00" in html
    assert "0.50" in html
    assert "completion_wall_time" in html
    assert "completed_steps" in html
    assert "Max RSS MiB" in html
    assert "Max threads" in html
    assert "tensordot" in html
    assert "Profile Root Cause Summary" in html
    assert "Small GEMM" in html
    assert "100.0%" in html
    assert "12,48,12 x4" in html


def test_generate_sbm_batched_grouped_report_separates_path_and_performance(tmp_path):
    from renormalizer.backend.example_benchmark import generate_sbm_batched_grouped_report

    stage_a = tmp_path / "grouped-events-summary.json"
    stage_a.write_text(
        json.dumps(
            {
                "grouped_gemm_execute_count": 2,
                "events": [
                    {
                        "num_tasks": 301,
                        "num_groups": 3,
                        "group_sizes": [1, 299, 1],
                        "policy": "bucketed_loop_matmul",
                        "batched_kernel_calls": 0,
                        "loop_kernel_calls": 301,
                        "bucket_fallback_reasons": [
                            "task_count below pack_threshold",
                            "flops below copy threshold",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage_b = tmp_path / "results.jsonl"
    records = [
        {
            "case": "sbm",
            "suite": "cpu",
            "backend": "numpy",
            "device": "cpu",
            "threads": 1,
            "status": "sampled",
            "wall_s": 120.0,
            "max_rss_kb": 1024,
            "max_threads": 1,
            "command": ["python", "example/sbm.py"],
            "thread_env": {"OMP_NUM_THREADS": "1"},
            "progress": {
                "completed_steps": 14,
                "step_throughput_per_s": 0.116,
                "mean_logged_step_s": 7.9,
            },
        },
        {
            "case": "sbm",
            "suite": "cpu",
            "backend": "numpy",
            "device": "cpu",
            "threads": 48,
            "status": "sampled",
            "wall_s": 120.0,
            "max_rss_kb": 2048,
            "max_threads": 95,
            "command": ["python", "example/sbm.py"],
            "thread_env": {"OMP_NUM_THREADS": "48"},
            "progress": {
                "completed_steps": 14,
                "step_throughput_per_s": 0.116,
                "mean_logged_step_s": 7.9,
            },
        },
    ]
    stage_b.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    output = tmp_path / "sbm_report.html"

    summary = generate_sbm_batched_grouped_report(
        stage_a,
        stage_b,
        output,
        metadata={"git_commit": "abc123", "dependencies": {"numpy": "2.2.6"}},
    )

    html = output.read_text(encoding="utf-8")
    assert summary["classification"] == "entered grouped path, but CPU thread scaling remains poor"
    assert summary["stage_a"]["grouped_gemm_execute_count"] == 2
    assert summary["stage_b"]["speedup_48_vs_1"] == 1.0
    assert "Stage A: Path Evidence" in html
    assert "Stage B: Performance Evidence" in html
    assert "grouped_gemm_execute events" in html
    assert "bucketed_loop_matmul" in html
    assert "batched kernel calls" in html
    assert "loop kernel calls" in html
    assert "48 vs 1 speedup" in html
    assert "Benchmark command and thread environment" in html
    assert "abc123" in html


def test_example_runner_keeps_profile_enabled_when_example_resets_log_level(tmp_path):
    script_path = tmp_path / "example_resets_log.py"
    run_dir = tmp_path / "run"
    event_path = run_dir / "profile-events.jsonl"
    run_dir.mkdir()
    script_path.write_text(
        "\n".join(
            [
                "import logging",
                "from renormalizer.utils import log, profiling",
                "log.init_log(logging.INFO)",
                "profiling.record('tensordot', wall_s=0.01, input_shapes=[(2, 2), (2, 2)], output_shape=(2, 2))",
            ]
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["RENO_LOG_LEVEL"] = "PROFILING"
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "renormalizer.backend.example_runner",
            "--backend",
            "numpy",
            "--device",
            "cpu",
            "--cwd",
            str(run_dir),
            "--script",
            str(script_path),
            "--profile-events",
            str(event_path),
            "--",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in events] == ["tensordot"]
