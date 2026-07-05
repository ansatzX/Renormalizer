# SBM Batched/Grouped GEMM Test Design

## Purpose

Use the SBM example as the first controlled case to determine whether the current batched/grouped GEMM strategy is useful for real Renormalizer workloads.

The test must separate two questions that were previously mixed:

1. Does the SBM path actually enter batched or grouped GEMM execution?
2. If it does, does that improve CPU performance from 1 core to 48 cores?

No conclusion about batched/grouped GEMM effectiveness is valid until both questions are answered separately.

## Context

The existing sampled CPU benchmark for `example/sbm.py` recorded:

- 1 core: 14 completed steps in about 1800 s.
- 48 cores: 14 completed steps in about 1800 s.
- 48-core process reached about 95 threads, so thread controls were active.
- Profiling was off, so those results do not prove whether batched/grouped GEMM was entered.

The FMO profile showed many small GEMM/GEMV contractions and no observed grouped GEMM events, but that is not enough to infer SBM behavior.

## Scope

This design covers only the SBM example on the NumPy2 CPU backend.

Backends out of scope:

- CuPy
- Torch
- JAX
- cupynumeric

Examples out of scope:

- FMO
- transport Kubo
- TTNS examples
- SSH
- H2O QC

## Stage A: Path Evidence

Create a short SBM profiling probe that follows the same model construction and algorithmic path as `example/sbm.py`, but uses a shorter evolution length.

Reference SBM configuration:

- `alpha = 0.05`
- `raw_delta = Quantity(1)`
- `raw_omega_c = Quantity(20)`
- `n_phonons = 300`
- `renormalization_p = 1`
- `CompressConfig(threshold=1e-4)`
- `EvolveConfig(adaptive=True, guess_dt=0.1)`
- `evolve_dt = 0.1`

Probe duration:

- Target 5 to 10 completed evolution steps.
- Keep the same `evolve_dt`.
- Shorten only `evolve_time`.

Profiling:

- Set `RENO_LOG_LEVEL=PROFILING`.
- Write profiling events to JSONL.
- Keep resource sampling enabled.

Required Stage A metrics:

- `batched_gemm` event count.
- `grouped_gemm_execute` event count.
- `hmm_num_batched_gemm`.
- `num_batched_gemm`.
- `num_grouped_tasks`.
- batch/group sizes where available.
- grouped pack strategy where available.
- pack time, compute time, scatter time, and fallback reason where available.
- dominant contraction shapes and small-GEMM indicators.

Stage A decision:

- If no batched/grouped events are observed, the conclusion is: the current strategy did not enter the SBM path under this probe.
- If batched/grouped events are observed, continue to Stage B and compare performance without full profiling overhead.

## Stage B: Performance Evidence

Run the original `example/sbm.py` path as a sampled CPU benchmark:

- backend: `numpy`
- device: `cpu`
- thread counts: `1` and `48`
- profiling mode: `off`
- resource sampling: on

The run does not need to finish all 201 expected steps if it is sampled consistently. The comparison metric is step throughput.

Required Stage B metrics:

- completed steps.
- step throughput per second.
- wall time.
- max RSS.
- max thread count.
- context switch samples if available from existing resource sampling.
- benchmark command and thread environment.

Stage B decision:

- If Stage A observed batched/grouped events and Stage B speedup is significant, the strategy is promising for SBM and can be expanded to other examples.
- If Stage A observed batched/grouped events but Stage B speedup is poor, inspect Stage A timing breakdown to decide whether pack/scatter, small kernels, or fallback explains the loss.
- If Stage A observed no batched/grouped events, do not claim the strategy is ineffective; instead treat it as a path-coverage failure.

## Reporting

Generate one SBM-specific HTML report with two sections:

1. Path evidence from Stage A.
2. Performance evidence from Stage B.

The report must keep path evidence and performance evidence separate. It must not state that batched/grouped GEMM is ineffective unless the path evidence proves the strategy was entered.

## Success Criteria

The test is complete when the report can answer these questions:

1. Did SBM enter batched/grouped GEMM execution?
2. If yes, how many batched/grouped operations were executed?
3. If yes, what were the dominant group sizes and pack strategies?
4. Did 48 CPU cores improve sampled SBM step throughput versus 1 CPU core?
5. Is the result a path-coverage failure, an ineffective strategy, or a promising strategy?

## Non-Goals

This design does not implement a new GEMM backend.

This design does not tune all examples.

This design does not claim FMO and SBM have the same bottleneck.

This design does not use GPU profiling.

## Risks

Full profiling may slow SBM enough to distort performance. This is why Stage A is only for path evidence and Stage B disables full profiling.

A very short probe may not cover all later-time SBM behavior. The first pass only needs to determine whether the strategy can enter the path at all.

If Stage A does not observe batched/grouped events, the next design should focus on path coverage before performance tuning.
