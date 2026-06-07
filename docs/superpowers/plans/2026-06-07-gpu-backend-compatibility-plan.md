# GPU Backend Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the FMO workload operationally supported on `cupy/gpu`, `jax/gpu`, and `torch/gpu`, and either enable or intentionally fence `cupynumeric` based on current backend limits.

**Architecture:** Keep all compatibility work isolated in `backend-gpufix` while the CPU formal campaign continues in `backend-np2`. Use socket1/all-GPU screening for evidence gathering, then apply narrowly scoped backend-boundary fixes only where the evidence shows a concrete incompatibility.

**Tech Stack:** Python 3.10, Renormalizer backend adapters, existing `.codex` smoke helpers, socket1/GPU screening jobs, JSON smoke artifacts, pytest.

---

## Phase 0: Safety And Evidence Baseline

**Files:**
- Read: `.codex/traj-cupy-gpu-np2/cupy_gpu_smoke.json`
- Read: `.codex/traj-jax-gpu-np2/jax_gpu_smoke.json`
- Read: `.codex/traj-torch-gpu-np2/torch_gpu_smoke.json`
- Read: `.codex/traj-cupynumeric-cpu-np2/cupynumeric_cpu_smoke.json`
- Read: `docs/superpowers/specs/2026-06-07-gpu-backend-compatibility-spec.md`

- [ ] **Step 1: Preserve the active CPU formal benchmark boundary**

Do not modify files in:

```text
/home/interns/data/code/Renormalizer/.worktrees/backend-np2
```

Expected result: all GPU compatibility code changes stay inside `backend-gpufix`.

- [x] **Step 2: Confirm the current support matrix from existing smoke artifacts**

Record:

```text
cupy/gpu: pass
jax/gpu: pass
torch/gpu: pass
cupynumeric: fail on non-affine view conversion
```

Expected result: later code changes are judged against a known baseline.


## Phase 1: GPU Screening On Socket1 / All GPUs

**Files:**
- Modify or Create: `.codex/run_gpu_backend_array.py`
- Modify or Create: `.codex/build_gpu_backend_screening_array.py`
- Reuse: `.codex/run_backend_fmo_smoke.py`
- Reuse: `.codex/backend_telemetry.py`

- [x] **Step 1: Define a screening array that uses socket1-only host slices**

Use:

```text
GPU0 -> CPUs 48-53
GPU1 -> CPUs 54-59
GPU2 -> CPUs 60-65
GPU3 -> CPUs 66-71
GPU4 -> CPUs 72-77
GPU5 -> CPUs 78-83
GPU6 -> CPUs 84-89
GPU7 -> CPUs 90-95
```

The array must only schedule GPU rows onto socket1 host CPUs.

- [x] **Step 2: Build an initial all-GPU screening matrix**

First screening batch should include:

```text
cupy/gpu
jax/gpu
torch/gpu
```

with tiny smoke and at least one benchmark-like larger-FMO point per backend.

Expected result: a quick answer to whether current support survives beyond the tiny already-saved smoke.

- [x] **Step 3: Launch only screening jobs, not formal timing jobs**

The screening runner may use all 8 GPUs concurrently, but its outputs must be labeled screening/probe, not formal benchmark.

Expected result: compatibility evidence without polluting the CPU formal timing run.


## Phase 2: Backend-Specific Failure Triage

**Files:**
- Read/Modify: `renormalizer/backend/cupynumeric_backend.py`
- Read/Modify: `renormalizer/mps/matrix.py`
- Read/Modify: backend-specific helper code only if required by evidence

- [x] **Step 1: Reproduce concrete failures before changing code**

For every failing backend row, capture:

```text
backend
device
input shape / MS
failing function
traceback
last emitted artifact
```

Expected result: no speculative backend patching.

- [x] **Step 2: Treat `cupy/gpu`, `jax/gpu`, and `torch/gpu` as stability targets**

If any of those regress in larger smoke, fix only the backend-specific boundary or dtype/device issue proven by the traceback.

- [x] **Step 3: Treat `cupynumeric` as a capability probe with explicit decision points**

Try the least-invasive compatibility path first:

```text
backend-local eager copy fallback before cnp.asarray() on unsafe host views
```

If that fails or causes broader breakage, stop and convert `cupynumeric` to an explicit unsupported path for FMO.


## Phase 3: Implement Narrow Fixes

**Files:**
- Modify: `renormalizer/backend/cupynumeric_backend.py`
- Possibly Modify: `renormalizer/mps/matrix.py`
- Modify: `renormalizer/mps/tests/test_backend_modernization.py`

- [x] **Step 1: Add failing tests that match the reproduced incompatibility**

At minimum, add a test covering the `cupynumeric` non-affine-view conversion path.

- [x] **Step 2: Apply the narrowest backend-boundary fix**

Preferred order:

```text
cupynumeric backend-local fallback
then matrix-boundary normalization only if backend-local fix is insufficient
```

- [x] **Step 3: Re-run the relevant unit tests and smoke**

Only the backend implicated by the change needs to be rerun first, followed by regression smoke on the already-working GPU backends.


## Phase 4: Re-Screen And Classify Support

**Files:**
- Output: `.codex/gpu_screening_runs/...`
- Update: `docs/superpowers/specs/2026-06-07-gpu-backend-compatibility-spec.md`

- [x] **Step 1: Re-run tiny GPU smoke for `cupy/jax/torch`**

Expected result: confirm no regression.

- [x] **Step 2: Re-run the larger socket1/GPU screening rows**

Expected result: classify each backend as:

```text
supported
conditionally supported
unsupported with intentional guardrail
```

- [x] **Step 3: Record the final `cupynumeric` outcome explicitly**

Do not leave `cupynumeric` in an ambiguous state.


## Phase 5: Final Review And Merge Readiness

**Files:**
- Review: code changes in `backend-gpufix`
- Review: smoke outputs and test evidence
- Update: support documentation if needed

- [ ] **Step 1: Review only for blocking correctness issues**

Focus on:

```text
wrong device placement
silent CPU fallback
dtype corruption
backend-specific regression
ambiguous unsupported behavior
```

- [ ] **Step 2: Summarize what is truly fixed**

The summary must separate:

```text
tiny smoke support
larger screening support
formal benchmark readiness
cupynumeric final status
```

- [ ] **Step 3: Stop short of claiming completion without evidence**

Completion requires current smoke/test artifacts proving the chosen support classification.

## Current Execution Notes

- `r1` screening showed `cupy/gpu` and `jax/gpu` passing tiny, `MS=64`, and `MS=128` rows, while `torch/gpu` failed on a real scalar `.imag` assumption and later mixed dtype/device issues.
- Current-code fixes were applied in:
  - `renormalizer/mps/matrix.py`
  - `renormalizer/mps/mps.py`
  - `renormalizer/tn/tree.py`
  - `renormalizer/backend/cupynumeric_backend.py`
- Added targeted regression tests in:
  - `renormalizer/mps/tests/test_backend_modernization.py`
- `r2` screening then passed all six rows:
  - `cupy/gpu`: tiny, `MS=64`
  - `jax/gpu`: tiny, `MS=64`
  - `torch/gpu`: tiny, `MS=64`
- `r3` screening then passed all three `MS=128` rows on the current code state:
  - `cupy/gpu`: `MS=128`
  - `jax/gpu`: `MS=128`
  - `torch/gpu`: `MS=128`
- `r4` screening then passed all three short-trajectory rows on the current code state:
  - `cupy/gpu`: `MS=64, steps=3`
  - `jax/gpu`: `MS=64, steps=3`
  - `torch/gpu`: `MS=64, steps=3`
- `cupynumeric` is now intentionally classified as unsupported for the current FMO workload, with a backend-boundary fast-fail for rank-`>4` contractions instead of a deep Legate runtime crash.
