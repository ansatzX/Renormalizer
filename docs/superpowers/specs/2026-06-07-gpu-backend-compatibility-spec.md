# GPU Backend Compatibility Specification

## Scope

This specification covers GPU-backend compatibility work in the isolated worktree:

```text
/home/interns/data/code/Renormalizer/.worktrees/backend-gpufix
```

The running CPU formal benchmark remains isolated in:

```text
/home/interns/data/code/Renormalizer/.worktrees/backend-np2
```

No code changes for GPU compatibility may be made in `backend-np2` while the CPU formal campaign is active.

## Objective

Make the FMO workload practically supported on the non-NumPy backends under the current backend modernization model:

```text
cupy/gpu
jax/gpu
torch/gpu
```

If feasible, improve `cupynumeric` support enough to run the FMO workload; otherwise, clearly define and implement the correct degraded behavior and failure boundary.

This work is not complete when a backend merely imports or passes tiny unit tests. Completion requires current-state evidence that the FMO workflow can initialize and execute on the target backend/device configuration under the benchmark harness.

## Current Evidence

### Existing small-FMO smoke results

The following tiny trajectory smoke results already exist in the copied workspace state:

```text
.codex/traj-cupy-gpu-np2/cupy_gpu_smoke.json
.codex/traj-jax-gpu-np2/jax_gpu_smoke.json
.codex/traj-torch-gpu-np2/torch_gpu_smoke.json
.codex/traj-cupynumeric-cpu-np2/cupynumeric_cpu_smoke.json
```

Observed status:

- `cupy/gpu`: pass
- `jax/gpu`: pass
- `torch/gpu`: pass
- `cupynumeric/cpu`: fail

### What the current passes do and do not prove

The `cupy/gpu`, `jax/gpu`, and `torch/gpu` smoke files prove:

- backend selection works
- FMO cache prepare works on a tiny case
- short trajectory evolution works on a tiny case
- numerical error against NumPy1 truth is small enough to continue

They do **not** yet prove:

- larger FMO resource-ramp stability
- socket1 host orchestration correctness under formal GPU runners
- full benchmark compatibility under long-running or large-`MS` conditions
- robustness across all code paths exercised by larger production-like FMO cases

### Screening evidence gathered in `backend-gpufix`

Two screening rounds were run from the isolated GPU worktree:

```text
.codex/gpu_screening_runs/2026-06-07-gpu-screening-r1
.codex/gpu_screening_runs/2026-06-07-gpu-screening-r2
```

`r1` established that current larger-FMO screening on socket1/all-GPU host slices was already good for:

- `cupy/gpu`: tiny pass, `MS=64` pass, `MS=128` pass
- `jax/gpu`: tiny pass, `MS=64` pass, `MS=128` pass

`r1` also exposed real `torch/gpu` compatibility failures on both tiny and `MS=64` screening rows.

After narrow backend-boundary fixes, `r2` re-ran the key rows on the current code state and all six rows passed:

- `cupy/gpu`: tiny pass, `MS=64` pass
- `jax/gpu`: tiny pass, `MS=64` pass
- `torch/gpu`: tiny pass, `MS=64` pass

An additional `r3` screening then re-ran `MS=128` on the current code state for all three GPU backends, and all three rows passed:

- `cupy/gpu`: `MS=128` pass
- `jax/gpu`: `MS=128` pass
- `torch/gpu`: `MS=128` pass

An additional `r4` screening then ran short trajectory smoke on the current code state:

- `cupy/gpu`: `MS=64, steps=3` pass
- `jax/gpu`: `MS=64, steps=3` pass
- `torch/gpu`: `MS=64, steps=3` pass

This means the present code state has current evidence for:

- `cupy/gpu`: supported for tiny smoke, larger smoke through `MS=128`, and short trajectory smoke at `MS=64`
- `jax/gpu`: supported for tiny smoke, larger smoke through `MS=128`, and short trajectory smoke at `MS=64`
- `torch/gpu`: supported for tiny smoke, larger smoke through `MS=128`, and short trajectory smoke at `MS=64`

It does **not** yet mean formal GPU benchmark readiness. The evidence level is still screening/smoke, not long-run formal timing.

### cuPyNumeric failure boundary

The original `cupynumeric` failure occurred during FMO initialization, before trajectory evolution:

```text
NotImplementedError:
cuPyNumeric does not currently know how to attach to array views
that are not affine transforms of their parent array.
```

The traceback shows the failing path:

```text
transport/dynamics.py:create_electron_fc
-> mpo.py:apply
-> matrix.py:tensordot
-> matrix.py:asxp
-> backend.to_backend
-> cupynumeric_backend.py:to_backend
-> cnp.asarray(view)
```

That blocker was reproduced and narrowed to conversion of intermediate non-affine NumPy views into cuPyNumeric arrays.

The current code now applies a backend-local copy fallback for that view-attachment case, which allows execution to proceed farther into the FMO path. The next concrete blocker is deeper and more fundamental for this workload:

```text
cupynumeric backend does not support tensor contractions producing rank > 4
for the current FMO workload.
```

This is now surfaced as an intentional fast-fail at the backend boundary instead of a deep Legate runtime crash. For the current FMO workload, `cupynumeric` should be treated as unsupported with an explicit guardrail.

## Constraints

### Workspace isolation

- CPU formal execution continues in `backend-np2`.
- GPU compatibility code work happens only in `backend-gpufix`.
- Shared conclusions and scripts may later be copied back after review, but not while CPU results are being collected in the live benchmark worktree.

### Benchmark isolation

When GPU backend runs are launched for evidence gathering:

- use socket1 host CPUs `48-95`
- use all GPUs only for screening/smoke
- do not share socket0 with GPU host orchestration during CPU formal timing

### Completion standard

For `cupy/gpu`, `jax/gpu`, and `torch/gpu`, compatibility is considered real only after:

1. tiny FMO smoke still passes after any code changes
2. at least one larger FMO run path used by the benchmark harness still initializes and runs
3. backend-specific failures, if any, are documented and reproducible

For `cupynumeric`, completion requires one of two explicit outcomes:

1. **Supported path:** FMO smoke passes under the selected `cupynumeric` mode
2. **Unsupported path with correct handling:** the backend fails fast with a precise, intentional error message that documents the effective FMO limitation and avoids ambiguous partial execution

## Current Support Classification

Based on the current isolated-worktree evidence:

- `cupy/gpu`: `supported` at screening level through `MS=128`, with multi-step smoke evidence at `MS=64`
- `jax/gpu`: `supported` at screening level through `MS=128`, with multi-step smoke evidence at `MS=64`
- `torch/gpu`: `supported` at screening level through `MS=128`, with multi-step smoke evidence at `MS=64` after current-code compatibility fixes
- `cupynumeric`: `unsupported with intentional guardrail` for the current FMO workload

## Problem Decomposition

This work splits into four subproblems.

### 1. GPU runner compatibility

Ensure the benchmark helper and backend runtime still operate correctly for:

```text
cupy/gpu
jax/gpu
torch/gpu
```

under socket1 host binding and current telemetry rules.

### 2. Larger-FMO backend stability

Determine whether the current tiny smoke success extends to at least one benchmark-like FMO path beyond `n_phonons=2, max_bonddim=8`.

This is an evidence problem first, not an optimization problem.

### 3. cuPyNumeric conversion semantics

Decide how Renormalizer should handle non-affine host views when the selected backend is `cupynumeric`.

There are three possible policies:

#### Policy A: eager-copy fallback at backend boundary

If `cnp.asarray(x)` fails for a host-side view, convert with:

```text
np.array(x, copy=True)
-> cnp.asarray(copied_array)
```

Pros:

- smallest surface-area change
- likely enough to unblock FMO smoke
- backend-local fix

Cons:

- may hide performance penalties
- may still fail on deeper code paths
- does not make cuPyNumeric truly view-compatible

#### Policy B: normalize views before backend conversion in matrix helpers

Teach `matrix.asxp()` or nearby call sites to recognize unsafe view patterns and materialize them before calling backend conversion.

Pros:

- centralizes cross-backend boundary behavior
- may help more than just cuPyNumeric

Cons:

- broader behavioral surface
- risks changing semantics for already-working backends

#### Policy C: explicit unsupported classification

Do not attempt to emulate unsupported view semantics. Instead, detect the FMO path early and fail with an intentional compatibility error that explains the limitation.

Pros:

- technically honest
- low risk to existing backends

Cons:

- does not improve capability
- leaves `cupynumeric` outside practical FMO support

### 4. Evidence-driven backend acceptance

After any change, rerun the relevant smoke or probe and decide whether the backend is:

```text
supported
conditionally supported
unsupported with intentional guardrail
```

## Recommended Technical Direction

Recommended order:

1. keep `cupy/gpu`, `jax/gpu`, and `torch/gpu` stable
2. verify at least one larger FMO path still works on those GPU backends
3. try `cupynumeric` **Policy A** first as the least invasive capability probe
4. if Policy A fails or causes broader regressions, fall back to explicit unsupported classification rather than broad unsafe rewrites

This recommendation is based on current evidence:

- three GPU backends already pass tiny FMO smoke
- `cupynumeric` has one concrete blocker with a narrow failing path
- the user asked for compatibility progress, not speculative broad refactoring

## Required Deliverables

1. isolated GPU compatibility worktree with current benchmark helper context
2. written compatibility spec
3. code changes, if needed, limited to the GPU compatibility worktree
4. smoke/probe evidence for:
   - `cupy/gpu`
   - `jax/gpu`
   - `torch/gpu`
   - `cupynumeric` chosen outcome
5. updated documentation of support status and remaining limits

## Explicit Non-Goals

- do not optimize GPU performance yet
- do not refactor unrelated tensor code while CPU benchmark is running
- do not claim `cupynumeric` support based only on import success or synthetic array tests
