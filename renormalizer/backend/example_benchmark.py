# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Small profile-off A/B benchmark for backend execution policies."""

import argparse
import json
from time import perf_counter
import warnings

from renormalizer import set_backend
from renormalizer.model import Phonon, SpinBosonModel
from renormalizer.sbm import SpinBosonDynamics
from renormalizer.utils import EvolveConfig, EvolveMethod, Quantity, profiling


_EVOLVE_DT = 0.05
_IDENTICAL_KRYLOV_WARNING = (
    r"^Precision loss occurred in moment calculation due to catastrophic cancellation\. "
    r"This occurs when the data are nearly identical\. Results may be unreliable\.$"
)


def _run_sbm(execution_policy, steps):
    set_backend("numpy", precision=64, execution_policy=execution_policy)
    model = SpinBosonModel(
        Quantity(0),
        Quantity(1),
        [Phonon.simple_phonon(Quantity(1), Quantity(0.1), 2)],
    )
    evolve_config = EvolveConfig(
        method=EvolveMethod.tdvp_ps,
        adaptive=False,
        guess_dt=_EVOLVE_DT,
    )
    job = SpinBosonDynamics(
        model,
        auto_expand=True,
        evolve_config=evolve_config,
    )
    started = perf_counter()
    # The tiny fixed problem gives identical Krylov iteration counts; SciPy warns
    # while computing their skew/kurtosis even though benchmark numerics are valid.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_IDENTICAL_KRYLOV_WARNING,
            category=RuntimeWarning,
        )
        job.evolve(evolve_dt=_EVOLVE_DT, nsteps=steps)
    wall_s = perf_counter() - started
    executed = len(job.evolve_times) - 1
    if executed != steps:
        raise RuntimeError(
            "SBM benchmark executed {} steps instead of {}".format(executed, steps)
        )
    return {
        "case": "sbm",
        "evolve_dt": _EVOLVE_DT,
        "evolve_time": float(job.evolve_times[-1]),
        "execution_policy": execution_policy,
        "observable": float(job.sigma_z[-1]),
        "profiling_enabled": bool(profiling.enabled()),
        "state_norm": float(job.latest_mps.mp_norm),
        "steps_executed": executed,
        "steps_requested": steps,
        "wall_s": float(wall_s),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("sbm",), required=True)
    parser.add_argument(
        "--execution-policy",
        choices=("legacy_oe", "execution_ir"),
        required=True,
    )
    parser.add_argument("--steps", type=int, required=True)
    options = parser.parse_args(argv)
    if options.steps < 0:
        parser.error("--steps must be non-negative")
    if profiling.enabled():
        parser.error("profile-free benchmark profiling must be disabled")
    payload = _run_sbm(options.execution_policy, options.steps)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
