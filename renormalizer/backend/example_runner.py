# -*- coding: utf-8 -*-

"""Execute one example after configuring backend and profiling.

This module is intentionally small because it runs in the benchmark child
process.  The parent benchmark process owns scheduling, environment variables,
resource sampling, and reporting.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def _install_profile_log_level_guard(log_module):
    """Keep profiling enabled even if an example calls ``log.init_log(INFO)``."""
    original_init_log = log_module.init_log

    def init_log_preserving_profiling(level=log_module.DEBUG):
        parsed = log_module.parse_log_level(level)
        return original_init_log(min(parsed, log_module.PROFILING))

    log_module.init_log = init_log_preserving_profiling
    log_module.init_log(log_module.PROFILING)
    return original_init_log


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--profile-events", default=None)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.script_args and args.script_args[0] == "--":
        args.script_args = args.script_args[1:]
    return args


def main(argv=None):
    args = _parse_args(argv)
    cwd = Path(args.cwd).resolve()
    script = Path(args.script).resolve()

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    profiling = None
    try:
        os.chdir(cwd)
        sys.argv = [str(script)] + list(args.script_args)

        import renormalizer
        from renormalizer.utils import log
        from renormalizer.utils import profiling as profiling_module

        renormalizer.set_backend(args.backend, device=args.device)
        profiling = profiling_module
        if args.profile_events:
            _install_profile_log_level_guard(log)
            profiling.register_event_output(args.profile_events)

        runpy.run_path(str(script), run_name="__main__")
    finally:
        if profiling is not None:
            profiling.flush_summaries()
            profiling.flush_event_output()
            profiling.close_event_output()
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
