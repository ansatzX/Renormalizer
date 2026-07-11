"""Deterministic workspace accounting for execution-plan metadata."""


MAX_WORKSPACE_BYTES = (1 << 63) - 1


def workspace_bytes_for_steps(steps, output_key):
    """Return a conservative sum of unique temporary output buffers."""
    sizes = {}
    for step in steps:
        for output in step.workspace_outputs:
            if output.key == output_key:
                continue
            previous = sizes.setdefault(output.key, output.spec.nbytes)
            if previous != output.spec.nbytes:
                raise ValueError("temporary buffer key has inconsistent specifications")
    total = 0
    for size in sizes.values():
        total += size
        if total > MAX_WORKSPACE_BYTES:
            raise OverflowError("workspace requirement exceeds the supported metadata bound")
    return total


__all__ = ["workspace_bytes_for_steps"]
