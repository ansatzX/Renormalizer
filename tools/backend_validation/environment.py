"""Report actual backend selection and installed versions, not environment labels."""
import argparse
import importlib.metadata as metadata
import json
import platform


def validate_identity(env_id, numpy_version, selected_backend):
    if env_id in ('numpy1', 'numpy2'):
        if numpy_version.split('.')[0] != env_id[-1]:
            return {'ok': False, 'reason': 'numpy_major_mismatch'}
        expected = 'numpy'
    elif env_id in ('jax', 'torch', 'cupy'):
        expected = env_id
    else:
        return {'ok': False, 'reason': 'unknown_environment'}
    if selected_backend != expected:
        return {'ok': False, 'reason': 'backend_mismatch'}
    return {'ok': True, 'reason': None}


def collect(env_id):
    import numpy as np
    from renormalizer.cons import set_backend

    selected = set_backend('numpy' if env_id.startswith('numpy') else env_id)
    array = selected.ones((2, 2), dtype=selected.float64)
    selected.sync()
    versions = {}
    for name in ('numpy', 'scipy', 'qutip', 'jax', 'jaxlib', 'torch', 'cupy-cuda12x'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    device = getattr(array, 'device', 'cpu')
    if callable(device):
        device = device()
    identity = validate_identity(env_id, np.__version__, selected.name)
    if str(array.dtype) not in ('float64', 'torch.float64'):
        identity = {'ok': False, 'reason': 'precision_mismatch'}
    return {
        **identity,
        'environment': env_id, 'python': platform.python_version(),
        'versions': versions, 'backend': selected.name,
        'array_type': f'{type(array).__module__}.{type(array).__qualname__}',
        'dtype': str(array.dtype), 'device': str(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('environment', choices=('numpy1', 'numpy2', 'jax', 'torch', 'cupy'))
    result = collect(parser.parse_args().environment)
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
