# Installation with pip and PyPI

`pyproject.toml` is the source of package metadata, supported Python versions,
runtime dependencies and backend extras. Normal installation uses pip; lock
files are optional recipes for reproducing a particular tested environment.
No uv configuration or platform-specific wheel URL is embedded in the published
package's dependency metadata.

**Release status:** the multi-backend changes in this branch are not yet on
PyPI. The existing PyPI release 0.0.11 predates these extras. Use the checkout
commands below now; the PyPI equivalents apply after a release containing these
changes. Publishing also requires a new, unused release version.

## Choose a backend

Create and activate an isolated environment first (for example,
`python3.11 -m venv .venv`, then `source .venv/bin/activate` on Linux/macOS).
From this branch's checkout, choose one row:

| Selection | Install this checkout now | PyPI command after the new release |
|---|---|---|
| NumPy CPU | `python -m pip install .` | `python -m pip install renormalizer` |
| Torch | `python -m pip install '.[torch]'` | `python -m pip install 'renormalizer[torch]'` |
| JAX CPU | `python -m pip install '.[jax]'` | `python -m pip install 'renormalizer[jax]'` |
| JAX CUDA12 | `python -m pip install '.[jax-cuda12]'` | `python -m pip install 'renormalizer[jax-cuda12]'` |
| CuPy CUDA12 | `python -m pip install '.[cupy]'` | `python -m pip install 'renormalizer[cupy]'` |
| Development and documentation | `python -m pip install '.[dev]'` | `python -m pip install 'renormalizer[dev]'` |

NumPy 1 and 2 use the same adapter. Extras install optional dependencies; they do
not select the computation device or guarantee that a GPU/driver is available.
The examples are alternatives, not a request to combine every backend in one
environment. Runtime dependencies retain their supported version ranges; the
reference profiles below record the exact combinations actually tested.

## Torch: CPU execution versus a CPU-only installation

The `torch` extra uses standard dependency resolution. The wheel available from
PyPI depends on the platform and Torch version; do not infer CUDA support just
from the package name. For example, the PyPI Torch 2.5.1 Linux x86_64 build has
CUDA dependencies, whereas its macOS build does not. A CUDA-enabled Torch build
can also run CPU computations. Choosing `device='cpu'` does not turn that build
into a smaller CPU-only installation.

For ordinary installation, the Torch row above is sufficient. If you explicitly
need a CPU-only build or a particular CUDA build, install that official Torch
build first in the new environment, then install the Renormalizer extra. These
Linux x86_64 / Python 3.11 examples use the separately validated 2.5.1 builds:

```bash
# CPU-only build: choose this command ...
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu

# ... OR CUDA12.4 build (requires a compatible NVIDIA driver).
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124

# Then install this branch. The already installed compatible Torch satisfies
# its dependency; use 'renormalizer[torch]' here after the new PyPI release.
python -m pip install '.[torch]'
python -m pip check
```

Do not apply the Torch-only index to the whole Renormalizer installation: other
project dependencies come from PyPI. Standard pip metadata has no per-extra
index selection, so `.[torch]` alone does not promise a CPU-only or specific CUDA
build. See the [official Torch build recipes](https://pytorch.org/get-started/previous-versions/)
for other supported platforms and releases.

## Developer installation and documentation

`.[dev]` includes tests, formatting and documentation dependencies. The former
separate docs extra is folded into dev; GPU libraries remain optional. Building
notebook documentation additionally requires the external **Pandoc** executable
on `PATH`; the Python `nbsphinx` dependency does not supply it. Install Pandoc
with your operating system's package manager or from its distribution, then run:

```bash
pandoc --version
python -m sphinx -b html doc/source doc/build/html
```

Notebook rendering is included. Review build warnings rather than skipping
notebooks to hide a missing executable.

## Reproduce a reference environment

Run from the checkout with Python 3.11. Keep pip-tools in a separate environment:
`pip-sync` removes packages absent from its input, so it must never target a shared
working environment accidentally.

```bash
python3.11 -m venv .venvs/compiler
.venvs/compiler/bin/python -m pip install 'pip-tools==7.6.1'
python3.11 -m venv .venvs/run
.venvs/compiler/bin/pip-sync --python-executable .venvs/run/bin/python constraints/multibackend/torch-cpu-py311.lock
.venvs/run/bin/python -m pip install --no-deps .
.venvs/run/bin/python -m pip check
```

Choose one lock: `torch-cpu-py311.lock`, `torch-cu124-py311.lock`,
`jax-cuda12-py311.lock`, or the existing `numpy1-py311.lock`, `numpy2-py311.lock`,
`jax-cpu-py311.lock`, `cupy-py311.lock`. The older four files are captured version
snapshots, not newly generated pip-tools locks; they include some developer
packages. New files carry generation commands and source selection. None locks the
kernel driver or provides universal portability; only the two Torch wheel URLs include artifact hashes; the other dependencies
are version pins, not a fully hashed supply-chain lock.

Recompile when changing dependencies, extras, Python or platform. Do not edit the
resolved lock by hand. Example Torch CPU/CUDA build profiles:

```bash
.venvs/compiler/bin/pip-compile --index-url https://pypi.org/simple --extra torch --output-file constraints/multibackend/torch-cpu-py311.lock pyproject.toml constraints/multibackend/torch-cpu-py311.in
.venvs/compiler/bin/pip-compile --index-url https://pypi.org/simple --extra torch --output-file constraints/multibackend/torch-cu124-py311.lock pyproject.toml constraints/multibackend/torch-cu124-py311.in
.venvs/compiler/bin/pip-compile --index-url https://pypi.org/simple --extra jax-cuda12 --output-file constraints/multibackend/jax-cuda12-py311.lock pyproject.toml constraints/multibackend/jax-cuda12-py311.in
```

For NumPy1/2, compile pyproject with `-c constraints/multibackend/numpy1.txt` or
`numpy2.txt`. Add `--extra dev` for a complete developer lock, or combine it with
a selected backend extra. Runtime profiles do not themselves install pytest or
Sphinx. A successful resolution is not an execution test.

## Verify the installed build and actual device

The check runs native matrix multiplication and SVD reconstruction, synchronizes,
compares against a host reference and records the **output array's** device. GPU
requests fail instead of accepting a CPU fallback. Run from the source checkout;
`tools` is a developer entry point, not part of normal solver imports.

```bash
.venvs/run/bin/python -m tools.backend_validation.environment numpy2 --device cpu
.venvs/run/bin/python -m tools.backend_validation.environment torch --device cpu
CUDA_VISIBLE_DEVICES=0 .venvs/run/bin/python -m tools.backend_validation.environment torch --device cuda:0
CUDA_VISIBLE_DEVICES=0 .venvs/run/bin/python -m tools.backend_validation.environment cupy --device cuda:0
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venvs/run/bin/python -m tools.backend_validation.environment jax --device cpu
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venvs/run/bin/python -m tools.backend_validation.environment jax --device cuda:0
```

Run only the command matching the environment. `cuda:0` is relative to visible
devices, not necessarily machine GPU0. JAX x64 must be enabled **before** import;
preallocation is disabled on shared GPUs. CuPy and Torch should likewise use an
explicitly selected device. This tiny smoke establishes installation/runtime
functionality, not every algorithm, performance, or memory-capacity guarantee.

For physical regression in an environment also containing pytest:

```bash
RENO_TEST_BACKEND=torch RENO_TEST_DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=0 .venvs/run/bin/python -m pytest renormalizer/mps/tests/test_multibackend_spin.py renormalizer/tn/tests/test_multibackend_spin.py -q
```

Use `numpy`/`jax`/`cupy` and the corresponding device for the other builds; retain
JAX's x64 setting. NumPy1 and NumPy2 need separate environments. These small spin
problems do not certify all models, CV paths or optional precision settings.

Optional `primme` remains available for static DMRG; see its installation guide
if a native build is required. It is not necessary to establish a backend install.
