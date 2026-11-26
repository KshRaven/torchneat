# Installation

TorchNEAT needs Python 3.11 or newer. The GPU backend needs an NVIDIA GPU with a CUDA 12 or 13 capable driver.

## PyTorch comes first

`torchneat` depends on a plain `torch`, so pip installs whatever PyPI serves for your platform:

| Platform | `pip install torch` gives |
|---|---|
| Linux | CUDA build (several GB of `nvidia-*` wheels), even when you only want the CPU backend |
| Windows, macOS | CPU build, even when you want the GPU backend |

Neither `torchneat` nor `torchneat[gpu]` can choose a PyTorch wheel index for you, so install PyTorch yourself from the
index you want before installing TorchNEAT:

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchneat

# NVIDIA GPU (cu132 = CUDA 13.2; use the "CUDA Version" shown by nvidia-smi or the closest older index)
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install "torchneat[gpu]"
```

Without the first line you get the platform default above. Installing `torchneat[gpu]` with `--extra-index-url` does not
replace an already installed `torch`, and pip keeps an installed build that satisfies `torch`.

## The `gpu` extra

`pip install "torchneat[gpu]"` runs `setup.py`, which reads the CUDA version your driver supports from `nvidia-smi`
(for example 13.2) and pins:

`numba-cuda[cu13]`, `cupy-cuda13x`, `cuda-python`, `cuda-bindings`, `cuda-toolkit`, `nvidia-nvvm`, `nvidia-nvjitlink`,
`nvidia-cuda-nvrtc` and `nvidia-cuda-runtime` to `13.2.*` (CUDA 12 pins `numba-cuda[cu12]`, `cupy-cuda12x`,
`cuda-python`, `cuda-bindings` and `cuda-toolkit`). Matching versions avoid `nvJitLink`, `cupy` and `numba.cuda` import
and compile errors.

- With no NVIDIA driver found, the extra falls back to an unpinned CUDA 13 set.
- Set `TORCHNEAT_CUDA=cpu`, `auto` or a version such as `12.8` before installing to override the detection.
- The pins are computed when the package is built. Installing from a checkout or source distribution uses your machine;
  a prebuilt wheel carries the pins of the machine that built it. Use `torchneat-install` on end-user machines.

## `torchneat-install`

Installed together with the package. It detects the driver's CUDA version, installs PyTorch from the matching index
(the newest CUDA index at or below the driver's version, then older ones, then CPU) and the pinned CUDA libraries:

```bash
torchneat-install                 # detect the system
torchneat-install --cpu           # CPU PyTorch only, no CUDA libraries
torchneat-install --cuda 12.8     # target a specific CUDA version
torchneat-install --no-torch      # leave PyTorch alone
torchneat-install --dry-run       # print the pip commands only
```

An installed PyTorch is replaced only when its kind does not match (CPU instead of CUDA, or another CUDA major version).

## Development install

```bash
pip install -e ".[gpu,dev]"
```

Use an editable install. A plain `pip install .` copies the package into `site-packages`, and that copy keeps running
instead of your working tree until you reinstall it.

## Verify

```bash
python -c "import torch, torchneat; print(torch.cuda.is_available(), torchneat.cuda_is_available())"
pytest -m "not gpu"
pytest -m gpu
```

## Devices and memory

`torchneat.set_device("cuda")` (or `TORCHNEAT_DEVICE=cuda`) selects the NEAT backend: initialization, crossover,
mutation and speciation kernels. It does not move your model. Create the `NeatParameter`s and `NeatModule`s on the same
device (`device="cuda"`), so the backend turns the PyTorch weights into CuPy arrays without a copy and the new weights
are written back in place. A module left on the CPU while the backend is on `cuda` copies every parameter to the GPU
and back each generation.

The design assumes GPU memory is shared with the rest of the pipeline:

- Model weights live on the GPU permanently; environments, post-rollout trainer calculations and the NEAT backend all use them.
- Rollouts are collected into CPU buffers, since they can hold thousands of large records, and moved to the GPU in batches.
- The backend handles one parameter at a time and releases temporary arrays (`gc.collect()` and the CuPy memory pool)
  after initialization, reproduction and speciation, leaving as much VRAM as possible for the other stages.

The numba CUDA low-occupancy warnings are hidden. Set `TORCHNEAT_PERF_WARNINGS=1` or run code inside
`with torchneat.perf_warnings():` to show them.
