# Troubleshooting

## Installation

**Linux installed several GB of CUDA wheels, but I only want the CPU backend.**
`pip install torch` is the CUDA build on Linux. Install `torch` from the CPU index first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install torchneat`. To undo it, run
`torchneat-install --cpu`.

**Windows (or macOS) has `torch.cuda.is_available() == False` although I installed `torchneat[gpu]`.**
`[gpu]` installs the numba/cupy/CUDA libraries only. Install the CUDA PyTorch wheel yourself:
`pip install torch --index-url https://download.pytorch.org/whl/cu132 --force-reinstall --no-deps` (use the index that
matches `nvidia-smi`), or run `torchneat-install`.

**`pip install ".[gpu]"` keeps my old PyTorch build.**
`--extra-index-url` only adds a place to look; an installed `torch` that satisfies the requirement is kept. Reinstall
it from the index you want with `--index-url` and `--force-reinstall --no-deps`.

**`cuda-toolkit` is not recognized as a command.**
It is a pip package, not a program. Check versions with `pip list` and the driver with `nvidia-smi`.

**`ImportError: cannot import name ... from 'torchneat' (unknown location)`.**
Python found a plain `torchneat` folder instead of the installed package (for example after uninstalling, or running
from the folder that contains the repository). Install it (`pip install -e .`) and run from another directory.

**My edits to the source are not used.**
A plain `pip install .` installed a copy. Use `pip install -e ".[dev]"`.

**The `gpu` extra pulled CUDA 13 packages but my driver supports CUDA 12 only.**
A prebuilt wheel carries the pins of the machine that built it. Run `torchneat-install` (or `torchneat-install --cuda 12.8`),
or install from source on your machine.

## CUDA errors

**`AttributeError: module 'numpy' has no attribute 'row_stack'`.**
NumPy 2.4 removed `np.row_stack`, which older `numba-cuda` releases still use. TorchNEAT aliases it to `np.vstack`
when the GPU backend loads; updating `numba-cuda` or pinning `numpy<2.4` also works.

**`nvJitLinkError: ERROR_OUTDATED_LIBRARY` / `NVVM_ERROR_INVALID_INPUT`.**
The CUDA libraries are newer than the driver supports, or are a mix of versions. Read the "CUDA Version" in the top
right of `nvidia-smi` and pin every CUDA package to it with `pip install "torchneat[gpu]"` (which does this) or
`torchneat-install`. Check with `pip list` that `cuda-toolkit`, `nvidia-nvvm`, `nvidia-nvjitlink`, `cuda-bindings` and
`cuda-python` share the same `MAJOR.MINOR`. A PyTorch build for a newer CUDA minor version still runs on the same major
version.

**`CUDA out of memory`.**
Lower the population size or the module size. Each parameter is processed on its own, so the largest single parameter
(population size times its element count) must fit in VRAM next to your model and rollout data. Release memory held by
your own code between stages.

## Performance

**The GPU backend is slower than the CPU.**
- The module is still on the CPU: create it with `device="cuda"` so the weights are used in place.
- Populations of around 100 genomes and small modules are dominated by fixed per-generation cost; the GPU pays off at
  populations in the thousands. See [Benchmarking](benchmarking.md).
- The first generation compiles the kernels (several seconds); time later generations.
- Python-side evaluation code, not the backend, is usually the bottleneck of a full pipeline.

**`NumbaPerformanceWarning: Grid size ... GPU under-utilization`.**
Hidden by default. If it still appears, you are running an older copy of the package (see "My edits are not used").
Set `TORCHNEAT_PERF_WARNINGS=1` to show it on purpose.

## Populations

**`Population(100, genera=2)` has 200 genomes.**
The population size applies to each genus. See [Configuration](configuration.md).
