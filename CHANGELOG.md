# Changelog

## Unreleased

## 0.2.3
- Supported Python versions are 3.10 to 3.14; CI and release runs test every version on Linux, Windows and macOS.
- Added a documentation site published to GitHub Pages.

## 0.2.2
- Added `setup.py`: the `gpu` extra pins `numba-cuda`, `cupy`, `cuda-python`, `cuda-bindings`, `cuda-toolkit` and the `nvidia-*` libraries to the CUDA version of the installed NVIDIA driver (`TORCHNEAT_CUDA=cpu|auto|X.Y` overrides it).
- Added the `torchneat-install` command, which installs the PyTorch wheel (CPU or CUDA index) and CUDA libraries matching the system.
- Fixed the GPU backend failing to compile kernels with NumPy 2.4+ (`np.row_stack` was removed).
- `Population(n, ..., genera=g)` now creates `n` genomes per genus (`n * g` in total); `genus_sizes` no longer has to sum to `n`.
- Reproduction keeps every genus at no fewer than `min_pop_size` genomes, and species spawn totals now add up exactly to the target size.
- Hid the numba CUDA low-occupancy warnings; `torchneat.perf_warnings()` and `TORCHNEAT_PERF_WARNINGS=1` show them again.
- Added `scripts/benchmark_backends.py` to compare the CPU and GPU backends.
- Documented installation, PyTorch wheel indexes, the memory model, benchmarking and troubleshooting in `docs/`.

## 0.2.1
- Replaced reflected lists with typed lists and arrays in Numba reproduction, speciation and best-genome selection calls.
- Fixed NEAT and PPO secondary-buffer updates for partially evaluated populations.
- Added detailed validation errors for non-finite episode statistics.
- Added RL statistics regression tests.

## 0.2.0
- `general.pop_size` is now `general.min_pop_size`; the `genomes` argument of `Population` is optional and overrides it. `general.genera` sets the number of genera, overridden by the `genera` option.
- `Species` is now `Specie` and `SpeciesSet` is now `Species`.
- Genomes track `created`, `age`, `best_fitness` and `improved`; species track `age` and `stagnant_time`.
- Added `Ancestry` (`Population.ancestry`) for optional key-only lineage tracking.
- Added `GeneBank` (`Population.genebank`) to keep elite genomes as parents beyond culling.
- Species keep a stable representative across generations, so ids are no longer recreated whenever the best genome changes.
- `GeneBank` limits apply per specie by default; `scope='genus'` applies them per genus.
- Documented `Ancestry` and `GeneBank` in `docs/ancestry-genebank.md`.
- Added `torchneat.experimental.GraphNetwork`, a node and connection feed-forward network on fixed-shape parameters.

## 0.1.0
- First release of `torchneat`, a restructured package built from an earlier private project (its history is not carried over).
- New layout: `core/` (genome, species, stagnation, reproduction, population), `backends/` (device-independent
  orchestration plus `cpu/` and `gpu/` kernels), `nn/` (evolvable modules, layers, blocks, wrapped PyTorch modules),
  `rl/` (NEAT and PPO trainers), `config/`, `optim/`, `reporting/` and `utils/`.
- Added PyTorch activations, regularisation, container and shape modules re-exported from `torchneat.nn`, so user code no
  longer has to import `torch.nn` next to `torchneat`.
- Moved the old NEAT-style activations to `torchneat.nn.legacy`.
- GPU tests are marked `gpu` and skip themselves when no usable CUDA device is present.
- Removed unused code: the stale `cuda` stub, the unused reproduction draft, multiprocessing inference helpers, the legacy RL trainers and
  unused helpers. The examples and the NeatBoard dashboard now live outside this package.
