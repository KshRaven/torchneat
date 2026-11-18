# Changelog

## Unreleased

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
