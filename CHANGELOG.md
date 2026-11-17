# Changelog

## Unreleased

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
