# TorchNEAT

TorchNEAT is a PyTorch-based neuroevolution library. It evolves the parameters of `NeatModule` models through
initialization, mutation, speciation and reproduction, on the CPU (numba) or on an NVIDIA GPU (numba.cuda / cupy).
RL trainers (`NEAT`, `PPO`) are included; environment design and fitness evaluation remain the caller's job.

> TorchNEAT grew out of an earlier private project of the same author. This repository is a clean restructure; the old
> history is not carried over.

## Install

Python 3.11 or newer.

```bash
pip install torchneat                 # CPU backend
pip install "torchneat[gpu]"          # + CUDA backend (NVIDIA GPU)
pip install "torchneat[logging]"      # + runlytics logging for the RL trainers
```

## Quick start

```python
import torch
import torchneat as neat
from torchneat import nn


class Policy(neat.nn.Model):
    def __init__(self, inputs=4, hidden=16, outputs=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden), nn.Tanh(),     # NEAT-evolvable layer + wrapped PyTorch activation
            nn.Linear(hidden, outputs),
        )

    def forward(self, state, keys=None):
        return self.net(state, keys=keys)


config = neat.Config(file_name="demo", directory="demo")
population = neat.Population(100, Policy(), config)
```

Everything you normally take from `torch.nn` (activations, dropout, flatten, containers, ...) is re-exported from
`torchneat.nn`. Parameters that evolve use `NeatParameter` /
`NeatModule`; everything else is plain PyTorch.

## Package layout

| Package | Contents |
|---|---|
| `torchneat.core` | `Genome`, `Species`/`SpeciesSet`, `Stagnation`, `Reproduction`, `Population` |
| `torchneat.backends` | device-independent init/reproduce/speciate orchestration, `cpu/` and `gpu/` kernels, seeding |
| `torchneat.nn` | `NeatParameter`, `NeatModule`, `Model`; layers, attention/transformer blocks, activations, PyTorch re-exports; `legacy/` |
| `torchneat.rl` | `NEAT` and `PPO` trainers, rollout buffers, logger |
| `torchneat.config` | `Config` and its sections |
| `torchneat.optim` | hyper-parameter schedulers |
| `torchneat.reporting` | reporters and loggers |
| `torchneat.utils` | storage, console colours, timing, helpers |

## Devices

CPU is the default. Choose the device with `torchneat.set_device("cuda")` or the `TORCHNEAT_DEVICE` environment
variable; `torchneat.cuda_is_available()` tells whether the GPU backend is usable.

## Saving and loading

```python
population.save(filename="experiment", directory="neat_save")
population.load(filename="experiment", directory="neat_save")
state = population.state_dict()
```

Checkpoints use `pickle`; only load files you trust.

## Documentation

- [Configuration](docs/configuration.md): experiment settings and schedulers.
- [Reinforcement learning](docs/reinforcement-learning.md): trainers, rollout behaviour and metrics.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not gpu"     # CPU tests
pytest -m gpu           # GPU tests (skipped when no CUDA device is usable)
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md) and [TODO.md](TODO.md).
