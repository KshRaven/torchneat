# TorchNEAT

TorchNEAT is a PyTorch-based neuroevolution library. It evolves the parameters of `NeatModule` models through
initialization, mutation, speciation and reproduction, on the CPU (numba) or on an NVIDIA GPU (numba.cuda / cupy).
RL trainers (`NEAT`, `PPO`, `A2C`, `TRPO`, `DQN`) are included; environment design and fitness evaluation remain the caller's job.

## Install

Python 3.10+. Install PyTorch first, from the wheel index that matches what you want, then TorchNEAT:

```bash
# CPU only (any OS; on Linux this avoids the multi-GB CUDA wheels)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchneat

# NVIDIA GPU: use your driver's CUDA version from `nvidia-smi` (cu132 = CUDA 13.2), required on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install "torchneat[gpu]"

pip install "torchneat[logging]"      # + runlytics logging for the RL trainers
```

`pip install torchneat` alone takes whatever PyTorch PyPI serves: the CUDA build on Linux, the CPU build on Windows
and macOS. `[gpu]` adds the numba/cupy/CUDA libraries pinned to the CUDA version of your NVIDIA driver, but never
changes your PyTorch build. `torchneat-install` detects the system and sets up PyTorch and the CUDA libraries for you
(`--cpu` to force CPU). See [Installation](docs/installation.md) and [Troubleshooting](docs/troubleshooting.md).

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
| `torchneat.core` | `Genome`, `Specie`/`Species`, `Stagnation`, `Reproduction`, `Population` |
| `torchneat.backends` | device-independent init/reproduce/speciate orchestration, `cpu/` and `gpu/` kernels, seeding |
| `torchneat.nn` | `NeatParameter`, `NeatModule`, `Model`; layers, attention/transformer blocks, activations, PyTorch re-exports; `legacy/` |
| `torchneat.rl` | `NEAT` and `PPO` trainers, rollout buffers, logger |
| `torchneat.config` | `Config` and its sections |
| `torchneat.optim` | hyper-parameter schedulers |
| `torchneat.reporting` | reporters and loggers |
| `torchneat.utils` | storage, console colours, timing, helpers |

## Devices

CPU is the default. Choose the NEAT backend with `torchneat.set_device("cuda")` or the `TORCHNEAT_DEVICE` environment
variable; `torchneat.cuda_is_available()` tells whether the GPU backend is usable. `set_device` does not move your
model: create the `NeatParameter`s / `NeatModule`s on the same device (`device="cuda"`) so the backend can use the
weights in place. The numba CUDA occupancy warnings are hidden; set `TORCHNEAT_PERF_WARNINGS=1` or use
`with torchneat.perf_warnings():` to show them.

## Saving and loading

```python
population.save(filename="experiment", directory="neat_save")
population.load(filename="experiment", directory="neat_save")
state = population.state_dict()
```

Checkpoints use `pickle`; only load files you trust.

## Documentation

- [Documentation site](https://kshraven.github.io/torchneat/)
- [Installation](docs/installation.md): PyTorch wheel indexes, the `gpu` extra, `torchneat-install`, memory model.
- [Troubleshooting](docs/troubleshooting.md): install, CUDA and performance problems.
- [Benchmarking](docs/benchmarking.md): CPU versus GPU backend benchmark script.
- [Configuration](docs/configuration.md): experiment settings and schedulers.
- [Ancestry and GeneBank](docs/ancestry-genebank.md): lineage tracking and elite parent storage.
- [Reinforcement learning](docs/reinforcement-learning.md): trainers, rollout behaviour and metrics.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not gpu"     # CPU tests
pytest -m gpu           # GPU tests (skipped when no CUDA device is usable)
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md) and [TODO.md](TODO.md).
