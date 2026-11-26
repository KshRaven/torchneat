# TorchNEAT

TorchNEAT is a PyTorch-based neuroevolution library. It evolves the parameters of `NeatModule` models through initialization, mutation, speciation and reproduction, on the CPU (numba) or on an NVIDIA GPU (numba.cuda / cupy). RL trainers (`NEAT`, `PPO`) are included; environment design and fitness evaluation remain the caller's job.

Supports Python 3.10 to 3.14 on Linux, Windows and macOS.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchneat
```

| You want to | Read |
| --- | --- |
| Install PyTorch, the `gpu` extra and `torchneat-install` | [Installation](installation.md) |
| Set up experiment settings and schedulers | [Configuration](configuration.md) |
| Track lineage and store elite parents | [Ancestry and GeneBank](ancestry-genebank.md) |
| Train populations with `NEAT` and `PPO` | [Reinforcement learning](reinforcement-learning.md) |
| Compare the CPU and GPU backends | [Benchmarking](benchmarking.md) |
| Fix install, CUDA and performance problems | [Troubleshooting](troubleshooting.md) |
