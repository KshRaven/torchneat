from __future__ import annotations

from . import a2c, backend, base, dqn, functional, neat, ppo, replay, trainer, trpo
from .base import Algorithm, RunlyticsLogger, TensorboardLogger, TerminalReporter
from .backend import CpuRLBackend, EpisodeRuns, RLBackend, get_rl_backend
from .neat import NEAT, Ghost
from .a2c import A2C
from .dqn import DQN
from .ppo import PPO
from .trainer import PolicyTrainer
from .trpo import TRPO
from .replay import LogBuffer, ReplayBuffer, RolloutBuffer, read_replay, write_replay

__all__ = [
    # submodules
    'a2c', 'backend', 'base', 'dqn', 'functional', 'neat', 'ppo', 'replay', 'trainer', 'trpo',
    # algorithms
    'Algorithm', 'NEAT', 'Ghost', 'PolicyTrainer', 'PPO', 'A2C', 'TRPO', 'DQN',
    # logging
    'RunlyticsLogger', 'TensorboardLogger', 'TerminalReporter',
    # buffers
    'ReplayBuffer', 'RolloutBuffer', 'LogBuffer', 'read_replay', 'write_replay',
    # backend
    'EpisodeRuns', 'RLBackend', 'CpuRLBackend', 'CudaRLBackend', 'get_rl_backend',
]


def __getattr__(name):
    if name == 'CudaRLBackend':
        from .backend import CudaRLBackend
        return CudaRLBackend
    raise AttributeError(name)
