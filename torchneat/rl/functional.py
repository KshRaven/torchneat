"""Compatibility import path of the RL backends; they live in ``torchneat.rl.backend``."""
from __future__ import annotations

from .backend import CpuRLBackend, EpisodeRuns, RLBackend, get_rl_backend

__all__ = ['EpisodeRuns', 'RLBackend', 'CpuRLBackend', 'CudaRLBackend', 'get_rl_backend']  # noqa: F822


def __getattr__(name):
    if name == 'CudaRLBackend':
        from .backend import CudaRLBackend
        return CudaRLBackend
    raise AttributeError(name)
