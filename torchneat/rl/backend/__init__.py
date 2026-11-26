"""
RL backends: the device the trainers run their post-rollout bookkeeping on.

The device follows ``torchneat.set_device()`` (the same setting the NEAT backends use), so it never has to be given
to a trainer; a trainer can still override it with its ``device`` argument. Nothing GPU related is imported until a
CUDA backend is requested.
"""
from __future__ import annotations

import warnings

from .base import ACCURACY_MODES, POLICY_MODES, EpisodeRuns, RLBackend
from .cpu import CpuRLBackend

__all__ = [
    'ACCURACY_MODES', 'POLICY_MODES', 'CpuRLBackend', 'CudaRLBackend', 'EpisodeRuns', 'RLBackend',
    'get_rl_backend', 'resolve_device',
]

_BACKENDS: dict[str, RLBackend] = {}


def __getattr__(name):
    if name == 'CudaRLBackend':
        from .gpu import CudaRLBackend
        return CudaRLBackend
    raise AttributeError(name)


def resolve_device(device=None) -> str:
    """``'cpu'`` or ``'cuda'``: an explicit ``device`` (str or ``torch.device``), else the global torchneat device."""
    if device is None:
        from ...backends import get_device
        return get_device()
    name = str(getattr(device, 'type', device)).lower()
    if 'cuda' in name:
        return 'cuda'
    if 'cpu' in name:
        return 'cpu'
    raise ValueError(f"Unsupported device: '{device}'")


def get_rl_backend(device=None) -> RLBackend:
    """Backend for ``device`` (default: the device ``torchneat.set_device`` selected), cached per device name."""
    name = resolve_device(device)
    if name == 'cuda' and name not in _BACKENDS:
        try:
            from ...backends import cuda_is_available
            if not cuda_is_available():
                raise RuntimeError("no usable CUDA device")
            from .gpu import CudaRLBackend
            _BACKENDS[name] = CudaRLBackend()
        except Exception as error:
            warnings.warn(f"CUDA RL backend unavailable ({error}); using the CPU backend.", RuntimeWarning,
                          stacklevel=2)
            name = 'cpu'
    if name not in _BACKENDS:
        _BACKENDS[name] = CpuRLBackend()
    return _BACKENDS[name]
