from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import torch

from ..core.population import Population
from ..optim.scheduler import Scheduler
from .trainer import PolicyTrainer

__all__ = ['A2C']


class A2C(PolicyTrainer):
    """
    Evolutionary Advantage Actor-Critic.

    The policy loss is ``-mean(log_prob * advantage)`` (no probability ratio). The critic is off by default, which
    makes the advantage the raw return; with ``use_critic=True`` the advantage is the GAE(``gae_lambda``) estimate
    from ``Model.get_value``. Options are documented on ``PolicyTrainer``.
    """
    policy_mode = 'a2c'

    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=None, dtype=torch.float32, use_critic: bool | None = None, gae_lambda: float = 0.95,
                 **options):
        """:param gae_lambda: GAE smoothing used when the critic is enabled."""
        self.gae_lambda: float = gae_lambda
        super().__init__(population, schedulers, device, dtype, use_critic, **options)

    def advantages(self, returns: np.ndarray, values: np.ndarray | None, keys: list[int], rows: np.ndarray,
                   lengths: np.ndarray) -> np.ndarray:
        if values is None:
            return returns
        prim = self.primary
        return self.ops.compute_gae(
            prim.data['reward'], values, prim.data['ep_map'], lengths, rows, self.gamma, self.gae_lambda, 1.0
        )[1]
