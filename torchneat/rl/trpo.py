from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import torch

from ..core.population import Population
from ..optim.scheduler import Scheduler
from .trainer import PolicyTrainer

__all__ = ['TRPO']


class TRPO(PolicyTrainer):
    """
    Evolutionary Trust Region Policy Optimization.

    The loss is the surrogate ``-mean(ratio * advantage)`` plus ``kl_coef * max(KL - max_kl, 0)`` per record,
    where KL is the sampled estimate of ``KL(old || new)``. Genomes whose policy drifts further than ``max_kl``
    from the one that collected the rollout are penalized instead of constrained. The critic is off by default.
    Options are documented on ``PolicyTrainer``.
    """
    policy_mode = 'trpo'

    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=None, dtype=torch.float32, use_critic: bool | None = None, max_kl: float = 0.01,
                 kl_coef: float = 10.0, **options):
        """
        :param max_kl: KL divergence tolerated before the penalty applies.
        :param kl_coef: Weight of the penalty above ``max_kl``.
        """
        self.max_kl: float = max_kl
        self.kl_coef: float = kl_coef
        super().__init__(population, schedulers, device, dtype, use_critic, **options)

    def penalty(self, kl: np.ndarray) -> np.ndarray:
        return self.kl_coef * np.maximum(kl - self.max_kl, 0.0)
