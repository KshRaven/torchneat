from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import torch

from ..core.population import Population
from ..optim.scheduler import Scheduler
from .trainer import PolicyTrainer

__all__ = ['DQN']


class DQN(PolicyTrainer):
    """
    Evolutionary Deep Q-Network for discrete actions.

    ``Model.get_q_values`` supplies the action values. Each genome's loss is the squared one-step TD error
    ``(r + gamma * max_a' Q(s', a') - Q(s, a))^2`` over its own records (the last record of an episode is terminal).
    There is no target network or critic; exploration (epsilon-greedy and similar) belongs to the evaluation function.
    ``ent_reg`` (default 0) uses the entropy of ``softmax(Q)``; ``pol_reg`` has no log-probabilities to compare and
    stays 0. Options are documented on ``PolicyTrainer``.
    """
    primary_buffers = ('state', 'action', 'reward', 'ep_map')
    defaults = {'pol_reg': 0.0, 'ent_reg': 0.0, 'use_entropy': True}

    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=None, dtype=torch.float32, **options):
        options.pop('use_critic', None)
        super().__init__(population, schedulers, device, dtype, False, **options)

    def update(self, observations, actions, rewards, terminated: Union[bool, list[bool]], force_stop=False,
               reset_mapping: bool | list[bool] = False, session_dim: int | None = None) -> bool:
        """
        Records one environment step for every environment instance; ``actions`` are the taken action indices.
        See ``PPO.update`` for the other arguments. Returns whether the rollout buffer has been filled.
        """
        return self._record_step(
            dict(state=observations, action=actions, reward=rewards), terminated, force_stop, reset_mapping, session_dim
        )

    def record_terms(self, keys, rows, lengths, advantages, returns):
        prim = self.primary
        q = self.forward_pass('get_q_values', ['state'], keys, lengths, strict=True)[0]
        td = self.ops.td_terms(q, prim.data['action'], prim.data['reward'], prim.data['ep_map'], lengths, rows, self.gamma)
        flat = q.reshape(q.shape[0], q.shape[1], -1).astype(np.float64)
        flat = flat - flat.max(axis=2, keepdims=True)
        prob = np.exp(flat)
        prob /= prob.sum(axis=2, keepdims=True)
        entropy = -(prob * np.log(prob + 1e-12)).sum(axis=2)
        return td, np.zeros_like(td), entropy

    def std_terms(self, keys, rows, lengths):
        return np.zeros((len(self.primary.lengths), self.primary.max_size()), np.float64)

    def policy_accuracy(self, key, row, lengths, error, kind) -> float:
        q = self.forward_pass('get_q_values', ['state'], [key], lengths, strict=True)[0]
        n = int(lengths[row])
        taken = self.primary.data['action'][row, :n].reshape(n, -1)[:, 0].astype(np.int64)
        return float(np.mean(q[row, :n].reshape(n, -1).argmax(axis=1) == taken))
