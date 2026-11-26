from __future__ import annotations

from .trainer import PolicyTrainer

__all__ = ['PPO']


class PPO(PolicyTrainer):
    """
    Evolutionary Proximal Policy Optimization.

    There is no clipped surrogate update and no optimizer step: each genome's return and its policy loss
    ``-mean(ratio * advantage)`` under its own policy rank the genomes. ``Model.evaluate_action`` supplies the
    log-probabilities. The critic is off by default (``use_critic=False``): the raw returns are the advantages.

    ``recurrent=True`` evaluates whole episodes in step order, for recurrent policies (see ``PolicyTrainer``).
    Options are documented on ``PolicyTrainer``.
    """
    policy_mode = 'ppo'
