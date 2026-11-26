from types import SimpleNamespace

import numpy as np
import pytest

from torchneat import rl


@pytest.mark.parametrize("cls, mode", [(rl.PPO, 'ppo'), (rl.A2C, 'a2c'), (rl.TRPO, 'trpo')])
def test_policy_modes(cls, mode):
    assert issubclass(cls, rl.PolicyTrainer)
    assert cls.policy_mode == mode
    assert cls.critic_default is False


def test_dqn_records_without_log_prob():
    assert issubclass(rl.DQN, rl.PolicyTrainer)
    assert 'log_prob' not in rl.DQN.primary_buffers
    assert rl.DQN.defaults['pol_reg'] == 0.0


def test_trpo_penalty_hinge():
    trainer = object.__new__(rl.TRPO)
    trainer.max_kl, trainer.kl_coef = 0.01, 10.0
    out = trainer.penalty(np.array([0.0, 0.01, 0.03]))
    assert np.allclose(out, [0.0, 0.0, 0.2])


def test_short_rollout_uses_capacity_sized_placeholder_records():
    class ReachedValueAggregation(Exception):
        pass

    capacity, actual_length = 6000, 2294
    rows = np.array([0])
    runs = SimpleNamespace(
        counts=np.array([1]),
        slot=np.full((1, capacity), -1, dtype=np.int64),
    )
    primary = SimpleNamespace(
        data={
            'ep_map': np.zeros((1, capacity), dtype=np.int64),
            'reward': np.zeros((1, capacity, 1), dtype=np.float64),
        },
        lengths=np.array([actual_length]),
        max_size=lambda: actual_length,
    )

    class Ops:
        def __init__(self):
            self.segment_calls = 0

        def record_mean(self, values, *_args):
            return np.zeros(values.shape[:2], dtype=np.float64)

        def segment_mean(self, values, _runs, _rows):
            assert values.shape == runs.slot.shape
            self.segment_calls += 1
            if self.segment_calls == 5:
                raise ReachedValueAggregation
            return np.zeros((1, 1), dtype=np.float64)

    trainer = object.__new__(rl.PPO)
    trainer.primary = primary
    trainer.secondary = object()
    trainer.ops = Ops()
    trainer.log_path = 'test'
    trainer.gamma = trainer.kappa = 0.0
    trainer.use_critic = False
    trainer.loss_reg = 1.0
    trainer.pol_reg = trainer.ent_reg = trainer.val_reg = 0.0
    trainer.use_entropy = True
    trainer.epsilon = 1e-12
    trainer._collect_rollout = lambda *_args: ([0], 0.0)
    trainer.prepare_rollout = lambda *_args, **_kwargs: (rows, runs)
    trainer.compute_returns = lambda *_args, **_kwargs: np.zeros((1, capacity, 1))
    trainer.episode_means = lambda *_args, **_kwargs: np.zeros((1, 1))
    trainer.advantages = lambda *_args, **_kwargs: np.zeros((1, capacity, 1))
    trainer.record_terms = lambda *_args, **_kwargs: tuple(
        np.zeros((1, capacity), dtype=np.float64) for _ in range(3)
    )

    with pytest.raises(ReachedValueAggregation):
        trainer.learn(lambda: None, steps=capacity, epochs=1)

    assert trainer.ops.segment_calls == 5


def test_top_level_exports():
    import torchneat

    for name in ('A2C', 'TRPO', 'DQN'):
        assert getattr(torchneat, name) is getattr(rl, name)
