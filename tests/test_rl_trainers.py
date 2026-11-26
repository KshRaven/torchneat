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


def test_top_level_exports():
    import torchneat

    for name in ('A2C', 'TRPO', 'DQN'):
        assert getattr(torchneat, name) is getattr(rl, name)
