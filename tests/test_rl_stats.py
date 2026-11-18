from types import SimpleNamespace

import numpy as np
import pytest

from torchneat.rl.base import Algorithm
from torchneat.rl.functional import CpuRLBackend
from torchneat.rl.replay import RolloutBuffer


def test_row_stats_reports_non_finite_episode_values():
    trainer = SimpleNamespace(ops=CpuRLBackend())
    matrix = np.array([[1.0, 3.0], [2.0, np.nan]])
    counts = np.array([2, 2])
    rows = np.array([0, 1])

    with pytest.raises(ValueError, match=r"row 1 .*non-finite episode indices=\[1\]"):
        Algorithm.row_stats(trainer, matrix, counts, rows)


def test_row_stats_returns_finite_statistics():
    trainer = SimpleNamespace(ops=CpuRLBackend())
    matrix = np.array([[1.0, 3.0], [4.0, 4.0]])
    counts = np.array([2, 2])
    rows = np.array([0, 1])

    mean, std = Algorithm.row_stats(trainer, matrix, counts, rows)

    np.testing.assert_allclose(mean[rows], [2.0, 4.0])
    np.testing.assert_allclose(std[rows], [1.0, 0.0])


def test_secondary_buffer_excludes_unevaluated_genomes_from_normalization():
    trainer = SimpleNamespace(
        secondary=RolloutBuffer(buffers=["score", "ep_map"], dtype=np.float64),
        ops=CpuRLBackend(),
    )
    trainer.secondary.update_mapping({10: 0, 20: 1})

    scores = np.array([[[1.0], [2.0]], [[-np.inf], [-np.inf]]])
    episodes = np.array([[0, 1], [0, 1]])
    Algorithm.append_secondary_block(trainer, np.array([0]), score=scores, ep_map=episodes)

    scores = np.array([[[-np.inf], [-np.inf]], [[3.0], [4.0]]])
    Algorithm.append_secondary_block(trainer, np.array([1]), score=scores, ep_map=episodes)

    assert trainer.secondary.lengths.tolist() == [2, 2]
    returns = trainer.ops.compute_returns(
        trainer.secondary.data["score"],
        trainer.secondary.data["ep_map"],
        trainer.secondary.lengths,
        np.array([0, 1]),
        np.array([0, 0]),
        normalize=3,
    )
    assert np.isfinite(returns[[0, 1], :2]).all()
