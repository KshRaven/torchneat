"""Hyper-parameter schedulers."""

import pytest

from torchneat.optim import BinaryAnnealing, CosineAnnealing, RandomAnnealing, Scheduler


@pytest.mark.parametrize("cls", [BinaryAnnealing, CosineAnnealing, RandomAnnealing])
def test_schedulers_are_scheduler_subclasses(cls):
    assert issubclass(cls, Scheduler)
