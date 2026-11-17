"""GPU backend tests. Marked ``gpu``: skipped automatically (see ``tests/conftest.py``) when no usable CUDA device exists."""

import numpy as np
import pytest
import torch

import torchneat
from torchneat import Config, NeatModule, NeatParameter, Population

pytestmark = pytest.mark.gpu


class GpuModule(NeatModule):
    def __init__(self, dim_size: int = 32, dtype=torch.float32):
        super().__init__()
        self.weight = NeatParameter((dim_size, dim_size), dtype=dtype)
        self.bias = NeatParameter((dim_size,), dtype=dtype)


def make_config() -> Config:
    config = Config("gpu_smoke", "tests")
    config.general.seed = 1
    config.genome.init_type = "normal"
    config.genome.weight_init_std = 0.5
    config.genome.weight_min_value = -np.inf
    config.genome.weight_max_value = np.inf
    config.reproduction.min_species_size = 5
    config.reproduction.purge = 0
    config.species.compatibility_threshold = np.inf
    return config


def evaluate(population, **_options):
    for index, genome in enumerate(population.genomes.values()):
        genome.fitness = float(index)


def test_cuda_device_can_be_selected():
    torchneat.set_device("cuda")
    assert torchneat.get_device() == "cuda"


def test_population_evolves_on_gpu():
    torchneat.set_device("cuda")
    module = GpuModule()
    population = Population(20, module, make_config(), verbose=False)
    population.run(evaluate, generations=2, verbose=False)
    assert len(population.genomes) == 20
    assert all(parameter.shape[0] == 20 for parameter in module.neat_parameters())
