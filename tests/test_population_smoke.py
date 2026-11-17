"""End-to-end smoke test: build a population and evolve it for a few generations on the CPU backend."""

import numpy as np
import torch

import torchneat.nn as nn
from torchneat import Config, NeatModule, NeatParameter, Population, Reproduction
from torchneat.backends import seeding


class TinyModule(NeatModule):
    def __init__(self, dim_size: int = 16, dtype=torch.float64):
        super().__init__()
        self.weight = NeatParameter((dim_size, dim_size), dtype=dtype)
        self.bias = NeatParameter((dim_size,), dtype=dtype)
        self.linear = nn.Linear(dim_size, dim_size, True, dtype=dtype)


def make_config() -> Config:
    config = Config("smoke", "tests")
    config.general.seed = 1
    config.genome.init_type = "normal"
    config.genome.weight_init_std = 0.5
    config.genome.weight_min_value = -np.inf
    config.genome.weight_max_value = np.inf
    config.genome.weight_mutate_rate = 0.65
    config.genome.param_epsilon = 1e-12
    config.reproduction.min_species_size = 5
    config.reproduction.purge = 0
    config.species.compatibility_threshold = np.inf
    return config


def evaluate(population, **_options):
    for index, genome in enumerate(population.genomes.values()):
        genome.fitness = float(index)


def test_population_evolves_three_generations():
    module = TinyModule()
    population = Population(20, module, make_config(), verbose=False)
    population.run(evaluate, generations=3, verbose=False)
    assert len(population.genomes) == 20
    assert all(parameter.shape[0] == 20 for parameter in module.neat_parameters())


def test_seeded_runs_are_reproducible():
    def run():
        # The backend streams keep advancing within a process; restart them (and the genome id counter) per run
        seeding.set_seed(1)
        Reproduction.genome_indexer.set(0)
        module = TinyModule()
        population = Population(10, module, make_config(), verbose=False)
        population.run(evaluate, generations=2, verbose=False)
        return [parameter.detach().clone() for parameter in module.neat_parameters()]

    first, second = run(), run()
    assert all(torch.equal(a, b) for a, b in zip(first, second))
