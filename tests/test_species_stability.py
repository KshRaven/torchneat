"""Species keep their identity across generations because their representative always carries over."""

import numpy as np
import torch

from torchneat import Config, NeatModule, NeatParameter, Population


class Module(NeatModule):
    def __init__(self):
        super().__init__()
        self.weight = NeatParameter((6, 6), dtype=torch.float64)
        self.bias = NeatParameter((6,), dtype=torch.float64)


def make_config() -> Config:
    config = Config("species_test", "tests")
    config.general.seed = 3
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


def test_species_id_is_stable():
    population = Population(10, Module(), make_config(), verbose=False, init_reporter=False)
    first_ids = set(population.species_set.species.keys())
    assert len(first_ids) == 1
    for _ in range(8):
        population.run(evaluate, generations=1, verbose=False)
        assert set(population.species_set.species.keys()) == first_ids
    specie = next(iter(population.species_set.species.values()))
    assert specie.age >= 6
    assert specie.created == 0


def test_previous_representative_carries_over_and_stays_a_member():
    population = Population(10, Module(), make_config(), verbose=False, init_reporter=False)
    for _ in range(6):
        specie = next(iter(population.species_set.species.values()))
        previous = specie.representative.key
        population.run(evaluate, generations=1, verbose=False)
        specie = next(iter(population.species_set.species.values()))
        assert previous in population.genomes
        assert previous in specie.members
        assert specie.representative.key in population.genomes
        assert specie.representative.key != previous  # drifts to the nearest genome instead of staying fixed


def test_population_size_is_kept():
    population = Population(10, Module(), make_config(), verbose=False, init_reporter=False)
    for _ in range(5):
        population.run(evaluate, generations=1, verbose=False)
        assert len(population.genomes) == 10
