"""Ancestry: recording, switching on and off, lineage queries, persistence and no effect on evolution."""

import numpy as np
import pytest
import torch

from torchneat import Ancestry, Config, NeatModule, NeatParameter, Population, Reproduction
from torchneat.backends import seeding


class Module(NeatModule):
    def __init__(self):
        super().__init__()
        self.weight = NeatParameter((6, 6), dtype=torch.float64)
        self.bias = NeatParameter((6,), dtype=torch.float64)


def make_config() -> Config:
    config = Config("ancestry_test", "tests")
    config.general.seed = 1
    config.genome.weight_init_std = 0.5
    config.genome.weight_min_value = -np.inf
    config.genome.weight_max_value = np.inf
    config.genome.weight_mutate_rate = 0.65
    config.reproduction.min_species_size = 5
    config.reproduction.purge = 0
    config.species.compatibility_threshold = np.inf
    return config


def evaluate(population, **_options):
    for index, genome in enumerate(population.genomes.values()):
        genome.fitness = float(index)


def make_population(**options) -> Population:
    return Population(10, Module(), make_config(), verbose=False, init_reporter=False, **options)


def test_disabled_by_default_records_nothing():
    population = make_population()
    assert not population.ancestry.enabled
    population.run(evaluate, generations=3, verbose=False)
    assert len(population.ancestry) == 0


def test_config_enables_ancestry():
    config = make_config()
    config.ancestry.enabled = True
    population = Population(10, Module(), config, verbose=False, init_reporter=False)
    population.run(evaluate, generations=2, verbose=False)
    assert population.ancestry.enabled and len(population.ancestry) > 0


def test_records_every_new_genome_with_valid_parents():
    population = make_population(ancestry=True)
    initial = set(population.genomes.keys())
    start = population.reproduction.genome_indexer.get()
    population.run(evaluate, generations=4, verbose=False)
    ancestry = population.ancestry

    assert len(ancestry) == population.reproduction.genome_indexer.get() - start
    for key in population.genomes:
        if key in initial:
            assert key not in ancestry
            continue
        first, second = ancestry.parents(key)
        assert first < key and second < key
        assert ancestry.generation(key) >= 1


def test_lineage_graph():
    population = make_population(ancestry=True)
    initial = set(population.genomes.keys())
    population.run(evaluate, generations=4, verbose=False)
    ancestry = population.ancestry
    newest = max(population.genomes)
    lineage = ancestry.get(newest)

    assert lineage.root == newest
    assert lineage.depth >= 1
    assert set(lineage.roots) <= initial
    assert all(parent < child for parent, child in lineage.edges)
    assert all(parent in lineage.nodes and child in lineage.nodes for parent, child in lineage.edges)
    assert ancestry.get(population.genomes[newest]).keys == lineage.keys  # Genome objects work like keys
    assert lineage.tree()['key'] == newest

    limited = ancestry.get(newest, depth=1)
    assert limited.depth <= 1
    assert set(limited.nodes) <= set(lineage.nodes)

    first, second = ancestry.parents(newest)
    assert newest in ancestry.children(first)
    assert newest in ancestry.children(second)


def test_lineage_survives_genomes_leaving_the_population():
    population = make_population(ancestry=True)
    initial = set(population.genomes.keys())
    population.run(evaluate, generations=4, verbose=False)
    gone = [key for key in initial if key not in population.genomes]
    assert gone, "the original genomes should have been replaced"
    newest = max(population.genomes)
    assert set(population.ancestry.get(newest).roots) <= initial


def test_disable_keeps_records_and_enable_resumes():
    population = make_population(ancestry=True)
    population.run(evaluate, generations=2, verbose=False)
    ancestry = population.ancestry
    before = len(ancestry)
    snapshot = {key: ancestry.parents(key) for key in list(population.genomes) if key in ancestry}
    assert before > 0

    ancestry.disable()
    population.run(evaluate, generations=2, verbose=False)
    assert len(ancestry) == before
    assert all(ancestry.parents(key) == parents for key, parents in snapshot.items())

    unknown = max(population.genomes)
    assert unknown not in ancestry
    lineage = ancestry.get(unknown)
    assert lineage.roots == [unknown] and lineage.edges == []  # untracked, not an error

    ancestry.enable()
    population.run(evaluate, generations=2, verbose=False)
    assert len(ancestry) > before
    assert all(ancestry.parents(key) == parents for key, parents in snapshot.items())


def test_explicit_ancestry_instance_is_used():
    ancestry = Ancestry()
    population = make_population(ancestry=ancestry)
    assert population.ancestry is ancestry
    population.run(evaluate, generations=2, verbose=False)
    assert len(ancestry) > 0
    off = make_population(ancestry=Ancestry(enabled=False))
    off.run(evaluate, generations=2, verbose=False)
    assert len(off.ancestry) == 0


def test_state_dict_round_trip():
    population = make_population(ancestry=True)
    population.run(evaluate, generations=3, verbose=False)
    state = population.state_dict()
    assert 'ancestry' in state

    restored = make_population()
    restored.load(state_dict=state, strict=False, verbose=False)
    assert len(restored.ancestry) == len(population.ancestry)
    for key in population.genomes:
        assert restored.ancestry.parents(key) == population.ancestry.parents(key)
    # The enabled flag follows the restored population's own setting, the records are kept
    assert not restored.ancestry.enabled
    restored.ancestry.enable()
    restored.run(evaluate, generations=1, verbose=False)
    assert len(restored.ancestry) > len(population.ancestry)


def test_old_state_without_ancestry_loads():
    population = make_population(ancestry=True)
    population.run(evaluate, generations=2, verbose=False)
    state = population.state_dict()
    del state['ancestry']
    restored = make_population(ancestry=True)
    restored.load(state_dict=state, strict=False, verbose=False)
    assert len(restored.ancestry) == 0


def test_ancestry_does_not_change_evolution():
    def run(ancestry):
        seeding.set_seed(1)
        Reproduction.genome_indexer.set(0)
        module = Module()
        population = Population(10, module, make_config(), verbose=False, init_reporter=False, ancestry=ancestry)
        population.run(evaluate, generations=3, verbose=False)
        return [parameter.detach().clone() for parameter in module.neat_parameters()], list(population.genomes)

    (off_params, off_keys), (on_params, on_keys) = run(False), run(True)
    assert off_keys == on_keys
    assert all(torch.equal(a, b) for a, b in zip(off_params, on_params))


def test_merge_keeps_existing_records():
    first = make_population(ancestry=True)
    first.run(evaluate, generations=2, verbose=False)
    second = make_population(ancestry=True)
    second.run(evaluate, generations=2, verbose=False)
    keys = set(first.ancestry.state_dict()['child'].tolist())
    keep = {key: first.ancestry.parents(key) for key in keys}
    first.ancestry.merge(second.ancestry)
    assert all(first.ancestry.parents(key) == parents for key, parents in keep.items())
    child = first.ancestry.state_dict()['child']
    assert np.all(child[1:] > child[:-1])


def test_unknown_key_is_not_found():
    ancestry = Ancestry()
    assert ancestry.parents(12345) is None
    assert 12345 not in ancestry
    assert ancestry.get(12345).roots == [12345]
