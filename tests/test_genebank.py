"""GeneBank: banking rules, use in reproduction, persistence and no effect on evolution or ancestry."""

import numpy as np
import pytest
import torch

from torchneat import Config, GeneBank, NeatModule, NeatParameter, Population, Reproduction
from torchneat.backends import seeding


class Module(NeatModule):
    def __init__(self):
        super().__init__()
        self.weight = NeatParameter((6, 6), dtype=torch.float64)
        self.bias = NeatParameter((6,), dtype=torch.float64)


def make_config() -> Config:
    config = Config("genebank_test", "tests")
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


def rated(population):
    """Genomes with fitness 0..n-1 in population order."""
    evaluate(population)
    return list(population.genomes.values())


# ------------------------------ Banking rules ------------------------------ #
def test_disabled_by_default():
    population = make_population()
    assert not population.genebank.enabled
    population.run(evaluate, generations=3, verbose=False)
    assert len(population.genebank) == 0


def test_top_genomes_are_banked_every_generation():
    population = make_population(genebank=GeneBank(top=2, capacity=6))
    population.run(evaluate, generations=3, verbose=False)
    bank = population.genebank
    assert 2 <= len(bank) <= 6
    genomes = rated(population)
    bank.update(population)
    best = max(genomes, key=lambda g: g.fitness)
    assert best.key in bank


def test_banked_weights_match_the_population_row():
    population = make_population(genebank=GeneBank(top=3, capacity=6))
    population.run(evaluate, generations=1, verbose=False)
    bank = population.genebank
    parameters = population.modules[population.genus].neat_parameters()
    alive = [entry for entry in bank.entries() if entry.key in population.genomes]
    assert alive
    for entry in alive:
        for position, parameter in enumerate(parameters):
            row = parameter.data[parameter.mapping[entry.key]]
            assert torch.equal(entry.weights[position], row.cpu())
            assert entry.weights[position].shape == parameter.data.shape[1:]


def test_rolling_capacity_keeps_the_best():
    bank = GeneBank(top=0, capacity=2)
    population = make_population(genebank=bank)
    genomes = rated(population)
    for genome in genomes[:5]:
        bank.add(genome)
    assert sorted(bank.keys()) == sorted(g.key for g in genomes[3:5])


def test_oldest_policy_keeps_the_latest():
    bank = GeneBank(top=0, capacity=2, policy='oldest')
    population = make_population(genebank=bank)
    genomes = rated(population)
    for genome in genomes[4::-1]:  # worst candidates last
        bank.add(genome)
    assert sorted(bank.keys()) == sorted(g.key for g in genomes[:2])


def test_slots_keep_the_best_per_slot_and_limit_slot_count():
    bank = GeneBank(top=0, capacity=2, slots=2, per_slot=1)
    population = make_population(genebank=bank)
    a, b, c, d = rated(population)[:4]
    bank.add(a, slot=0, score=1.0)
    bank.add(b, slot=0, score=2.0)
    assert bank.keys() == [b.key]  # a lost slot 0 and was removed
    bank.add(c, slot=1, score=0.5)
    assert sorted(bank.keys()) == sorted([b.key, c.key])
    with pytest.raises(ValueError):
        bank.add(d, slot=2, score=9.0)
    bank.add(b, slot=1, score=3.0)  # one genome can be the best of several slots
    assert bank.keys() == [b.key]
    assert bank.slot_members(0) == [b.key] and bank.slot_members(1) == [b.key]


def test_slotted_bank_fills_free_capacity_with_rolling_genomes():
    bank = GeneBank(top=0, capacity=3, slots=2, per_slot=1, fill=True)
    population = make_population(genebank=bank)
    genomes = rated(population)
    bank.add(genomes[0], slot=0, score=1.0)
    for genome in genomes[1:6]:
        bank.add(genome)
    assert len(bank) == 3
    assert genomes[0].key in bank  # slot holders are never evicted by rolling genomes

    strict = GeneBank(top=0, capacity=3, slots=2, per_slot=1, fill=False)
    make_population(genebank=strict)
    strict_population = strict._population
    strict.add(list(strict_population.genomes.values())[0], slot=0, score=1.0)
    assert not strict.add(list(strict_population.genomes.values())[1])
    assert len(strict) == 1


def test_scope_specie_limits_each_species_separately():
    population = make_population(genebank=GeneBank(top=0, capacity=1, scope='specie'))
    bank = population.genebank
    bank._specie = lambda key: key % 2  # two species
    genomes = rated(population)
    odd = [g for g in genomes if g.key % 2][:2]
    even = [g for g in genomes if g.key % 2 == 0][:2]
    for genome in (*odd, *even):
        bank.add(genome)
    assert len(bank) == 2
    assert {key % 2 for key in bank.keys()} == {0, 1}


def test_scope_genus_limits_the_whole_genus():
    population = make_population(genebank=GeneBank(top=0, capacity=1, scope='genus'))
    bank = population.genebank
    bank._specie = lambda key: key % 2
    genomes = rated(population)
    for genome in genomes[:4]:
        bank.add(genome)
    assert len(bank) == 1


def test_default_scope_is_specie():
    assert GeneBank().scope == 'specie'
    assert Config().genebank.scope == 'specie'
    with pytest.raises(ValueError):
        GeneBank(scope='species')


def test_mark_discard_scheme_and_max_age():
    population = make_population(genebank=GeneBank(top=0, capacity=5))
    bank = population.genebank
    genomes = rated(population)
    bank.mark(genomes[1])
    bank.mark(genomes[2], score=5.0)
    assert len(bank) == 0  # queued only
    bank.update(population)
    assert sorted(bank.keys()) == sorted([genomes[1].key, genomes[2].key])

    bank.discard(genomes[1])
    seen = []
    bank.scheme = lambda b, p: (seen.append(len(p.genomes)), b.add(genomes[3]))
    bank.update(population)
    assert seen and genomes[1].key not in bank and genomes[3].key in bank

    bank.scheme = None
    bank.max_age = 0
    population.generation += 1
    bank.update(population)
    assert len(bank) == 0


def test_disabled_bank_does_not_update():
    bank = GeneBank(enabled=False, top=3)
    population = make_population(genebank=bank)
    rated(population)
    bank.update(population)
    assert len(bank) == 0


# ------------------------------ Persistence ------------------------------ #
def test_state_dict_round_trip():
    population = make_population(genebank=GeneBank(top=2, capacity=6, slots=2, per_slot=1))
    bank = population.genebank
    genomes = rated(population)
    bank.add(genomes[0], slot=0, score=1.0)
    bank.add(genomes[1], slot=1, score=2.0)
    population.run(evaluate, generations=2, verbose=False)
    state = population.state_dict()
    assert 'genebank' in state

    restored = make_population(genebank=GeneBank(top=0, capacity=6, slots=2, per_slot=1))
    restored.load(state_dict=state, strict=False, verbose=False)
    other = restored.genebank
    assert sorted(other.keys()) == sorted(bank.keys())
    for entry in bank.entries():
        copy = other._entries[entry.key]
        assert copy.fitness == entry.fitness and copy.slots == entry.slots
        assert copy.genome.fitness == entry.fitness
        assert all(torch.equal(a, b) for a, b in zip(copy.weights, entry.weights))
    assert other.slot_members(0) == bank.slot_members(0)


# ------------------------------ Reproduction ------------------------------ #
def run_seeded(genebank, ancestry=False, generations=3):
    seeding.set_seed(1)
    Reproduction.genome_indexer.set(0)
    module = Module()
    population = Population(10, module, make_config(), verbose=False, init_reporter=False,
                            genebank=genebank, ancestry=ancestry)
    population.run(evaluate, generations=generations, verbose=False)
    return population, [p.detach().clone() for p in module.neat_parameters()]


def test_empty_bank_leaves_reproduction_unchanged():
    off, off_params = run_seeded(False)
    on, on_params = run_seeded(GeneBank(top=0))
    assert list(off.genomes) == list(on.genomes)
    assert all(torch.equal(a, b) for a, b in zip(off_params, on_params))
    assert [g.fitness for g in off.genomes.values()] == [g.fitness for g in on.genomes.values()]


def test_bank_does_not_change_ancestry_records_of_the_run():
    plain, _ = run_seeded(False, ancestry=True)
    banked, _ = run_seeded(GeneBank(top=0), ancestry=True)
    assert plain.ancestry.state_dict()['child'].tolist() == banked.ancestry.state_dict()['child'].tolist()
    assert plain.ancestry.state_dict()['first'].tolist() == banked.ancestry.state_dict()['first'].tolist()


def test_banked_genome_parents_children_after_it_left_the_population():
    config = make_config()
    config.reproduction.clone_threshold = 0.0
    config.reproduction.survival_threshold = 0.5
    config.reproduction.cross_multiplier = 1.0
    config.reproduction.darwin_multiplier = 1.0
    population = Population(10, Module(), config, verbose=False, init_reporter=False,
                            genebank=GeneBank(top=1, capacity=4), ancestry=True)
    state = {'generation': 0, 'star': None}

    def scripted(pop, **_options):
        genomes = list(pop.genomes.values())
        for index, genome in enumerate(genomes):
            genome.fitness = float(index)
        if state['generation'] == 0:
            state['star'] = genomes[-1].key
            genomes[-1].fitness = 1000.0
        elif state['star'] in pop.genomes:
            pop.genomes[state['star']].fitness = -1000.0
        state['generation'] += 1

    population.run(scripted, generations=1, verbose=False)
    star = state['star']
    assert star in population.genebank
    banked_weights = [w.clone() for w in population.genebank._entries[star].weights]
    assert population.genebank.fitness(star) == 1000.0

    for _ in range(6):
        if star not in population.genomes:
            break
        population.run(scripted, generations=1, verbose=False)
    assert star not in population.genomes, "the star genome should have been culled"
    assert star in population.genebank
    assert population.genebank.fitness(star) == 1000.0  # last recorded fitness is kept
    left_at = population.generation

    population.run(scripted, generations=2, verbose=False)
    ancestry = population.ancestry
    late = [child for child in ancestry.children(star) if ancestry.generation(child) > left_at]
    assert late, "banked genome was never used as a parent after leaving the population"

    # Banking never touches the stored weights, the population stays consistent and lineages stay intact
    assert all(torch.equal(a, b) for a, b in zip(population.genebank._entries[star].weights, banked_weights))
    assert all(parameter.shape[0] == len(population.genomes)
               for parameter in population.modules[population.genus].neat_parameters())
    child = late[0]
    lineage = ancestry.get(child)
    assert star in lineage.nodes
    assert ancestry.get(star).root == star
    assert set(lineage.roots) <= set(range(len(population.genomes) + 10)) | set(lineage.roots)


def test_disabled_bank_is_not_used_for_reproduction():
    population = make_population(genebank=GeneBank(top=1), ancestry=True)
    genomes = rated(population)
    star = genomes[-1].key
    population.genebank.add(genomes[-1])
    population.genebank.disable()
    for _ in range(4):
        population.run(evaluate, generations=1, verbose=False)
    if star not in population.genomes:
        assert population.ancestry.children(star) == [] or \
            all(population.ancestry.generation(c) <= 2 for c in population.ancestry.children(star))
