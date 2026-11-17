# Ancestry and GeneBank

Both are optional parts of a `Population` (`population.ancestry`, `population.genebank`). Each can be switched on in the config, overridden in the `Population` constructor, passed in as an object, and toggled at runtime with `enable()` / `disable()`. Both are saved and loaded with the population state.

## Ancestry

Records which two genomes produced every new genome. Only keys, generations and genera are stored (flat integer arrays, no genomes or weights), filled by one jitted pass per generation while enabled. Lineages are built on demand.

```python
population = Population(100, module, config, ancestry=True)
population.run(evaluate, generations=50)

lineage = population.ancestry.get(best_genome)       # a Genome or its key
lineage.depth                                        # longest recorded parent chain
lineage.roots                                        # keys without recorded parents
lineage.layers()                                     # {generation: [keys]}
lineage.tree()                                       # nested {'key', 'generation', 'parents': [...]}
lineage.edges                                        # [(parent, child), ...]
graph = lineage.to_networkx()                        # needs networkx

population.ancestry.get(best_genome, depth=3)        # only the last 3 generations of parents
population.ancestry.parents(key)                     # (first, second) or None
population.ancestry.children(key)                    # keys of every recorded child
population.ancestry.generation(key)                  # birth generation or None
population.ancestry.lineages(genomes, depth=None)    # {key: Lineage}
```

Enabling:

- `config.ancestry.enabled = True`
- `Population(..., ancestry=True | False)` overrides the config
- `Population(..., ancestry=Ancestry(capacity=4096))` passes an own instance
- `population.ancestry.disable()` / `.enable()` at any time; records made so far are kept while disabled, and genomes born during the gap are simply unrecorded (they show up as roots)
- `population.ancestry.clear()` drops everything

Genomes that are no longer in the population stay in the records, and lineages of banked genomes are kept.

## GeneBank

Stores elite genomes together with their weights. A banked genome survives culling, elitism cut-offs and `to_delete`. It is never evaluated or inserted into the population again; reproduction adds it to the parent pool of its genus next to the surviving members. It is scored with the fitness recorded when it was banked, so the darwin multiplier still applies to it. The genome keeps its lineage in `Ancestry`.

```python
bank = GeneBank(capacity=10, top=1, scope='specie')
population = Population(100, module, config, genebank=bank)
```

Settings (`config.genebank` or constructor arguments):

| Setting | Meaning |
|---|---|
| `enabled` | Whether the bank is updated and used |
| `capacity` | Genomes kept per `scope` |
| `top` | Best genomes of every genus banked automatically each generation (`0` disables the automatic filling) |
| `scope` | `'specie'` (default): limits apply to every species separately. `'genus'`: limits apply to the whole genus |
| `slots`, `per_slot` | Keep the best `per_slot` genomes for each of `slots` slot ids (episode index, seed, scenario) |
| `fill` | Fill capacity the slots leave free with the rolling top genomes |
| `policy` | `'fitness'` evicts the worst, `'oldest'` evicts the earliest added |
| `max_age` | Remove genomes banked more than this many generations ago |
| `scheme` | Callable `scheme(bank, population)` run right before reproduction |

Weights are copied to the CPU when a genome enters the bank; a genome that is already banked costs nothing.

### Filling

Rolling: the best `capacity` genomes seen per scope. Slotted: the best genomes per slot id, for example one slot per generation seed so the bank ends up with the best genomes of each seed, which can be the same genome for several seeds.

```python
def evaluate(population, **options):
    for seed in range(10):
        run_episode(population, seed)
        for genome in population.genomes.values():
            population.genebank.mark(genome, slot=seed, score=genome.episode_score)
```

### When it is updated

`genebank.update(population)` runs automatically right before reproduction, after the fitnesses of the generation are final (this includes fitnesses set by an RL trainer after the evaluation function). In order it applies: queued `discard`s, the `scheme`, queued `mark`s, the automatic `top` genomes, `max_age`, and then the limits.

Ways to control it:

- During evaluation: `bank.mark(genome, slot=None, score=None)`, `bank.unmark(genome)`, `bank.discard(genome)`. `score` ranks the genome inside its slot or among the rolling genomes and defaults to the fitness.
- Before reproduction with final fitnesses: a `scheme`, for example

```python
def scheme(bank, population):
    for genome in population.genomes.values():
        if genome.fitness is not None and genome.fitness > threshold:
            bank.add(genome, slot=genome.key % 3)
    bank.remove(worst_key)

bank.scheme = scheme
```

- Directly: `bank.add(genome, slot=None, score=None)` returns whether it is banked, `bank.remove(genome)`, `bank.clear(slot=None)`.

### Inspecting

```python
len(bank); key in bank
bank.keys(genus=None); bank.entries(genus=None)   # entries: key, genus, fitness, score, added, slots, genome, weights
bank.fitness(genome)                              # last recorded fitness
bank.slot_members(slot)
```

### Notes

- With an empty or disabled bank, reproduction and ancestry records are identical to a run without a bank.
- `Population.state_dict()` contains `'genebank'` and `'ancestry'`; loading keeps your current settings unless you pass `load_settings=True` / `load_enabled=True` to the object's `load_state_dict`.
