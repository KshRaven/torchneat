# Configuration guide

torchneat uses a JSON-backed configuration system to persist experiment settings for evolution runs. The main entry point is the Config wrapper exposed by the package.

## Main classes

The configuration stack is implemented in the config package:

- Configuration: base helper for loading, updating, and saving section data
- GeneralConfig: global run settings
- GenomeConfig: mutation, initialization, and weight settings
- SpeciesConfig: species compatibility rules
- StagnationConfig: rules for species stagnation
- ReproductionConfig: reproduction and selection behavior
- RLConfig: open-ended section for RL algorithm hyper-parameters (initially empty)
- Config: top-level wrapper that groups all sections together

## Creating and saving a config

```python
import torchneat as neat

config = neat.Config(file_name="demo", directory="demo")
config.general.min_pop_size = 120
config.genome.weight_mutate_rate = 0.5
config.genome.weight_mutate_power = 0.2
config.save(verbose=True)
```

This stores the configuration as a JSON file under the storage directory for the selected experiment folder.

## Section overview

### GeneralConfig

Controls the overall evolution loop:

- fitness criterion and threshold
- minimum (initial) population size, overridden by the `genomes` argument of `Population`
- number of genera, overridden by the `genera` option of `Population`

The population size applies to every genus: `Population(100, module, config, genera=2)` creates 200 genomes, 100 per
genus, and reproduction keeps each genus at no fewer than `min_pop_size` genomes. Pass `genus_sizes={genus: size}` to
`Population` to give genera different initial sizes.
- reset-on-extinction behavior
- random seed

### GenomeConfig

Controls how genomes are initialized and mutated:

- weight initialization mean and standard deviation
- weight mutation power and rate
- add/delete mutation probabilities
- structural mutation behavior
- compatibility scaling factors

### SpeciesConfig

Controls speciation:

- compatibility threshold for assigning genomes to species

### StagnationConfig

Controls how species are handled when progress stalls:

- species fitness function
- maximum stagnation period
- elitism within species

### ReproductionConfig

Controls reproduction and selection:

- elitism ratio
- clone threshold
- survival threshold
- crossover behavior and multipliers
- minimum species size
- purge behavior

### AncestryConfig

Controls lineage tracking (`Population.ancestry`):

- `enabled`: record the parents of every new genome (off by default)

`Population(..., ancestry=True | False | Ancestry(...))` overrides it, and `population.ancestry.enable()` / `.disable()` switch it at any time. Records already stored are kept while it is off. Usage is described in [Ancestry and GeneBank](ancestry-genebank.md).

### GeneBankConfig

Controls the bank of elite genomes that only act as parents (`Population.genebank`):

- `enabled`, `capacity` (genomes kept per `scope`), `top` (best genomes banked every generation)
- `scope`: `'specie'` (default, limits apply to each species) or `'genus'`
- `slots`, `per_slot`, `fill`: keep the best genomes per slot id (episode, seed), the remaining capacity is filled by the rolling top genomes when `fill` is set
- `policy` (`'fitness'` or `'oldest'`), `max_age`

`Population(..., genebank=True | False | GeneBank(...))` overrides it. Genomes can be queued with `genebank.mark(genome, slot=None, score=None)` and `genebank.discard(genome)` during evaluation, or handled before reproduction by a `genebank.scheme(bank, population)` callable. Usage is described in [Ancestry and GeneBank](ancestry-genebank.md).

### RLConfig

Holds hyper-parameters of RL algorithms (for example `rew_reg`, `ent_reg`, `gamma` for PPO) so they can be scheduled and persisted like any other setting. It is stored under `'rl config'` in the same `.mnc.json` file.

Unlike the other sections it has **no fixed fields**. It starts empty, and each algorithm registers only the variables it needs, since different algorithms use different attributes and values. Anything registered is saved, updated and loaded together with the rest of the config.

```python
config.rl.add("ent_reg", 1.0)             # register a variable (no-op if it already exists)
config.rl.add("ent_reg", 0.5, overwrite=True)
config.rl.has("ent_reg")                  # True
config.rl.pull(ppo, ["rew_reg", "ent_reg"])  # copy an algorithm's own starting values into the config
config.rl.apply(ppo)                      # push config values onto attributes the algorithm already has
config.save()
```

Variable names must not start with an underscore.

## Schedulers

Schedulers (`CosineAnnealing`, `RandomAnnealing`, `BinaryAnnealing`) modify config values over time and can target any variable in any section, including `config.rl`.

A scheduled variable that does not exist in any section is **created in `config.rl`** instead of raising an error.

Schedulers are bound to the algorithm they are passed to (`scheduler.bind(algorithm)`, done by the algorithm's constructor). After that no extra code is needed:

- Variables that don't exist in the config take their starting value from the algorithm's own attribute of the same name (or from `defaults={"ent_reg": 1.0}` if you give one).
- Every `step()` and `reset()` pushes the scheduled values onto the algorithm's attributes (`config.rl.apply`), so `self.ent_reg` etc. follow the schedule.

```python
schedulers = [
    CosineAnnealing(config, 20, 0.1, "ent_reg"),
    BinaryAnnealing(config, 15, 0.0, "pol_reg"),
]
ppo = PPO(population, schedulers=schedulers, ent_reg=1.0, pol_reg=0.33)   # binds them
# ... the algorithm steps the schedulers each epoch; call config.update() to persist values
```

The schedulers must be built from the same `Config` instance the algorithm's population uses. If a variable has no default, no config value and no matching algorithm attribute, it is created as `0.0` and a warning is emitted at the first step. A value already in the config (for example loaded from file) takes priority over the algorithm's constructor value.

## Practical notes

- Use descriptive config names for each experiment.
- Call save() after changing values if you want the settings persisted.
- Call load() to restore values from disk when continuing an experiment.
- Config files saved before `RLConfig` existed have no `'rl config'` entry; `load()` reports it as not found until you call `save()` or `update()` once.
- Only variables registered in `config.rl` at save time are written to the file.

The configuration behavior is defined in the config package and is used by the population and reproduction logic during evolution.