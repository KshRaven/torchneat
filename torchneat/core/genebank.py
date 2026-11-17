"""Storage of elite genomes (identity and weights) that only take part in reproduction as parents.

A banked genome survives culling, elitism cut-offs and ``to_delete``. It is never evaluated or placed in the
population again; reproduction draws it into the parent pool of its genus next to the surviving members, scored with
the fitness recorded when it was banked, so the darwin multiplier keeps working for it.

Two ways of filling, which can be mixed:

* rolling: the best ``capacity`` genomes seen per ``scope`` (``top`` of them are added automatically every generation).
* slotted: the best ``per_slot`` genomes for every slot id (an episode index, a seed ...) per ``scope``, set with
  ``mark(genome, slot=...)`` or ``add``. Whatever capacity the slots leave free is filled by the rolling candidates
  when ``fill`` is set.

``scope`` is ``'specie'`` (default: limits apply to every species separately) or ``'genus'``.

Weights are cloned to the CPU when a genome enters the bank; genomes that are already banked cost nothing.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Iterable

import torch
from numba.typed import List

from .genome import Genome, load_genome

GENOME = Genome.class_type.instance_type
_SETTINGS = ('capacity', 'slots', 'per_slot', 'fill', 'scope', 'policy', 'max_age', 'top')


class BankEntry:
    __slots__ = ('key', 'genus', 'fitness', 'score', 'added', 'rolling', 'slots', 'specie', 'genome', 'weights')

    def __init__(self, key, genus, fitness, score, added, rolling, specie, genome, weights):
        self.key, self.genus, self.fitness, self.score = key, genus, fitness, score
        self.added, self.rolling, self.specie = added, rolling, specie
        self.slots: set[int] = set()
        self.genome: Genome = genome
        self.weights: list[torch.Tensor] = weights


class GeneBank:
    def __init__(self, enabled: bool = True, capacity: int = 10, slots: int | None = None, per_slot: int = 1,
                 fill: bool = True, scope: str = 'specie', policy: str = 'fitness',
                 max_age: int | None = None, top: int = 1,
                 scheme: Callable[['GeneBank', Any], None] | None = None):
        if scope not in ('specie', 'genus'):
            raise ValueError(f"Unknown scope '{scope}', use 'specie' or 'genus'")
        if policy not in ('fitness', 'oldest'):
            raise ValueError(f"Unknown policy '{policy}', use 'fitness' or 'oldest'")
        self._enabled = bool(enabled)
        self.slots, self.per_slot = slots, max(1, int(per_slot))
        self.capacity = max(1, int(capacity))
        if slots is not None:
            self.capacity = max(self.capacity, int(slots) * self.per_slot)
        self.fill, self.scope, self.policy, self.max_age = bool(fill), scope, policy, max_age
        self.top = max(0, int(top))
        self.scheme = scheme
        self._entries: dict[int, BankEntry] = {}
        self._slot_table: dict[tuple[tuple[int, int | None], int], dict[int, float]] = {}
        self._pending: list[tuple[int, int | None, float | None]] = []
        self._removals: set[int] = set()
        self._population = None

    @classmethod
    def from_config(cls, config) -> 'GeneBank':
        values = {name: getattr(config, name) for name in _SETTINGS if hasattr(config, name)}
        return cls(enabled=config.enabled, **values)

    # ------------------------------ State ------------------------------ #
    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True
        return self

    def disable(self):
        self._enabled = False
        return self

    def attach(self, population):
        """Population whose module weights are read when genomes are banked (set by ``Population``)."""
        self._population = population
        return self

    @property
    def _minimize(self) -> bool:
        return self._population is not None and self._population.config.general.fitness_criterion == 'min'

    def _rank(self, value: float) -> float:
        return -value if self._minimize else value

    def __len__(self):
        return len(self._entries)

    def __contains__(self, item) -> bool:
        return int(getattr(item, 'key', item)) in self._entries

    def keys(self, genus: int | None = None) -> list[int]:
        return [k for k, e in self._entries.items() if genus is None or e.genus == genus]

    def entries(self, genus: int | None = None) -> list[BankEntry]:
        return [e for e in self._entries.values() if genus is None or e.genus == genus]

    def fitness(self, genome) -> float | None:
        entry = self._entries.get(int(getattr(genome, 'key', genome)))
        return None if entry is None else entry.fitness

    def slot_members(self, slot: int, genus: int | None = None) -> list[int]:
        return [key for (group, s), table in self._slot_table.items()
                if s == slot and genus in (None, group[0]) for key in table]

    # ------------------------------ Queueing (cheap, usable inside the evaluation function) ------------------------------ #
    def mark(self, genome, slot: int | None = None, score: float | None = None):
        """Queue a genome for banking. It is stored before the next reproduction, with its fitness at that time.

        ``score`` ranks the genome inside its slot (or among the rolling genomes); it defaults to the fitness."""
        self._pending.append((int(getattr(genome, 'key', genome)), slot, score))

    def unmark(self, genome):
        key = int(getattr(genome, 'key', genome))
        self._pending = [item for item in self._pending if item[0] != key]

    def discard(self, genome):
        """Queue a banked genome for removal before the next reproduction."""
        self._removals.add(int(getattr(genome, 'key', genome)))

    # ------------------------------ Direct changes ------------------------------ #
    def add(self, genome, slot: int | None = None, score: float | None = None) -> bool:
        """Bank ``genome`` now. Returns whether it is in the bank afterwards."""
        population = self._population
        key = int(getattr(genome, 'key', genome))
        if not isinstance(genome, Genome):
            genome = population.genomes.get(key) if population is not None else None
        if genome is None:
            return False
        fitness = genome.fitness if genome.fitness is not None else score
        if fitness is None or fitness != fitness:
            return False
        value = float(fitness if score is None else score)
        genus = genome.genus
        rank = self._rank(value)
        entry = self._entries.get(key)
        group = self._group(genus, self._specie(key) if entry is None else entry.specie)

        if slot is None:
            if self.slots is not None and not self.fill:
                return entry is not None
            if entry is None and not self._rolling_fits(group, rank):
                return False
        else:
            if self.slots is not None and (group, slot) not in self._slot_table and \
                    sum(1 for g, _ in self._slot_table if g == group) >= self.slots:
                raise ValueError(f"The bank has no free slot for {slot!r} (slots={self.slots})")
            table = self._slot_table.get((group, slot), {})
            if key not in table and len(table) >= self.per_slot and \
                    rank <= min(self._rank(v) for v in table.values()):
                return entry is not None

        if entry is None:
            entry = self._create(genome, fitness, value, slot is None)
        else:
            if slot is None:
                entry.rolling = True
            if rank > self._rank(entry.score):
                entry.score, entry.fitness = value, float(fitness)
                entry.genome.fitness = float(fitness)
        if slot is not None:
            table = self._slot_table.setdefault((group, slot), {})
            table[key] = value
            entry.slots.add(slot)
            if len(table) > self.per_slot:
                worst = min(table, key=lambda k: self._rank(table[k]))
                self._unslot(group, slot, worst)
        self._trim(group)
        return key in self._entries

    def remove(self, genome):
        key = int(getattr(genome, 'key', genome))
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        group = self._group(entry.genus, entry.specie)
        for slot in list(entry.slots):
            table = self._slot_table.get((group, slot))
            if table is not None:
                table.pop(key, None)
                if not table:
                    del self._slot_table[(group, slot)]
        self._pending = [item for item in self._pending if item[0] != key]

    def clear(self, slot: int | None = None):
        """Empty the bank, or only the genomes held by ``slot``."""
        if slot is None:
            self._entries.clear()
            self._slot_table.clear()
            self._pending.clear()
            self._removals.clear()
            return
        for key in self.slot_members(slot):
            self.remove(key)

    def drop_genera(self, genera: Iterable[int]):
        genera = set(genera)
        for key in [k for k, e in self._entries.items() if e.genus in genera]:
            self.remove(key)

    def merge(self, other: 'GeneBank'):
        for key, entry in other._entries.items():
            if key in self._entries:
                continue
            self._entries[key] = entry
            group = self._group(entry.genus, entry.specie)
            for slot in entry.slots:
                self._slot_table.setdefault((group, slot), {})[key] = other._slot_table[(group, slot)][key]

    # ------------------------------ Internals ------------------------------ #
    def _group(self, genus: int, specie) -> tuple[int, int | None]:
        return (genus, specie if self.scope == 'specie' else None)

    def _specie(self, key: int):
        population = self._population
        return None if population is None else population.species_set.genome_to_species.get(key)

    def _create(self, genome: Genome, fitness: float, score: float, rolling: bool) -> BankEntry:
        population = self._population
        if population is None:
            raise RuntimeError("The bank is not attached to a Population, so weights cannot be read")
        module = population.modules[genome.genus]
        weights = []
        for parameter in module.neat_parameters():
            weights.append(parameter.data[parameter.mapping[genome.key]].detach().clone().cpu())
        copy = Genome(genome.key, genome.genus)
        copy.fitness = float(fitness)
        copy.created, copy.age, copy.improved = genome.created, genome.age, genome.improved
        copy.best_fitness = genome.best_fitness
        entry = BankEntry(genome.key, genome.genus, float(fitness), score, population.generation, rolling,
                          self._specie(genome.key), copy, weights)
        self._entries[genome.key] = entry
        return entry

    def _unslot(self, group: tuple[int, int | None], slot: int, key: int):
        table = self._slot_table.get((group, slot))
        if table is not None:
            table.pop(key, None)
            if not table:
                del self._slot_table[(group, slot)]
        entry = self._entries.get(key)
        if entry is not None:
            entry.slots.discard(slot)
            if not entry.slots and not entry.rolling:
                del self._entries[key]

    def _entry_group(self, entry: BankEntry) -> tuple[int, int | None]:
        return self._group(entry.genus, entry.specie)

    def _free(self, group: tuple[int, int | None]) -> int:
        protected = sum(1 for e in self._entries.values() if self._entry_group(e) == group and e.slots)
        return max(0, self.capacity - protected)

    def _rolling(self, group: tuple[int, int | None]) -> list[BankEntry]:
        return [e for e in self._entries.values() if self._entry_group(e) == group and not e.slots and e.rolling]

    def _rolling_fits(self, group: tuple[int, int | None], rank: float) -> bool:
        rolling = self._rolling(group)
        if len(rolling) < self._free(group) or self.policy == 'oldest':
            return self._free(group) > 0
        return bool(rolling) and rank > min(self._rank(e.score) for e in rolling)

    def _trim(self, group: tuple[int, int | None]):
        rolling = self._rolling(group)
        excess = len(rolling) - self._free(group)
        if excess > 0:
            order = sorted(rolling, key=(lambda e: self._rank(e.score)) if self.policy == 'fitness'
                           else (lambda e: e.added))
            for entry in order[:excess]:
                self._evict(entry)

    def _evict(self, entry: BankEntry):
        entry.rolling = False
        if not entry.slots:
            self._entries.pop(entry.key, None)

    # ------------------------------ Population hook ------------------------------ #
    def update(self, population=None):
        """Apply queued removals and marks, the custom scheme and the automatic top genomes, then enforce the limits.
        Called by ``Population`` right before reproduction, once the fitnesses of the generation are final."""
        if not self._enabled:
            return
        if population is not None:
            self._population = population
        population = self._population
        for key in self._removals:
            self.remove(key)
        self._removals.clear()
        if self.scheme is not None:
            self.scheme(self, population)
        pending, self._pending = self._pending, []
        for key, slot, score in pending:
            self.add(key, slot, score)
        if self.top > 0:
            for genus in population.genera:
                candidates = [g for g in population.genomes.values() if g.genus == genus and g.fitness is not None
                              and g.fitness == g.fitness]
                pick = heapq.nsmallest if self._minimize else heapq.nlargest
                for genome in pick(self.top, candidates, key=lambda g: g.fitness):
                    self.add(genome)
        if self.max_age is not None:
            for entry in [e for e in self._entries.values() if population.generation - e.added > self.max_age]:
                self.remove(entry.key)
        self._regroup(population)
        for group in {self._entry_group(e) for e in self._entries.values()}:
            self._trim(group)

    def _regroup(self, population):
        """Follow the species ids: living genomes take their current species, others lose a species that is gone."""
        if self.scope != 'specie':
            return
        species = population.species_set
        moved = []
        for entry in self._entries.values():
            current = species.genome_to_species.get(entry.key)
            if current is None and entry.specie is not None and entry.specie not in species.species:
                current = -1
            if current is not None and current != entry.specie:
                moved.append((entry, None if current == -1 else current))
        for entry, specie in moved:
            old = self._entry_group(entry)
            entry.specie = specie
            new = self._entry_group(entry)
            for slot in entry.slots:
                table = self._slot_table.get((old, slot))
                if table is not None and entry.key in table:
                    self._slot_table.setdefault((new, slot), {})[entry.key] = table.pop(entry.key)
                    if not table:
                        del self._slot_table[(old, slot)]

    # ------------------------------ Reproduction access ------------------------------ #
    def members(self, genus: int) -> List:
        """Genome records of ``genus`` (typed list) carrying the fitness they were banked with."""
        genomes = List.empty_list(GENOME)
        for entry in self._entries.values():
            if entry.genus == genus:
                genomes.append(entry.genome)
        return genomes

    def stack(self, keys: list[int], position: int, like: torch.Tensor) -> torch.Tensor:
        """Weights of parameter ``position`` for the banked ``keys``, as rows matching the tensor ``like``."""
        rows = torch.stack([self._entries[key].weights[position] for key in keys])
        return rows.to(device=like.device, dtype=like.dtype)

    # ------------------------------ Persistence ------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        entries = []
        for entry in self._entries.values():
            group = self._entry_group(entry)
            slots = [(slot, self._slot_table[(group, slot)][entry.key]) for slot in entry.slots]
            entries.append({
                'key': entry.key, 'genus': entry.genus, 'fitness': entry.fitness, 'score': entry.score,
                'added': entry.added, 'rolling': entry.rolling, 'slots': slots, 'specie': entry.specie,
                'created': entry.genome.created, 'age': entry.genome.age, 'improved': entry.genome.improved,
                'best_fitness': entry.genome.best_fitness,
                'weights': [w.clone() for w in entry.weights],
            })
        return {
            'enabled': self._enabled,
            'settings': {name: getattr(self, name) for name in _SETTINGS},
            'entries': entries,
        }

    def load_state_dict(self, state: dict[str, Any], load_settings: bool = False, load_enabled: bool = False):
        """Replace the banked genomes with a saved state. Settings and the enabled flag stay as they are
        unless ``load_settings`` / ``load_enabled``."""
        if load_settings:
            for name, value in state.get('settings', {}).items():
                setattr(self, name, value)
        if load_enabled:
            self._enabled = bool(state.get('enabled', self._enabled))
        self.clear()
        for data in state.get('entries', []):
            genome = load_genome({
                'key': data['key'], 'genus': data['genus'], 'fitness': data['fitness'],
                'created': data['created'], 'age': data['age'], 'improved': data['improved'],
                'best_fitness': data['best_fitness'],
            })
            entry = BankEntry(data['key'], data['genus'], data['fitness'], data['score'], data['added'],
                              data['rolling'], data['specie'], genome, [w.clone() for w in data['weights']])
            self._entries[entry.key] = entry
            group = self._entry_group(entry)
            for slot, score in data['slots']:
                entry.slots.add(slot)
                self._slot_table.setdefault((group, slot), {})[entry.key] = score
        return self
