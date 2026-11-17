from numba import njit, objmode, optional, typeof, types
from numba.experimental import jitclass
from numba.typed import Dict

FLOAT   = types.float64
INT     = types.int64
STR     = typeof('str')
BOOL    = types.boolean


@jitclass([
    ('key', INT),
    ('fitness', optional(FLOAT)),
    ('_actual', optional(FLOAT)),
    ('genus', INT),
    ('created', INT),
    ('age', INT),
    ('best_fitness', optional(FLOAT)),
    ('improved', INT),
])
class Genome(object):
    """
    Identity and fitness of an individual. Its parameters (weights/biases) live in the NeatModule tensors,
    indexed through ``NeatModule.mapping[genome.key]``; crossover, mutation, distance and initialization run in
    ``torchneat.base`` (CPU / GPU backends).
    """
    def __init__(self, key: int, genus: int):
        self.key = key
        self.fitness: float | None = None
        self._actual: float | None = None
        self.genus = genus
        self.created = 0
        self.age = 0
        self.best_fitness: float | None = None
        self.improved = 0

    @property
    def actual_fitness(self):
        return self.fitness if self._actual is None else self._actual

    def __str__(self):
        with objmode(res='unicode_type'):
            res = (
                "Genome(key="+str(self.key)+", genus="+str(self.genus)+", "
                "fitness="+str(self.fitness)+", age="+str(self.age)+")"
            )
        return res


@njit(nogil=True, cache=True)
def update_genome_stats(genomes: dict[int, Genome], generation: int, minimize: bool):
    """Per generation bookkeeping: age, best fitness seen and the generation it last improved."""
    for genome in genomes.values():
        genome.age = generation - genome.created
        fitness = genome.fitness
        if fitness is None:
            continue
        best = genome.best_fitness
        if best is None or (fitness < best if minimize else fitness > best):
            genome.best_fitness = fitness
            genome.improved = generation


@njit(nogil=True, cache=True)
def mark_created(genomes: dict[int, Genome], first_key: int, generation: int):
    """Stamp the genomes spawned since ``first_key`` with the generation they first take part in."""
    for genome in genomes.values():
        if genome.key >= first_key:
            genome.created = generation
            genome.age = 0


def load_genome(struct: Dict):
    key             = struct['key']
    genus           = struct['genus']
    genome          = Genome(key, genus)
    genome.fitness  = struct['fitness']
    genome.created  = struct.get('created', 0)
    genome.age      = struct.get('age', 0)
    genome.improved = struct.get('improved', 0)
    best            = struct.get('best_fitness')
    genome.best_fitness = best if best is not None else genome.fitness
    return genome
