# ---------- CPU (numba, prange) implementation of the genome methods ---------- #
# Device independent logic lives in torchneat.base.{initialization, reproduction, speciation, util, functional};
# this package only holds the numba kernels and the CPU backend that plugs them into it.

from ...config import Config
from ...core.species import Species
from ...core.stagnation import Stagnation
from ...nn import Genome, NeatModule
from ...reporting import ReporterSet
from .. import initialization as _initialization
from .. import reproduction as _reproduction
from .. import speciation as _speciation
from . import initialization, mutation, reproduction, speciation
from .grid import grid, gridsize
from .backend import BACKEND, CpuBackend
from .mutation import mutate

__all__ = [
    'BACKEND', 'CpuBackend', 'grid', 'gridsize', 'initialize', 'initialization', 'mutate', 'mutation',
    'reproduce', 'reproduction', 'speciate', 'speciation',
]


def initialize(config: Config, module: NeatModule, tpb=1, verbose: bool | int = False):
    return _initialization.initialize(BACKEND, config, module, tpb, verbose)


def reproduce(
        config: Config, genera: list[int], modules: dict[int, NeatModule], population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], generation: int, to_delete: list[int], genome_indexer: int,
        stagnation: Stagnation, species_set: Species, reporters: ReporterSet,
        tpb=1, verbose: bool | int = False, genebank=None):
    return _reproduction.reproduce(
        BACKEND, config, genera, modules, population, ancestors, generation, to_delete, genome_indexer,
        stagnation, species_set, reporters, tpb, verbose, genebank
    )


def speciate(config: Config, genera: list[int], modules: dict[int, NeatModule],
             species_set: Species, population: dict[int, Genome],
             generation: int, tpb=10, verbose: bool | int = False):
    return _speciation.speciate(BACKEND, config, genera, modules, species_set, population, generation, tpb, verbose)


# Keep the old module level entry points (e.g. `base.cpu.reproduction.reproduce`) available
initialization.initialize = initialize
reproduction.reproduce = reproduce
speciation.speciate = speciate
