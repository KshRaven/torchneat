"""TorchNEAT: CPU/GPU accelerated neuroevolution for PyTorch modules.

Typical use::

    import torchneat as tn

    config = tn.Config("demo", "demo")
    population = tn.Population(100, MyPolicy(), config)

``torchneat.nn`` mirrors ``torch.nn`` (evolvable layers plus the common PyTorch modules re-exported), so model code
only needs ``import torchneat.nn as nn``.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from . import _perf
from ._version import __version__

_perf.silence()

# name -> submodule that defines it. Everything is imported lazily (PEP 562) so ``import torchneat`` stays cheap and
# the optional GPU / logging dependencies are never touched until used.
_EXPORTS = {
    # configuration
    "Config": "config",
    # modules
    "Genome": "core.genome",
    "Ancestry": "core.ancestry",
    "GeneBank": "core.genebank",
    "Model": "nn.model",
    "NeatModule": "nn.module",
    "NeatParameter": "nn.parameter",
    # evolution
    "Population": "core.population",
    "Reproduction": "core.reproduction",
    "Specie": "core.species",
    "Species": "core.species",
    "Stagnation": "core.stagnation",
    # reporting
    "BaseReporter": "reporting",
    "ReporterSet": "reporting",
    "StdOutReporter": "reporting",
    # reinforcement learning
    "NEAT": "rl",
    "PPO": "rl",
    "A2C": "rl",
    "TRPO": "rl",
    "DQN": "rl",
    # backends
    "cuda_is_available": "backends",
    "get_device": "backends",
    "get_tpb": "backends",
    "set_device": "backends",
    "set_tpb": "backends",
    # warnings
    "perf_warnings": "_perf",
    # utils
    "CM": "utils",
    "Fore": "utils",
}
_SUBMODULES = ("backends", "config", "core", "nn", "optim", "reporting", "rl", "utils")

__all__ = ["__version__", *_EXPORTS, *_SUBMODULES]

if TYPE_CHECKING:  # pragma: no cover
    from . import backends, config, core, nn, optim, reporting, rl, utils
    from .backends import cuda_is_available, get_device, get_tpb, set_device, set_tpb
    from .config import Config
    from .core.ancestry import Ancestry
    from .core.genebank import GeneBank
    from .core.genome import Genome
    from .core.population import Population
    from .core.reproduction import Reproduction
    from .core.species import Specie, Species
    from .core.stagnation import Stagnation
    from ._perf import perf_warnings
    from .nn.model import Model
    from .nn.module import NeatModule
    from .nn.parameter import NeatParameter
    from .reporting import BaseReporter, ReporterSet, StdOutReporter
    from .rl import A2C, DQN, NEAT, PPO, TRPO
    from .utils import CM, Fore


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    elif name in _SUBMODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
