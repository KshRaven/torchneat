"""Evolution core: genomes, species, stagnation, reproduction state and the :class:`Population`.

Names are imported lazily (PEP 562) so that low level packages (``nn``, ``backends``) can import
``torchneat.core.genome`` without pulling in the whole evolution stack, which keeps the import graph free of cycles.
"""

from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "Genome": "genome",
    "Species": "species",
    "SpeciesSet": "species",
    "Stagnation": "stagnation",
    "Reproduction": "reproduction",
    "Population": "population",
}
_SUBMODULES = ("constants", "genome", "population", "reproduction", "species", "stagnation")

__all__ = [*_EXPORTS, *_SUBMODULES]

if TYPE_CHECKING:  # pragma: no cover
    from .genome import Genome
    from .population import Population
    from .reproduction import Reproduction
    from .species import Species, SpeciesSet
    from .stagnation import Stagnation


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    elif name in _SUBMODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
