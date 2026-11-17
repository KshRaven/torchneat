"""Optional, key-only lineage tracking.

Reproduction already knows the parents of every child of a generation. :class:`Ancestry` copies just those keys into
flat arrays (no genomes, no weights) with one jitted pass per generation, and only while it is enabled. Lineages are
built on demand by :meth:`Ancestry.get`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _collect(ancestors, genomes, child, first, second, genus, start):
    index = start
    for key, pair in ancestors.items():
        child[index]  = key
        first[index]  = pair[0].key
        second[index] = pair[1].key
        genus[index]  = genomes[key].genus if key in genomes else pair[0].genus
        index += 1
    return index


def _key_of(item) -> int:
    return int(item.key if hasattr(item, 'key') else item)


@dataclass
class Lineage:
    """Ancestors of one genome. ``nodes`` maps key -> (parent1, parent2, generation, genus); roots have parents ``None``."""
    root: int
    nodes: dict[int, tuple[int | None, int | None, int | None, int | None]] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)  # (parent, child)

    @property
    def keys(self) -> list[int]:
        return list(self.nodes)

    @property
    def roots(self) -> list[int]:
        """Keys with no recorded parents (initial genomes, or genomes born while tracking was off)."""
        return [key for key, node in self.nodes.items() if node[0] is None]

    @property
    def depth(self) -> int:
        """Longest chain of recorded parents above the root."""
        parents_of: dict[int, list[int]] = {}
        for parent, child in self.edges:
            parents_of.setdefault(child, []).append(parent)
        levels: dict[int, int] = {}
        for key in sorted(self.nodes, key=lambda k: -1 if self.nodes[k][2] is None else self.nodes[k][2]):
            parents = parents_of.get(key, [])
            levels[key] = 1 + max(levels.get(p, 0) for p in parents) if parents else 0
        return levels[self.root]

    def layers(self) -> dict[int | None, list[int]]:
        """Keys grouped by the generation they were born in."""
        result: dict[int | None, list[int]] = {}
        for key, node in self.nodes.items():
            result.setdefault(node[2], []).append(key)
        return result

    def tree(self, key: int | None = None) -> dict[str, Any]:
        """Nested ``{'key', 'parents': [...]}`` form (shared ancestors are repeated)."""
        key = self.root if key is None else key
        parent1, parent2 = self.nodes[key][:2]
        parents = [] if parent1 is None else [parent1] if parent1 == parent2 else [parent1, parent2]
        return {'key': key, 'generation': self.nodes[key][2], 'parents': [self.tree(p) for p in parents]}

    def to_dict(self) -> dict[str, Any]:
        return {'root': self.root, 'nodes': dict(self.nodes), 'edges': list(self.edges)}

    def to_networkx(self):
        import networkx as nx
        graph = nx.DiGraph()
        for key, (_, _, generation, genus) in self.nodes.items():
            graph.add_node(key, generation=generation, genus=genus)
        graph.add_edges_from(self.edges)
        return graph


class Ancestry:
    def __init__(self, enabled: bool = True, capacity: int = 1024):
        self._enabled = bool(enabled)
        self._n = 0
        self._child  = np.empty(capacity, np.int64)
        self._first  = np.empty(capacity, np.int64)
        self._second = np.empty(capacity, np.int64)
        self._gen    = np.empty(capacity, np.int32)
        self._genus  = np.empty(capacity, np.int32)

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

    def clear(self):
        self._n = 0

    def __len__(self):
        return self._n

    def __contains__(self, item) -> bool:
        return self._row(_key_of(item)) >= 0

    # ------------------------------ Recording ------------------------------ #
    def _reserve(self, extra: int):
        needed = self._n + extra
        capacity = len(self._child)
        if needed <= capacity:
            return
        capacity = max(needed, capacity * 2)
        for name in ('_child', '_first', '_second', '_gen', '_genus'):
            old = getattr(self, name)
            new = np.empty(capacity, old.dtype)
            new[:self._n] = old[:self._n]
            setattr(self, name, new)

    def record(self, ancestors, genomes, generation: int):
        """Store the parents of the children of ``generation`` (the typed dict filled by reproduction)."""
        if not self._enabled or len(ancestors) == 0:
            return
        self._reserve(len(ancestors))
        start = self._n
        end = _collect(ancestors, genomes, self._child, self._first, self._second, self._genus, start)
        self._gen[start:end] = generation
        self._n = end
        batch = self._child[start:end]
        if (len(batch) > 1 and not np.all(batch[1:] > batch[:-1])) or (start > 0 and batch[0] <= self._child[start - 1]):
            self._sort()  # keeps key lookups valid for out of order keys

    def _sort(self):
        order = np.argsort(self._child[:self._n], kind='stable')
        for name in ('_child', '_first', '_second', '_gen', '_genus'):
            array = getattr(self, name)
            array[:self._n] = array[:self._n][order]

    def merge(self, other: 'Ancestry'):
        """Add the records of another ancestry (records already present are kept)."""
        if other._n == 0:
            return
        known = set(self._child[:self._n].tolist())
        rows = [i for i, key in enumerate(other._child[:other._n].tolist()) if key not in known]
        if not rows:
            return
        self._reserve(len(rows))
        end = self._n + len(rows)
        for name in ('_child', '_first', '_second', '_gen', '_genus'):
            getattr(self, name)[self._n:end] = getattr(other, name)[rows]
        self._n = end
        self._sort()

    # ------------------------------ Queries ------------------------------ #
    def _row(self, key: int) -> int:
        row = int(np.searchsorted(self._child[:self._n], key))
        return row if row < self._n and self._child[row] == key else -1

    def parents(self, genome) -> tuple[int, int] | None:
        row = self._row(_key_of(genome))
        return None if row < 0 else (int(self._first[row]), int(self._second[row]))

    def generation(self, genome) -> int | None:
        row = self._row(_key_of(genome))
        return None if row < 0 else int(self._gen[row])

    def children(self, genome) -> list[int]:
        key = _key_of(genome)
        rows = np.flatnonzero((self._first[:self._n] == key) | (self._second[:self._n] == key))
        return self._child[rows].tolist()

    def get(self, genome, depth: int | None = None) -> Lineage:
        """Lineage of a genome (a ``Genome`` or its key), at most ``depth`` generations of parents deep."""
        root = _key_of(genome)
        lineage = Lineage(root)
        queue = deque([(root, 0)])
        while queue:
            key, level = queue.popleft()
            if key in lineage.nodes:
                continue
            row = self._row(key)
            if row < 0:
                lineage.nodes[key] = (None, None, None, None)
                continue
            first, second = int(self._first[row]), int(self._second[row])
            lineage.nodes[key] = (first, second, int(self._gen[row]), int(self._genus[row]))
            if depth is not None and level >= depth:
                continue
            for parent in dict.fromkeys((first, second)):
                lineage.edges.append((parent, key))
                queue.append((parent, level + 1))
        return lineage

    def lineages(self, genomes: Iterable, depth: int | None = None) -> dict[int, Lineage]:
        return {_key_of(g): self.get(g, depth) for g in genomes}

    # ------------------------------ Persistence ------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        n = self._n
        return {
            'enabled': self._enabled,
            'child': self._child[:n].copy(), 'first': self._first[:n].copy(), 'second': self._second[:n].copy(),
            'generation': self._gen[:n].copy(), 'genus': self._genus[:n].copy(),
        }

    def load_state_dict(self, state: dict[str, Any], load_enabled: bool = False):
        """Replace the records with a saved state. The enabled flag is kept unless ``load_enabled``."""
        n = len(state['child'])
        capacity = max(n, 1024)
        self._child  = np.empty(capacity, np.int64)
        self._first  = np.empty(capacity, np.int64)
        self._second = np.empty(capacity, np.int64)
        self._gen    = np.empty(capacity, np.int32)
        self._genus  = np.empty(capacity, np.int32)
        self._child[:n], self._first[:n], self._second[:n] = state['child'], state['first'], state['second']
        self._gen[:n], self._genus[:n] = state['generation'], state['genus']
        self._n = n
        if load_enabled:
            self._enabled = bool(state.get('enabled', self._enabled))
        return self
