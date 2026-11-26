"""
``grid()`` for the CPU kernels: the ``numba.cuda.grid`` of a ``prange``.

A CUDA thread asks ``cuda.grid(ndim)`` for its position in the launch grid. A ``prange`` iteration only has a flat
index, so ``grid(ndim, idx, dims)`` turns that index into the ``ndim`` coordinates of the cell it stands for. Looping
``prange(gridsize(dims))`` and calling ``grid`` inside runs every ``(x, y, z)`` cell as its own work item, with no
per-row load imbalance and nothing to synchronize.

Coordinates follow the C order of the arrays the kernels index: the last dimension changes fastest.

    @njit(parallel=True, cache=True)
    def kernel(array):
        for i in prange(gridsize(array.shape)):
            x, y, z = grid(3, i, array.shape)
            array[x, y, z] += 1

Do not combine ``parallel=True`` with ``nogil=True``: it serializes the parallel region on several platforms.
"""

import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.extending import overload

__all__ = ["grid", "gridsize"]


def grid(ndim, idx, dims):
    """Coordinates of flat index ``idx`` in a grid of shape ``dims``: an int for ``ndim == 1``, else a tuple."""
    idx = int(idx)
    if ndim == 1:
        return idx
    if ndim == 2:
        return idx // int(dims[1]), idx % int(dims[1])
    if ndim == 3:
        plane = int(dims[1]) * int(dims[2])
        return idx // plane, (idx % plane) // int(dims[2]), idx % int(dims[2])
    raise ValueError(f"grid supports 1 to 3 dimensions, got {ndim}")


def gridsize(dims):
    """Number of cells of a grid of shape ``dims``."""
    total = 1
    for dim in dims:
        total *= int(dim)
    return total


@overload(grid, prefer_literal=True)
def _grid_overload(ndim, idx, dims):
    if not isinstance(dims, types.BaseTuple):
        raise TypingError("grid() needs the grid shape as a tuple of integers")
    count = len(dims)
    if isinstance(ndim, types.IntegerLiteral) and ndim.literal_value != count:
        raise TypingError(f"grid({ndim.literal_value}, ...) was given a shape with {count} dimensions")

    if count == 1:
        def impl(ndim, idx, dims):
            return np.int64(idx)
    elif count == 2:
        def impl(ndim, idx, dims):
            i = np.int64(idx)
            d1 = np.int64(dims[1])
            return i // d1, i % d1
    elif count == 3:
        def impl(ndim, idx, dims):
            i = np.int64(idx)
            d1 = np.int64(dims[1])
            d2 = np.int64(dims[2])
            plane = d1 * d2
            return i // plane, (i % plane) // d2, i % d2
    else:
        raise TypingError(f"grid supports 1 to 3 dimensions, got {count}")
    return impl


@overload(gridsize)
def _gridsize_overload(dims):
    if not isinstance(dims, types.BaseTuple):
        raise TypingError("gridsize() needs the grid shape as a tuple of integers")
    count = len(dims)

    if count == 1:
        def impl(dims):
            return np.int64(dims[0])
    elif count == 2:
        def impl(dims):
            return np.int64(dims[0]) * np.int64(dims[1])
    elif count == 3:
        def impl(dims):
            return np.int64(dims[0]) * np.int64(dims[1]) * np.int64(dims[2])
    else:
        raise TypingError(f"gridsize supports 1 to 3 dimensions, got {count}")
    return impl
