import numpy as np
import pytest
from numba import njit, prange

from torchneat.backends.cpu import grid, gridsize


@njit(parallel=True, cache=True)
def _fill3(out):
    for i in prange(gridsize(out.shape)):
        x, y, z = grid(3, i, out.shape)
        out[x, y, z] = i


@njit(parallel=True, cache=True)
def _fill2(out):
    for i in prange(gridsize(out.shape)):
        x, y = grid(2, i, out.shape)
        out[x, y] = x * 1000 + y


@njit(parallel=True, cache=True)
def _fill1(out):
    for i in prange(gridsize(out.shape)):
        out[grid(1, i, out.shape)] = i


def test_grid_covers_every_cell_in_c_order():
    out = np.full((3, 4, 5), -1, np.int64)
    _fill3(out)
    np.testing.assert_array_equal(out.ravel(), np.arange(60))

    out = np.full((6, 7), -1, np.int64)
    _fill2(out)
    expected = np.add.outer(np.arange(6) * 1000, np.arange(7))
    np.testing.assert_array_equal(out, expected)

    out = np.full(9, -1, np.int64)
    _fill1(out)
    np.testing.assert_array_equal(out, np.arange(9))


def test_grid_matches_unravel_index_in_python():
    for idx in range(24):
        assert grid(3, idx, (2, 3, 4)) == tuple(int(v) for v in np.unravel_index(idx, (2, 3, 4)))
        assert grid(2, idx, (4, 6)) == tuple(int(v) for v in np.unravel_index(idx, (4, 6)))
    assert grid(1, 5, (9,)) == 5
    assert gridsize((2, 3, 4)) == 24


def test_grid_rejects_unsupported_ndim():
    with pytest.raises(ValueError):
        grid(4, 0, (1, 1, 1, 1))
