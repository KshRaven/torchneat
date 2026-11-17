"""CPU (numba) helpers: RNG state sampling and element accessors. Device independent ones are re-exported from ``..functional``."""

import numpy as np
from numba import njit
from numpy import ndarray as CPUArray

from .. import seeding
from ..functional import calc, calc_grid, clamp, set_seed, threads_total  # noqa: F401  (re-exported)

__all__ = [
    "calc",
    "calc_grid",
    "clamp",
    "gauss",
    "get_enumeration",
    "get_rng_states",
    "normal",
    "prob",
    "set_seed",
    "uniform",
]


def get_rng_states(kernel_shape: tuple, seed: int = None, get_normal=True, dtype=np.float64) -> tuple[np.ndarray, int]:
    # `seed` is applied through ``seeding.configure(config.general.seed)``; an explicit value re-seeds the backend
    if seed is not None:
        seeding.configure(seed)
    rng = seeding.numpy_rng()
    total = threads_total(kernel_shape)
    if get_normal:
        rng_states = rng.standard_normal(size=total).astype(dtype)
    else:
        rng_states = rng.random(size=total).astype(dtype)
    return rng_states, total


@njit(cache=True)
def gauss(states: CPUArray, index: int) -> float:
    """
    Get the value at a given index for a Normal Distribution CPU Array
    :param states: 1D CPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@njit(cache=True)
def prob(states: CPUArray, index: int) -> float:
    """
    Get the value at a given index for a Probability (Uniform[0, 1]) CPU Array
    :param states: 1D CPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@njit(cache=True)
def normal(states: CPUArray, index: int, mean: float, std: float) -> float:
    return mean + gauss(states, index) * std


@njit(cache=True)
def uniform(states: CPUArray, index: int, minimum: float, maximum: float) -> float:
    difference = maximum - minimum
    return minimum + prob(states, index) * difference


@njit(cache=True)
def get_enumeration(array: CPUArray):
    positions: list[tuple[int, ...]] = [index for index, value in np.ndenumerate(array)]
    count = len(positions)
    return positions, count
