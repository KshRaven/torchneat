"""GPU (numba.cuda / cupy) helpers: RNG state sampling and element accessors. Device independent ones are re-exported from ``..functional``."""

from typing import Union

import cupy as cp
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray

from .. import seeding
from ..functional import calc, calc_grid, clamp, set_seed, threads_total  # noqa: F401  (re-exported)

__all__ = [
    "calc",
    "calc_grid",
    "clamp",
    "gauss",
    "get_rng_states",
    "get_value",
    "normal",
    "prob",
    "set_seed",
    "set_value",
    "uniform",
]


def get_rng_states(kernel_shape: tuple, seed: int = None, get_normal=True, use_cuda=True) -> tuple[cp.ndarray, int]:
    if seed is not None:
        seeding.configure(seed)
    total = threads_total(kernel_shape)
    if use_cuda:
        rng = seeding.cupy_rng()
        rng_states = rng.standard_normal(size=total, dtype=cp.float32) if get_normal \
            else rng.random(size=total, dtype=cp.float32)
    else:
        rng = seeding.numpy_rng()
        rng_states = (rng.standard_normal(size=total) if get_normal else rng.random(size=total)).astype(cp.float32)
    return rng_states, total


@cuda.jit(device=True)
def gauss(states: GPUArray, index: int) -> float:
    """
    Get the value at a given index for a Normal Distribution GPU Array
    :param states: 1D GPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@cuda.jit(device=True)
def prob(states: GPUArray, index: int) -> float:
    """
    Get the value at a given index for a Probability (Uniform[0, 1]) GPU Array
    :param states: 1D GPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@cuda.jit(device=True)
def normal(states: GPUArray, index: int, mean: float, std: float) -> float:
    return mean + gauss(states, index) * std


@cuda.jit(device=True)
def uniform(states: GPUArray, index: int, minimum: float, maximum: float) -> float:
    difference = maximum - minimum
    return minimum + prob(states, index) * difference


@cuda.jit(device=True)
def handle_invalids(value: float):
    if value == np.nan:
        raise ValueError(f"nan value found")
    elif abs(value) == np.inf:
        raise ValueError(f"inf value found")
    else:
        return value


@cuda.jit(device=True)
def get_value(source: GPUArray, g: int, x: int, y: int) -> Union[int, float, bool]:
    if source.ndim == 3:
        return handle_invalids(source[g, x, y])
    elif source.ndim == 2:
        return handle_invalids(source[g, x])
    elif source.ndim == 1:
        return handle_invalids(source[g])
    else:
        raise NotImplementedError('Cannot get source array value')


@cuda.jit(device=True)
def set_value(source: GPUArray, g: int, x: int, y: int, value: Union[int, float, bool]):
    if source.ndim == 3:
        source[g, x, y] = value
    elif source.ndim == 2:
        source[g, x] = value
    elif source.ndim == 1:
        source[g] = value
    else:
        raise NotImplementedError('Cannot set value to update array')
