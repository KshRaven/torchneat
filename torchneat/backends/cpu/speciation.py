"""CPU distance kernels (numba, prange). The species grouping lives in ``torchneat.base.speciation``."""

import numpy as np
from numba import njit, prange
from numpy import ndarray as CPUArray

from ...config import Config
from ...nn.module import NeatModule
from ..util import parameter_dtype, reshape_to_ndim
from .grid import grid, gridsize


@njit(cache=True)
def calc_distance(parameter: CPUArray, total_distance: CPUArray,
                  genome0: int, genome1: int, x: int,
                  compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    """
    Returns the genetic distance between this genome and the other. This distance value
    is used to compute genome compatibility for speciation.
    """

    value0 = parameter[genome0, x]
    value1 = parameter[genome1, x]

    current_distance = 0 # total_distance[genome0, genome1]
    disjoint_value   = 0
    if value0 == 0:
        disjoint_value += 1
    if value1 == 0:
        disjoint_value = disjoint_value + 1
    if value0 != 0 and value1 != 0:
        current_distance = current_distance + (abs(value0 - value1) * compatibility_weight_coefficient)
    current_distance = current_distance + disjoint_value * compatibility_disjoint_coefficient

    # Parameter shape = (genomes, x, y | 1)
    x_s = parameter.shape[1] # [:]
    size = x_s # * y_s
    # if size != 0:
    current_distance = current_distance / size

    return current_distance


@njit(parallel=True, cache=True)
def get_distance(parameter: CPUArray, total_distance: CPUArray,
                 compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    ret_code: int = 0
    if parameter.ndim != 2: ret_code = 1

    if not ret_code:
        g_lim = total_distance.shape[0]
        x_lim = parameter.shape[1]
        dims = (g_lim, g_lim)

        # Lower triangle: one flat prange over every (genome0, genome1) cell. Each cell owns its own element and
        # accumulates over the parameter in a local, so there is no cross-thread write conflict.
        for idx in prange(gridsize(dims)):
            genome0, genome1 = grid(2, idx, dims)
            if genome1 > genome0:
                acc = total_distance[genome0, genome1]
                for x in range(x_lim):
                    acc += calc_distance(
                        parameter, total_distance, genome0, genome1, x,
                        compatibility_weight_coefficient, compatibility_disjoint_coefficient
                    )
                total_distance[genome0, genome1] = acc
        # Upper triangle: mirror the finished cells
        for idx in prange(gridsize(dims)):
            genome0, genome1 = grid(2, idx, dims)
            if genome1 < genome0:
                total_distance[genome0, genome1] = total_distance[genome1, genome0]

    return ret_code


def compute_distances(config: Config, module: NeatModule, dtype, tpb: int = 10, verbose: bool | int = False) -> np.ndarray:
    """Genetic distance matrix of the module's genomes (symmetric, zero diagonal), summed over all parameters."""
    total_distance = np.zeros((module.genome_num, module.genome_num), dtype)

    for param in module.neat_parameters():
        array = np.asarray(reshape_to_ndim(param.data.to(parameter_dtype(param.data.dtype)), 2)[0])

        try:
            ret_code = get_distance(
                array, total_distance,
                config.genome.compatibility_weight_coefficient,
                config.genome.compatibility_disjoint_coefficient
            )
            if ret_code == 1: raise ValueError(f"Expected a 2D array, got {array.ndim}")
            if ret_code == 2: raise ValueError("Out of bounds")
        except Exception as e:
            print(param.dtype, param.device, array.shape, param.data.shape)
            raise e

        # Release memory
        del array

    return total_distance
