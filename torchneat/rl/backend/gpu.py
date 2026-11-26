"""
CUDA RL backend: CuPy arrays on the device.

The array kernels of ``RLBackend`` run as CuPy operations; the per-episode recurrences (discounted returns and GAE) are
``numba.cuda`` kernels with one thread per (row, episode, feature) cell, the same cells the CPU backend gives to its
``prange`` grid. The records never leave the device between kernels: arrays are moved over once per public call.
"""
from __future__ import annotations

import cupy as cp
import numpy as np
from numba import cuda

from .base import RLBackend
from .cpu import _rank_order as _rank_order_host

__all__ = ['CudaRLBackend']

_TPB = 128


def _launch(kernel, cells: int, *args):
    if cells <= 0:
        return
    kernel[(cells + _TPB - 1) // _TPB, _TPB](*args)
    cuda.synchronize()


@cuda.jit(device=True)
def _norm(v, mode, gmean, gstd, gmin, gmax):
    if mode == 1:
        return (v - gmean) / (gstd + 1e-9)
    elif mode == 2:
        return (v - gmin) / (gmax - gmin)
    elif mode == 3:
        return -1.0 + 2.0 * ((v - gmin) / (gmax - gmin))
    elif mode == 4:
        return v - gmin
    return v


@cuda.jit
def _returns_kernel(rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                    run_factors, mult, do_alpha, gamma, kappa):
    E = starts.shape[1]
    F = rew.shape[2]
    idx = cuda.grid(1)
    if idx >= rows.shape[0] * E * F:
        return
    f = idx % F
    r = (idx // F) % E
    i = idx // (F * E)
    g = rows[i]
    if r >= counts[g]:
        return
    s = starts[g, r]
    e = ends[g, r]
    w = 1.0
    if do_alpha:
        w = mult[g] ** run_factors[g, r]
    nm = norm_mode[g]
    mu = gmean[g]
    sd = gstd[g]
    lo = gmin[g]
    hi = gmax[g]
    use_g = gamma > 0.0
    use_k = kappa > 0.0
    if use_k:
        acc = 0.0
        for t in range(s, e):
            acc = _norm(np.float64(rew[g, t, f]), nm, mu, sd, lo, hi) * w + acc * kappa
            out[g, t, f] = acc
    if use_g:
        acc = 0.0
        for t in range(e - 1, s - 1, -1):
            acc = _norm(np.float64(rew[g, t, f]), nm, mu, sd, lo, hi) * w + acc * gamma
            if use_k:
                out[g, t, f] = (acc + out[g, t, f]) / 2.0
            else:
                out[g, t, f] = acc
    if not use_g and not use_k:
        for t in range(s, e):
            out[g, t, f] = _norm(np.float64(rew[g, t, f]), nm, mu, sd, lo, hi) * w


@cuda.jit
def _gae_kernel(rew, val, starts, ends, counts, rows, adv, ret, gamma, lam, alpha, reverse):
    E = starts.shape[1]
    F = rew.shape[2]
    idx = cuda.grid(1)
    if idx >= rows.shape[0] * E * F:
        return
    f = idx % F
    r = (idx // F) % E
    i = idx // (F * E)
    g = rows[i]
    if r >= counts[g]:
        return
    factor = r
    if reverse:
        factor = counts[g] - 1 - r
    w = alpha ** factor
    fv = 0.0
    fa = 0.0
    for t in range(ends[g, r] - 1, starts[g, r] - 1, -1):
        v = np.float64(val[g, t, f])
        delta = np.float64(rew[g, t, f]) * w + gamma * fv - v
        fv = v
        fa = delta + gamma * lam * fa
        adv[g, t, f] = fa
        ret[g, t, f] = fa + v


class CudaRLBackend(RLBackend):
    """CuPy / numba.cuda implementation of the RL bookkeeping."""
    name = 'cuda'
    xp = cp

    def __init__(self):
        if not cuda.is_available():
            raise RuntimeError("The CUDA RL backend needs a usable CUDA device")

    def asarray(self, array):
        if isinstance(array, cp.ndarray):
            return array
        return cp.asarray(self.to_host(array))

    def release(self):
        cp.get_default_memory_pool().free_all_blocks()

    def _k_rank_order(self, scores, lengths, rows, descending):
        # Rare (alpha orders 2 to 5) and needs a stable sort: done on the host
        order = _rank_order_host(scores.get(), lengths.get(), rows.get(), bool(descending))
        return cp.asarray(order)

    def _k_returns(self, rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                   run_factors, mult, do_alpha, gamma, kappa):
        cells = int(rows.shape[0]) * int(starts.shape[1]) * int(rew.shape[2])
        _launch(_returns_kernel, cells, rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                run_factors, mult, bool(do_alpha), float(gamma), float(kappa))

    def _k_gae(self, rew, val, starts, ends, counts, rows, adv, ret, gamma, lam, alpha, reverse):
        cells = int(rows.shape[0]) * int(starts.shape[1]) * int(rew.shape[2])
        _launch(_gae_kernel, cells, rew, val, starts, ends, counts, rows, adv, ret,
                float(gamma), float(lam), float(alpha), bool(reverse))
