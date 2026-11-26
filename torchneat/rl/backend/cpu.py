"""
CPU RL backend: numba kernels over ``(genomes, capacity, features)`` NumPy arrays.

Every kernel is one flat ``prange`` over its (row, record) / (row, episode, feature) cells; ``grid`` (the CPU
``cuda.grid``) turns the flat index into those coordinates, so every cell is an independent work item. The kernels use
``parallel=True`` without ``nogil``.
"""
from __future__ import annotations

import gc

import numpy as np
from numba import njit, prange

from ...backends.cpu.grid import grid, gridsize
from .base import RLBackend

__all__ = ['CpuRLBackend']

_JIT = dict(parallel=True, cache=True)


# ============================================================================================================ #
#                                                  Kernels                                                     #
# ============================================================================================================ #

@njit(cache=True)
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


@njit(**_JIT)
def _run_flags(ep, lengths, rows, width):
    """1 where a record opens a new episode run (flattened (row, record), every cell independent)."""
    flags = np.zeros(ep.shape, np.int64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        if t == 0 or ep[g, t] != ep[g, t - 1]:
            flags[g, t] = 1
    return flags


@njit(**_JIT)
def _run_starts(ep, lengths, rows, width, flags, slot, starts, ids):
    """Every run start writes its own (row, run) slot: no two work items touch the same cell."""
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t < lengths[g] and flags[g, t] == 1:
            starts[g, slot[g, t]] = t
            ids[g, slot[g, t]] = ep[g, t]


@njit(**_JIT)
def _run_ends(starts, counts, lengths, rows, ends):
    """A run ends where the next one starts (or at the row's length); flattened (row, run)."""
    E = starts.shape[1]
    dims = (rows.shape[0], E)
    for idx in prange(gridsize(dims)):
        i, r = grid(2, idx, dims)
        g = rows[i]
        if r >= counts[g]:
            continue
        ends[g, r] = starts[g, r + 1] if r + 1 < counts[g] else lengths[g]


def _find_runs(ep, lengths, rows, width):
    """Contiguous episode runs of every row: ``(counts, starts, ends, ids)``; three flattened passes + one cumsum."""
    flags = _run_flags(ep, lengths, rows, width)
    counts = flags.sum(axis=1).astype(np.int64)
    emax = int(counts[rows].max()) if len(rows) else 0
    slot = np.cumsum(flags, axis=1) - 1
    G = ep.shape[0]
    starts = np.zeros((G, emax), np.int64)
    ends = np.zeros((G, emax), np.int64)
    ids = np.full((G, emax), -1, np.int64)
    _run_starts(ep, lengths, rows, width, flags, slot, starts, ids)
    _run_ends(starts, counts, lengths, rows, ends)
    return counts, starts, ends, ids, slot


@njit(**_JIT)
def _record_mean(arr, lengths, rows, width):
    G, cap, F = arr.shape
    out = np.zeros((G, cap), np.float64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                       # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        s = 0.0
        for f in range(F):
            s += float(arr[g, t, f])
        out[g, t] = s / F
    return out


@njit(**_JIT)
def _segment_mean(vals, starts, ends, counts, rows):
    G = vals.shape[0]
    E = starts.shape[1]
    out = np.full((G, E), np.nan, np.float64)
    dims = (rows.shape[0], E)
    for idx in prange(gridsize(dims)):                           # flattened (row, episode)
        i, r = grid(2, idx, dims)
        g = rows[i]
        if r >= counts[g]:
            continue
        s = 0.0
        for t in range(starts[g, r], ends[g, r]):
            s += vals[g, t]
        out[g, r] = s / (ends[g, r] - starts[g, r])
    return out


@njit(**_JIT)
def _row_mean_std(mat, counts, rows, ddof):
    G = mat.shape[0]
    mean = np.full(G, np.nan, np.float64)
    std = np.zeros(G, np.float64)
    for i in prange(rows.shape[0]):
        g = rows[i]
        n = counts[g]
        if n == 0:
            continue
        s = 0.0
        for t in range(n):
            s += mat[g, t]
        m = s / n
        mean[g] = m
        if n > 1 and n - ddof > 0:
            v = 0.0
            for t in range(n):
                v += (mat[g, t] - m) ** 2
            std[g] = np.sqrt(v / (n - ddof))
    return mean, std


@njit(**_JIT)
def _record_stats(arr, lengths, rows, width):
    """Per record (flattened (row, record)): sum, min and max over the features."""
    G, cap, F = arr.shape
    rs = np.zeros((G, cap), np.float64)
    lo = np.full((G, cap), np.inf, np.float64)
    hi = np.full((G, cap), -np.inf, np.float64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        for f in range(F):
            v = float(arr[g, t, f])
            rs[g, t] += v
            if v < lo[g, t]: lo[g, t] = v
            if v > hi[g, t]: hi[g, t] = v
    return rs, lo, hi


@njit(**_JIT)
def _record_sq_dev(arr, lengths, rows, width, mean):
    """Per record (flattened (row, record)): sum over the features of the squared deviation from the row mean."""
    G, cap, F = arr.shape
    out = np.zeros((G, cap), np.float64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        for f in range(F):
            out[g, t] += (float(arr[g, t, f]) - mean[g]) ** 2
    return out


def _flat_stats(arr, lengths, rows):
    """Per row: mean, unbiased std, min, max over every element (records x features)."""
    G, cap, F = arr.shape
    width = int(lengths[rows].max()) if len(rows) else 0
    rs, lo, hi = _record_stats(arr, lengths, rows, width)
    n = lengths.astype(np.float64) * F
    mean = np.zeros(G, np.float64)
    std = np.zeros(G, np.float64)
    mn = np.full(G, np.inf, np.float64)
    mx = np.full(G, -np.inf, np.float64)
    for g in rows:  # 1-D reductions over the records, cheap next to the passes above
        if n[g] > 0:
            mean[g] = rs[g, :lengths[g]].sum() / n[g]
            mn[g] = lo[g, :lengths[g]].min()
            mx[g] = hi[g, :lengths[g]].max()
    sq = _record_sq_dev(arr, lengths, rows, width, mean)
    for g in rows:
        if n[g] > 1:
            std[g] = np.sqrt(sq[g, :lengths[g]].sum() / (n[g] - 1))
    return mean, std, mn, mx


@njit(**_JIT)
def _returns_kernel(rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                    run_factors, mult, do_alpha, gamma, kappa):
    """
    Normalize -> per-episode alpha scaling -> gamma (reward-to-go) and/or kappa (forward) discounting, per element.
    ``run_factors[g, r]`` is the alpha exponent of the r-th episode (run) of row g. The discounting recurrences only run
    along time inside one episode, so every (row, episode, feature) triple is an independent work item.
    ``norm_mode`` / ``gmean`` / ``gstd`` / ``gmin`` / ``gmax`` are arrays (genomes,): the normalization statistics of the
    group (genus, or the whole population) each row belongs to.
    """
    E = starts.shape[1]
    F = rew.shape[2]
    use_g = gamma > 0.0
    use_k = kappa > 0.0
    dims = (rows.shape[0], E, F)
    for idx in prange(gridsize(dims)):                       # flattened (row, episode, feature)
        i, r, f = grid(3, idx, dims)
        g = rows[i]
        if r >= counts[g]:
            continue
        s = starts[g, r]
        e = ends[g, r]
        w = 1.0
        if do_alpha:
            w = mult[g] ** run_factors[g, r]
        nm, mu, sd, lo, hi = norm_mode[g], gmean[g], gstd[g], gmin[g], gmax[g]
        if use_k:
            acc = 0.0
            for t in range(s, e):
                acc = _norm(float(rew[g, t, f]), nm, mu, sd, lo, hi) * w + acc * kappa
                out[g, t, f] = acc
        if use_g:
            acc = 0.0
            for t in range(e - 1, s - 1, -1):
                acc = _norm(float(rew[g, t, f]), nm, mu, sd, lo, hi) * w + acc * gamma
                if use_k:
                    out[g, t, f] = (acc + out[g, t, f]) / 2.0
                else:
                    out[g, t, f] = acc
        if not use_g and not use_k:
            for t in range(s, e):
                out[g, t, f] = _norm(float(rew[g, t, f]), nm, mu, sd, lo, hi) * w


@njit(**_JIT)
def _gae_kernel(rew, val, starts, ends, counts, rows, adv, ret, gamma, lam, alpha, reverse):
    E = starts.shape[1]
    F = rew.shape[2]
    dims = (rows.shape[0], E, F)
    for idx in prange(gridsize(dims)):                       # flattened (row, episode, feature)
        i, r, f = grid(3, idx, dims)
        g = rows[i]
        if r >= counts[g]:
            continue
        factor = r if not reverse else counts[g] - 1 - r
        w = alpha ** factor
        fv = 0.0
        fa = 0.0
        for t in range(ends[g, r] - 1, starts[g, r] - 1, -1):
            v = float(val[g, t, f])
            delta = float(rew[g, t, f]) * w + gamma * fv - v
            fv = v
            fa = delta + gamma * lam * fa
            adv[g, t, f] = fa
            ret[g, t, f] = fa + v


@njit(cache=True)
def _bipolar_zone(v, thr):
    """The 3 zones of a bipolar output: 2 when ``v >= thr``, 0 when ``v <= -thr``, 1 in between."""
    if v >= thr:
        return 2
    if v <= -thr:
        return 0
    return 1


@njit(**_JIT)
def _accuracy_kernel(pred, target, lengths, rows, err, mode, width):
    """Per record accuracy (mean over features). mode: 0 continuous, 1 binary, 2 discrete, 3 bipolar."""
    G, cap, Fp = pred.shape
    Ft = target.shape[2]
    F = max(Fp, Ft)
    out = np.zeros((G, cap), np.float64)
    bipolar_thr = max(0.0, 1.0 - 2.0 * err)                         # 0.8 for err=0.1
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                       # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        s = 0.0
        for f in range(F):
            p = float(pred[g, t, f if Fp > 1 else 0])
            a = float(target[g, t, f if Ft > 1 else 0])
            if mode == 0:
                ok = (p <= a * (1.0 + err)) and (p >= a * (1.0 - err))
            elif mode == 1:
                ok = (p >= (1.0 - err)) == (a >= (1.0 - err))
            elif mode == 2:
                ok = p == a
            else:
                ok = _bipolar_zone(p, bipolar_thr) == _bipolar_zone(a, bipolar_thr)
            if ok:
                s += 1.0
        out[g, t] = s / F
    return out


@njit(**_JIT)
def _policy_terms(new_lp, old_lp, adv, ent, lengths, rows, width, mode):
    """
    Per record: objective, log-prob MSE, entropy and sampled KL (see ``RLBackend.policy_terms``). ``mode``: 0 / 2 use
    ``ratio * mean(adv)``, 1 uses ``log_prob * mean(adv)``.

    The consistency term uses the LOG-probability difference, not the difference of the probabilities: with a tiny std
    the density is ~e^(-log std) per action dimension, so (p_new - p_old)^2 overflows to inf. The log difference is
    scale free, 0 = identical policy, and is clamped to +-1e3 (NaN -> 1e3) so one broken genome is penalised instead
    of crashing the run.
    """
    G, cap, A = new_lp.shape
    Fa = adv.shape[2]
    Fe = ent.shape[2]
    o_obj = np.zeros((G, cap), np.float64)
    o_sq = np.zeros((G, cap), np.float64)
    o_ent = np.zeros((G, cap), np.float64)
    o_kl = np.zeros((G, cap), np.float64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                              # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        am = 0.0
        for f in range(Fa):
            am += float(adv[g, t, f])
        am /= Fa
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for a in range(A):
            n = float(new_lp[g, t, a])
            d = n - float(old_lp[g, t, a])
            if mode == 1:
                s1 += n * am
            else:
                s1 += np.exp(d) * am
            if d != d:
                c = 1.0e3
            elif d > 1.0e3:
                c = 1.0e3
            elif d < -1.0e3:
                c = -1.0e3
            else:
                c = d
            s2 += c * c
            k = min(max(c, -20.0), 20.0)
            s3 += np.exp(k) - 1.0 - k
        o_obj[g, t] = s1 / A
        o_sq[g, t] = s2 / A
        o_kl[g, t] = s3 / A
        e = 0.0
        for f in range(Fe):
            e += float(ent[g, t, f])
        o_ent[g, t] = e / Fe
    return o_obj, o_sq, o_ent, o_kl


@njit(**_JIT)
def _td_terms(q, act, rew, ep, lengths, rows, gamma, width):
    """Per record squared one-step TD error; the next record of the same episode supplies the bootstrap value."""
    G, cap, A = q.shape
    F = rew.shape[2]
    out = np.zeros((G, cap), np.float64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                              # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        n = lengths[g]
        if t >= n:
            continue
        a = min(max(int(act[g, t]), 0), A - 1)
        r = 0.0
        for f in range(F):
            r += float(rew[g, t, f])
        target = r / F
        if t + 1 < n and ep[g, t + 1] == ep[g, t]:
            best = float(q[g, t + 1, 0])
            for k in range(1, A):
                v = float(q[g, t + 1, k])
                if v > best:
                    best = v
            target += gamma * best
        diff = target - float(q[g, t, a])
        out[g, t] = diff * diff
    return out


@njit(**_JIT)
def _value_terms(values, rets, lengths, rows, eps, width):
    G, cap, Fv = values.shape
    Fr = rets.shape[2]
    F = max(Fv, Fr)
    out = np.zeros((G, cap), np.float64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                       # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        s = 0.0
        for f in range(F):
            d = float(values[g, t, f if Fv > 1 else 0]) - float(rets[g, t, f if Fr > 1 else 0])
            s += np.log10(d * d + eps)
        out[g, t] = s / F
    return out


@njit(**_JIT)
def _ghost_reduction(means, src_rows, use_src, lengths, rows, discrete, width):
    """
    Per record partial sums of the squared distance between each row's policy means and its ghost sources' policy
    means (all sources, all action dims; softmax over the last dim first when ``discrete``). Returns
    ``(sum, count)`` shaped (genomes, capacity); the caller reduces them over the records.
    """
    G, cap, A = means.shape
    K = src_rows.shape[1]
    psum = np.zeros((G, cap), np.float64)
    pcnt = np.zeros((G, cap), np.float64)
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                       # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g] or not use_src[g]:
            continue
        v = np.zeros(A, np.float64)
        s = np.zeros(A, np.float64)
        for a in range(A):
            v[a] = float(means[g, t, a])
        if discrete:
            mx = v.max()
            z = 0.0
            for a in range(A):
                v[a] = np.exp(v[a] - mx)
                z += v[a]
            for a in range(A):
                v[a] /= z
        total = 0.0
        cnt = 0.0
        for k in range(K):
            sr = src_rows[g, k]
            if sr < 0 or t >= lengths[sr]:
                continue
            for a in range(A):
                s[a] = float(means[sr, t, a])
            if discrete:
                mx2 = s.max()
                z2 = 0.0
                for a in range(A):
                    s[a] = np.exp(s[a] - mx2)
                    z2 += s[a]
                for a in range(A):
                    s[a] /= z2
            for a in range(A):
                total += (v[a] - s[a]) ** 2
                cnt += 1.0
        psum[g, t] = total
        pcnt[g, t] = cnt
    return psum, pcnt


@njit(**_JIT)
def _rank_order(scores, lengths, rows, descending):
    G, cap = scores.shape
    order = np.zeros((G, cap), np.int64)
    for i in prange(rows.shape[0]):
        g = rows[i]
        n = lengths[g]
        if descending:
            order[g, :n] = np.argsort(-scores[g, :n], kind='mergesort')
        else:
            order[g, :n] = np.argsort(scores[g, :n], kind='mergesort')
    return order


@njit(**_JIT)
def _gather_rows(arr, order, lengths, rows, out, width):
    F = arr.shape[2]
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                       # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        o = order[g, t]
        for f in range(F):
            out[g, t, f] = arr[g, o, f]


@njit(**_JIT)
def _scatter_rows(arr, order, lengths, rows, out, width):
    F = arr.shape[2]
    dims = (rows.shape[0], width)
    for idx in prange(gridsize(dims)):                       # flattened (row, record)
        i, t = grid(2, idx, dims)
        g = rows[i]
        if t >= lengths[g]:
            continue
        o = order[g, t]
        for f in range(F):
            out[g, o, f] = arr[g, t, f]


# ============================================================================================================ #
#                                                  Backend                                                     #
# ============================================================================================================ #

class CpuRLBackend(RLBackend):
    """Numba (flat ``prange`` grid) implementation of the RL bookkeeping, running on NumPy arrays."""
    name = 'cpu'
    xp = np

    def asarray(self, array):
        return np.asarray(self.to_host(array))

    def release(self):
        gc.collect()

    def _k_find_runs(self, ep, lengths, rows):
        cap = ep.shape[1]
        counts, starts, ends, ids, slot = _find_runs(ep, lengths, rows, self.width(lengths, rows))
        selected = np.zeros(ep.shape[0], dtype=bool)
        selected[rows] = True
        valid = (np.arange(cap)[None, :] < lengths[:, None]) & selected[:, None]
        return counts, starts, ends, ids, np.where(valid, slot, -1)

    def _k_record_mean(self, arr, lengths, rows):
        return _record_mean(arr, lengths, rows, self.width(lengths, rows))

    def _k_segment_mean(self, vals, slot, starts, ends, counts, rows):
        return _segment_mean(vals, starts, ends, counts, rows)

    def _k_row_mean_std(self, mat, counts, rows, ddof):
        return _row_mean_std(mat, counts, rows, ddof)

    def _k_flat_stats(self, arr, lengths, rows):
        return _flat_stats(arr, lengths, rows)

    def _k_rank_order(self, scores, lengths, rows, descending):
        return _rank_order(scores, lengths, rows, bool(descending))

    def _k_gather(self, arr, order, lengths, rows):
        out = np.zeros_like(arr)
        _gather_rows(arr, order, lengths, rows, out, self.width(lengths, rows))
        return out

    def _k_scatter(self, arr, order, lengths, rows):
        out = np.zeros_like(arr)
        _scatter_rows(arr, order, lengths, rows, out, self.width(lengths, rows))
        return out

    def _k_accuracy(self, pred, target, lengths, rows, err, mode):
        return _accuracy_kernel(pred, target, lengths, rows, err, mode, self.width(lengths, rows))

    def _k_policy_terms(self, new_lp, old_lp, adv, ent, lengths, rows, mode):
        return _policy_terms(new_lp, old_lp, adv, ent, lengths, rows, self.width(lengths, rows), mode)

    def _k_value_terms(self, values, rets, lengths, rows, eps):
        return _value_terms(values, rets, lengths, rows, eps, self.width(lengths, rows))

    def _k_td_terms(self, q, actions, rewards, ep, lengths, rows, gamma):
        return _td_terms(q, actions, rewards, ep, lengths, rows, gamma, self.width(lengths, rows))

    def _k_ghost_terms(self, means, src_rows, use_src, lengths, rows, discrete):
        return _ghost_reduction(means, src_rows, use_src, lengths, rows, discrete, self.width(lengths, rows))

    def _k_returns(self, rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                   run_factors, mult, do_alpha, gamma, kappa):
        _returns_kernel(rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                        run_factors, mult, do_alpha, gamma, kappa)

    def _k_gae(self, rew, val, starts, ends, counts, rows, adv, ret, gamma, lam, alpha, reverse):
        _gae_kernel(rew, val, starts, ends, counts, rows, adv, ret, gamma, lam, alpha, reverse)
