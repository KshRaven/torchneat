"""
Backend interface of the RL trainers.

``RLBackend`` holds everything that is the same on every device: the public methods (host arrays in, host arrays out),
the orchestration of returns / GAE / normalization, and a device agnostic reference implementation of every kernel
written with the array module ``xp`` (numpy or cupy). A device backend only overrides the kernels it can run better:
``CpuRLBackend`` with numba ``prange`` kernels, ``CudaRLBackend`` with ``numba.cuda`` kernels for the sequential
per-episode recurrences.

Conventions
-----------
* ``rows``    : int64 array of the buffer rows (genomes) a kernel works on.
* ``lengths`` : int64 array (genomes,), records stored per row. Anything at ``index >= lengths[row]`` is never read.
* ``ep``      : int64 array (genomes, capacity), episode id of every record. Must be sorted per row
                (``RolloutBuffer.sort_by('ep_map')``) before returns / episode reductions.
* Public methods take and return host (NumPy) arrays; the ``_k_*`` kernels work on arrays of the backend's own device.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

__all__ = ['EpisodeRuns', 'RLBackend', 'ACCURACY_MODES', 'POLICY_MODES']

ACCURACY_MODES = {'continuous': 0, 'binary': 1, 'discrete': 2, 'bipolar': 3}
POLICY_MODES = {'ppo': 0, 'a2c': 1, 'trpo': 2}


@dataclass
class EpisodeRuns:
    """Contiguous episode segments of every row: records ``starts[g, r]:ends[g, r]`` belong to episode ``ids[g, r]``."""
    starts: np.ndarray
    ends: np.ndarray
    ids: np.ndarray
    counts: np.ndarray
    #: (genomes, capacity) index of the run every record belongs to, -1 for padding
    slot: np.ndarray | None = None

    @property
    def emax(self) -> int:
        return int(self.starts.shape[1])

    @property
    def lengths(self) -> np.ndarray:
        """(rows, emax) episode lengths (0 where an episode does not exist)."""
        return self.ends - self.starts


class RLBackend(object):
    """Device independent RL bookkeeping; see the module docstring."""
    name = 'abstract'
    xp = np

    # ================================================ Arrays ================================================ #
    @staticmethod
    def to_host(array) -> np.ndarray:
        """Torch (cpu / cuda) tensors and CuPy arrays -> NumPy (zero copy when already on the host)."""
        if isinstance(array, np.ndarray):
            return array
        if hasattr(array, 'detach'):  # torch
            return array.detach().cpu().numpy()
        if hasattr(array, 'get'):  # cupy
            return array.get()
        return np.asarray(array)

    def asarray(self, array):
        """Host array -> array of this backend's device."""
        return self.xp.asarray(self.to_host(array))

    def release(self):
        """Free temporary device memory."""

    @staticmethod
    def width(lengths: np.ndarray, rows: np.ndarray) -> int:
        """Grid width of the (row, record) launches: the longest valid row (no work items spent on padding)."""
        return int(lengths[rows].max()) if len(rows) else 0

    @staticmethod
    def flat3(arr):
        """View a ``(genomes, capacity, *features)`` array as ``(genomes, capacity, F)`` (never copies C arrays)."""
        if arr.ndim == 3:
            return arr
        return arr.reshape(arr.shape[0], arr.shape[1], -1)

    def _mask(self, lengths, rows, cap: int):
        """(genomes, capacity) bool: valid records of the selected rows."""
        xp = self.xp
        selected = xp.zeros(lengths.shape[0], dtype=bool)
        selected[rows] = True
        return (xp.arange(cap)[None, :] < lengths[:, None]) & selected[:, None]

    def _quiet(self):
        """Silences floating point warnings of the reference kernels (numpy only)."""
        return np.errstate(all='ignore') if self.xp is np else contextlib.nullcontext()

    # ============================================= Public methods ============================================ #
    def episode_runs(self, ep: np.ndarray, lengths: np.ndarray, rows: np.ndarray) -> EpisodeRuns:
        counts, starts, ends, ids, slot = self._k_find_runs(self.asarray(ep), self.asarray(lengths), self.asarray(rows))
        h = self.to_host
        return EpisodeRuns(h(starts), h(ends), h(ids), h(counts), h(slot))

    def record_mean(self, arr: np.ndarray, lengths: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """(genomes, capacity, *features) -> (genomes, capacity) mean over the features of every record."""
        return self.to_host(self._k_record_mean(self.asarray(self.flat3(arr)), self.asarray(lengths), self.asarray(rows)))

    def segment_mean(self, per_record: np.ndarray, runs: EpisodeRuns, rows: np.ndarray) -> np.ndarray:
        """(genomes, capacity) -> (genomes, episodes) mean per episode (NaN where the episode does not exist)."""
        d = self.asarray
        return self.to_host(self._k_segment_mean(
            d(per_record), d(runs.slot), d(runs.starts), d(runs.ends), d(runs.counts), d(rows)
        ))

    def row_mean_std(self, mat: np.ndarray, counts: np.ndarray, rows: np.ndarray, ddof: int = 0):
        """Mean and std of the first ``counts[g]`` entries of every row. ``std`` is 0 for single entries."""
        d = self.asarray
        mean, std = self._k_row_mean_std(d(np.ascontiguousarray(mat, dtype=np.float64)), d(counts), d(rows), int(ddof))
        return self.to_host(mean), self.to_host(std)

    def accuracy(self, pred, target, lengths, rows, error=0.10, kind='continuous') -> np.ndarray:
        """
        Per record accuracy (mean over features), shaped (genomes, capacity).

        :param kind: ``'continuous'``: prediction within ``error`` (relative) of the target. ``'binary'``: prediction
            and target on the same side of ``1 - error``. ``'bipolar'``: three zones split at ``+-(1 - 2 * error)``
            (above, within, below); correct when prediction and target fall in the same zone. ``'discrete'``: exact match.
        """
        if kind not in ACCURACY_MODES:
            raise ValueError(f"Unsupported accuracy type '{kind}'")
        d = self.asarray
        return self.to_host(self._k_accuracy(
            d(self.flat3(pred)), d(self.flat3(target)), d(lengths), d(rows), float(error), ACCURACY_MODES[kind]
        ))

    def policy_terms(self, new_lp, old_lp, adv, ent, lengths, rows, mode: str = 'ppo'):
        """
        Per record ``(objective, log-prob MSE, entropy, KL)``, each shaped (genomes, capacity).

        ``mode='ppo'`` / ``'trpo'``: objective is ``mean(ratio * mean(adv))``; ``'a2c'``: ``mean(log_prob * mean(adv))``.
        The log-prob MSE is ``mean((logp_new - logp_old)^2)`` (clamped to +-1e3, NaN -> 1e3). The KL is the sampled
        ``(ratio - 1) - log(ratio)`` estimate of ``KL(old || new)`` (log ratio clamped to +-20).
        """
        if mode not in POLICY_MODES:
            raise ValueError(f"Unknown policy mode '{mode}'; use one of {list(POLICY_MODES)}")
        if new_lp.shape[2:] != old_lp.shape[2:]:
            raise ValueError(f"log_prob shape mismatch: new {new_lp.shape[2:]} vs old {old_lp.shape[2:]}")
        d, f = self.asarray, self.flat3
        out = self._k_policy_terms(
            d(f(new_lp)), d(f(old_lp)), d(f(adv)), d(f(ent)), d(lengths), d(rows), POLICY_MODES[mode]
        )
        return tuple(self.to_host(o) for o in out)

    def ppo_terms(self, new_lp, old_lp, adv, ent, lengths, rows):
        """``policy_terms`` in ``'ppo'`` mode without the KL: ``(objective, log-prob MSE, entropy)``."""
        return self.policy_terms(new_lp, old_lp, adv, ent, lengths, rows, 'ppo')[:3]

    def value_terms(self, values, rets, lengths, rows, eps) -> np.ndarray:
        d, f = self.asarray, self.flat3
        return self.to_host(self._k_value_terms(d(f(values)), d(f(rets)), d(lengths), d(rows), float(eps)))

    def td_terms(self, q_values, actions, rewards, ep, lengths, rows, gamma: float = 0.99) -> np.ndarray:
        """
        Per record squared one-step TD error ``(r + gamma * max_a' Q(s', a') - Q(s, a))^2`` shaped (genomes, capacity).
        ``s'`` is the next record of the same episode; the last record of an episode is terminal (``target = r``).
        ``actions`` hold the taken (discrete) action index of every record.
        """
        d, f = self.asarray, self.flat3
        actions = np.ascontiguousarray(f(self.to_host(actions))[:, :, 0].astype(np.int64))
        return self.to_host(self._k_td_terms(
            d(f(q_values)), d(actions), d(f(rewards)), d(ep), d(lengths), d(rows), float(gamma)
        ))

    def ghost_reduction(self, means, src_rows, use_src, lengths, rows, discrete=False) -> np.ndarray:
        d = self.asarray
        psum, pcnt = self._k_ghost_terms(
            d(self.flat3(means)), d(src_rows), d(np.asarray(use_src, dtype=bool)), d(lengths), d(rows), bool(discrete)
        )
        total, count = self.to_host(psum).sum(axis=1), self.to_host(pcnt).sum(axis=1)
        return np.where(count > 0, total / np.maximum(count, 1.0), 0.0)

    # ================================================= Returns ================================================ #
    @staticmethod
    def _run_factors(counts: np.ndarray, emax: int, rows: np.ndarray, order: int) -> np.ndarray:
        G = len(counts)
        base = np.broadcast_to(np.arange(emax, dtype=np.float64), (G, emax)).copy()
        if order == 1:
            base = counts[:, None].astype(np.float64) - 1.0 - base
        elif order in (4, 5, 6):
            for g in rows:
                R = int(counts[g])
                remaining = list(range(R))[::-1]
                popped = []
                for _ in range(R):
                    if len(remaining) == 1:
                        pos = 0
                    else:
                        ef = np.arange(len(remaining))
                        ef_ = ef[::-1] + 1
                        pos = int(np.random.choice(ef, p=(ef_ / ef_.sum()) if order in (4, 5) else None))
                    popped.append(remaining.pop(pos))
                # The first popped factor belongs to the last episode, the last popped to the first
                for r in range(R):
                    base[g, r] = popped[R - 1 - r]
        return base

    def compute_returns(
            self, rewards: np.ndarray, ep: np.ndarray, lengths: np.ndarray, rows: np.ndarray, row_genus: np.ndarray,
            gamma: float = 0.97, kappa: float = 0.0, alpha: float = 1.0, order: int = 0, inverse: bool = False,
            normalize: int = 0, genus_filter=None, beta: float = 1.0, per_genus: bool = True,
    ) -> np.ndarray:
        """
        Reward normalization (per genus by default, else global over ``rows``; see ``per_genus``), per-episode
        ``alpha`` / ``beta`` scaling, then gamma (reward-to-go) and kappa (forward) discounting.
        Returns float64 of the shape of ``rewards``; the input is never modified.
        """
        if not (-1 <= order <= 6):
            raise ValueError(f"Invalid alpha order: '{order}'")
        normalize = int(normalize)
        shape = rewards.shape
        G, cap = shape[0], shape[1]
        if len(rows) == 0:
            return np.zeros(shape, np.float64)
        xp, d, h = self.xp, self.asarray, self.to_host
        r3 = d(self.flat3(rewards))
        ep_d, len_d, rows_d = d(ep), d(lengths), d(rows)
        out = xp.zeros(r3.shape, xp.float64)
        genus_filter = [] if genus_filter is None else list(genus_filter)

        # Normalization statistics (mean of means, std of per-genome stds, min of mins, max of maxes), one set per
        # group: every genus on its own (``per_genus``, so a weak genus is ranked against itself and not against a
        # stronger one) or the whole population. Stored per row so the kernel just indexes its own row.
        nmode = np.zeros(G, np.int64)
        nmean = np.zeros(G, np.float64)
        nstd = np.ones(G, np.float64)
        nmin = np.zeros(G, np.float64)
        nmax = np.ones(G, np.float64)
        if normalize in (1, 2, 3, 4):
            means, stds, mins, maxs = (h(v) for v in self._k_flat_stats(r3, len_d, rows_d))
            groups = [rows[row_genus[rows] == g] for g in np.unique(row_genus[rows])] if per_genus else [rows]
            for grp in groups:
                if normalize == 1:
                    gmean = float(np.mean(means[grp]))
                    gstd = float(np.std(stds[grp])) if len(grp) > 1 else 0.0
                    nmean[grp], nstd[grp] = gmean, gstd
                    nmode[grp] = 1 if gstd != 0.0 else 0
                else:
                    gmin, gmax = float(np.min(mins[grp])), float(np.max(maxs[grp]))
                    nmin[grp], nmax[grp] = gmin, gmax
                    nmode[grp] = 4 if normalize == 4 else (normalize if (gmax - gmin) != 0.0 else 0)

        # Score based ordering of the records (episodes) before applying the episode factors
        perm = None
        ep_used = ep_d
        if order in (2, 3, 4, 5):
            scores = self._k_record_mean(r3, len_d, rows_d)
            perm = self._k_rank_order(scores, len_d, rows_d, order in (3, 5))
            r3 = self._k_gather(r3, perm, len_d, rows_d)
            ep_used = self._k_gather(ep_d.reshape(G, cap, 1), perm, len_d, rows_d).reshape(G, cap)

        counts, starts, ends, _, _ = self._k_find_runs(ep_used, len_d, rows_d)
        emax = int(starts.shape[1])
        factors = self._run_factors(h(counts), emax, rows, order)

        mult = np.full(G, float(alpha), np.float64)
        if beta != 1.0 and genus_filter:
            mult[np.isin(row_genus, genus_filter)] = float(beta)
        if inverse:
            mult = 1.0 / mult
        do_alpha = (alpha is not None and alpha != 1.0) or beta != 1.0

        self._k_returns(
            r3, starts, ends, counts, rows_d, out, d(nmode), d(nmean), d(nstd), d(nmin), d(nmax), d(factors), d(mult),
            bool(do_alpha), float(gamma), float(kappa)
        )
        if perm is not None:  # back to the original record positions
            out = self._k_scatter(out, perm, len_d, rows_d)
        return h(out).reshape(shape)

    def compute_gae(self, rewards, values, ep, lengths, rows, gamma=0.95, gae_lambda=0.91, alpha=1.10, reverse=False):
        """Returns ``(returns, advantages)`` (float64, shaped like ``rewards``) with per-episode ``alpha`` scaling."""
        xp, d, h = self.xp, self.asarray, self.to_host
        r3, v3 = d(self.flat3(rewards)), d(self.flat3(values))
        rows_d = d(rows)
        counts, starts, ends, _, _ = self._k_find_runs(d(ep), d(lengths), rows_d)
        adv, ret = xp.zeros(r3.shape, xp.float64), xp.zeros(r3.shape, xp.float64)
        self._k_gae(r3, v3, starts, ends, counts, rows_d, adv, ret,
                    float(gamma), float(gae_lambda), float(alpha), bool(reverse))
        return h(ret).reshape(rewards.shape), h(adv).reshape(rewards.shape)

    # ======================================= Kernels (reference, via xp) ====================================== #
    def _k_find_runs(self, ep, lengths, rows):
        """Contiguous episode runs of every row: ``(counts, starts, ends, ids, slot)``."""
        xp = self.xp
        G, cap = ep.shape
        mask = self._mask(lengths, rows, cap)
        previous = xp.concatenate([ep[:, :1] - 1, ep[:, :-1]], axis=1)  # the first record always opens a run
        flags = (mask & (ep != previous)).astype(xp.int64)
        counts = flags.sum(axis=1)
        emax = int(counts.max()) if G else 0
        cumulative = xp.cumsum(flags, axis=1) - 1
        slot = xp.where(mask, cumulative, -1)
        starts = xp.zeros((G, emax), xp.int64)
        ids = xp.full((G, emax), -1, xp.int64)
        gi, ti = xp.nonzero(flags)
        starts[gi, cumulative[gi, ti]] = ti
        ids[gi, cumulative[gi, ti]] = ep[gi, ti]
        if emax:
            following = xp.concatenate([starts[:, 1:], lengths[:, None].astype(xp.int64)], axis=1)
            position = xp.arange(emax)[None, :]
            ends = xp.where(position + 1 < counts[:, None], following, lengths[:, None].astype(xp.int64))
            ends = xp.where(position < counts[:, None], ends, 0)
        else:
            ends = xp.zeros((G, 0), xp.int64)
        return counts, starts, ends, ids, slot

    def _k_record_mean(self, arr, lengths, rows):
        xp = self.xp
        mask = self._mask(lengths, rows, arr.shape[1])
        return xp.where(mask, arr.astype(xp.float64).mean(axis=2), 0.0)

    def _k_segment_mean(self, vals, slot, starts, ends, counts, rows):
        xp = self.xp
        G = vals.shape[0]
        E = starts.shape[1]
        valid = slot >= 0
        flat = (xp.arange(G)[:, None] * E + slot)[valid]
        sums = xp.bincount(flat, weights=vals[valid].astype(xp.float64), minlength=G * E).reshape(G, E)
        length = (ends - starts).astype(xp.float64)
        exists = xp.arange(E)[None, :] < counts[:, None]
        return xp.where(exists, sums / xp.maximum(length, 1.0), xp.nan)

    def _k_row_mean_std(self, mat, counts, rows, ddof):
        xp = self.xp
        G, E = mat.shape
        selected = xp.zeros(G, dtype=bool)
        selected[rows] = True
        valid = xp.arange(E)[None, :] < counts[:, None]
        n = counts.astype(xp.float64)
        mean_all = xp.where(valid, mat, 0.0).sum(axis=1) / xp.maximum(n, 1.0)
        var = xp.where(valid, (mat - mean_all[:, None]) ** 2, 0.0).sum(axis=1) / xp.maximum(n - ddof, 1.0)
        mean = xp.where(selected & (counts > 0), mean_all, xp.nan)
        std = xp.where(selected & (counts > 1) & (n - ddof > 0), xp.sqrt(var), 0.0)
        return mean, std

    def _k_flat_stats(self, arr, lengths, rows):
        """Per row: mean, unbiased std, min, max over every element (records x features)."""
        xp = self.xp
        a = arr.astype(xp.float64)
        valid = self._mask(lengths, rows, a.shape[1])
        mask = valid[:, :, None]
        n = valid.sum(axis=1).astype(xp.float64) * a.shape[2]
        mean = xp.where(n > 0, xp.where(mask, a, 0.0).sum(axis=(1, 2)) / xp.maximum(n, 1.0), 0.0)
        low = xp.where(mask, a, xp.inf).min(axis=(1, 2))
        high = xp.where(mask, a, -xp.inf).max(axis=(1, 2))
        squares = xp.where(mask, (a - mean[:, None, None]) ** 2, 0.0).sum(axis=(1, 2))
        std = xp.where(n > 1, xp.sqrt(squares / xp.maximum(n - 1.0, 1.0)), 0.0)
        return mean, std, low, high

    def _k_rank_order(self, scores, lengths, rows, descending):
        xp = self.xp
        mask = self._mask(lengths, rows, scores.shape[1])
        key = -scores if descending else scores
        key = xp.where(mask, key, xp.inf)
        order = xp.argsort(key, axis=1, kind='stable').astype(xp.int64)
        return xp.where(mask, order, 0)

    def _k_gather(self, arr, order, lengths, rows):
        """``out[g, t] = arr[g, order[g, t]]`` for the valid records of the selected rows."""
        xp = self.xp
        out = xp.zeros_like(arr)
        gi, ti = xp.nonzero(self._mask(lengths, rows, arr.shape[1]))
        out[gi, ti] = arr[gi, order[gi, ti]]
        return out

    def _k_scatter(self, arr, order, lengths, rows):
        """``out[g, order[g, t]] = arr[g, t]`` for the valid records of the selected rows."""
        xp = self.xp
        out = xp.zeros_like(arr)
        gi, ti = xp.nonzero(self._mask(lengths, rows, arr.shape[1]))
        out[gi, order[gi, ti]] = arr[gi, ti]
        return out

    def _k_accuracy(self, pred, target, lengths, rows, err, mode):
        xp = self.xp
        p, a = pred.astype(xp.float64), target.astype(xp.float64)
        if mode == 0:
            ok = (p <= a * (1.0 + err)) & (p >= a * (1.0 - err))
        elif mode == 1:
            ok = (p >= (1.0 - err)) == (a >= (1.0 - err))
        elif mode == 2:
            ok = p == a
        else:
            threshold = max(0.0, 1.0 - 2.0 * err)

            def zone(v):
                return xp.where(v >= threshold, 2, xp.where(v <= -threshold, 0, 1))
            ok = zone(p) == zone(a)
        mask = self._mask(lengths, rows, p.shape[1])
        return xp.where(mask, ok.astype(xp.float64).mean(axis=2), 0.0)

    def _k_policy_terms(self, new_lp, old_lp, adv, ent, lengths, rows, mode):
        xp = self.xp
        n, o = new_lp.astype(xp.float64), old_lp.astype(xp.float64)
        mask = self._mask(lengths, rows, n.shape[1])
        with self._quiet():
            delta = n - o
            advantage = adv.astype(xp.float64).mean(axis=2)[:, :, None]
            objective = (n * advantage if mode == 1 else xp.exp(delta) * advantage).mean(axis=2)
            clamped = xp.where(xp.isnan(delta), 1.0e3, xp.clip(delta, -1.0e3, 1.0e3))
            sq = (clamped * clamped).mean(axis=2)
            small = xp.clip(clamped, -20.0, 20.0)
            kl = (xp.exp(small) - 1.0 - small).mean(axis=2)
        entropy = ent.astype(xp.float64).mean(axis=2)
        return (xp.where(mask, objective, 0.0), xp.where(mask, sq, 0.0), xp.where(mask, entropy, 0.0),
                xp.where(mask, kl, 0.0))

    def _k_value_terms(self, values, rets, lengths, rows, eps):
        xp = self.xp
        diff = values.astype(xp.float64) - rets.astype(xp.float64)
        mask = self._mask(lengths, rows, diff.shape[1])
        return xp.where(mask, xp.log10(diff * diff + eps).mean(axis=2), 0.0)

    def _k_td_terms(self, q, actions, rewards, ep, lengths, rows, gamma):
        xp = self.xp
        q = q.astype(xp.float64)
        G, cap, A = q.shape
        mask = self._mask(lengths, rows, cap)
        taken = xp.clip(actions, 0, A - 1)[:, :, None]
        q_taken = xp.take_along_axis(q, taken, axis=2)[:, :, 0]
        q_best = q.max(axis=2)
        next_best = xp.concatenate([q_best[:, 1:], xp.zeros((G, 1))], axis=1)
        same = xp.concatenate([ep[:, 1:] == ep[:, :-1], xp.zeros((G, 1), dtype=bool)], axis=1)
        has_next = same & (xp.arange(cap)[None, :] + 1 < lengths[:, None])
        target = rewards.astype(xp.float64).mean(axis=2) + gamma * xp.where(has_next, next_best, 0.0)
        return xp.where(mask, (target - q_taken) ** 2, 0.0)

    def _k_ghost_terms(self, means, src_rows, use_src, lengths, rows, discrete):
        xp = self.xp
        m = means.astype(xp.float64)
        if discrete:
            e = xp.exp(m - m.max(axis=2, keepdims=True))
            m = e / e.sum(axis=2, keepdims=True)
        G, cap, A = m.shape
        base = self._mask(lengths, rows, cap) & use_src[:, None]
        position = xp.arange(cap)[None, :]
        total = xp.zeros((G, cap), xp.float64)
        count = xp.zeros((G, cap), xp.float64)
        for k in range(src_rows.shape[1]):
            source = src_rows[:, k]
            safe = xp.maximum(source, 0)
            ok = base & (source >= 0)[:, None] & (position < lengths[safe][:, None])
            distance = ((m - m[safe]) ** 2).sum(axis=2)
            total += xp.where(ok, distance, 0.0)
            count += xp.where(ok, float(A), 0.0)
        return total, count

    # ===================================== Kernels (sequential, per device) =================================== #
    def _k_returns(self, rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                   run_factors, mult, do_alpha, gamma, kappa):
        """Normalize -> per-episode alpha scaling -> gamma / kappa discounting, written into ``out``."""
        raise NotImplementedError

    def _k_gae(self, rew, val, starts, ends, counts, rows, adv, ret, gamma, lam, alpha, reverse):
        """Lambda-return and GAE advantage per episode, written into ``adv`` / ``ret``."""
        raise NotImplementedError
