"""Multi-head attention with differential attention (AttentionLambda), branch gains and a KV cache."""

from __future__ import annotations

import math
from typing import Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import device as DEVICE
from torch import dtype as DTYPE

from ...utils.console import CM, Fore
from ...utils.helpers import manage_params
from ..layers import LayerNorm, Linear, Polynomial
from ..module import NeatModule
from ..parameter import NeatParameter
from ..utils import get_tensor_info
from .encoding import Repositioning, RoPE


class AttentionLambda(NeatModule):
    """
    Learnable per-head scalar gains used by Differential Attention.

    Produces a tensor shaped (genomes, batch_size, heads, lambdas, 1, 1) that is
    broadcast over the (..., query_len, key_len) attention map.

    The attention energy is combined as  e_0 + sum_k lambda_k * e_k  (k = 1..lambdas).

    mode='simplex' (default in Attention):
        lambda_k = -beta * w_k,   w = softmax(s_1..s_L),   beta = sigmoid(b + beta_bias)
        => every extra branch is subtracted, sum(lambda) = -beta in [-1, 0], so the common-mode gain
           1 + sum(lambda) = 1 - beta is in [0, 1] for ANY parameter values (saturation cannot break it).
           beta -> 1 gives exact common-mode rejection; beta -> 0 degrades to plain attention.
        scalars: (heads, lambdas + 1); the last column is the beta logit.
    mode='legacy':
        lambda_k = (-1)^k * gain_k^k, gain_k = 1 + mean_d(tanh(s)) in [0, max_gain]; scalars: (heads, head_dim, lambdas).
    """

    u_lim = 0.8
    l_lim = 0.2

    def __init__(
        self, heads: int, head_dim: int, layer_idx: int = None,
        lambdas: int = 1, init_mean: float = 0., init_std: float = 0.5,
        max_gain: float = 2.0, affine: bool = True, epsilon: float = 1e-8,
        mode: str = 'legacy', beta_bias: float = 3.0,
        device: DEVICE = 'cpu', dtype: DTYPE = torch.float32
    ):
        super().__init__(
            heads=heads, head_dim=head_dim, coeffs=lambdas,
            base_limits=(self.u_lim, self.l_lim), base_affine=affine
        )

        self.heads      = heads
        self.head_dim   = head_dim
        self.layer_idx  = layer_idx if layer_idx is not None else 0
        self.lambdas    = lambdas
        self.max_gain   = max_gain
        self.base_affine = affine
        self.epsilon    = epsilon
        self.range      = abs(self.u_lim - self.l_lim)

        # self.q1 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.q2 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.k1 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.k2 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.init = (
        #     self.u_lim - (self.range * math.exp(-0.3 * self.layer_idx))
        #     if not affine else
        #     NeatParameter((heads, 1), requires_grad=False, device=device, dtype=dtype) # .unsqueeze(2)
        # )
        if mode not in ('legacy', 'simplex'):
            raise ValueError(f"AttentionLambda mode must be 'legacy' or 'simplex', got '{mode}'")
        self.mode = mode
        self.beta_bias = beta_bias    # sigmoid(0 + 3) ~ 0.95: starts close to exact common-mode rejection
        scalar_shape = (heads, lambdas + 1) if mode == 'simplex' else (heads, head_dim, lambdas)
        self.scalars = NeatParameter(scalar_shape, requires_grad=False, device=device, dtype=dtype)
        # shape (1, lambdas) — broadcasts with gain (heads, lambdas)
        self.exponents = (torch.arange(lambdas, device=device, dtype=dtype) + 1).unsqueeze(0).unsqueeze(1)
        self.multipliers = torch.pow(-1, self.exponents)

        self.mean = init_mean
        self.std  = init_std

        # Register a hook to modify gradients
        # TODO: Might be useless since NEAT backend has its own way initializing Parameter values, 
        #       and happens after the model has already been initialized. Just ensure the output
        #       gain is within the realistic limit of [0, 1]
        for param in self.neat_parameters(recurse=False):
            with torch.no_grad():
                param.data.normal_(init_mean, init_std)
        self.min_val = init_mean - init_std * 2
        self.max_val = init_mean + init_std * 2

        # with torch.no_grad():
        #     for p in [self.q1, self.q2, self.k1, self.k2]:
        #         p.data.normal_(init_mean, init_std)

    def extra_repr(self):
        return (
            f"heads={self.heads}, head_dim={self.head_dim}, coeffs={self.lambdas}, "
            f"base_limits={(self.u_lim, self.l_lim)}, base_affine={self.base_affine}"
        )

    def init_affine(self, keys: Union[int, list[int], None]):
        # if isinstance(self.init, NeatParameter):
        #     return self.l_lim + (self.range * torch.sigmoid(self.init[keys]))
        # else:
        #     return self.init
        return 0.0

    def post_attention_shift(self, keys: Union[int, list[int], None], offset: int | None = None, verbose: bool | int = False):
        """Returns a scalar/tensor broadcastable with attended (batch, q_len, heads, head_dim)."""
        # # TODO: Fix the parameterized implementation of init
        # # return self.init if not isinstance(self.init, NeatParameter) else self.expand(self.init[keys], tensor, offset=offset, keys=keys)
        # if not isinstance(self.init, NeatParameter):
        #     shift = self.init
        # else:
        #     # reshape (heads, 1) -> (1, 1, heads, 1) to broadcast over (b, q, h, d)
        #     parameters = self.init[keys]
        #     shift = self.l_lim + (self.range * torch.sigmoid(parameters))
        #     shift = shift.unsqueeze(1).unsqueeze(2)
        shift = torch.tensor(0.0, device=self.scalars.device, dtype=self.scalars.dtype)
        if verbose:
            print(get_tensor_info(shift, 'Shift', verbose))
        return shift

    # def update_limit(self):
    #     for param in self.neat_parameters():
    #         with torch.no_grad():
    #             param.data[:] = torch.clamp(param.data, self.min_val, self.max_val)

    @torch.no_grad()
    def diagnostics(self, keys: Union[int, Iterable[int], None] = None) -> dict[str, Tensor]:
        """
        Read-only view of the differential factors, for logging (does not change any behaviour).

        Returns tensors with the genomes on dim 0:
            ``lambdas``  (genomes, heads, lambdas)  the coefficients applied to the extra branches (a_0 = 1 is implicit)
            ``lambda_sum`` (genomes, heads)         sum_k lambda_k
            ``cm_gain``  (genomes, heads)           |1 + sum_k lambda_k|: gain of a component shared by every branch
                                                    (0 = perfect common-mode rejection, 1 = no rejection)
        and in 'simplex' mode additionally ``beta`` (genomes, heads) and ``weights`` (genomes, heads, lambdas).
        """
        lam = self.forward(keys)[:, 0, :, :, 0, 0]                  # (genomes, heads, lambdas)
        out = {'lambdas': lam, 'lambda_sum': lam.sum(-1), 'cm_gain': (1.0 + lam.sum(-1)).abs()}
        if self.mode == 'simplex':
            s = self.scalars[keys]
            out['weights'] = torch.softmax(s[..., :self.lambdas], dim=-1)
            out['beta'] = torch.sigmoid(s[..., self.lambdas] + self.beta_bias)
        return out

    def forward(self, keys: Union[int, Iterable[int]] = None):
        """
        Returns:
            (1, heads, lambdas, 1, 1)
            Broadcasts over energy shaped (batch, heads, coeffs, q_len, k_len).
        """
        if self.mode == 'simplex':
            s = self.scalars[keys]                                   # (genomes, heads, lambdas + 1)
            w = torch.softmax(s[..., :self.lambdas], dim=-1)         # (genomes, heads, lambdas), sums to 1
            beta = torch.sigmoid(s[..., self.lambdas:] + self.beta_bias)   # (genomes, heads, 1), in (0, 1)
            coeffs = -beta * w                                       # all subtracted; sum = -beta
            return coeffs.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)   # (genomes, 1, heads, lambdas, 1, 1)
        # query:     (batch_size, q_len, heads, head_dim)
        # key:       (batch_size, k_len, heads, head_dim)
        # attention: (batch_size, heads, q_len, k_len)
        # # TODO: Might want to verify the need of clamping the parameters
        # q1 = self.q1[keys] # * self.std # F.tanh(self.q1[keys], self.min_val, self.max_val)
        # k1 = self.k1[keys] # * self.std # F.tanh(self.k1[keys], self.min_val, self.max_val)
        # q2 = self.q2[keys] # * self.std # F.tanh(self.q2[keys], self.min_val, self.max_val)
        # k2 = self.k2[keys] # * self.std # F.tanh(self.k2[keys], self.min_val, self.max_val)
        # gain = (
        #     torch.exp(torch.sum(q1 * k1, dim=-2)) -
        #     torch.exp(torch.sum(q2 * k2, dim=-2))
        # )
        # gain = torch.sigmoid(gain) * self.max_gain
        # TODO: Most implementations of NEAT weight mutation do not scale well so exponential calculations are to be avoided
        gain = torch.clamp(torch.mean(torch.tanh(self.scalars[keys]) * self.max_gain, dim=-2), -self.max_gain, +self.max_gain)
        gain = (gain + self.max_gain) / 2 # Ensures positive gain
        assert torch.all((gain >= 0) & (gain <= self.max_gain))
        # self.exponents: (1, lambdas), self._init_affine(): (heads, 1) or scalar
        scalars = (
            # (base + self.biases(base, keys, 0)) * self.multipliers # ** self.exponents * self.multipliers
            ((gain + self.init_affine(keys)) ** self.exponents) * self.multipliers
        )
        # Returns shape (genomes, batch_size=1, heads, lambdas, query_len=1, key_len=1)
        return scalars.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)


class BranchGain(NeatModule):
    """
    'Tied' differential branches: every extra branch is the SAME q/k projection as branch 0 with its own per-head-dim
    multiplicative gain, ``g = exp(strength * tanh(s))`` in [exp(-strength), exp(strength)] (saturation-safe).

    Why: with independent q/k projections per branch nothing makes the branches share a component, so there is no
    'common mode' to reject; noise entering the input hits all branches identically only when they share a projection.
    Here it does by construction, and the extra parameters are tiny vectors (heads * branches * head_dim) that NEAT mutates well.

    forward: (genomes, batch, seq, heads, head_dim) -> (genomes, batch, seq, heads, (1 + branches) * head_dim)
    """

    def __init__(self, heads: int, head_dim: int, branches: int, strength: float = 0.7,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super().__init__(heads=heads, head_dim=head_dim, branches=branches)
        self.heads, self.head_dim, self.branches, self.strength = heads, head_dim, branches, strength
        self.scalars = NeatParameter((heads, branches, head_dim), requires_grad=False, device=device, dtype=dtype)
        for param in self.neat_parameters(recurse=False):
            with torch.no_grad():
                param.data.normal_(0.0, 0.5)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        gain = torch.exp(self.strength * torch.tanh(self.scalars[keys]))      # (genomes, heads, branches, head_dim)
        gain = gain.unsqueeze(1).unsqueeze(1)                                  # (genomes, 1, 1, heads, branches, head_dim)
        base = tensor.unsqueeze(-2)                                            # (g, b, s, heads, 1, head_dim)
        out  = torch.cat([base, base * gain], dim=-2)                          # (g, b, s, heads, 1 + branches, head_dim)
        return out.reshape(*tensor.shape[:-1], (1 + self.branches) * self.head_dim)


class Attention(NeatModule):
    """
    Multi-head (grouped-query) self-attention with:
      - Arbitrary extra dims: (genomes, *extra_dims, seq_len, dim)
      - RoPE positional embeddings
      - Optional differential attention (λ-weighted multi-head combination)
      - Optional KV-cache with dequeue-based overflow handling
      - Pre-RMSNorm + residual connection
    """

    def __init__(
            self, max_seq_len: int, dim_size: int, heads: int = None, kv_heads: int = None,
            differential: int | bool = None, layer_idx: int = None, causal_mask: bool = True,
            bias: bool = False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options
    ):
        super().__init__()

        # ---- head / dim bookkeeping ----------------------------------------
        if heads is None:
            heads = 1
        if dim_size % heads != 0:
            raise ValueError("dim_size must be divisible by heads")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        if isinstance(differential, bool):
            differential = int(differential)

        # ---- options -------------------------------------------------------
        self.auto_single    = manage_params(options, 'auto_single', False)
        self.constant       = manage_params(options, 'constant', 10_000)
        self.epsilon        = manage_params(options, ['eps', 'epsilon'], 1e-9)
        self.affine         = manage_params(options, 'affine', True)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], True)
        self.normalize      = manage_params(options, 'normalize', True)
        self.cache          = manage_params(options, ['cache', 'kv_cache'], True)
        inputs: int         = manage_params(options, 'inputs', None)
        outputs: int        = manage_params(options, 'outputs', None)
        self.repo_enabled   = manage_params(options, 'reposition', True)
        self.lambda_mode    = manage_params(options, 'lambda_mode', 'simplex')   # 'simplex' | 'legacy'
        # diff_mode 'energy': e_0 + sum(l_k e_k) BEFORE the softmax. NOTE: with free branch projections this is algebraically
        #   one ordinary attention with a (1+L)x wider head (beta/lambda are absorbed by the k weights): only extra parameters.
        # diff_mode 'softmax': softmax(e_0) + sum(l_k softmax(e_k)), l_k < 0 (paper form). Cannot be written as one softmax
        #   (signed weights, cancels the diffuse background of the maps), so it is a functional difference.
        self.diff_mode      = manage_params(options, 'diff_mode', 'energy')
        # branch_mode 'free': independent q/k projection per branch | 'tied': one projection + BranchGain per branch
        self.branch_mode    = manage_params(options, 'branch_mode', 'free')
        # kv_exact: K/V of a cached token depend on that token alone (see TransformerBase share_kv): shifts are exact
        self.kv_exact       = manage_params(options, ['kv_exact', 'exact_cache'], False)
        self.qk_norm        = manage_params(options, 'qk_norm', False)           # RMS-normalise q/k per branch before RoPE
        self.branch_rope    = manage_params(options, 'branch_rope', True)        # rotate every differential branch with the SAME frequencies
        self.proj_coeffs    = int(manage_params(options, ['proj_coeffs', 'coefficients'], 1))   # >1: Polynomial instead of Linear
        self.poly_projs     = manage_params(options, 'poly_projs', 'qkvo')       # which projections: letters of q, k, v, o
        self.repo_coeffs    = int(manage_params(options, 'repo_coeffs', self.proj_coeffs))      # >1: Polynomial in the cache Repositioning converters
        if self.diff_mode not in ('energy', 'softmax') or self.branch_mode not in ('free', 'tied'):
            raise ValueError(f"diff_mode must be 'energy'|'softmax' and branch_mode 'free'|'tied', got '{self.diff_mode}'/'{self.branch_mode}'")
        if self.proj_coeffs < 1 or self.repo_coeffs < 1:
            raise ValueError(f"proj_coeffs/repo_coeffs must be >= 1, got {self.proj_coeffs}/{self.repo_coeffs}")

        # ---- attributes ----------------------------------------------------
        self.bias        = bias
        self.heads       = heads
        self.dim_size    = dim_size
        self.head_dim    = dim_size // heads
        self.kv_heads    = kv_heads
        self.q_kv_ratio  = heads // kv_heads
        self.max_seq_len = max_seq_len
        self.causal_mask = causal_mask
        self.differential      = differential
        self.mult              = 1 if not differential else 1 + int(differential)
        self.att_coeff_num     = None if not differential else 1 + int(differential)
        self.att_coeff_indices = None if not differential else torch.arange(differential, device=device) + 1
        self.tied              = bool(differential) and self.branch_mode == 'tied'
        self.proj_mult         = 1 if self.tied else self.mult   # q/k projection width multiplier (tied: expanded after)
        self.layer_idx = 0 if layer_idx is None else layer_idx
        self.device = device
        self.dtype  = dtype
        # noise probe (see set_noise_probe)
        self.probe_sigma = 0.0
        self.noise_sink = None
        self._probing = False

        # ---- sub-modules ---------------------------------------------------
        in_dim  = dim_size if inputs  is None else inputs
        out_dim = dim_size if outputs is None else outputs

        # self.pre_norm = RMSNorm(dim_size, self.epsilon, self.affine, device, dtype) if self.normalize else None
        self.pre_norm = LayerNorm(dim_size, self.epsilon, self.affine, bias, device, dtype) if self.normalize else None

        self.query_proj = self._projection('q', in_dim, heads * self.head_dim * self.proj_mult, bias, device, dtype)
        self.key_proj   = self._projection('k', in_dim, kv_heads * self.head_dim * self.proj_mult, bias, device, dtype)
        self.value_proj = self._projection('v', in_dim, kv_heads * self.head_dim, bias, device, dtype)
        self.out_proj   = self._projection('o', dim_size, out_dim, bias, device, dtype)

        # With branch_rope each differential branch is a head_dim-wide vector rotated with identical frequencies, so a
        # signal common to the branches stays common (incl. its relative-position part) and can be rejected.
        rope_size = dim_size if self.branch_rope else dim_size * self.mult
        self.rotary_embedding = RoPE(self.max_seq_len, rope_size, heads, self.constant, device, dtype)
        self.softmax    = nn.Softmax(dim=-1)
        # self.head_norm  = RMSNorm(self.head_dim, self.epsilon, self.affine, device, dtype) if self.normalize else None
        self.head_norm  = LayerNorm(self.head_dim, self.epsilon, self.affine, bias, device, dtype) if self.normalize else None
        self.diff_lambda = AttentionLambda(
            heads, self.head_dim, layer_idx, differential, 0.0, 0.1, 2.0, True, self.epsilon, 
            mode=self.lambda_mode, device=device, dtype=dtype,
        ) if differential else None
        self.gain_q = BranchGain(heads, self.head_dim, differential, device=device, dtype=dtype) if self.tied else None
        self.gain_k = BranchGain(kv_heads, self.head_dim, differential, device=device, dtype=dtype) if self.tied else None

        # Repositioning helpers for KV-cache dequeue. Keys are RoPE-rotated, values are not.
        self.repo_k = Repositioning(
            max_seq_len, self.head_dim * self.mult, self.layer_idx, bias, self.rotary_embedding, device, dtype,
            coefficients=self.repo_coeffs, exact=self.kv_exact, **options
        ) if self.cache else None
        self.repo_v = Repositioning(
            max_seq_len, self.head_dim, self.layer_idx, bias, None, device, dtype,
            coefficients=self.repo_coeffs, exact=self.kv_exact, **options
        ) if self.cache else None

        # ---- KV-cache state ------------------------------------------------
        self.cache_k: Tensor | None = None  # (genomes, batch_size, max_seq_len, heads, head_dim*mult)
        self.cache_v: Tensor | None = None  # (genomes, batch_size, max_seq_len, heads, head_dim)
        self.cache_size: int = 0 # TODO: Maybe make cache size a tensor that references each genome (and batch index maybe)
        self.forced_cache: bool = False

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
        
    def force_cache(self, enable: bool = True):
        if enable and not self.cache:
            raise ValueError("force_cache needs the repositioning modules; build with cache=True")
        self.forced_cache = enable
        if enable:
            if self.cache_k is None:
                self._init_cache(None, 1)
        else:
            # if not self.cache:
            # Save memory by clearing any cache, especially on GPU memory. NOTE: cuda.empty_cache()
            self.cache_k = self.cache_v = None
            self.cache_size = 0
        
    def empty_cache(self):
        """Reset cache to initial state."""
        if self.cache_k is None:
            self._init_cache(None, 1)
        if self.cache_k is not None:
            self.cache_k.zero_()
        if self.cache_v is not None:
            self.cache_v.zero_()
        self.cache_size = 0
    
    def _init_cache(self, genomes: int | None = None, batch_size: int = 1):
        """Resize cache if batch dimensions changed."""
        # TODO: Modify parameters entered to cater for extra dims
        if genomes is None:
            genomes = self.genome_num if self.genome_num is not None else 1
        self.cache_k = torch.zeros(
            genomes, batch_size, self.max_seq_len, self.kv_heads, self.head_dim * self.mult,
            device=self.device, dtype=self.dtype
        )
        self.cache_v = torch.zeros(
            genomes, batch_size, self.max_seq_len, self.kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        self.cache_size = 0
    
    def _ensure_cache(self, genomes: int, batch_size: int):
        if (self.cache_k is None or self.cache_k.shape[0] != genomes or self.cache_k.shape[1] != batch_size):
            self._init_cache(genomes, batch_size)

    def _update_cache(self, key: Tensor, value: Tensor, pos_idx: int,
                      keys: Union[int, Iterable[int]] = None, verbose: bool | int = False):
        """
        forced_cache=True : append after the current cache; on overflow dequeue the oldest `excess`
                            entries (reposition the survivors) and append at the end.
        forced_cache=False: write at pos_idx and truncate everything after it.
        key: (g, b, s, kv_heads, head_dim*mult)   value: (g, b, s, kv_heads, head_dim)
        """
        g, b, s = key.shape[:3]
        if s > self.max_seq_len:
            raise ValueError(f"Input sequence length {s} > max_seq_len {self.max_seq_len}")
        self._ensure_cache(g, b)

        if self.forced_cache:
            start = self.cache_size
        else:
            if not 0 <= pos_idx < self.max_seq_len:
                raise ValueError(f"Position index '{pos_idx}' must be within [0, {self.max_seq_len - 1}]")
            if pos_idx > self.cache_size:
                raise ValueError(f"Position index '{pos_idx}' leaves a gap after cache_size={self.cache_size}")
            start = pos_idx
        end = start + s
        if not self.forced_cache and end > self.max_seq_len:
            raise ValueError(f"pos_idx {pos_idx} + seq_len {s} exceeds max_seq_len {self.max_seq_len}")

        excess = max(0, end - self.max_seq_len)      # only ever > 0 in forced mode
        if verbose: print(f"Shifting cache by {excess} position(s)")
        if excess:
            keep = self.cache_size - excess          # survivors; <= 0 means the whole cache is replaced
            if keep > 0:
                # clone: source and destination slices overlap
                self.cache_k[:, :, :keep] = self.repo_k(
                    self.cache_k[:, :, excess:self.cache_size].clone(), excess, keys=keys, verbose=verbose)
                self.cache_v[:, :, :keep] = self.repo_v(
                    self.cache_v[:, :, excess:self.cache_size].clone(), excess, keys=keys)
            start -= excess
            end   -= excess
        if verbose: print(f"Setting position {start} to {end}")
        self.cache_k[:, :, start:end] = key
        self.cache_v[:, :, start:end] = value
        self.cache_size = end

    # ------------------------------------------------------------------
    # GQA helper
    # ------------------------------------------------------------------

    def set_noise_probe(self, sigma: float = 0.0, sink=None):
        """
        Attention-noise probe for full-sequence passes: with ``sigma > 0`` every (uncached) forward also runs once on an
        input perturbed by ``sigma * token_std * N(0, 1)`` and reports ``sink(keys, noise, batch)`` where ``noise`` is
        (genomes,): mean total-variation distance 0.5 * sum_k |scores - scores_noisy| of the attention maps.
        0 = the maps ignore the perturbation, larger = noise moves the attention. Works for plain and differential
        attention alike, so 'does differential attention reduce noise' is a direct comparison.
        """
        self.probe_sigma = float(sigma)
        self.noise_sink = sink if sigma > 0 else None

    @torch.no_grad()
    def _probe(self, tensor: Tensor, prompt: Tensor | None, keys, pos_idx: int, scores: Tensor):
        self._probing = True
        try:
            sigma = self.probe_sigma
            noisy = tensor + sigma * tensor.std(-1, keepdim=True) * torch.randn_like(tensor)
            nprompt = None if prompt is None else prompt + sigma * prompt.std(-1, keepdim=True) * torch.randn_like(prompt)
            _, noisy_scores = self.forward(noisy, keys=keys, prompt=nprompt, pos_idx=pos_idx, get=True)
        finally:
            self._probing = False
        tv = 0.5 * (scores - noisy_scores).abs().sum(-1)            # (g, b, h, q)
        self.noise_sink(keys, tv.mean(dim=(1, 2, 3)).float(), tensor.shape[1])

    def _projection(self, tag: str, inputs: int, outputs: int, bias: bool, device, dtype):
        """Linear, or Polynomial(proj_coeffs) for the projections listed in poly_projs.
        Polynomial(1 coefficient) is exactly a Linear layer, so proj_coeffs=1 keeps Linear."""
        if self.proj_coeffs > 1 and tag in self.poly_projs:
            return Polynomial(inputs, outputs, self.proj_coeffs, bias, device, dtype)
        return Linear(inputs, outputs, bias, device, dtype)

    def _qk_norm(self, tensor: Tensor):
        """Parameter-free RMS normalisation of every branch of q/k over head_dim.
        Bounds the energies regardless of how large the q/k projection weights get (RoPE preserves the norm)."""
        shape = tensor.shape                                     # (..., heads, head_dim * mult)
        tensor = tensor.reshape(*shape[:-1], self.mult, self.head_dim)
        tensor = tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.epsilon)
        return tensor.reshape(shape)

    def _rope(self, tensor: Tensor, pos_idx: int, verbose: bool | int = False):
        """RoPE over (g, b, seq, heads, head_dim * mult); with branch_rope every branch gets its own full set of frequencies."""
        if not self.branch_rope or self.mult == 1:
            return self.rotary_embedding(tensor, pos_idx, verbose)
        shape = tensor.shape
        folded = tensor.reshape(*shape[:-2], shape[-2] * self.mult, self.head_dim)
        return self.rotary_embedding(folded, pos_idx, verbose).reshape(shape)

    def repeat_kv(self, tensor: Tensor):
        """Expand key/value heads to match query heads (GQA)."""
        # tensor: (genomes, *extra_dims, seq_len, kv_heads, head_dim)
        if self.q_kv_ratio == 1:
            return tensor
        genomes, pre, kv, post = tensor.shape[0], tensor.shape[1:-2], tensor.shape[-2], tensor.shape[-1]
        return (
            tensor
            .unsqueeze(-2)
            .expand(genomes, *pre, kv, self.q_kv_ratio, post)
            .reshape(genomes, *pre, kv * self.q_kv_ratio, post)
        )

    # ------------------------------------------------------------------
    # Core attention computation
    # ------------------------------------------------------------------

    def attention(
        self, query: Tensor, key: Tensor, value: Tensor,
        keys: Union[int, Iterable[int]], mask: bool, verbose: bool | int = False
    ):
        """
        Args:
            query:  (genomes, batch, q_len,  heads, head_dim [* mult])
            key:    (genomes, batch, kv_len, heads, head_dim [* mult])
            value:  (genomes, batch, kv_len, heads, head_dim)
        Returns:
            (scores, attended_values) with attended shaped (batch, q_len, heads, head_dim)
        """
        if self.differential:
            query = query.view(*query.shape[:-1], self.att_coeff_num, -1)
            key   = key.view(*key.shape[:-1], self.att_coeff_num, -1)

        # Get the attention score (energy)
        eqn = "...qhd,...khd->...hqk" if not self.differential else "...qhcd,...khcd->...hcqk"
        energy = torch.einsum(eqn, [query, key])
        # queries shape: (genomes, batch_size, query_len, heads, *coeffs, head_dim)
        # key shape:     (genomes, batch_size, key_len, heads, *coeffs, head_dim)
        # energy shape:  (genomes, batch_size, heads, *coeffs, query_len, key_len)
        if verbose:
            print(get_tensor_info(energy, 'Energy', verbose))

        softmax_diff = bool(self.differential) and self.diff_mode == 'softmax'
        if self.differential:
            lambdas: Tensor = self.diff_lambda(keys)
            if verbose and verbose >= 2:
                print(get_tensor_info(torch.round(lambdas, decimals=4), 'Lambdas', verbose+2))
                print(get_tensor_info(self.att_coeff_indices, 'Coeff Indices', verbose+2))
            if not softmax_diff:
                # Index '-3' is the lambdas' coefficients dimension
                # Collapse the coefficient dimension: (g, b, h, c, q, k) -> (g, b, h, q, k)
                energy = torch.select(energy, dim=-3, index=0) + torch.sum(
                    torch.index_select(energy, dim=-3, index=self.att_coeff_indices) * lambdas, dim=-3
                )

        if mask:
            q_len, k_len = energy.shape[-2:]
            if q_len > k_len:
                raise ValueError(f"query_len {q_len} > key_len {k_len}: first queries would have no keys")
            # the last query sees the last key, so query i sees keys <= i + (k_len - q_len)
            mask_ = torch.ones(q_len, k_len, device=energy.device, dtype=torch.bool).triu(1 + k_len - q_len)
            energy = energy.masked_fill(mask_, -torch.inf)
            if verbose and verbose >= 2:
                print(get_tensor_info(mask_, 'Mask', verbose))
                print(get_tensor_info(energy, 'Masked Energy', verbose))

        # Get the softmax of the energy: scores is always (g, b, h, q, k) at this point
        scores = self.softmax(energy / math.sqrt(self.head_dim))
        if softmax_diff:
            # scores (g, b, h, c, q, k): one map per branch -> map_0 + sum_k lambda_k * map_k  (lambda_k <= 0)
            scores = torch.select(scores, dim=-3, index=0) + torch.sum(
                torch.index_select(scores, dim=-3, index=self.att_coeff_indices) * lambdas, dim=-3
            )
        if verbose:
            print(get_tensor_info(torch.round(scores, decimals=4), 'Attention Score', verbose))

        # Get the weighted sum of the values and reshape to remove heads
        attention = torch.einsum("...hqv,...vhd->...qhd", [scores, value])
        if self.head_norm is not None:
            attention = self.head_norm(attention, keys=keys)
        # scores shape:    (genomes, batch_size, heads, query_len, value_len)
        # values shape:    (genomes, batch_size, value_len, heads, head_dim)
        # attention shape: (genomes, batch_size, query_len, heads, head_dim) then concat last 2 dim
        if self.differential:
            # attention = attention * (1 - self.diff_lambda.biases(attention, keys, 2))
            attention = attention * (1 - self.diff_lambda.post_attention_shift(keys, verbose=verbose))

        return scores, attention

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, prompt: Tensor = None,
                pos_idx: int | None = 0, verbose: bool | int = False, get: bool = False):
        """
        Accepts (genomes, *extra_dims, seq_len, dim) for any number of extra dims (batch, processes, ...): attention only
        mixes along seq_len, so every extra dim is an independent batch. They are folded into one batch axis for the
        computation (and for the KV cache, which therefore needs the same extra dims on every call) and restored on the
        output. `prompt` must carry the same extra dims. With ``get`` the returned scores are
        (genomes, *extra_dims, heads, q_len, k_len).
        """
        if tensor.ndim < 4:
            raise ValueError(f"Attention expects (genomes, *extra_dims, seq_len, dim) with at least one extra dim, got {tuple(tensor.shape)}")
        if tensor.ndim == 4:
            return self._forward(tensor, keys, prompt, pos_idx, verbose, get)
        g, extra = tensor.shape[0], tuple(tensor.shape[1:-2])
        if prompt is not None:
            if tuple(prompt.shape[:-2]) != (g, *extra):
                raise ValueError(f"prompt dims {tuple(prompt.shape[:-2])} must match the input's leading dims {(g, *extra)}")
            prompt = prompt.reshape(g, -1, *prompt.shape[-2:])
        out = self._forward(tensor.reshape(g, -1, *tensor.shape[-2:]), keys, prompt, pos_idx, verbose, get)
        if get:
            out, scores = out
            return out.reshape(g, *extra, *out.shape[-2:]), scores.reshape(g, *extra, *scores.shape[2:])
        return out.reshape(g, *extra, *out.shape[-2:])

    def _forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, prompt: Tensor = None,
                 pos_idx: int | None = 0, verbose: bool | int = False, get: bool = False):
        """
        tensor:  (genomes, batch, seq_len, dim)  -> supplies K and V (and Q unless `prompt` is given)
        prompt:  optional (genomes, batch, q_len, dim) with q_len <= seq_len; they are the LAST q_len queries
        pos_idx: absolute position of the first query token (negative = relative to seq_len).
                 Keys are placed so the last key and last query coincide (what the causal mask assumes).
                 Ignored in forced-cache mode, where positions come from the cache.
        """
        if self.auto_single and prompt is None:
            prompt = tensor.select(-2, -1).unsqueeze(-2)
        seq_len = tensor.shape[-2]
        raw_tensor, raw_prompt = tensor, prompt
        if pos_idx is None:
            pos_idx = 0
        elif pos_idx < 0:
            pos_idx += seq_len
        if not 0 <= pos_idx < self.max_seq_len:
            raise ValueError(f"pos_idx {pos_idx} must be within [0, {self.max_seq_len - 1}]")
        if verbose:
            print(f'\n{CM("Executing Self Attention", Fore.LIGHTBLUE_EX)}')
            print(get_tensor_info(tensor, f'Input', verbose, Fore.LIGHTRED_EX))
            if prompt is not None:
                print(get_tensor_info(prompt, f'Prompt', verbose, Fore.LIGHTRED_EX))

        # Get residue and pre-normalize
        residue = tensor if prompt is None else prompt
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
            if prompt is not None:
                prompt = self.pre_norm(prompt, keys=keys)

        # Linearize Q, K, V
        query: Tensor   = self.query_proj(tensor if prompt is None else prompt, keys=keys)
        key: Tensor     = self.key_proj(tensor, keys=keys)
        value: Tensor   = self.value_proj(tensor, keys=keys)
        if verbose:
            print(CM('Post Linearization =>'))
            print(get_tensor_info(query, 'Query', verbose, Fore.LIGHTRED_EX))
            print(get_tensor_info(key, 'Key', verbose, Fore.LIGHTGREEN_EX))
            print(get_tensor_info(value, 'Value', verbose, Fore.LIGHTBLUE_EX))

        # Reshape Q, K, V for each rep head
        g, b, q_len, _ = query.shape
        query   = query.view(g, b, q_len, self.heads, self.head_dim*self.proj_mult)
        key     = key.view(g, b, -1, self.kv_heads, self.head_dim*self.proj_mult)
        value   = value.view(g, b, -1, self.kv_heads, self.head_dim)
        if self.tied:   # one projection, (1 + L) gain-scaled branches -> (.., head_dim * mult)
            query, key = self.gain_q(query, keys), self.gain_k(key, keys)
        if verbose:
            print(get_tensor_info(query, 'Q after Reshaping', verbose))
            print(get_tensor_info(key, 'K after Reshaping', verbose))
        if self.qk_norm:
            query, key = self._qk_norm(query), self._qk_norm(key)
        key_len = key.shape[-3]
        if q_len > key_len:
            raise ValueError(f"query_len {q_len} > key_len {key_len}")
        lead = key_len - q_len        # queries are the last q_len of the incoming key chunk
        if key_len > self.max_seq_len:
            raise ValueError(f"seq_len {key_len} > max_seq_len {self.max_seq_len}")

        # Absolute position of the first incoming key
        use_cache = self.cache_k is not None and (self.forced_cache or self.cache)
        if self.forced_cache:
            start = self.cache_size if (self.cache_size + key_len) <= self.max_seq_len else (self.max_seq_len - key_len)
        else:
            start = max(0, pos_idx - lead)
            if start + key_len > self.max_seq_len:
                raise ValueError(f"positions {start}..{start + key_len} exceed max_seq_len {self.max_seq_len}")
        q_pi, k_pi = start + lead, start
        query = self._rope(query, q_pi, verbose)
        key   = self._rope(key, k_pi)
        if verbose:
            print(get_tensor_info(query, f'Q after Rotary Embedding({q_pi}, {q_len})', verbose))
            print(get_tensor_info(key, f'K after Rotary Embedding({k_pi}, {key_len})', verbose))

        # KV-cache (forced: deque + reposition | normal: positional write + truncate)
        if use_cache:
            self._update_cache(key, value, start, keys, verbose)
            key   = self.cache_k[:, :, :self.cache_size]
            value = self.cache_v[:, :, :self.cache_size]
            if verbose:
                print(f'Using cached K, V with cache_size={self.cache_size}')
                print(get_tensor_info(key, 'Cached K', verbose))
                print(get_tensor_info(value, 'Cached V', verbose))

        # GQA expansion
        key   = self.repeat_kv(key)
        value = self.repeat_kv(value)
        if verbose:
            print(get_tensor_info(key, 'Duplicated K', verbose))
            print(get_tensor_info(value, 'Duplicated V', verbose))

        # Apply attention
        attention_scores, attention = self.attention(query, key, value, keys, self.causal_mask, verbose)
        attention = attention.reshape(g, b, q_len, self.dim_size)
        # out_view shape:  (genomes, batch_size, seq_len, dim_size)
        if verbose:
            print(get_tensor_info(attention, 'Attended Values', verbose))

        # Output projection + residual
        tensor: Tensor = self.out_proj(attention, keys=keys)
        if self.skip_connection:
            tensor = tensor + residue
        if verbose:
            print(get_tensor_info(tensor, 'Output Projection', verbose))

        if self.probe_sigma > 0 and self.noise_sink is not None and not self._probing and not use_cache:
            self._probe(raw_tensor, raw_prompt, keys, pos_idx, attention_scores)

        # Store attention for debugging
        return tensor if not get else (tensor, attention_scores)

    def extra_repr(self) -> str:
        return (f"*** "
                f"residual={self.skip_connection}, normalize={self.normalize}, "
                f"auto_single={self.auto_single}. causal_mask={self.causal_mask}"
                f"***")
