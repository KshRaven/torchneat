"""Convolutional self- and cross-attention."""

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
from ..layers import Conv1d, Conv2d, Conv3d, GroupNorm, RMSNorm
from ..module import NeatModule
from ..utils import calc_padding, display, get_tensor_info
from ._common import get_convolution_class
from .attention import AttentionLambda
from .encoding import RoPE

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.dim_size}, pixels={self.pixels}, kernel_size={self.kernel_size}, " \
    #            f"bias={self.bias}{display('hd', self._heads)}{display('kv_hd', self._kv_heads)}" \
    #            f"{display('diffs', self.att_coeff_num-1 if self.att_coeff_num else None)}" \
    #            f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
    #            f"{display('pre_norm_ng', self.norm_groups)}{display('residual', self.skip_connection)}" \
    #            f"{display('as', self.auto_single)}"

# TODO: Convoluted attention is not updated yet

class ConvSelfAttention(NeatModule):
    def __init__(
            self, max_pixels: list[int], dim_size: int, kernel_size: int, heads: int = None, kv_heads: int = None,
            differential: int = None, layer_idx: int = None, causal_mask=False,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(ConvSelfAttention, self).__init__()
        self._heads = heads
        self._kv_heads = kv_heads
        if heads is None:
            heads = 1
        if dim_size % heads != 0:
            raise ValueError(f"Embedding dimensions must be a multiple of heads num")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError(f"Query heads num must be a multiple of number of Key-Value heads num")
        if isinstance(differential, bool):
            differential = int(differential)
        if not isinstance(max_pixels, (list, tuple)):
            raise ValueError(f"pixels parameter must be Iterable got '{type(max_pixels)}'")

        # OPTIONS
        self.auto_single = manage_params(options, 'auto_single', False)
        if self.auto_single:
            kernel_size = 1
        self.padding = manage_params(options, 'padding', calc_padding(kernel_size, stride=1, dilation=1))
        self.padding_mode = manage_params(options, 'padding_mode', 'zeros')
        self.constant = manage_params(options, 'constant', 10000)
        self.epsilon = manage_params(options, 'epsilon', 1e-9)
        self.affine = manage_params(options, 'affine', True)
        self.norm_groups = manage_params(options, 'norm_groups', 1)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], True)

        # ATTRIBUTES
        self.heads      = heads
        self.dim_size   = dim_size
        self.head_dim   = dim_size // heads
        self.kv_heads   = kv_heads
        self.q_kv_ratio = heads // kv_heads
        self.pixels     = max_pixels
        self.differential = differential
        self.att_coeff_num = None if not differential else 1+differential
        self.att_coeff_indices = None if not differential else torch.arange(differential, device=device) + 1
        self.causal_mask = causal_mask
        self.pixels_total = math.prod(max_pixels)
        self.kernel_size = kernel_size
        self.bias = bias

        # MODULES
        self.pre_norm = GroupNorm(
            self.norm_groups, dim_size, self.epsilon, self.affine, False, device=device, dtype=dtype
        ) if self.norm_groups else None
        if len(max_pixels) == 1:
            Convolution = Conv1d
        elif len(max_pixels) == 2:
            Convolution = Conv2d
        elif len(max_pixels) == 3:
            Convolution = Conv3d
        else:
            raise ValueError(f"Unsupported num of image dimension '{len(max_pixels)}'")
        self.mult = 1+differential if differential else 1
        self.query_proj = Convolution(dim_size, heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.key_proj   = Convolution(dim_size, kv_heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.value_proj = Convolution(dim_size, kv_heads * self.head_dim, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.out_proj   = Convolution(dim_size, dim_size, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.rotary_embedding = RoPE(self.pixels_total, dim_size * self.mult, heads, self.constant, device, dtype)
        self.softmax    = nn.Softmax(-1)
        self.head_norm  = RMSNorm(self.head_dim, self.epsilon, self.affine, device, dtype)
        self.diff_lambda = AttentionLambda(
            heads, self.head_dim, layer_idx, differential, 0.0, 0.1, True, self.epsilon, device, dtype
        ) if differential else None

        # STATES
        self.device = device
        self.dtype  = dtype

    def repeat_kv(self, tensor: Tensor):
        genomes, batch_size, seq_len, kv_heads, head_dim = tensor.shape
        if self.q_kv_ratio == 1:
            return tensor
        else:
            return tensor.unsqueeze(-2).expand(genomes, batch_size, seq_len, kv_heads, self.q_kv_ratio, head_dim).\
                contiguous().view(genomes, batch_size, seq_len, kv_heads * self.q_kv_ratio, head_dim)

    @staticmethod
    def convert(image: Tensor, heads: int, head_dim: int, multiplier: int = 1):
        (g, b), c, p = image.shape[:2], image.shape[2] // multiplier, image.shape[3:]
        assert c == heads * head_dim
        # Return shape (genomes, batch_size, pixels, heads, head_dim)
        return image.view(g, b, c, -1).transpose(-1, -2).contiguous().view(g, b, -1, heads, head_dim*multiplier), (g, b, c, p)

    @staticmethod
    def revert(tensor: Tensor, batch_size: int, channels: int, pixels: list[int]):
        # Return shape (genomes, batch_size, channels, *pixels)
        return tensor.transpose(-1, -2).contiguous().view(-1, batch_size, channels, *pixels)

    def attention(self, query: Tensor, key: Tensor, value: Tensor, keys: Union[int, Iterable[int]], mask: bool, verbose: bool | int = False):
        if self.differential:
            query   = query.view(*query.shape[:-1], self.att_coeff_num, -1)
            key     = key.view(*key.shape[:-1], self.att_coeff_num, -1)
        # Get the attention score (energy)
        energy = torch.einsum("...qhd,...khd->...hqk" if not self.differential else "...qhcd,...khcd->...hcqk", [query, key])
        # queries shape: (genomes, batch_size, query_len, heads, *coeffs, head_dim)
        # key shape:     (genomes, batch_size, key_len, heads, *coeffs, head_dim)
        # energy shape:  (genomes, batch_size, heads, *coeffs, query_len, key_len)
        if verbose:
            print(get_tensor_info(energy, 'Energy', verbose))

        if mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask_ = torch.ones_like(energy, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            energy.masked_fill_(mask_, -torch.inf)
            if verbose and verbose >= 2 and not self.differential:
                print(get_tensor_info(mask_, 'Mask', verbose))
                print(get_tensor_info(energy, 'Masked Energy', verbose))

        # Get the softmax of the energy
        scores = self.softmax(energy / math.sqrt(self.head_dim))

        if self.differential:
            lambdas: Tensor = self.diff_lambda(keys)
            if verbose and verbose >= 2:
                print(get_tensor_info(torch.round(lambdas, decimals=4), 'Lambdas', verbose+2))
                print(get_tensor_info(self.att_coeff_indices, 'Coeff Indices', verbose+2))
            scores = torch.select(scores, dim=-3, index=0) + torch.sum(
                torch.index_select(scores, dim=-3, index=self.att_coeff_indices) * lambdas, dim=-3
            ) # * lambdas)

            if mask:
                scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
                if verbose and verbose >= 2 and not self.differential:
                    print(get_tensor_info(scores, 'Differential Masked Energy', verbose))
            scores = self.softmax(scores)

        if verbose:
            print(get_tensor_info(torch.round(scores, decimals=4), 'Attention Score', verbose))

        # Get the weighted sum of the values and reshape to remove heads
        attention = self.head_norm(torch.einsum("...hqv,...vhd->...qhd", [scores, value]), keys=keys)
        # scores shape:    (genomes, batch_size, heads, query_len, value_len)
        # values shape:    (genomes, batch_size, value_len, heads, head_dim)
        # attention shape: (genomes, batch_size, query_len, heads, head_dim) then concat last 2 dim
        if self.differential:
            attention = attention * (1 - self.diff_lambda.biases(attention, keys, 2))

        return scores, attention

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, 
                pretext: Tensor = None, pos_idx: int = None, verbose: bool | int = False, get=False):
        if self.auto_single and len(self.pixels) == 1 and pretext is None:
            pretext = tensor.select(-1, -1).unsqueeze(-1)
        if pos_idx is not None:
            pos_idx = self.pixels_total + pos_idx if pos_idx < 0 else pos_idx
            assert 0 < pos_idx < self.pixels_total
        if verbose:
            print(f'\n{CM("Executing Self Attention", Fore.LIGHTBLUE_EX)}')
            print(get_tensor_info(tensor, f'Input', verbose, Fore.LIGHTRED_EX))
            if pretext is not None:
                print(get_tensor_info(pretext, f'Pretext', verbose, Fore.LIGHTRED_EX))

        # Get residue and pre-normalize
        residue = tensor if pretext is None else pretext
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
            if pretext is not None:
                pretext = self.pre_norm(pretext, keys=keys)

        # Convolve Q, K, V
        query: Tensor   = self.query_proj(tensor if pretext is None else pretext, keys=keys)
        key: Tensor     = self.key_proj(tensor, keys=keys)
        value: Tensor   = self.value_proj(tensor, keys=keys)
        if verbose:
            print(CM('Post Linearization =>'))
            print(get_tensor_info(query, 'Query', verbose, Fore.LIGHTRED_EX))
            print(get_tensor_info(key, 'Key', verbose, Fore.LIGHTGREEN_EX))
            print(get_tensor_info(value, 'Value', verbose, Fore.LIGHTBLUE_EX))

        batch_size, q_seq_len = query.shape[:2]

        # Reshape Q, K, V for each rep head
        query, (g, b, c, p) = self.convert(query, self.heads, self.head_dim, self.mult)
        key     = self.convert(key, self.kv_heads, self.head_dim, self.mult)[0]
        value   = self.convert(value, self.kv_heads, self.head_dim)[0]
        if verbose:
            print(get_tensor_info(query, 'Q after Reshaping', verbose))
            print(get_tensor_info(key, 'K after Reshaping', verbose))

        # Apply Rotary Embeddings
        query = self.rotary_embedding(query, pos_idx if len(self.pixels) == 1 else None, verbose)
        key   = self.rotary_embedding(key, None)
        if verbose:
            print(get_tensor_info(query, 'Q after Rotary Embedding', verbose))
            print(get_tensor_info(key, 'K after Rotary Embedding', verbose))

        # Duplicate K and V for kv heads num per query head
        key   = self.repeat_kv(key)
        value = self.repeat_kv(value)
        if verbose:
            print(get_tensor_info(key, 'Duplicated K', verbose))
            print(get_tensor_info(value, 'Duplicated V', verbose))

        # Apply attention
        attention_scores, attention = self.attention(query, key.contiguous(), value.contiguous(), keys,
                                                     self.causal_mask and pretext is None, verbose)
        attention = attention.reshape(g, b, -1, self.dim_size)
        # out_view shape:  (genomes, batch_size, *pixels, channels)
        if verbose:
            print(get_tensor_info(attention, 'Attented Values', verbose))

        # Apply output projection
        try:
            # Apply weights
            tensor: Tensor = self.out_proj(self.revert(attention, b, c, p), keys=keys)
        except Exception as e:
            print(CM(f"batch_size={b}, channels={c}, pixels={p}, attention={attention.shape}"
                       f"\nquery={query.shape}, key={key.shape}, value={value.shape}\n", Fore.LIGHTRED_EX))
            raise e
        # Add the residue
        if self.skip_connection:
            tensor = tensor + residue
        if verbose:
            print(get_tensor_info(tensor, 'Output Projection', verbose))

        # Store attention for debugging
        if not get:
            return tensor
        else:
            return tensor, attention_scores


    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.dim_size}, pixels={self.pixels}, kernel_size={self.kernel_size}, " \
    #            f"bias={self.bias}{display('hd', self._heads)}{display('kv_hd', self._kv_heads)}" \
    #            f"{display('diffs', self.att_coeff_num-1 if self.att_coeff_num else None)}" \
    #            f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
    #            f"{display('pre_norm_ng', self.norm_groups)}{display('residual', self.skip_connection)}" \
    #            f"{display('as', self.auto_single)}"


class ConvCrossAttention(NeatModule):
    def __init__(
            self, max_pixels: list[int], cross_max_pixels: list[int], dim_size: int, kernel_size: int,
            heads: int = None, kv_heads: int = None, differential: int = None, layer_idx: int = None, causal_mask=False,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(ConvCrossAttention, self).__init__()
        self._heads = heads
        self._kv_heads = kv_heads
        if heads is None:
            heads = 1
        if dim_size % heads != 0:
            raise ValueError(f"Embedding dimensions must be a multiple of heads num")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError(f"Query heads num must be a multiple of number of Key-Value heads num")
        if isinstance(differential, bool):
            differential = int(differential)
        if not isinstance(max_pixels, (list, tuple)):
            raise ValueError(f"pixels parameter must be Iterable got '{type(max_pixels)}'")
        if not isinstance(cross_max_pixels, (list, tuple)):
            raise ValueError(f"pixels parameter must be Iterable got '{type(cross_max_pixels)}'")

        # OPTIONS
        self.padding = manage_params(options, 'padding', calc_padding(kernel_size, stride=1, dilation=1))
        self.padding_mode = manage_params(options, 'padding_mode', 'zeros')
        self.constant = manage_params(options, 'constant', 10000)
        self.epsilon = manage_params(options, 'epsilon', 1e-8)
        self.affine = manage_params(options, 'affine', True)
        self.norm_groups = manage_params(options, 'norm_groups', None)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], False)

        # ATTRIBUTES
        self.heads      = heads
        self.dim_size   = dim_size
        self.head_dim   = dim_size // heads
        self.kv_heads   = kv_heads
        self.q_kv_ratio = heads // kv_heads
        self.pixels     = max_pixels
        self.cross_max_pixels = cross_max_pixels
        self.differential = differential
        self.att_coeff_num = None if not differential else 1+differential
        self.att_coeff_indices = None if not differential else torch.arange(differential, device=device) + 1
        self.causal_mask = causal_mask
        self.pixels_total = max(math.prod(max_pixels), math.prod(cross_max_pixels))
        self.kernel_size = kernel_size
        self.bias = bias

        # torchneat
        self.pre_norm = GroupNorm(
            self.norm_groups, dim_size, self.epsilon, self.affine, False, device=device, dtype=dtype
        ) if self.norm_groups else None
        Convolution = get_convolution_class(max_pixels)
        CrossConvolution = get_convolution_class(cross_max_pixels)
        self.mult = 1+differential if differential else 1
        self.query_proj = Convolution(dim_size, heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.key_proj   = CrossConvolution(dim_size, kv_heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                           padding=self.padding, padding_mode=self.padding_mode,
                                           bias=bias, device=device, dtype=dtype)
        self.value_proj = CrossConvolution(dim_size, kv_heads * self.head_dim, kernel_size=kernel_size,
                                           padding=self.padding, padding_mode=self.padding_mode,
                                           bias=bias, device=device, dtype=dtype)
        self.out_proj   = Convolution(dim_size, dim_size, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.rotary_embedding = RoPE(self.pixels_total, dim_size * self.mult, heads, self.constant, device, dtype)
        self.softmax    = nn.Softmax(-1)
        self.head_norm  = RMSNorm(self.head_dim, self.epsilon, self.affine, device, dtype)
        self.diff_lambda = AttentionLambda(
            heads, self.head_dim, layer_idx, differential, 0.0, 0.1, True, self.epsilon, device, dtype
        ) if differential else None

        # STATES
        self.device = device
        self.dtype  = dtype

    def repeat_kv(self, tensor: Tensor):
        genomes, batch_size, seq_len, kv_heads, head_dim = tensor.shape
        if self.q_kv_ratio == 1:
            return tensor
        else:
            return tensor.unsqueeze(-2).expand(genomes, batch_size, seq_len, kv_heads, self.q_kv_ratio, head_dim).\
                reshape(batch_size, seq_len, kv_heads * self.q_kv_ratio, head_dim)

    @staticmethod
    def convert(image: Tensor, heads: int, head_dim: int, multiplier: int = 1):
        (g, b), c, p = image.shape[:2], image.shape[2] // multiplier, image.shape[3:]
        assert c == heads * head_dim
        # Return shape (genomes, batch_size, pixels, heads, head_dim)
        return image.view(g, b, c, -1).transpose(-1, -2).contiguous().view(g, b, -1, heads, head_dim*multiplier), (b, c, p)

    @staticmethod
    def revert(tensor: Tensor, batch_size: int, channels: int, pixels: list[int]):
        return tensor.transpose(-1, -2).view(-1, batch_size, channels, *pixels)

    def attention(self, query: Tensor, key: Tensor, value: Tensor, keys: Union[int, Iterable[int]], mask: bool, verbose: bool | int = False):
        if self.differential:
            query   = query.view(*query.shape[:-1], self.att_coeff_num, -1)
            key     = key.view(*key.shape[:-1], self.att_coeff_num, -1)
        # Get the attention score (energy)
        energy = torch.einsum("...qhd,...khd->...hqk" if not self.differential else "...qhcd,...khcd->...hcqk", [query, key])
        # queries shape: (batch_size, query_len, heads, head_dim)
        # key shape:     (batch_size, key_len, heads, head_dim)
        # energy shape:  (batch_size, heads, query_len, key_len)
        if verbose:
            print(get_tensor_info(energy, 'Energy', verbose))

        if mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask_ = torch.ones_like(energy, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            energy.masked_fill_(mask_, -torch.inf)
            if verbose and verbose >= 2 and not self.differential:
                print(get_tensor_info(mask_, 'Mask', verbose))
                print(get_tensor_info(energy, 'Masked Energy', verbose))

        # Get the softmax of the energy
        scores = self.softmax(energy / math.sqrt(self.head_dim))

        if self.differential:
            lambdas: Tensor = self.diff_lambda()
            if verbose and verbose >= 2:
                print(get_tensor_info(torch.round(lambdas, decimals=4), 'Lambdas', verbose+2))
                print(get_tensor_info(self.att_coeff_indices, 'Coeff Indices', verbose+2))
            scores = torch.select(scores, dim=-3, index=0) + torch.sum(
                torch.index_select(scores, dim=-3, index=self.att_coeff_indices) * lambdas, dim=-3
            ) # * lambdas)

            if mask:
                scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
                if verbose and verbose >= 2 and not self.differential:
                    print(get_tensor_info(scores, 'Differential Masked Energy', verbose))
            scores = self.softmax(scores)

        if verbose:
            print(get_tensor_info(torch.round(scores, decimals=4), 'Attention Score', verbose))

        # Get the weighted sum of the values and reshape to remove heads
        attention = self.head_norm(torch.einsum("...hqv,...vhd->...qhd", [scores, value]), keys=keys)
        # scores shape:    (batch_size, heads, query_len, value_len)
        # values shape:    (batch_size, value_len, heads, head_dim)
        # attention shape: (batch_size, query_len, heads, head_dim) then concat last 2 dim
        if self.differential:
            attention = attention * (1 - self.diff_lambda.biases(attention, keys, 2))

        return scores, attention

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None,
                context: Tensor = None, pos_idx: int = None, verbose: bool | int = False, get=False):
        if pos_idx is not None and len(self.pixels) == 1:
            pos_idx = self.pixels_total + pos_idx if pos_idx < 0 else pos_idx
            assert 0 < pos_idx < self.pixels_total
        if verbose:
            print(f'\n{CM("Executing Cross Attention", Fore.LIGHTBLUE_EX)}')

        residue = tensor
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
            if context is not None:
                context = self.pre_norm(context, keys=keys)

        # Convolve Q, K, V
        try:
            query: Tensor   = self.query_proj(tensor, keys=keys)
            key: Tensor     = self.key_proj(tensor if context is None else context, keys=keys)
            value: Tensor   = self.value_proj(tensor if context is None else context, keys=keys)
        except Exception as e:
            print("\n")
            print(CM(self.query_proj, Fore.LIGHTRED_EX))
            print(CM(self.key_proj, Fore.LIGHTRED_EX))
            print(CM(self.value_proj, Fore.LIGHTRED_EX))
            print(get_tensor_info(tensor, 'Tensor', 3))
            if context is not None:
                print(get_tensor_info(context, 'Context', 3))
            raise e
        if verbose:
            print(CM('Post Linearization =>'))
            print(get_tensor_info(query, 'Query', verbose, Fore.LIGHTRED_EX))
            print(get_tensor_info(key, 'Key', verbose, Fore.LIGHTGREEN_EX))
            print(get_tensor_info(value, 'Value', verbose, Fore.LIGHTBLUE_EX))

        batch_size, q_seq_len = query.shape[:2]

        # Reshape Q, K, V for each rep head
        query, (b, c, p) = self.convert(query, self.heads, self.head_dim, self.mult)
        key     = self.convert(key, self.kv_heads, self.head_dim, self.mult)[0]
        value   = self.convert(value, self.kv_heads, self.head_dim)[0]
        if verbose:
            print(get_tensor_info(query, 'Q after Reshaping', verbose))
            print(get_tensor_info(key, 'K after Reshaping', verbose))

        # Apply Rotary Embeddings
        query = self.rotary_embedding(query, pos_idx if len(self.pixels) == 1 else None, verbose)
        key   = self.rotary_embedding(key, None)
        if verbose:
            print(get_tensor_info(query, 'Q after Rotary Embedding', verbose))
            print(get_tensor_info(key, 'K after Rotary Embedding', verbose))

        # Duplicate K and V for kv heads num per query head
        key   = self.repeat_kv(key)
        value = self.repeat_kv(value)
        if verbose:
            print(get_tensor_info(key, 'Duplicated K', verbose))
            print(get_tensor_info(value, 'Duplicated V', verbose))

        attention_scores, attention = self.attention(
            query, key.contiguous(), value.contiguous(), keys,
            self.causal_mask and context is None and len(self.pixels) == 1, verbose
        )
        attention = attention.reshape(batch_size, -1, self.dim_size)
        # out_view shape:  (batch_size, query_len, embed_size)
        if verbose:
            print(get_tensor_info(attention, 'Attented Values', verbose))

        try:
            # Apply weights
            tensor: Tensor = self.out_proj(self.revert(attention, b, c, p), keys=keys)
        except Exception as e:
            print(CM(f"batch_size={b}, channels={c}, pixels={p}, attention={attention.shape}"
                       f"\nquery={query.shape}, key={key.shape}, value={value.shape}\n", Fore.LIGHTRED_EX))
            raise e
        if self.skip_connection:
            tensor = tensor + residue
        if verbose:
            print(get_tensor_info(tensor, 'Output Projection', verbose))

        if not get:
            return tensor
        else:
            return tensor, attention_scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dim_size}, pixels={self.pixels}, kernel_size={self.kernel_size}, " \
               f"bias={self.bias}{display('hd', self._heads)}{display('kvhd', self._kv_heads)}" \
               f"{display('diffs', self.att_coeff_num-1 if self.att_coeff_num else None)}" \
               f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
               f"{display('pre_norm_ng', self.norm_groups)}{display('residual', self.skip_connection)}"
