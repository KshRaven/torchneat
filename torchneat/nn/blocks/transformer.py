"""Transformer blocks, the stacked TransformerBase and the Transformer model."""

from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import device as DEVICE
from torch import dtype as DTYPE

from ...utils.helpers import manage_params
from ..module import NeatModule
from ..utils import get_tensor_info
from .attention import Attention
from .feedforward import SwiGLU

# --------------------------------------------- #
# Transformers                                  #
# --------------------------------------------- #


class TransformerBlock(NeatModule):
    def __init__(
            self, max_seq_len: int, dim_size: int, heads: int = None, kv_heads: int = None,
            differential=False, layer_idx: int = None, causal_mask=True,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(TransformerBlock, self).__init__()

        # ATTRIBUTES
        self.max_seq_len = max_seq_len
        self.dim_size = dim_size
        self.heads      = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        fwd_func = manage_params(options, ['fwd_func', 'ff'], None)

        # BUILD
        # options['residual'] = True
        self.self_attention = Attention(
            max_seq_len, dim_size, heads, kv_heads, differential, layer_idx, causal_mask,
            bias, device, dtype, **options
        )
        self.feedforward = SwiGLU(dim_size, bias, device, dtype, **options) if fwd_func is None else fwd_func
        self.dropout = nn.Dropout(manage_params(options, 'dropout', 0))
        # TODO: Create cross attention module

        # STATES
        self.self_attention_tensor: Tensor = None
        # self.cross_attention_tensor: Tensor = None
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None,
                prompt: Tensor = None, context: Tensor = None,
                pos_idx: int = None, verbose: bool | int = False, get=False):
        # Normalize then get the self attention
        attention = self.self_attention(
                tensor, keys=keys, prompt=prompt, pos_idx=pos_idx, verbose=verbose, get=get
        )
        if get:
            attention, self.self_attention_tensor = attention
        # Apply dropout
        tensor = self.dropout(attention)
        # Normalize then get the cross attention
        # if self.cross_attention is not None and context is not None:
        #     attention = self.cross_attention(
        #             tensor, keys=keys, context=context, pos_idx=pos_idx, verbose=verbose, get=get
        #     )
        #     if get:
        #         attention, self.cross_attention_tensor = attention
        #     # Apply dropout
        #     tensor = self.dropout(attention)
        # Pass through feed forward
        activation = self.feedforward(tensor, keys=keys)
        # Dropout
        tensor = self.dropout(activation)
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Output', verbose))

        return tensor


class TransformerBase(NeatModule):
    def __init__(
            self, max_seq_len: int, dim_size: int, layers: int, heads: int = None, kv_heads: int = None,
            differential: int = None, causal_mask=True,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(TransformerBase, self).__init__()

        # ATTRIBUTES
        self.max_seq_len    = max_seq_len
        self.dim_size       = dim_size
        self.layer_num      = layers
        self.heads          = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        self.dropout        = manage_params(options, 'dropout', 0)
        self.auto_single    = manage_params(options, 'auto_single', False)
        # share_kv: every layer attends to K/V projected from the INPUT tokens (a shared, per-token memory) and only the
        # newest token's state flows through the layers (it re-reads the memory once per layer). A cached K/V is then a
        # pure function of one token, so deque + RoPE re-rotation is EXACT: the cached rollout and the full-sequence
        # pass give the same output by construction, with no repositioning to learn and no policy-accuracy term needed.
        self.share_kv       = bool(manage_params(options, ['share_kv', 'memory_kv'], False))

        # BUILD
        options['auto_single'] = False # NOTE: Disables the auto-single directly used in the Attention module
        if self.share_kv:
            options['kv_exact'] = True
        self.layers: list[TransformerBlock] = nn.ModuleList()
        for layer_idx in range(layers):
            # if layer_idx == layers-1:
            #     if self.auto_single:
            #         options['auto_single'] = self.auto_single
            self.layers.append(
                TransformerBlock(
                    max_seq_len, dim_size, heads, kv_heads, differential, layer_idx,
                    causal_mask, bias, device, dtype, **options
                )
            )

        # STATE
        self.device     = device
        self.dtype      = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None,
                prompt: Tensor = None, context: Tensor = None, pos_idx: int = None,
                verbose: bool | int = False, get=False, single=False):
        ndim = tensor.ndim
        rem_batch = rem_seq = False
        if ndim == 2: # No batch and sequence dim
            rem_batch = rem_seq = True
            tensor = tensor.unsqueeze(1).unsqueeze(2)
        elif ndim == 3: # No batch or sequence dim (assume sequence is missing)
            rem_seq = True
            tensor = tensor.unsqueeze(2)
        if self.share_kv:
            # memory = the (embedded) input tokens; state = newest token, (genomes, batch, 1, dim). `prompt` is ignored.
            memory = tensor
            tensor = torch.select(memory, -2, -1).unsqueeze(-2)
            for layer in self.layers:
                tensor = layer(memory, keys=keys, prompt=tensor, context=context,
                               pos_idx=memory.shape[-2] - 1, verbose=verbose, get=get)
            if rem_seq:
                tensor = tensor.squeeze(2)
            if rem_batch:
                tensor = tensor.squeeze(1)
            return tensor
        # Pass through the encoder blocks
        for layer_idx, layer in enumerate(self.layers):
            # Single mode is only when on final layer and instance (seq_len=1) output is required
            single_fetch = (
                ((single or self.auto_single) and layer_idx == len(self.layers) - 1)
                or
                (self.auto_single and layer.self_attention.forced_cache)
            )
            # shape (genomes, batch_size, seq_len, dim_size)
            if single_fetch:
                assert tensor.ndim >= 4, f"expected (genomes, *extra_dims, seq_len, dim), got {tuple(tensor.shape)}"
                # Using last token index in sequence to get the next token
                # shape (genomes, batch_size, seq_len, dim_size)
                set_prompt = torch.select(tensor, -2, -1).unsqueeze(-2)
            else:
                set_prompt = prompt
            tensor = layer(tensor, keys=keys, prompt=set_prompt, context=context,
                           pos_idx=pos_idx if not single_fetch else tensor.shape[-2]-1, # deque mode corrects the position regardless
                           verbose=verbose, get=get)
        if rem_seq:
            tensor = tensor.squeeze(2)
        if rem_batch:
            tensor = tensor.squeeze(1)
        return tensor

    def get_attention(self):
        return [(
            layer.self_attention_tensor, # layer.cross_attention_tensor
        ) for layer in self.layers]

    def force_cache(self, enable: bool = True):
        for layer in self.layers:
            layer.self_attention.force_cache(enable)
    
    def empty_cache(self):
        for layer in self.layers:
            layer.self_attention.empty_cache()


class Transformer(NeatModule):
    def __init__(
            self, inputs: int, outputs: int, embed_size: int, max_seq_len: int, layers: int,
            heads: int = None, kv_heads: int = None, differential=True, dropout: int = None,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Transformer, self).__init__()
        # self.distribution       = manage_params(options, 'distribution', 'normal')
        # self.fwd_exp            = manage_params(options, 'fwd_exp', None)
        # self.epsilon            = manage_params(options, 'epsilon', 1e-8)
        # self.constant           = manage_params(options, 'constant', 10000)
        # self.affine             = manage_params(options, 'affine', True)
        # self.causal_mask        = manage_params(options, 'causal_mask', True)
        # self.primary_activation = manage_params(options, 'pri_actv', nn.SiLU())
        # self.secondary_activation = manage_params(options, 'sec_actv', None)
        #
        # # torchneat
        # self.embedder    = BufferEmbedding(inputs, embed_size, bias, device, dtype)
        # self.encoder     = BufferEncoding(max_seq_len, embed_size, bias, device, dtype)
        # self.transformer = TransformerBase(
        #     max_seq_len, embed_size, layers, heads, kv_heads, self.fwd_exp, differential,
        #     self.constant, self.epsilon, self.affine, self.causal_mask, dropout, bias, device, dtype
        # )
        # self.dec_norm   = RMSNorm(embed_size, self.epsilon, self.affine, device, dtype)
        # output_dim      = outputs if self.distribution != 'discrete' else 2 ** outputs
        # self.decode     = Linear(embed_size, output_dim, bias, device, dtype)
        # if dropout is None:
        #     dropout = 0
        # self.dropout    = nn.Dropout(dropout)
        #
        # # STATE
        # self.device = device
        # self.dtype  = dtype
        # self.eval()
        #
        # # ATTRIBUTES
        # self.max_seq_len = max_seq_len
        raise NotImplementedError()
