"""Convolutional transformer ("Conver") blocks and models."""

from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import device as DEVICE
from torch import dtype as DTYPE

from ...utils.helpers import manage_params
from ..layers import (
    Conv1d,
    GroupNorm,
    Ignore,
    Sequential,
    Transpose,
)
from ..module import NeatModule
from ..utils import get_tensor_info
from .conv_attention import ConvCrossAttention, ConvSelfAttention
from .encoding import SequenceEncoding
from .feedforward import ConvSwiGLU
from .residual import ResidualBlock


class ConverBlock(NeatModule):
    def __init__(self, max_pixels: tuple[int, ...],  dim_size: int, kernel_size: int, norm_groups: int,
                 heads: int = None, kv_heads: int = None, differential: int = None, layer_idx: int = None,
                 causal_mask=False, bias=False, device='cpu', dtype=torch.float32, **options):
        super(ConverBlock, self).__init__()

        # ATTRIBUTES
        self.max_pixels     = max_pixels
        self.cross_max_pixels = manage_params(options, ['cross_max_pixels', 'cross_pixels'], None)
        self.dim_size       = dim_size
        self.kernel_size    = kernel_size
        self.norm_groups    = norm_groups
        self.heads          = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        self.image_ndim     = len(max_pixels)

        # torchneat
        options['norm_groups'] = norm_groups
        options['skip_connection'] = True
        self.self_attention = ConvSelfAttention(
            max_pixels, dim_size, kernel_size, heads, kv_heads, differential, layer_idx, causal_mask,
            bias, device, dtype, **options
        )
        self.cross_attention = ConvCrossAttention(
            max_pixels, self.cross_max_pixels, dim_size, kernel_size, heads, kv_heads, differential, layer_idx,
            False, bias, device, dtype, **options
        ) if self.cross_max_pixels is not None else None
        options['image_ndim'] = 1
        self.feedforward = ConvSwiGLU(dim_size, kernel_size, bias, device, dtype, **options)
        # if self.image_ndim == 1:
        #     Dropout = nn.Dropout1d
        # elif self.image_ndim == 2:
        #     Dropout = nn.Dropout2d
        # elif self.image_ndim == 3:
        #     Dropout = nn.Dropout3d
        # else:
        #     raise ValueError(f"Unsupported num of image dimension '{self.image_ndim}'")
        self.dropout = nn.Dropout(manage_params(options, 'dropout', 0))

        # STATES
        self.self_attention_tensor: Tensor = None
        self.cross_attention_tensor: Tensor = None
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, pretext: Tensor = None, context: Tensor = None,
                pos_idx: int = None, verbose: bool | int = False, get=False):
        # Normalize then get the self attention
        attention = self.self_attention(
                tensor, keys=keys, pretext=pretext, pos_idx=pos_idx, verbose=verbose, get=get
        )
        if get:
            attention, self.self_attention_tensor = attention
        # Apply dropout
        tensor = self.dropout(attention)
        # Normalize then get the cross attention
        if self.cross_attention is not None and context is not None:
            attention = self.cross_attention(
                    tensor, keys=keys, context=context, pos_idx=pos_idx, verbose=verbose, get=get
            )
            if get:
                attention, self.cross_attention_tensor = attention
            # Apply dropout
            tensor = self.dropout(attention)
        # Pass through feed forward
        activation = self.feedforward(tensor, keys=keys)
        # Dropout
        tensor = self.dropout(activation)
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Output', verbose))

        return tensor


class ConverBase(NeatModule):
    def __init__(self, max_pixels: tuple[int, ...],  dim_size: int, kernel_size: int, norm_groups: int, layers: int,
                 heads: int = None, kv_heads: int = None, differential: int = None,
                 causal_mask=False, bias=False, device='cpu', dtype=torch.float32, **options):
        super(ConverBase, self).__init__()

        # ATTRIBUTES
        self.max_pixels     = max_pixels
        self.dim_size       = dim_size
        self.kernel_size    = kernel_size
        self.norm_groups    = norm_groups
        self.layer_num      = layers
        self.heads          = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        self.image_ndim     = len(max_pixels)
        self.dropout        = manage_params(options, 'dropout', 0)
        self.auto_single    = manage_params(options, 'auto_single', False)

        # torchneat
        options['auto_single'] = False
        self.positional_encoding = SequenceEncoding(max_pixels, dim_size, True, device, dtype)
        self.layers: list[ConverBlock] = nn.ModuleList()
        for layer_idx in range(layers):
            if layer_idx == layers-1:
                options['auto_single'] = self.auto_single
                kernel_size = 1
                options['padding'] = -1
            self.layers.append(
                ConverBlock(
                    max_pixels, dim_size, kernel_size,
                    norm_groups, heads, kv_heads, differential, layer_idx,
                    causal_mask, bias, device, dtype, **options
                )
            )

        # STATE
        self.device     = device
        self.dtype      = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, pretext: Tensor = None, context: Tensor = None, pos_idx: int = None,
                verbose: bool | int = False, get=False, single=False):
        # Pass through the encoder blocks
        for layer_idx, layer in enumerate(self.layers):
            # Single mode is only when on final layer, pixels span 1 dimension and tensor has 3 dimensions only
            single_fetch = (single or self.auto_single) and layer_idx == len(self.layers) - 1 and len(self.max_pixels) == 1 and tensor.ndim == 4
            # shape (batch_size, channels, *pixels)
            tensor = self.positional_encoding(tensor, keys=keys, offset=None, verbose=verbose)
            if single_fetch:
                # Using last token index in sequence to get the next token
                set_pretext = torch.select(tensor, -1, -1).unsqueeze(-1)
            else:
                set_pretext = pretext
                if set_pretext is not None:
                    set_pretext = self.positional_encoding(tensor, keys=keys, offset=None, verbose=verbose)
            tensor = layer(tensor, keys=keys, pretext=set_pretext, context=context, pos_idx=pos_idx,
                           verbose=verbose, get=get)

        return tensor

    def get_attention(self):
        return [(layer.self_attention_tensor, layer.cross_attention_tensor) for layer in self.layers]


    # @property
    # def genomes_total(self):
    #     return self.decode.genome_num
    #
    # def forward(self, tensor: Tensor, pos_idx: int = None, keys: Union[int, Iterable[int]] = None,
    #             verbose: bool | int = False, get=False, single=False):
    #     if pos_idx is not None:
    #         tensor = tensor[:, :pos_idx+1]
    #     if verbose:
    #         print(f"\nTransformer Input =>\n{tensor}\n\tdim = {tensor.shape}")
    #
    #     tensor = self.embedder(tensor, keys=keys, verbose=verbose)
    #     if self.primary_activation is not None:
    #         tensor = self.primary_activation(tensor)
    #     tensor = self.dropout(self.encoder(tensor, keys=keys))
    #     tensor = self.transformer(tensor, keys=keys, verbose=verbose, get=get, single=single)
    #     tensor = self.decode(self.dec_norm(tensor, keys=keys), keys=keys)
    #     if self.secondary_activation is not None:
    #         tensor = self.secondary_activation(tensor)
    #     if verbose:
    #         print(f"\nTransformer Output =>\n{tensor}\n\tdim = {tensor.shape}")
    #     if self.distribution == 'discrete':
    #         tensor = torch.argmax(tensor, -1)
    #     return tensor
    #
    # def get_attention(self):
    #     a, v = [], []
    #     for ai, vi in self.transformer.get_attention():
    #         a.append(ai)
    #         v.append(vi)
    #     return a, v
    #
    # def infer(self, tensor: Tensor, pos_idx: int = None, keys: Union[int, Iterable[int]] = None, verbose: bool | int = False):
    #     pos_idx = self.max_seq_len + pos_idx if pos_idx is not None and pos_idx < 0 else pos_idx
    #     if pos_idx is None:
    #         pos_idx = self.max_seq_len-1
    #     sequence_dim = -2 if self.distribution == 'discrete' else -1
    #     tokens_current = min(tensor.shape[sequence_dim], pos_idx+1)
    #     tensor = tensor[..., :tokens_current+1, :]
    #     for idx in range(tokens_current):
    #         current_idx = tokens_current+idx
    #         token = self.forward(tensor, current_idx, keys, verbose)
    #         tensor = torch.cat((tensor, token), dim=sequence_dim)
    #     return tensor


class Conver(NeatModule):
    def __init__(
            self, inputs: int, outputs: int, max_seq_len: int, dim_size: int, kernel_size: int, layers: int,
            norm_groups: int, channels: Union[int, list[int]] = None,
            heads: int = None, kv_heads: int = None, differential=True,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Conver, self).__init__()
        if channels is None:
            channels = dim_size
        self.enc_layers         = manage_params(options, ['enc_layers', 'encoder_layers'], 0)
        self.dec_layers         = manage_params(options, ['dec_layers', 'decoder_layers'], 0) + 1
        self.distribution       = manage_params(options, 'distribution', 'normal')
        self.causal_mask        = manage_params(options, 'causal_mask', True)
        self.epsilon            = manage_params(options, 'epsilon', 1e-9)
        self.affine             = manage_params(options, 'affine', True)
        self.probabilistic      = manage_params(options, ['prob', 'probabilistic'], False) and self.distribution != 'discrete'
        self.lower_clip         = manage_params(options, 'lower_clip', -20)
        self.upper_clip         = manage_params(options, 'upper_clip', 20)
        self.feedback           = manage_params(options, 'feedback', False)
        self.init_kernel_size   = manage_params(options, 'init_kernel_size', kernel_size)
        self.trans_kernel_size  = manage_params(options, 'trans_kernel_size', 1)

        # ATTRIBUTES
        self.inputs = inputs if not self.feedback else inputs - self.feedback
        self.outputs = outputs

        # BUILD
        self.pri_actv = self.activation = manage_params(options, ['activation', 'actv', 'pri_actv'], nn.SiLU())
        self.sec_actv = manage_params(options, 'sec_actv', None)
        options['image_ndim'] = 1
        self.feedback_gain = Sequential(
            Transpose(),
             Conv1d(self.feedback, dim_size, 1, stride=1, padding=-1,
                 padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                 bias=True, device=device, dtype=dtype),
            GroupNorm(dim_size, dim_size, self.epsilon, self.affine, False, device, dtype),
             Conv1d(dim_size, self.feedback, 1, stride=1, padding=-1,
                 padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                 bias=bias, device=device, dtype=dtype),
            nn.Tanh(),
            Transpose(),
        ) if self.feedback and manage_params(options, 'feedback_gain', False) else None
        self.feedback_dropout = Ignore(manage_params(options, ['feedback_drop', 'feedback_dropout'], 0))
        self.encoder = Sequential(
            Transpose(),
            Conv1d(inputs, dim_size, self.init_kernel_size, 1, -1,
                   padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                   bias=True, device=device, dtype=dtype),
            *[
                ResidualBlock(dim_size, dim_size, kernel_size, norm_groups, bias, device, dtype, **options)
                for _ in range(self.enc_layers)
            ],
        )
        self.transformer = ConverBase(
            (max_seq_len,), dim_size, self.trans_kernel_size, norm_groups, layers, heads, kv_heads, differential,
            self.causal_mask, False, device, dtype, **options
        )
        decoder = []
        for layer_idx in range(self.dec_layers):
            if layer_idx != self.dec_layers - 1:
                decoder.append(
                    ResidualBlock(dim_size, dim_size, 1, norm_groups, bias, device, dtype, **options)
                )
            else:
                decoder.extend([
                    GroupNorm(norm_groups, dim_size, self.epsilon, self.affine, False, device, dtype),
                    self.activation,
                    Conv1d(
                        dim_size, outputs, 1, 1,
                        padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                        bias=bias, device=device, dtype=dtype
                    ),

                ])
                if self.sec_actv is not None:
                    decoder.append(self.sec_actv)
                decoder.append(Transpose(-1, -2))
        self.decoder = Sequential(*decoder)
        self.selector = torch.arange(max_seq_len, device=device, dtype=torch.int)

        # STATE
        self.device = device
        self.dtype  = dtype
        self.train()
        self.requires_grad_(False)

    def handle_feedback(self, tensor: Tensor, keys: Union[int, list[int]] = None):
        if self.feedback:
            tensor, feedback = torch.split(tensor, self.inputs, -1)
            try:
                assert feedback.shape[-1] == self.feedback
            except Exception as e:
                print(f"feedback = {feedback.shape}, feedback_dims = {self.feedback}")
                raise e
            feedback = self.feedback_dropout(feedback)
            if self.feedback_gain is not None:
                feedback = feedback * self.feedback_gain(feedback, keys=keys)
            tensor = torch.cat((tensor, feedback), -1)
        return tensor

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, pos_idx: int = None,
                verbose: bool | int = False, get=False, single=False, context: Tensor = None, noise: Tensor = None, logits=False):
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Input', verbose))
        squeeze = tensor.ndim == 3
        if squeeze:
            tensor = tensor.unsqueeze(0)
        if pos_idx is not None:
            position = min(tensor.shape[-2], pos_idx+1)
            tensor = tensor.index_select(-2, self.selector[:position])

        # Expected input shape (genomes, batch_size, seq_len, features), therefore transpose it
        tensor = self.encoder(self.handle_feedback(tensor, keys=keys), keys=keys) # , verbose=verbose)
        tensor = self.transformer(tensor, keys=keys, context=context, pos_idx=pos_idx, verbose=verbose, get=get, single=single)
        # Expected transform shape (genomes, batch_size, dim_size, seq_len), therefore transpose it
        tensor = self.decoder(tensor, keys=keys)

        if squeeze:
            tensor = tensor.squeeze(0)
        if self.probabilistic:
            # Get the distribution
            mean, log_variance = torch.chunk(tensor, 2, dim=-1)
            log_variance = torch.clamp(log_variance, self.lower_clip, self.upper_clip)
            std = log_variance.exp().sqrt()
            if noise is None:
                noise = torch.randn_like(std, device=std.device, dtype=std.dtype)
            tensor = mean + std * noise
        if self.distribution == 'discrete' and not logits:
            tensor = torch.argmax(tensor, -1)
        else:
            if self.sec_actv is not None:
                tensor = self.sec_actv(tensor)
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Output', verbose))

        return tensor

    def get_attention(self):
        a, v = [], []
        for ai, vi in self.transformer.get_attention():
            a.append(ai)
            v.append(vi)
        return a, v

    def infer(self, inputs: Tensor, keys: Union[int, list[int]] = None, pos_idx: int = None, get=False, single=True, verbose: bool | int = False):
        with torch.no_grad():
            if single:
                pos_idx = -1
            if pos_idx is not None:
                seq_len = inputs.shape[-2]
                pos_idx = seq_len + pos_idx if pos_idx < 0 else pos_idx
                assert 0 < pos_idx < seq_len
            outputs = self.forward(inputs, keys, pos_idx, verbose, get, single)
            return outputs
