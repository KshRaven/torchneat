"""Token/buffer/sequence embeddings and positional encodings (sinusoidal, RoPE, repositioning)."""

from __future__ import annotations

import math
from typing import Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import device as DEVICE
from torch import dtype as DTYPE

from ...utils.console import Fore
from ...utils.helpers import manage_params
from ..layers import Conv1d, Conv2d, Conv3d, Embedding, Linear, Polynomial, RMSNorm
from ..module import NeatModule
from ..utils import get_tensor_info


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, inp_features: int = None, depth=1, fwd_exp=4,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(TokenEmbedding, self).__init__()
        # torchneat
        self.embedding  = nn.Embedding(vocab_size, embed_size, device=device, dtype=dtype)
        self.reduction  = None if inp_features is None else nn.Sequential(
            nn.Linear(inp_features, embed_size*fwd_exp, False, device, dtype),
            *[nn.Linear(embed_size*fwd_exp, embed_size*fwd_exp, False, device, dtype) for _ in range(depth)],
            nn.Linear(embed_size*fwd_exp, 1, False, device, dtype)
        )

        # ATTRIBUTES
        self.embed_size = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, verbose: bool | int = False):
        # # Check if input is a tensor
        # if not isinstance(tensor, Tensor):
        #     raise InvalidValueError(f"Input Sequence entered is not a Tensor.")
        # Expands input to embedding space; [records, sequence] to [records, sequence, embed_size]
        # or (batch_size, seq_len, features) to (batch_size, seq_len, features, embed_size)
        tensor = self.embedding(tensor)
        if self.reduction is not None:
            # Convert (batch_size, seq_len, features, embed_size) to (batch_size, seq_len, embed_size)
            tensor = self.reduction(torch.transpose(tensor, -1, -2)).squeeze(-1)
        if verbose:
            print(verbose)
            print(f"\nEmbedded Data =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class SinusoidalEncoding(nn.Module):
    def __init__(self, seq_length: int, embed_size: int, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(SinusoidalEncoding, self).__init__()
        # torchneat - [records, sequence, embed_size], EMBEDDING - [sequence, embed_size]
        self.positional_encoding = self._generate_encoding(seq_length, embed_size).\
            to(device=device, dtype=dtype)
        # EMBEDDING - [1, sequence, embed_size]
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

        # ATTRIBUTES
        self.max_seq_length = seq_length
        self.embed_size     = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, verbose: bool | int = False):
        # Get the dimension shape of the input
        records, seq_length, embed_size = tensor.size()
        # Expanding positional encoding to shape of input
        positional_encoding = self.positional_encoding.expand(records, -1, embed_size)
        # Add encoding to tensor
        tensor = tensor + positional_encoding[:, :seq_length]
        if verbose:
            print(f"\nPositional Encoding =>\n{positional_encoding}\n\tdim = {positional_encoding.shape}")
            print(f"\nEncoded Sequences =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor

    @staticmethod
    def _generate_encoding(max_seq_length: int, embed_size: int, constant=10000.0):
        encoding = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.multiply(
                torch.arange(0, embed_size, 2),
                (-torch.log(torch.tensor(constant)) / embed_size)
            )
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding


class BufferEmbedding(NeatModule):
    def __init__(self, features: int, embed_size: int, bias=False, type='continuous',
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEmbedding, self).__init__()
        # BUILD
        if type == 'continuous':
            self.embedding = Linear(features, embed_size, bias, device, dtype)
        elif type == 'discrete':
            raise NotImplementedError(f"nn.Embedding not yet implemented for NEAT modules")
        else:
            raise NotImplementedError(f"Unsupported embedding type: '{type}'")

        # ATTRIBUTES
        self.input_dim  = features
        self.embed_size = embed_size
        self.type       = type

        # STATES
        self.device = device
        self.dtype  = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, verbose: bool | int = False):
        # Expand input to embedding space; [batch_size, sequence, *features] to [batch_size, sequence, embed_size]
        # print(f"forward={self.embedding.weights.data.shape, tensor.shape}")
        tensor = self.embedding(tensor, keys=keys)
        if verbose:
            print(get_tensor_info(tensor, "Embedded Tensor", verbose))

        return tensor


class BufferEncoding(NeatModule):
    def __init__(self, max_seq_len: int, embed_size: int, bias=True, type='discrete',
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEncoding, self).__init__()
        # BUILD
        assert max_seq_len >= 1
        if type == 'continuous':
            self.positions = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(1).unsqueeze(-1)
            if max_seq_len > 1:
                self.positions /= (max_seq_len - 1)
        elif type == 'discrete':
            self.positions = torch.arange(max_seq_len, device=device, dtype=torch.int32).unsqueeze(0).unsqueeze(1)
        else:
            raise NotImplementedError(f"Unsupported encoding type: '{type}'")
        # shape(genomes, batch_size, seq_len, features)
        self.selector       = torch.arange(max_seq_len, device=device, dtype=torch.long)
        if type == 'continuous':
            self.encoding = Linear(1, embed_size, bias, device, dtype)
        elif type == 'discrete':
            raise NotImplementedError(f"nn.Embedding not yet implemented for NEAT modules")
            self.encoding = Embedding(max_seq_len, embed_size, device=device, dtype=dtype)
        # self.activation     = nn.SiLU()

        # ATTRIBUTES
        self.max_seq_len    = max_seq_len
        self.embed_size     = embed_size
        self.encoding_memory: Tensor | None = None

        # STATES
        self.device  = device
        self.dtype   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, offset: tuple[int, int] = None, verbose: bool | int = False, hold=False):
        if offset is None:
            offset = 0
        if verbose and verbose >= 2:
            print(get_tensor_info(tensor, "Unencoded tensor", verbose))
        # tensor = (genomes, batch_size, seq_len, embed_size)
        genomes, _, seq_len, _ = tensor.shape
        # Expanding positional encoding to shape of input
        positions = self.positions.expand(genomes, *self.positions.shape[1:])
        positions = torch.index_select(positions, -2, self.selector[offset:offset+seq_len])
        if verbose and verbose >= 2:
            print(get_tensor_info(positions, "Positions", verbose))
        positional_encoding: Tensor = self.encoding(positions, keys=keys)
        # positional_encoding = self.activation(positional_encoding)

        # Add encoding to tensor
        tensor = tensor + positional_encoding
        if verbose:
            print(get_tensor_info(positional_encoding, "Positional Encoding", verbose))
            print(get_tensor_info(tensor, "Encoded tensor", verbose))

        return tensor


class SequenceEncoding(NeatModule):
    def __init__(self, max_pixels: tuple[int, ...], embed_size: int, bias=True, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(SequenceEncoding, self).__init__()
        self.pixels_total  = int(math.prod(max_pixels))
        self.positions      = torch.arange(self.pixels_total, device=device, dtype=dtype).reshape(max_pixels).\
                                  unsqueeze(0).unsqueeze(1).unsqueeze(2)
        if self.pixels_total > 1:
            self.positions /= (self.pixels_total - 1)
        # BUILD
        if len(max_pixels) == 1:
            Convolution = Conv1d
        elif len(max_pixels) == 2:
            Convolution = Conv2d
        elif len(max_pixels) == 3:
            Convolution = Conv3d
        else:
            raise ValueError(f"Unsupported num of image dimension '{len(max_pixels)}'")
        # shape(genomes, batch_size, channels, *pixels)
        self.encoding       = Convolution(1, embed_size, 1, bias=bias, device=device, dtype=dtype)
        # self.activation     = nn.SiLU()

        # ATTRIBUTES
        self.max_pixels     = max_pixels
        self.ndim           = len(max_pixels)
        self.embed_size     = embed_size
        self.selector       = torch.arange(int(max(self.max_pixels)), device=device, dtype=torch.int32)
        self.encoding_memory: Tensor|None = None

        # STATES
        self.device  = device
        self.dtype   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, offset: tuple[int, int] = None, verbose: bool | int = False, hold=False):
        if offset is None:
            offset = [0 for _ in range(self.ndim)]
        if verbose and verbose >= 2:
            print(get_tensor_info(tensor, "Unencoded tensor", verbose))
        # tensor = (genomes, batch_size, channels, *pixels)
        (genomes, _, channels), pixels = tensor.shape[:3], tensor.shape[3:]
        assert len(pixels) == len(offset) == self.ndim

        # Expanding positional encoding to shape of input
        positions = self.positions.expand(genomes, *self.positions.shape[1:])
        for dim, off in zip(pixels[::-1], offset[::-1]):
            positions = torch.index_select(positions, -1, self.selector[off:off+dim])
        if verbose and verbose >= 2:
            print(get_tensor_info(positions, "Positions", verbose))
        positional_encoding: Tensor = self.encoding(positions, keys=keys)
        # positional_encoding = self.activation(positional_encoding)

        # Add encoding to tensor
        tensor = tensor + positional_encoding
        if verbose:
            print(get_tensor_info(positional_encoding, "Positional Encoding", verbose))
            print(get_tensor_info(tensor, "Encoded tensor", verbose))

        return tensor


# --------------------------------------------- #
# Attention                                     #
# --------------------------------------------- #

class RoPE(NeatModule):
    """
    Rotary Position Embedding (RoPE).

    Operates on tensors shaped (genomes, batch_size, seq_len, heads, head_dim).
    """

    def __init__(self, max_seq_len: int, embed_size: int, heads: int,
                 constant: int = 10_000,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embed_size  = embed_size
        self.head_dim    = embed_size // heads
        self.heads       = heads
        self.constant    = constant
        self.device      = device
        self.dtype       = dtype

        # Use float32 for complex arithmetic when model dtype is lower precision
        self.conv_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32

        # embeddings (seq_len, head_dim / 2)
        self.complex_frequencies = self._generate_encoding(max_seq_len, embed_size // heads, constant, 0)
        # embeddings (genomes=1, batch_size=1, seq_len, heads=1, head_dim / 2)
        self.complex_frequencies = self.complex_frequencies.to(device=device).unsqueeze(0).unsqueeze(1).unsqueeze(-2)
        self.select = torch.arange(max_seq_len, device=device, dtype=torch.long) # Sequence selector

    def extra_repr(self) -> str:
        return (
            f"max_seq_len={self.max_seq_len}, embed_size={self.embed_size}, "
            f"heads={self.heads}, constant={self.constant}"
        )

    # @staticmethod
    def _generate_encoding(self, seq_length: int, head_dim: int, constant: float = 10000.0, verbose: bool | int = False):
        # Dimensions of embedding must be even
        assert head_dim % 2 == 0, "head_dim must be divisible by 2"
        # Get theta where theta_i = 10000 ^ (-2 * (i-1) / embedding) for i = [1, 2, ..., dim / 2]; [head_dim / 2]
        theta    = 1.0 / torch.pow(constant, torch.arange(0, head_dim, 2).float() / head_dim)
        # Get positions as m; [sequence]
        positions = torch.arange(seq_length)
        # Multiply theta by each position; [sequence] outer* [head_dim / 2] -> [sequence, head_dim / 2]
        angles   = torch.outer(positions, theta).to(dtype=self.conv_dtype)
        # We compute complex number in polar form c = R * exp(i * m * theta); [sequence, head_dim / 2]
        cf = torch.polar(torch.ones_like(angles), angles)  # (seq_len, head_dim/2)
        if verbose:
            print(f"\nTheta =>\n{theta}\n\tdim = {theta.shape}")
            print(f"\nPositions =>\n{positions}\n\tdim = {positions.shape}")
            print(f"\nAngles =>\n{angles}\n\tdim = {angles.shape}")
            print(f"\nComplex Frequencies init =>\n{cf}\n\tdim = {cf.shape}")
        return cf

    def forward(self, tensor: Tensor, pos_idx: int = 0, verbose: bool | int = False): # encode
        """
        Args:
            tensor:  (batch_size, seq_len, heads, head_dim)
            pos_idx: Starting position index.
        Returns:
            Same shape as input.
        """
        if tensor.numel() == 0:
            return tensor
        # NOTE: For uncached sequence, might want to start at 0.
        #       For instance use its absolute position in that sequence (be wary when the sequence is cached)
        # shape(genomes, batch_size, seq_len, heads, head_dim)
        seq_len = tensor.shape[-3]
        assert 0 < (pos_idx + seq_len) <= self.max_seq_len
        # [genomes, batch_size, sequence, heads, head_dim] -> [genomes, batch_size, sequence, heads, head_dim/2, 2]
        complex_tensor = torch.view_as_complex(tensor.view(*tensor.shape[:-1], -1, 2).to(self.conv_dtype))
        # [genomes, batch_size, sequence, heads, head_dim/2] * [1, 1, sequence, 1, head_dim/2] = [genomes, batch_size, sequence, heads, head_dim/2]
        complex_frequencies = torch.index_select(self.complex_frequencies, -3, self.select[pos_idx:pos_idx+seq_len])
        # for _ in range(max(0, complex_tensor.ndim - complex_frequencies.ndim)):
        #     complex_frequencies = complex_frequencies.unsqueeze(0)
        try:
            rotated_tensor = complex_tensor * complex_frequencies
        except Exception as e:
            print(get_tensor_info(self.complex_frequencies, 'Complex Frequencies', verbose))
            print(get_tensor_info(complex_frequencies, 'Selected Complex Frequencies', verbose))
            print(get_tensor_info(complex_tensor, 'Complex Tensor', verbose))
            raise e
        # [genomes, batch_size, sequence, heads, head_dim / 2] -> [genomes, batch_size, sequence, heads, head_dim / 2, 2]
        split_tensor = torch.view_as_real(rotated_tensor).to(self.dtype)
        # [records, sequence, heads, head_dim / 2, 2] -> [records, sequence, heads, head_dim]
        # [records, sequence, heads, head_dim] -> [records, sequence, embed_size]
        tensor = split_tensor.reshape(*tensor.shape).type_as(tensor)
        if verbose: #  and verbose >= 3:
            print(get_tensor_info(complex_tensor, 'Complex Tensor', verbose))
            print(get_tensor_info(complex_frequencies, 'Complex Frequencies', verbose))
            print(get_tensor_info(rotated_tensor, 'Rotated Tensor', verbose))
            print(get_tensor_info(split_tensor, 'Split Tensor', verbose))
            print(get_tensor_info(tensor, 'Encoded Tensor', verbose))

        return tensor

    def decode(self, tensor: Tensor, pos_idx: int = 0, verbose: bool | int = False):
        """Invert a RoPE-encoded tensor (multiply by complex conjugate)."""
        if tensor.numel() == 0:
            return tensor
        seq_len = tensor.shape[-3]

        # [..., head_dim] -> complex [..., head_dim/2]
        complex_tensor = torch.view_as_complex(
            tensor.view(*tensor.shape[:-1], -1, 2).to(self.conv_dtype)
        )

        complex_frequencies = torch.index_select(self.complex_frequencies, -3, self.select[pos_idx:pos_idx+seq_len])
        # for _ in range(max(0, complex_tensor.ndim - complex_frequencies.ndim)):
        #     complex_frequencies = complex_frequencies.unsqueeze(0)

        # Inverse rotation
        inverse_frequencies = torch.conj(complex_frequencies)

        try:
            restored_tensor = complex_tensor * inverse_frequencies
        except RuntimeError as e:
            print(f"")
            print(get_tensor_info(tensor, 'tensor', 1, Fore.LIGHTRED_EX))
            print(f"position_index = {pos_idx}")
            print(get_tensor_info(complex_tensor, 'complex_tensor', 1, Fore.LIGHTRED_EX))
            print(get_tensor_info(complex_frequencies, 'complex_frequencies', 1, Fore.LIGHTRED_EX))
            print(get_tensor_info(inverse_frequencies, 'inverse_frequencies', 1, Fore.LIGHTRED_EX))
            raise e

        split_tensor = torch.view_as_real(
            restored_tensor
        ).to(self.dtype)

        tensor = split_tensor.reshape(*tensor.shape).type_as(tensor)

        if verbose and verbose >= 3:
            print(get_tensor_info(complex_tensor, 'Encoded Complex Tensor', verbose))
            print(get_tensor_info(complex_frequencies, 'Complex Frequencies', verbose))
            print(get_tensor_info(inverse_frequencies, 'Inverse Frequencies', verbose))
            print(get_tensor_info(restored_tensor, 'Restored Tensor', verbose))
            print(get_tensor_info(split_tensor, 'Split Tensor', verbose))
            print(get_tensor_info(tensor, 'Decoded Tensor', verbose))

        return tensor


class Repositioning(NeatModule):
    """
    Guesses what the cached KV tensors would look like if shifted by `offset`
    positions (e.g. after a dequeue).  Useful for KV-cache management in
    multi-layer transformers beyond the first.
    """

    def __init__(
            self, max_seq_len: int, embed_size: int, layer_idx: int = None, bias: bool = True,
            rope: RoPE | None = None, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,
            coefficients: int = 1, exact: bool = False, **options
    ):
        # TODO: Is the the normalization even necessary. I don't think encoded tensors could have considerable
        #       value ranges beyond what is returned from QKV projection. The hope should just be that the 
        #       shift + converter can predict something close to the expected encoded tensors. Make it optional
        #       for now with default = False.
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embed_size  = embed_size
        self.coefficients = coefficients
        self.layer_idx   = layer_idx if layer_idx is not None else 1
        # exact: the cached tensors are a pure function of ONE token (no cross-token mixing), so a shift is only a RoPE
        #        re-rotation (keys) / identity (values): nothing to learn, no learned parameters are built.
        self.exact       = bool(exact)
        self.bias        = bias
        self.rope        = rope
        self.epsilon     = manage_params(options, ['eps', 'epsilon'], 1e-9)
        self.affine      = manage_params(options, 'affine', True)
        self.device      = device
        self.dtype       = dtype

        if not self.exact:
            self.embedder   = Embedding(max_seq_len, embed_size, None, device, dtype)
            self.norm       = RMSNorm(embed_size, self.epsilon, self.affine, device, dtype)
            # coefficients > 1: Polynomial converter (only used by the learned path: layer_idx > 0, or no RoPE)
            self.converter  = Linear(embed_size, embed_size, bias, device, dtype) if coefficients <= 1 else \
                Polynomial(embed_size, embed_size, coefficients, bias, device, dtype)


    def forward(self, tensor: Tensor, offset: int, keys: int | Iterable[int] = None, verbose: bool | int = False):
        """
        Args:
            tensor: (batch_size, seq_len, embed_size)
            offset: Number of positions to shift.
        Returns:
            Same shape as input.
        """
        if tensor.numel() == 0 or offset == 0: # No need for model to learn to guess the same values
            return tensor
        if (self.layer_idx == 0 or self.exact) and self.rope is None:
            # Layer-0 values carry no RoPE and no positional info, so a shift leaves them unchanged (exact)
            return tensor

        if (self.layer_idx == 0 or self.exact) and self.rope is not None:
            # NOTE: This assumes input is most probably idx(1:max_seq_len-1) being shifted to idx(0:max_seq_len-2)
            #       that was being cached somewhere
            # TODO: Decode RoPE then get the next ascending positions' encoding
            shape = tensor.shape
            fold = max(1, shape[-1] // self.rope.head_dim)   # per-branch RoPE: (.., kv_heads, head_dim*mult) -> (.., kv_heads*mult, head_dim)
            if fold > 1:
                tensor = tensor.reshape(*shape[:-2], shape[-2] * fold, shape[-1] // fold)
            raw_tensor = self.rope.decode(tensor, pos_idx=offset, verbose=verbose) # This should have no RoPE
            if verbose: #  and verbose >= 3:
                print(get_tensor_info(raw_tensor, 'Raw Tensor', verbose))
            repo_tensor = self.rope(raw_tensor, pos_idx=0)
            if fold > 1:
                repo_tensor = repo_tensor.reshape(shape)
        else:
            # NOTE: Guess possible future tensors with current tensor and offset, since with different positional
            #       embeddings the subsequent layer would have different tensors for queries keys and values
            shift = self.embedder(offset-1, keys=keys) # shape(genomes, dim_size)
            if verbose: #  and verbose >= 3:
                print(get_tensor_info(shift, f'Shift({offset})', verbose))
            # shift(genomes, dim_size) -> tensor(genomes, ..., dim_size)
            while shift.ndim < tensor.ndim:
                shift = shift.unsqueeze(1)
            repo_tensor = self.converter(self.norm(tensor + shift, keys=keys), keys=keys)
        if verbose: #  and verbose >= 3:
            print(get_tensor_info(repo_tensor, 'Repositioned Tensor', verbose))
        return repo_tensor
