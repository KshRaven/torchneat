"""Feed-forward blocks: MLP, SwiGLU and its convolutional variant."""

from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import device as DEVICE
from torch import dtype as DTYPE

from ...utils.helpers import manage_params
from ..layers import GroupNorm, LayerNorm, Linear
from ..module import NeatModule
from ..utils import calc_padding
from ._common import get_convolution_class


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size: int, fwd_exp: int = None, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(MultiLayerPerceptron, self).__init__()
        if fwd_exp is None:
            fwd_exp = 4
        # hidden_size = 4 * embed_size
        # hidden_size = int(2 * hidden_size / 3)
        # if fwd_exp is not None:
        #     hidden_size = int(fwd_exp * hidden_size)
        # hidden_size = mult * ((hidden_size + mult - 1) // mult)
        hidden_size = fwd_exp * embed_size

        # torchneat
        self.inp_proj = nn.Linear(embed_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, embed_size, bias=bias, device=device, dtype=dtype)
        self.activation = nn.SiLU()

    def forward(self, tensor: Tensor):
        tensor = self.out_proj(self.activation(self.inp_proj(tensor)))
        return tensor


# --------------------------------------------- #
# Gated Linear Units                            #
# --------------------------------------------- #

class SwiGLU(NeatModule):
    def __init__(self, dim_size: int, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(SwiGLU, self).__init__()

        # ATTRIBUTES
        self.dim_size       = dim_size
        self.bias           = bias
        self.epsilon        = manage_params(options, 'epsilon', 1e-8)
        self.affine         = manage_params(options, 'affine', True)
        self.normalize      = manage_params(options, 'normalize', True)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], True)
        self.fwd_exp        = manage_params(options, ['fwd_exp', 'forward_expansion'], 2)
        self.out_bias       = manage_params(options, 'out_bias', bias)

        # BUILD
        hidden_size = int(self.fwd_exp * dim_size)
        # self.pre_norm = RMSNorm(dim_size, self.epsilon, self.affine, device, dtype) if self.normalize else None
        self.pre_norm = LayerNorm(dim_size, self.epsilon, self.affine, bias, device, dtype) if self.normalize else None
        self.inp_proj = Linear(dim_size, hidden_size, bias, device, dtype)
        self.mul_proj = Linear(dim_size, hidden_size, bias, device, dtype)
        self.out_proj = Linear(hidden_size, dim_size, self.out_bias, device, dtype)
        self.activation = manage_params(options, ['actv', 'activation'], nn.SiLU())

        # STATES
        self.device = device
        self.dtype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        residue = tensor
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
        tensor = self.out_proj(self.activation(self.inp_proj(tensor, keys=keys)) * self.mul_proj(tensor, keys=keys), keys=keys)
        if self.skip_connection:
            tensor = tensor + residue
        return tensor

    def extra_repr(self) -> str:
        return (f"*** "
                f"residual={self.skip_connection}, normalize={self.normalize}, "
                f"***")


class ConvSwiGLU(NeatModule):
    def __init__(self, dim_size: int, kernel_size: int,  bias=False,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(ConvSwiGLU, self).__init__()

        # ATTRIBUTES
        self.auto_single = manage_params(options, 'auto_single', False)
        if self.auto_single:
            kernel_size = 1
        self.dim_size       = dim_size
        self.kernel_size    = kernel_size
        self.bias           = bias
        self.stride         = manage_params(options, 'stride', 1)
        self.dilation       = manage_params(options, 'dilation', 1)
        self.padding        = manage_params(options, 'padding', calc_padding(kernel_size, self.stride, self.dilation))
        self.padding_mode   = manage_params(options, 'padding_mode', 'zeros')
        self.epsilon        = manage_params(options, 'epsilon', 1e-8)
        self.affine         = manage_params(options, 'affine', True)
        self.norm_groups    = manage_params(options, 'norm_groups', None)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], False)
        self.fwd_exp        = manage_params(options, ['fwd_exp', 'forward_expansion'], 1)
        self.image_ndim     = manage_params(options, 'image_ndim', 2)
        # hidden_size = 4 * embed_size
        # hidden_size = int(2 * hidden_size / 3)
        # if fwd_exp is not None:
        #     hidden_size = int(fwd_exp * hidden_size)
        # hidden_size = mult * ((hidden_size + mult - 1) // mult)
        hidden_size = self.fwd_exp * dim_size

        # BUILD
        self.pre_norm = GroupNorm(
            self.norm_groups, dim_size, self.epsilon, self.affine, False, device=device, dtype=dtype
        ) if self.norm_groups else None
        Convolution = get_convolution_class(self.image_ndim)
        self.inp_proj = Convolution(dim_size, hidden_size, kernel_size, stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, padding_mode=self.padding_mode,
                                    bias=bias, device=device, dtype=dtype)
        self.mul_proj = Convolution(dim_size, hidden_size, kernel_size, stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, padding_mode=self.padding_mode,
                                    bias=bias, device=device, dtype=dtype)
        self.out_proj = Convolution(hidden_size, dim_size, kernel_size, stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, padding_mode=self.padding_mode,
                                    bias=bias, device=device, dtype=dtype)
        self.activation = manage_params(options, ['actv', 'activation'], nn.SiLU())

        # STATES
        self.device = device
        self.dtype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        residue = tensor
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
        tensor = self.out_proj(self.activation(self.inp_proj(tensor, keys=keys)) * self.mul_proj(tensor, keys=keys), keys=keys)
        if self.skip_connection:
            tensor = tensor + residue
        return tensor
