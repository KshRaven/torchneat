"""Residual (skip-connection) blocks."""

from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor

from ...utils.console import CM, Fore
from ...utils.helpers import manage_params
from ..layers import Conv1d, Conv2d, Conv3d, GroupNorm, Identity
from ..module import NeatModule
from ..utils import calc_padding, display

# --------------------------------------------- #
# Residuals                                     #
# --------------------------------------------- #

class ResidualBlock(NeatModule):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, norm_groups: int,
                 bias=False, device='cpu', dtype=torch.float32, **options):
        super().__init__()
        # Options
        self.activation: nn.Module = manage_params(options, ['activation', 'actv'], nn.SiLU())
        self.hidden_size: int   = manage_params(options, ['hidden_size', 'dim_size', 'embed_size'],
                                                channels_out if channels_out % norm_groups == 0 else channels_in)
        self.stride: int        = manage_params(options, 'stride', 1)
        self.dilation: int      = manage_params(options, 'dilation', 1)
        self.padding: Union[int, tuple[int, ...]] = manage_params(
            options, 'padding', calc_padding(kernel_size, stride=self.stride, dilation=self.dilation))
        self.padding_mode: str  = manage_params(options, 'padding_mode', 'zeros')
        self.epsilon: float     = manage_params(options, ['epsilon', 'eps'], 1e-5)
        self.affine: bool     = manage_params(options, 'affine', True)
        self.image_ndim: bool     = manage_params(options, 'image_ndim', 2)

        # Build
        try:
            if self.image_ndim == 1:
                Convolution = Conv1d
            elif self.image_ndim == 2:
                Convolution = Conv2d
            elif self.image_ndim == 3:
                Convolution = Conv3d
            else:
                raise ValueError(f"Unsupported num of image dimension '{self.image_ndim}'")
            self.norm1 = GroupNorm(norm_groups, channels_in, self.epsilon, self.affine, False, device, dtype)
            self.conv1 = Convolution(channels_in, self.hidden_size, kernel_size, self.stride, self.padding, self.dilation,
                                     bias=bias, device=device, dtype=dtype)

            self.norm2 = GroupNorm(norm_groups, self.hidden_size, self.epsilon, self.affine, False, device, dtype)
            self.conv2 = Convolution(self.hidden_size, channels_out, kernel_size, self.stride, self.padding, self.dilation,
                                     bias=bias, device=device, dtype=dtype)

            if not (channels_in != channels_out or self.stride > 1 or self.dilation > 1):
                self.residual_layer = Identity()
            else:
                self.residual_layer = Convolution(channels_in, channels_out, 1, self.stride,
                                                  calc_padding(1, self.stride, self.dilation), self.dilation,
                                                  bias=bias, device=device, dtype=dtype)
        except Exception as e:
            print(CM((channels_in, channels_out, kernel_size, norm_groups, bias, device, dtype, options), Fore.LIGHTRED_EX))
            raise e

        # Attributes
        self.in_channels = channels_in
        self.out_channels = channels_out
        self.kernel_size = kernel_size
        self.norm_groups = norm_groups
        self.bias = bias

        # States
        self.device = device
        self.stype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        # x: (Batch_Size, In_Channels, Height, Width)
        residue = tensor

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = self.conv1(self.activation(self.norm1(tensor, keys=keys)), keys=keys)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = self.conv2(self.activation(self.norm2(tensor, keys=keys)), keys=keys)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return tensor + self.residual_layer(residue, keys=keys)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}" \
               f"{display('stride', self.stride-1)}{display('dilation', self.dilation-1)}" \
               f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
               f", bias={self.bias}, ng={self.norm_groups}, actv={self.activation.__class__.__name__})"
