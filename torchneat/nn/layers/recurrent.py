"""Evolvable recurrent cells (``RNNCell``, ``GRUCell``, ``LSTMCell``) following the ``torch.nn`` gate equations.

Every cell is built from two evolvable :class:`~torchneat.nn.layers.linear.Linear` projections (input -> gates and
hidden -> gates), so all of its weights are NEAT parameters. Hidden state tensors have the same leading dimensions as
the input (genome dimension first), and default to zeros when no state is passed.
"""

from typing import Iterable, Union

import torch
from torch import Tensor

from ..module import NeatModule
from .linear import Linear

__all__ = ["GRUCell", "LSTMCell", "RNNCell"]


def _zeros(tensor: Tensor, size: int) -> Tensor:
    return tensor.new_zeros(*tensor.shape[:-1], size)


class RNNCell(NeatModule):
    """``h' = act(W_ih x + b_ih + W_hh h + b_hh)`` with ``act`` = tanh (default) or relu."""

    def __init__(self, inputs: int, hidden: int, bias: bool = True, nonlinearity: str = 'tanh',
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__(inputs=inputs, hidden=hidden, bias=bias, nonlinearity=nonlinearity)
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got '{nonlinearity}'")
        self.hidden = hidden
        self.nonlinearity = nonlinearity
        self.ih = Linear(inputs, hidden, bias, device, dtype)
        self.hh = Linear(hidden, hidden, bias, device, dtype)

    def forward(self, tensor: Tensor, state: Tensor = None, keys: Union[int, Iterable[int]] = None) -> Tensor:
        if state is None:
            state = _zeros(tensor, self.hidden)
        out = self.ih(tensor, keys=keys) + self.hh(state, keys=keys)
        return torch.tanh(out) if self.nonlinearity == 'tanh' else torch.relu(out)


class GRUCell(NeatModule):
    """Gated recurrent unit cell (same equations and gate order r, z, n as ``torch.nn.GRUCell``)."""

    def __init__(self, inputs: int, hidden: int, bias: bool = True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__(inputs=inputs, hidden=hidden, bias=bias)
        self.hidden = hidden
        self.ih = Linear(inputs, 3 * hidden, bias, device, dtype)
        self.hh = Linear(hidden, 3 * hidden, bias, device, dtype)

    def forward(self, tensor: Tensor, state: Tensor = None, keys: Union[int, Iterable[int]] = None) -> Tensor:
        if state is None:
            state = _zeros(tensor, self.hidden)
        i_r, i_z, i_n = self.ih(tensor, keys=keys).chunk(3, dim=-1)
        h_r, h_z, h_n = self.hh(state, keys=keys).chunk(3, dim=-1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        candidate = torch.tanh(i_n + reset * h_n)
        return (1 - update) * candidate + update * state


class LSTMCell(NeatModule):
    """LSTM cell (same equations and gate order i, f, g, o as ``torch.nn.LSTMCell``). Returns ``(h', c')``."""

    def __init__(self, inputs: int, hidden: int, bias: bool = True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__(inputs=inputs, hidden=hidden, bias=bias)
        self.hidden = hidden
        self.ih = Linear(inputs, 4 * hidden, bias, device, dtype)
        self.hh = Linear(hidden, 4 * hidden, bias, device, dtype)

    def forward(self, tensor: Tensor, state: tuple[Tensor, Tensor] = None,
                keys: Union[int, Iterable[int]] = None) -> tuple[Tensor, Tensor]:
        if state is None:
            state = (_zeros(tensor, self.hidden), _zeros(tensor, self.hidden))
        h, c = state
        gates = self.ih(tensor, keys=keys) + self.hh(h, keys=keys)
        i, f, g, o = gates.chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c
