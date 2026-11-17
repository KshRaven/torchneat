from typing import Iterable, Union

import torch
from torch import Tensor

from ..module import NeatModule
from ..parameter import NeatParameter


class Linear(NeatModule):
    """Fully connected linear layer."""
    
    def __init__(self, inputs: int, outputs: int, bias=True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(Linear, self).__init__(inputs=inputs, outputs=outputs, bias=bias)
        self.weights = NeatParameter((inputs, outputs), False, device, dtype)
        self.biases  = NeatParameter(outputs, False, device, dtype) if bias else None

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: bool | int = False):
        try:
            # print(f"lin_mod={self.weights.data.shape, tensor.shape}")
            w = self.expand(self.weights[keys], tensor, keys=keys)
            # print(w.shape, tensor.shape)
            tensor = torch.matmul(tensor, w)
            if self.biases is not None:
                b = self.expand(self.biases[keys], tensor, keys=keys)
                tensor = tensor + b
            else:
                b = None

            if not verbose:
                return tensor
            else:
                return tensor, (w, b)
        except ValueError as e:
            print(f"Input shape = {tensor.shape}")
            print(f"Weights shape = {self.weights.shape}")
            print(f"Biases shape = {self.biases.shape if self.biases is not None else None}")
            raise e


class Polynomial(NeatModule):
    """Polynomial layer with learnable coefficients.

    For `coefficients=n` the layer computes, per output:
        y = w[0]·x^n + w[1]·x^(n-1) + ... + w[n-1]·x^1 + b·x^0
    e.g. coeffs=2, bias=True, desc=True  ->  y = a·x² + b·x¹ + c·x⁰
    (the bias is the x⁰ term and is only present when bias=True).
    """

    def __init__(self, inputs: int, outputs: int, coefficients: int = 2, bias=True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, 
                 descending: bool = True):
        super(Polynomial, self).__init__(inputs=inputs, outputs=outputs, coefficients=coefficients, bias=bias)
        self.coefficients = coefficients
        self.has_bias = bias
        assert coefficients >= 1
        # Descending powers: [n, n-1, ..., 1]  ->  [2, 1] for coefficients=2
        factors = (coefficients, 0, -1) if descending else (1, coefficients + 1, 1)
        self.exponents = torch.arange(*factors, device=device, dtype=dtype)
        self.weights = NeatParameter((coefficients, inputs, outputs), False, device, dtype)
        self.biases  = NeatParameter(outputs, False, device, dtype) if bias else None

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: bool | int = False):
        # (G, ..., in) -> (G, ..., C, 1, in), holding x^e for each exponent e
        tensor = (tensor.unsqueeze(-2) ** self.exponents.unsqueeze(-1)).unsqueeze(-2)

        # (G, C, in, out) -> (G, 1, ..., 1, C, in, out), one singleton per extra input dim
        w = self.expand(self.weights[keys], tensor, keys=keys)
        try:
            # (G, ..., C, 1, in) @ (G, 1.., C, in, out) -> (G, ..., C, 1, out) -> sum over C -> (G, ..., out)
            tensor = torch.sum(torch.matmul(tensor, w).squeeze(-2), dim=-2)
            if self.has_bias:
                b = self.expand(self.biases[keys], tensor, keys=keys)
                tensor = tensor + b
            else:
                b = None

            if not verbose:
                return tensor
            else:
                return tensor, (w, b)
        except Exception as e:
            print(f"Input shape = {tensor.shape}")
            print(f"Weights shape = {self.weights.shape} -> {w.shape}")
            print(f"Biases shape = {self.biases.shape if self.biases is not None else None}")
            raise e
