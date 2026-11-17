"""NeatParameter: the evolvable parameter (one row per genome)."""

import copy
import sys
from itertools import count
from typing import Any, Iterable, Union

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray as CPUArray
from torch import Tensor

from ..core.genome import Genome

# GPU packages are never imported here (CPU-only machines may not have them). A cupy / numba.cuda array can only
# exist if its package is already loaded, so it is enough to look it up in sys.modules.
GPUArray = Any  # annotation only


def _is_cupy_array(array) -> bool:
    cupy = sys.modules.get('cupy')
    return cupy is not None and isinstance(array, cupy.ndarray)


def _is_numba_device_array(array) -> bool:
    module = sys.modules.get('numba.cuda.cudadrv.devicearray')
    return module is not None and isinstance(array, module.DeviceNDArray)


class NeatParameter(nn.Parameter):
    __indexer = count(0)

    def __new__(cls, 
                shape: int | Iterable[int],
                requires_grad: bool = False,
                device: torch.device = 'cpu',
                dtype: torch.dtype = torch.float32):
        
        if isinstance(shape, (int, float)):
            shape = [shape]
        
        # Tensor construction MUST happen here in __new__
        data = torch.randn(1, *shape, device=device, dtype=dtype)
        
        # Use _make_subclass to properly wrap data as this subclass type
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        return instance

    def __init__(self,
                 shape: int | Iterable[int],
                 requires_grad: bool = False,
                 device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32):
        
        # Tensor is already built — DO NOT call super().__init__(data=...) 
        super().__init__()  # object.__init__() with no args
        
        if isinstance(shape, (int, float)):
            shape = [shape]

        # Pure Python attribute setup only
        self.param_index: int = next(self.__indexer)
        self._module_index: int | None = None
        self.original_shape: tuple[int, ...] = tuple(shape)

        self.mapping: dict[int, int] = None
        self.genome_num: int = None
        
        # Debugging
        self.cd: CPUArray = None
        self.md: CPUArray = None
        
    def __deepcopy__(self, memo):
        """
        ``nn.Parameter.__deepcopy__`` rebuilds the parameter with ``type(self)(data, requires_grad)``, which would hit
        this class' ``__new__(shape, ...)`` with a tensor as ``shape``. Rebuild it here from the cloned data instead and
        carry over the python attributes (``param_index``, ``mapping``, ``original_shape`` ...).
        """
        if id(self) in memo:
            return memo[id(self)]
        result = torch.Tensor._make_subclass(
            type(self), self.data.clone(memory_format=torch.preserve_format), self.requires_grad
        )
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            result.__dict__[name] = copy.deepcopy(value, memo)
        return result

    @property
    def module_index(self):
        if self._module_index is None:
            raise ValueError(f"Module index has not been set. It is set every time Module.neat_parameters() is called.")
        else:
            return self._module_index

    def reset(self):
        self.data = torch.randn_like(self.data)
        self.requires_grad_(self.requires_grad)
        self.mapping = None
        self.genome_num = None
        self.cd = None
        self.md = None

    def _update_data(self, tensor: Tensor):
        remove_rg = self.requires_grad
        if remove_rg:
            self.requires_grad_(False)
        
        # Convert numpy to tensor if needed
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        
        # Handle shape mismatches by broadcasting
        if tensor.shape != self.data.shape:
            # If tensor is single genome (shape [1, ...]) and self.data is multi-genome (shape [N, ...])
            # broadcast/replicate the tensor across all genomes
            if len(tensor.shape) > 0 and tensor.shape[0] == 1 and len(self.data.shape) > 0 and self.data.shape[0] > 1:
                # Replicate the single genome weights across all genomes
                replicated = tensor.expand(self.data.shape[0], *tensor.shape[1:]).clone()
                self.data[:] = replicated
            else:
                # Try direct assignment (will error if incompatible)
                self.data[:] = tensor
        else:
            self.data[:] = tensor
        
        if remove_rg:
            self.requires_grad_(True)

    def update(self, genomes: dict[int, Genome], params: Union[Tensor, GPUArray] = None, verify=False):
        if params is None and self.mapping is not None and not all([key in self.mapping for key in genomes.keys()]):
            raise ValueError(f"Updating genomes of a module without parameters!")
        if isinstance(params, dict):
            params = params[self.param_index] # -> Any Array or Tensor
        if _is_numba_device_array(params):
            params = params.copy_to_host() # -> NumpyArray
        if isinstance(params, Tensor): # Handle Tensors first to avoid copying data twice
            params = params.clone().to(self.device, self.dtype).requires_grad_(self.requires_grad)
        if isinstance(params, CPUArray):
            params = torch.tensor(params, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        elif _is_cupy_array(params):
            params = torch.from_dlpack(params).to(self.device, self.dtype).requires_grad_(self.requires_grad)

        self.genome_num = len(genomes)

        if self.mapping is None:
            # Expand the tensor on first Population initialization
            if params is None:
                self.data = self.data.expand(self.genome_num, *self.original_shape)
                # size = self.data[0].numel()
                # self.data[:] = (torch.arange(size).view(self.data.shape[1:]) / size)
            # Load data from state dictionary
            else:
                d_shape = torch.tensor(self.original_shape)
                p_shape = torch.tensor(params.shape[-len(d_shape):])
                if not torch.all(d_shape == p_shape):
                    raise ValueError(f"Cannot initialize params of shape {p_shape.numpy()} "
                                     f"to data of shape {d_shape.numpy()}.")
                self.data = params.clone().to(self.device, self.dtype).requires_grad_(self.requires_grad)
        else:
            # Replace parameters with updated Population's parameters (ie. New genomes)
            if params is not None:
                # Verify pre-existing genomes
                new_indices = [index for index, genome in enumerate(genomes.values()) if genome.key in self.mapping]
                old_indices = [self.mapping[genome.key] for genome in genomes.values() if genome.key in self.mapping]
                try:
                    if verify and len(old_indices) > 0:
                        if not torch.all(self.data[old_indices] == (params[new_indices].to(self.data.device))):
                            # raise ValueError(f"Some values from new params are not in old params after update.")
                            pass
                except Exception as e:
                    print(new_indices)
                    print(old_indices)
                    print(list(genomes.keys()))
                    print(params.shape, self.data.shape)
                    raise e
                # Set population
                self.data = params

        # TODO: Rearrange genomes according to fitness and gid

        self.mapping = {genome.key: index for index, genome in enumerate(genomes.values())}

        return self.mapping, self.genome_num

    def get_list(self, keys: int | Iterable[int] = None):
        if isinstance(keys, (int, float)):
            keys = [int(keys)]
        elif keys is None or len(keys) == 0:
            return list(range(self.data.shape[0]))
        try:
            return [self.mapping[i] for i in keys]
        except KeyError as e:
            raise KeyError(f"One or more genome keys passed that are not in Population")

    def get(self, keys: int | Iterable[int] = None):
        if isinstance(keys, (int, float)):
            keys = [int(keys)]
        elif keys is None or len(keys) == 0:
            return self.data
        try:
            return self.data[[self.mapping[i] for i in keys]]
        except KeyError as e:
            raise KeyError(f"One or more genome keys passed that are not in Population; {e}")

    # TODO: Might raise issues with default get_item
    def __getitem__(self, keys: int | Iterable[int] = None):
        return self.get(keys)

    def __repr__(self):
        return f"NeatParameter(g={self.genome_num}, s={self.original_shape})"
