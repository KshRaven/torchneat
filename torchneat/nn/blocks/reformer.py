"""Reformer: an encoder-decoder autoencoder policy built on Conver."""

from typing import Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor

from ...utils.helpers import manage_params
from ..layers import Linear, RMSNorm, Sequential
from ..model import Model
from ..utils import get_tensor_info
from .conver import Conver


class Reformer(Model):
    def __init__(
            self, inputs: int, pol_out: int, val_out: int, max_seq_len: int, dim_size: int, layers: int,
            heads: int = None, kv_heads: int = None, differential=True,
            kernel_size=1, norm_groups=1, channels: Union[int, list[int]] = None,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Reformer, self).__init__()
        # ATTRIBUTES
        self.pol_size       = pol_out
        self.val_size       = val_out
        self.seq_len        = max_seq_len
        self.dim_size       = dim_size
        self.distribution   = manage_params(options, ['distribution', 'dist'], 'normal')
        self.epsilon        = manage_params(options, 'epsilon', 1e-8)
        self.affine         = manage_params(options, 'affine', True)
        self.probabilistic  = manage_params(options, ['probabilistic', 'prob'], True)
        self.clip_std_min   = manage_params(options, ['clip_min'], None)
        self.clip_std_max   = manage_params(options, ['clip_max'], None)
        self.clip_std       = self.clip_std_min is not None or self.clip_std_max is not None
        self.bias_enabled   = bias

        # BUILD
        self.pri_actv = self.activation = manage_params(options, ['activation', 'actv', 'pri_actv'], nn.SiLU())
        self.sec_actv = manage_params(options, 'sec_actv', None)
        options['distribution'] = options['dist'] = 'normal'
        options['sec_actv'] = None
        options['probabilistic'] = options['prob'] = False
        self.pol_proj = Conver(
            inputs, dim_size, max_seq_len, dim_size, kernel_size, layers, norm_groups, channels,
            heads, kv_heads, differential, bias, device, dtype, **options
        )
        self.mean_log_std = Sequential(
            RMSNorm(dim_size, self.epsilon, self.affine, device, dtype),
            self.pri_actv,
            Linear(dim_size, pol_out*(2 if self.probabilistic else 1), True, device, dtype),
        )
        if manage_params(options, 'get_values', False):
            self.val_proj = Conver(
                inputs, dim_size, max_seq_len, dim_size, kernel_size, layers, norm_groups, channels,
                heads, kv_heads, differential, bias, device, dtype, **options
            )
            self.decode = Linear(dim_size, val_out, True, device, dtype)
        else:
            self.val_proj = None
            self.decode = None
        self.mean_actv = manage_params(options, 'mean_actv', None)
        self.std_actv = manage_params(options, 'std_actv', None)

        # STATE
        self.single_pass    = False
        self.logits_pass    = True
        self.idx: int       = None
        self.device         = device
        self.dtype          = dtype
        self.train()

    def single_mode(self, enable=False):
        self.single_pass = enable

    def logits_mode(self, disable=False):
        self.logits_pass = not disable

    def position_idx_constant(self, index: int = None):
        self.idx = index

    def reduce(self, tensor: Tensor):
        if self.single_pass:
            tensor = tensor.squeeze(-2)
        # if not self.logits_pass and self.distribution == 'discrete':
        #     tensor = tensor.argmax(-1, False)
        return tensor

    def get_latent(self, module: Conver, state: Tensor, keys: Union[int, list[int]], pos_idx: int = None, verbose: bool | int = False, get=False, single=False):
        latent = module.forward(state, keys, pos_idx, verbose, get, single, logits=self.logits_pass)
        if self.pri_actv is not None:
            latent = self.pri_actv(latent)
        return latent

    def get_mean(self, latent: Tensor, keys: Union[int, Iterable[int]] = None) -> Tensor:
        mean = latent[..., :self.pol_size]
        if self.mean_actv is not None:
            mean = self.mean_actv(mean)
        return self.reduce(mean)

    def get_std(self, latent: Tensor, keys: Union[int, Iterable[int]] = None) -> Union[Tensor, None]:
        if self.probabilistic and self.distribution != 'discrete':
            log_std = latent[..., -self.pol_size:]
            if self.std_actv is not None:
                log_std = self.std_actv(log_std)
            if self.clip_std:
                if not isinstance(self.std_actv, nn.Sigmoid):
                    log_std = torch.clamp(log_std, self.clip_std_min, self.clip_std_max)
                else:
                    log_std = log_std * (self.clip_std_max-self.clip_std_min) + self.clip_std_min
            std = torch.pow(10.0, log_std)
            return self.reduce(std)
        else:
            return None

    def get_action(self, state: Tensor, keys: Union[int, Iterable[int]] = None) -> tuple[Tensor, Tensor]:
        latent      = self.get_latent(self.pol_proj, state, keys, self.idx, single=self.single_pass)
        source      = self.mean_log_std(latent, keys=keys)
        mean        = self.get_mean(source, keys)
        std         = self.get_std(source, keys)
        dist        = self.dist(mean, std, None, False)
        action      = dist.sample() if self.probabilistic else mean
        if self.sec_actv is not None:
            action = self.sec_actv(action)
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, Iterable[int]] = None) -> tuple[Tensor, Union[Tensor, None]]:
        latent      = self.get_latent(self.pol_proj, state, keys, self.idx, single=self.single_pass)
        source      = self.mean_log_std(latent, keys=keys)
        mean        = self.get_mean(source, keys)
        std         = self.get_std(source, keys)
        dist        = self.dist(mean, std, None, False)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, Iterable[int]] = None, **options) -> Tensor:
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', False)
        get     = manage_params(options, 'get', False)
        single  = manage_params(options, 'single', self.single_pass)

        latent  = self.get_latent(self.pol_proj, state, keys, pos_idx, verbose, get, single)
        source  = self.mean_log_std(latent, keys=keys)
        mean    = self.get_mean(source, keys)
        std     = self.get_std(source, keys)
        if verbose:
            print(get_tensor_info(mean, f"{self.__class__.__name__} Mean", verbose))
            if std is not None:
                print(get_tensor_info(std, f"{self.__class__.__name__} StdDev", verbose))
        dist    = self.dist(mean, std, None, False)
        action  = dist.sample() if self.probabilistic else mean
        if self.sec_actv is not None:
            action = self.sec_actv(action)
        if single:
            action = action.squeeze(-2)
        return action

    def get_value(self, state: Tensor, keys: Union[int, Iterable[int]] = None, **options) -> Tensor:
        if self.val_proj is not None:
            pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
            verbose = manage_params(options, 'verbose', False)
            get     = manage_params(options, 'get', False)
            single  = manage_params(options, 'single', self.single_pass)
            latent  = self.get_latent(self.val_proj, state, keys, pos_idx, verbose, get, single)
            value   = self.decode(latent, keys=keys)
            return self.reduce(value)
        else:
            raise NotImplementedError(f"Value modules were not initialized")

    @staticmethod
    def randomize(tensor: Tensor, noise: Tensor = None):
        if noise is None:
            return tensor * torch.randn_like(tensor)
        else:
            return tensor * noise

    def learn(self, inputs: Tensor, keys: Union[int, Iterable[int]] = None, verbose: bool | int = False):
        return self.forward(inputs, keys, verbose=verbose)

    def forward(self, inputs: Tensor, keys: Union[int, Iterable[int]] = None, **options):
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', False)
        get     = manage_params(options, 'get', False)
        single  = manage_params(options, 'single', False)
        randomize = manage_params(options, 'randomize', False)
        noise   = manage_params(options, 'noise', None)

        latent  = self.get_latent(self.pol_proj, inputs, keys, pos_idx, verbose, get, single)
        source  = self.mean_log_std(latent, keys=keys)
        mean    = self.get_mean(source, keys)
        std     = self.get_std(source, keys)

        if randomize:
            outputs = mean + self.randomize(std, noise)
        else:
            outputs = mean
        if self.sec_actv is not None:
            outputs = self.sec_actv(outputs)
        return outputs

    def infer(self, inputs: Tensor, keys: Union[int, Iterable[int]] = None, pos_idx: int = None, verbose: bool | int = False, get=False, single=True):
        with torch.no_grad():
            if single:
                pos_idx = -1
            if pos_idx is not None:
                seq_len = inputs.shape[-2]
                pos_idx = seq_len + pos_idx if pos_idx < 0 else pos_idx
                assert 0 < pos_idx < seq_len
            inputs = self.forward(inputs, keys=keys, pos_idx=pos_idx, verbose=verbose, get=get, single=single)
            return inputs

    def policy_params(self):
        params = list(self.pol_proj.parameters())
        params.extend(list(self.mean_log_std.parameters()))
        return params

    def value_params(self):
        params = list(self.val_proj.parameters())
        params.extend(list(self.decode.parameters()))
        return params
