"""Model: a NeatModule with a policy/value interface (distribution sampling, log-probabilities)."""

from typing import Iterable, Union

import torch
from torch import Tensor

from .distributions import SharpCategorical
from .module import NeatModule


class Model(NeatModule):
    def __init__(self, distribution: str = 'normal'):
        super().__init__()
        self.distribution: str = distribution
    
    def set_distribution(self, distribution: str):
        self.distribution = distribution

    set_ditribution = set_distribution  # backwards compatible spelling

    def dist(self, mean: Tensor, std: Union[Tensor, None], latent: Tensor = None, verbose: bool | int = False):
        mean, std = mean.float(), std.float() if std is not None else None
        def fill_std(std_dev: Tensor):
            if std_dev is not None:
                return std_dev
            else:
                return torch.full_like(mean, 1e-12)

        extra = {}
        if self.distribution == 'discrete':
            # `std` is passed straight into SharpCategorical, which internally scales
            # `mean` by it as a temperature and stores it as `.stddev_input`. This keeps
            # the std<->entropy relationship exact and documented inside the class
            # (pseudo_std == std / raw_margin) rather than baked into an external
            # division dist() does before the class ever sees it.
            distribution = SharpCategorical(logits=mean, std=std)
        elif self.distribution == 'normal':
            distribution = torch.distributions.Normal(mean, fill_std(std))
        elif self.distribution == 'mult_var_normal':
            std = fill_std(std)
            if latent is not None:
                try:
                    # Create and cache the correlation layer and its lower-triangular indices, if needed.
                    if not hasattr(self, 'corr'):
                        embedding, features = latent.shape[-1], std.shape[-1]
                        tril_params_num = (features * (features - 1)) // 2
                        self.corr = torch.nn.Linear(embedding, tril_params_num, device=std.device, dtype=std.dtype,
                                                    bias=self.bias_enabled)
                        self.corr_indices = torch.tril_indices(features, features, offset=-1, device=std.device)

                    # Compute the lower-triangular correlation parameters and reduce them to (-1, 1)
                    corr_params = torch.tanh(self.corr(latent))

                    # Determine the batch shape from the correlation output (could be one or more batch dims)
                    features_out = std.shape[-1]  # number of features (must equal 'features')

                    # Create a full identity matrix of shape (F, F) and expand it to the batch shape.
                    identity = torch.eye(features_out, device=corr_params.device, dtype=corr_params.dtype)
                    # Clone after expansion to ensure a writable (contiguous) tensor.
                    corr_matrix = identity.expand(*std.shape[:-1], features_out, features_out).clone()
                    epsilon = corr_matrix * 1e-3

                    # Fill the lower-triangular part (excluding the diagonal) with the computed correlations
                    # Enforce symmetry: copy the lower triangle to the upper triangle.
                    # We add the transpose and subtract the duplicate diagonal..
                    corr_matrix[..., self.corr_indices[0], self.corr_indices[1]] = corr_params
                    corr_matrix[..., self.corr_indices[1], self.corr_indices[0]] = corr_params

                    # Build the covariance matrix: std_diag * corr_matrix * std_diag.
                    assert torch.all(std >= 0)
                    std_diag = torch.diag_embed(std)
                    cov_matrix = std_diag @ corr_matrix @ std_diag

                    if verbose and verbose >= 2:
                        extra['corr_params'] = corr_params
                        extra['corr_matrix'] = corr_matrix
                        extra['std_diag'] = std_diag

                    cov_matrix = cov_matrix + epsilon
                except Exception:
                    cov_matrix = torch.diag_embed(std ** 2)
            else:
                cov_matrix = torch.diag_embed(std ** 2)
            try:
                distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
            except ValueError as e:
                if verbose < 2:
                    raise e
                else:
                    distribution = None
        else:
            raise NotImplementedError(f"Unsupported distribution")
        if verbose and verbose >= 2:
            return distribution, extra
        else:
            return distribution

    def _get_mean(self, latent: Tensor, keys: int | Iterable[int] = None) -> Tensor:
        raise NotImplementedError(f"No '_get_mean' method")

    def _get_std(self, latent: Tensor, keys: int | Iterable[int] = None) -> Tensor | None:
        raise NotImplementedError(f"No '_get_std' method")

    def _get_mean_std(self, latent: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError(f"No 'get_mean_std' method")

    # def get_mean(self, state: Tensor, keys: int | Iterable[int] = None) -> Tensor:
    #     raise NotImplementedError(f"No 'get_mean' method")

    # def get_std(self, state: Tensor, keys: int | Iterable[int] = None) -> Tensor | None:
    #     raise NotImplementedError(f"No 'get_std' method")

    # def get_mean_std(self, state: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor | None]:
    #     raise NotImplementedError(f"No 'get_mean_std' method")

    def get_action(self, state: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(f"No 'get_action' method")

    def evaluate_action(self, state: Tensor, action: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError(f"No 'evaluate_action' method")

    def get_policy(self, state: Tensor, keys: int | Iterable[int] = None, **options) -> Tensor:
        raise NotImplementedError(f"No 'get_policy' method")

    def get_value(self, state: Tensor, keys: int | Iterable[int] = None) -> Tensor:
        raise NotImplementedError(f"No 'get_value' method")

    def get_q_values(self, state: Tensor, keys: int | Iterable[int] = None) -> Tensor:
        raise NotImplementedError(f"No 'get_q_values' method")
    
    def extra_repr(self) -> str:
        return f"distribution='{self.distribution}'"
