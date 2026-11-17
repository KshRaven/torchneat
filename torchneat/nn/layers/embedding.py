import torch
from torch import Tensor

from ..module import NeatModule
from ..parameter import NeatParameter


class Embedding(NeatModule):
    def __init__(
        self, vocab_size: int, dim_size: int,  padding_idx: int | None = None, 
        device: torch.device = 'cpu', dtype: torch.dtype = torch.float32,
    ):
        super(Embedding, self).__init__(
            vocab_size=vocab_size, dim_size=dim_size, padding_idx=padding_idx
        )

        self.vocab_size = vocab_size
        self.dim_size = dim_size
        self.padding_idx = padding_idx

        # Learnable embedding matrix
        self.weights = NeatParameter((vocab_size, dim_size), False, device, dtype)
        # shape (genomes, vocab_size, dim_size)

        # Optional padding handling
        if padding_idx is not None:
            with torch.no_grad():
                self.weights.data[padding_idx].fill_(0)

    def forward(self, tensor: Tensor | int, keys: int | list[int] = None):
        """
        input_ids: LongTensor of shape (genomes, ...)
        returns: embeddings of shape (genomes, ..., dim_size)
        """
        resolved = self.weights.get_list(keys)  # list[int] of valid slot indices
        genomes = torch.tensor(
            resolved, device=self.weights.device, dtype=torch.long
        )

        if isinstance(tensor, (int, float)):
            tensor = torch.full((genomes.shape[0],), int(tensor),
                device=self.weights.device, dtype=torch.long)
        elif tensor.ndim == 0:
            tensor = tensor.unsqueeze(0).to(self.weights.device)

        if genomes.numel() == 0:
            raise ValueError(
                f"No valid genome slots resolved for keys={keys!r}; "
                f"weights tensor has {self.weights.data.shape[0]} genomes."
            )

        # Expand genome indices to match tensor shape
        while genomes.ndim < tensor.ndim:
            genomes = genomes.unsqueeze(-1)

        genomes = genomes.expand_as(tensor)

        # Advanced indexing
        return self.weights.data[genomes, tensor]
