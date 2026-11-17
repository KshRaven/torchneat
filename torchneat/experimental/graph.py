"""Classic NEAT style feed-forward network on fixed-shape parameters."""

from typing import Callable, Iterable, Union

import torch
from torch import Tensor

from ..nn.module import NeatModule
from ..nn.parameter import NeatParameter


class GraphNetwork(NeatModule):
    """
    Feed-forward network of individual nodes and connections, evolved with the fixed-shape NEAT parameters.

    Nodes are ordered ``[inputs | hidden | outputs]`` and node ``j`` may receive a connection from every earlier
    node ``i < j``. All genomes share the same ``hidden`` budget; one genome's topology is the set of its non-zero
    connections, so a genome that uses fewer nodes or layers just leaves the rest at zero and its inputs reach the
    outputs through whatever connections it kept (direct input -> output connections included), which makes
    identity padding unnecessary. Nodes are computed one after the other, batched over genomes.

    * ``weights``: one value per allowed connection, grouped by destination node.
    * ``biases``: one value per hidden and output node.

    Zero weights are only legitimate when ``config.genome.weight_del_prob > 0``; that is what lets connections be
    removed and added again. Every genome starts fully connected.
    """

    def __init__(self, inputs: int, outputs: int, hidden: int = 8, bias: bool = True,
                 activation: Callable[[Tensor], Tensor] = torch.tanh,
                 output_activation: Callable[[Tensor], Tensor] | None = None,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(GraphNetwork, self).__init__(inputs=inputs, outputs=outputs, hidden=hidden, bias=bias)
        if inputs < 1 or outputs < 1 or hidden < 0:
            raise ValueError("inputs and outputs must be positive and hidden non-negative")
        self.inputs, self.outputs, self.hidden = inputs, outputs, hidden
        self.nodes = inputs + hidden + outputs
        self.activation = activation
        self.output_activation = output_activation
        # Offset of each computed node's incoming connections inside ``weights``
        sizes = list(range(inputs, self.nodes))
        self.offsets = [sum(sizes[:step]) for step in range(len(sizes))]
        self.weights = NeatParameter(sum(sizes), False, device, dtype)
        self.biases  = NeatParameter(hidden + outputs, False, device, dtype) if bias else None

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: bool | int = False):
        weights = self.weights[keys]
        biases = self.biases[keys] if self.biases is not None else None
        out_start = self.nodes - self.outputs
        for step, node in enumerate(range(self.inputs, self.nodes)):
            w = self.expand(weights[:, self.offsets[step]:self.offsets[step] + node], tensor, keys=keys)
            value = (tensor * w).sum(dim=-1, keepdim=True)
            if biases is not None:
                value = value + self.expand(biases[:, step:step + 1], tensor, keys=keys)
            if node >= out_start:
                if self.output_activation is not None:
                    value = self.output_activation(value)
            else:
                value = self.activation(value)
            tensor = torch.cat([tensor, value], dim=-1)
        tensor = tensor[..., out_start:]
        return tensor if not verbose else (tensor, (weights, biases))

    def connections(self, key: int, epsilon: float = 0.0) -> list[tuple[int, int, float]]:
        """Active connections ``(source node, destination node, weight)`` of a genome (``|weight| > epsilon``)."""
        weights = self.weights[key][0]
        edges = []
        for step, node in enumerate(range(self.inputs, self.nodes)):
            row = weights[self.offsets[step]:self.offsets[step] + node]
            for source in torch.nonzero(row.abs() > epsilon).flatten().tolist():
                edges.append((source, node, float(row[source])))
        return edges
