"""Evolvable modules and PyTorch re-exports.

``torchneat.nn`` is meant to replace ``torch.nn`` imports in model code:

* the NEAT-evolvable layers/blocks (``Linear``, ``Conv2d``, ``LayerNorm``, ``Attention``, ``Transformer``, ...) are
  defined here and shadow the PyTorch classes of the same name;
* every other commonly used ``torch.nn`` module (activations, dropout, flatten, containers, losses) is re-exported
  unchanged from :mod:`torchneat.nn.wrapped`;
* the old scaled NEAT activations live in :mod:`torchneat.nn.legacy`.
"""

# Genome lives in ``torchneat.core``; it is re-exported here for convenience (and import-cycle safety, see core/__init__)
from ..core.genome import Genome
from . import activations, blocks, distributions, layers, legacy, utils, wrapped
from .blocks import (
    Attention,
    AttentionLambda,
    BranchGain,
    BufferEmbedding,
    BufferEncoding,
    ConvCrossAttention,
    Conver,
    ConverBase,
    ConverBlock,
    ConvSelfAttention,
    ConvSwiGLU,
    InvalidValueError,
    MultiLayerPerceptron,
    Reformer,
    Repositioning,
    ResidualBlock,
    RoPE,
    SequenceEncoding,
    SinusoidalEncoding,
    SwiGLU,
    TokenEmbedding,
    Transformer,
    TransformerBase,
    TransformerBlock,
)
from .distributions import SharpCategorical
from .layers import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    AvgPooling,
    BatchNorm,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    Convolution,
    Embedding,
    GroupNorm,
    GRUCell,
    Identity,
    Ignore,
    LayerNorm,
    Linear,
    LSTMCell,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MaxPooling,
    Permute,
    Polynomial,
    Pooling,
    RMSNorm,
    RNNCell,
    Sequential,
    Transpose,
)
from .model import Model
from .module import NeatModule, PseudoModule
from .parameter import NeatParameter
from .utils import calc_padding, display, get_tensor_info, has_parameter, model_params, model_size
from .wrapped import (
    CELU,
    ELU,
    GELU,
    GLU,
    SELU,
    AlphaDropout,
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    Flatten,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    HuberLoss,
    L1Loss,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    Module,
    ModuleDict,
    ModuleList,
    MSELoss,
    NLLLoss,
    Parameter,
    ParameterDict,
    ParameterList,
    PReLU,
    ReflectionPad1d,
    ReflectionPad2d,
    ReLU,
    ReLU6,
    ReplicationPad1d,
    ReplicationPad2d,
    RReLU,
    Sigmoid,
    SiLU,
    SmoothL1Loss,
    Softmax,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    Tanh,
    Tanhshrink,
    Threshold,
    Unflatten,
    Upsample,
    ZeroPad1d,
    ZeroPad2d,
    functional,
    init,
)

__all__ = [
    # core
    "Genome", "Model", "NeatModule", "NeatParameter", "PseudoModule", "SharpCategorical",
    # evolvable layers
    *[n for n in layers.__all__ if n[0].isupper()],
    # evolvable blocks
    *[n for n in blocks.__all__ if n[0].isupper()],
    # wrapped PyTorch modules
    *[n for n in wrapped.__all__ if n not in ("Identity",)],
    # helpers
    "calc_padding", "display", "get_tensor_info", "has_parameter", "model_params", "model_size",
    # sub-namespaces
    "activations", "blocks", "distributions", "layers", "legacy", "utils", "wrapped",
]
