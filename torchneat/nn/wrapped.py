"""PyTorch modules re-exported so user code needs a single ``torchneat.nn`` import.

Everything here is the unmodified ``torch.nn`` class (parameter-free modules, or modules whose parameters are not
evolved). They can be placed inside :class:`torchneat.nn.Sequential` next to NEAT-evolvable layers. NEAT-evolvable
counterparts of ``Linear``, ``Conv*``, ``*Norm``, ``Embedding`` and the pooling layers live in :mod:`torchneat.nn.layers`
and take priority for those names.
"""

import torch.nn.functional as functional
import torch.nn.init as init
from torch.nn import (
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
)

ACTIVATIONS = (
    "CELU", "ELU", "GELU", "GLU", "Hardshrink", "Hardsigmoid", "Hardswish", "Hardtanh", "LeakyReLU", "LogSigmoid",
    "LogSoftmax", "Mish", "PReLU", "RReLU", "ReLU", "ReLU6", "SELU", "SiLU", "Sigmoid", "Softmax", "Softmin",
    "Softplus", "Softshrink", "Softsign", "Tanh", "Tanhshrink", "Threshold",
)
REGULARIZATION = ("AlphaDropout", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d")
SHAPE = ("Flatten", "Unflatten", "Upsample", "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
         "ReplicationPad2d", "ZeroPad1d", "ZeroPad2d")
CONTAINERS = ("Module", "ModuleDict", "ModuleList", "Parameter", "ParameterDict", "ParameterList")
LOSSES = ("BCELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "HuberLoss", "L1Loss", "MSELoss", "NLLLoss",
          "SmoothL1Loss")

__all__ = [*ACTIVATIONS, *REGULARIZATION, *SHAPE, *CONTAINERS, *LOSSES, "functional", "init"]
