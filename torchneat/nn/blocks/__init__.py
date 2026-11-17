"""Composite evolvable blocks: attention, transformers, convolutional transformers and the Reformer."""

from . import attention, conv_attention, conver, encoding, feedforward, reformer, residual, transformer
from ._common import InvalidValueError
from .attention import Attention, AttentionLambda, BranchGain
from .conv_attention import ConvCrossAttention, ConvSelfAttention
from .conver import Conver, ConverBase, ConverBlock
from .encoding import (
    BufferEmbedding,
    BufferEncoding,
    Repositioning,
    RoPE,
    SequenceEncoding,
    SinusoidalEncoding,
    TokenEmbedding,
)
from .feedforward import ConvSwiGLU, MultiLayerPerceptron, SwiGLU
from .reformer import Reformer
from .residual import ResidualBlock
from .transformer import Transformer, TransformerBase, TransformerBlock

__all__ = [
    "ResidualBlock",
    "TokenEmbedding",
    "SinusoidalEncoding",
    "BufferEmbedding",
    "BufferEncoding",
    "SequenceEncoding",
    "RoPE",
    "Repositioning",
    "MultiLayerPerceptron",
    "SwiGLU",
    "ConvSwiGLU",
    "AttentionLambda",
    "BranchGain",
    "Attention",
    "ConvSelfAttention",
    "ConvCrossAttention",
    "TransformerBlock",
    "TransformerBase",
    "Transformer",
    "ConverBlock",
    "ConverBase",
    "Conver",
    "Reformer",
    "InvalidValueError",
    "attention", "conv_attention", "conver", "encoding", "feedforward", "reformer", "residual", "transformer",
]
