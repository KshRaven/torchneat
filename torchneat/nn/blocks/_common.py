"""Shared helpers of the block modules."""

from __future__ import annotations

from typing import Iterable

from ..layers import Conv1d, Conv2d, Conv3d


def get_convolution_class(dimensions: int | Iterable[int]):
    dimension_count = dimensions if isinstance(dimensions, int) else len(dimensions)
    convolution_classes = {1: Conv1d, 2: Conv2d, 3: Conv3d}
    try:
        return convolution_classes[dimension_count]
    except KeyError as error:
        raise ValueError(f"Unsupported number of image dimensions '{dimension_count}'") from error


class InvalidValueError(ValueError):
    pass
