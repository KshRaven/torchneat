"""Import surface: the package imports without optional dependencies and every lazy export resolves."""

import importlib

import pytest

import torchneat


def test_version():
    assert isinstance(torchneat.__version__, str) and torchneat.__version__


@pytest.mark.parametrize("name", torchneat.__all__)
def test_lazy_exports_resolve(name):
    assert getattr(torchneat, name) is not None


@pytest.mark.parametrize("module", ["torchneat.nn", "torchneat.rl", "torchneat.backends", "torchneat.core",
                                    "torchneat.reporting", "torchneat.optim", "torchneat.utils", "torchneat.config"])
def test_subpackages_import(module):
    importlib.import_module(module)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        torchneat.does_not_exist  # noqa: B018


def test_nn_reexports_torch_modules():
    import torch.nn as tnn

    import torchneat.nn as nn

    assert nn.ReLU is tnn.ReLU
    assert nn.Dropout is tnn.Dropout
    assert issubclass(nn.Linear, nn.NeatModule)


def test_cuda_probe_never_raises():
    assert isinstance(torchneat.cuda_is_available(), bool)
