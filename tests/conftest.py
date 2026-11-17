"""Shared pytest setup: tests run on the CPU backend, and ``@pytest.mark.gpu`` tests skip themselves without a GPU."""

import pytest

import torchneat


def pytest_collection_modifyitems(config, items):
    if torchneat.cuda_is_available():
        return
    skip = pytest.mark.skip(reason="no usable CUDA device (needs an NVIDIA GPU with cupy and numba.cuda)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _cpu_device():
    """Every test starts on the CPU backend and restores it afterwards."""
    torchneat.set_device("cpu")
    yield
    torchneat.set_device("cpu")
