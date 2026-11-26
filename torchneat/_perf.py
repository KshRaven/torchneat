import os
import sys
import warnings
from contextlib import contextmanager

ENV = "TORCHNEAT_PERF_WARNINGS"
_NUMBA_ENV = "NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"
_NUMBA_FLAG = "CUDA_LOW_OCCUPANCY_WARNINGS"
_CONFIGS = ("numba.cuda.core.config", "numba.core.config")
_MESSAGE = r"Grid size \d+ will likely result in GPU under-utilization"


def _set(enabled: bool):
    os.environ[_NUMBA_ENV] = "1" if enabled else "0"
    for name in _CONFIGS:
        config = sys.modules.get(name)
        if config is not None and hasattr(config, _NUMBA_FLAG):
            setattr(config, _NUMBA_FLAG, int(enabled))


def silence():
    if os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return
    _set(False)
    warnings.filterwarnings("ignore", message=_MESSAGE)


@contextmanager
def perf_warnings():
    """Show the numba CUDA performance warnings (hidden by default; ``TORCHNEAT_PERF_WARNINGS=1`` shows them globally)."""
    previous = os.environ.get(_NUMBA_ENV)
    _set(True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("always", message=_MESSAGE)
            yield
    finally:
        if previous is None:
            os.environ.pop(_NUMBA_ENV, None)
        else:
            os.environ[_NUMBA_ENV] = previous
        silence()
