"""Execution backends: device selection and the CPU (numba) / GPU (numba.cuda + cupy) kernels for
initialization, reproduction and speciation."""

from .execution import (
    DEVICE_ENV_NAME,
    TPB_ENV_NAME,
    cuda_is_available,
    get_device,
    get_device_env,
    get_tpb,
    get_tpb_env,
    initialize,
    reproduce,
    set_device,
    set_device_env,
    set_tpb,
    set_tpb_env,
    speciate,
)

__all__ = [
    'DEVICE_ENV_NAME', 'TPB_ENV_NAME', 'cuda_is_available',
    'get_device', 'get_device_env', 'get_tpb', 'get_tpb_env', 'initialize',
    'reproduce', 'set_device', 'set_device_env', 'set_tpb', 'set_tpb_env',
    'speciate',
]
