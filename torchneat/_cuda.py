import os
import re
import shutil
import subprocess
import sys
import urllib.request

TORCH_INDEX = "https://download.pytorch.org/whl"
_CPU_OFF = ("cpu", "none", "off", "0")


def driver_cuda_version() -> tuple[int, int] | None:
    """CUDA version supported by the installed NVIDIA driver, or None when no GPU is usable.

    ``TORCHNEAT_CUDA`` overrides detection: ``cpu`` or a version such as ``13.2``.
    """
    override = os.environ.get("TORCHNEAT_CUDA", "").strip().lower()
    if override in _CPU_OFF:
        return None
    if override and override != "auto":
        match = re.fullmatch(r"(\d+)\.(\d+)", override)
        if not match:
            raise ValueError(f"TORCHNEAT_CUDA must be 'cpu', 'auto' or 'MAJOR.MINOR', got {override!r}")
        return int(match[1]), int(match[2])
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(match[1]), int(match[2])) if match else None


def gpu_requirements(version: tuple[int, int] | None) -> list[str]:
    if version is None:
        return ["cuda-python>=12.0", "cuda-toolkit", "numba-cuda[cu13]", "cupy-cuda13x"]
    major, minor = version
    pin = f"{major}.{minor}.*"
    reqs = [f"numba-cuda[cu{major}]", f"cupy-cuda{major}x", f"cuda-python=={pin}", f"cuda-bindings=={pin}",
            f"cuda-toolkit=={pin}"]
    if major >= 13:
        reqs += [f"nvidia-nvvm=={pin}", f"nvidia-nvjitlink=={pin}", f"nvidia-cuda-nvrtc=={pin}",
                 f"nvidia-cuda-runtime=={pin}"]
    return reqs


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=15):
            return True
    except Exception:
        return False


def torch_index(version: tuple[int, int] | None, probe: bool = True) -> str | None:
    """PyTorch wheel index matching the driver, falling back to older builds, then CPU. None = default index."""
    if version is None:
        return None if sys.platform == "darwin" else f"{TORCH_INDEX}/cpu"
    major, minor = version
    candidates = [(major, m) for m in range(minor, -1, -1)] + [(major - 1, m) for m in range(9, -1, -1)]
    for maj, mnr in candidates:
        url = f"{TORCH_INDEX}/cu{maj}{mnr}"
        if not probe or _reachable(f"{url}/torch/"):
            return url
    return f"{TORCH_INDEX}/cpu"
