import importlib.util
from pathlib import Path

from setuptools import setup

_spec = importlib.util.spec_from_file_location("_torchneat_cuda", Path(__file__).parent / "torchneat" / "_cuda.py")
_cuda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cuda)

gpu = _cuda.gpu_requirements(_cuda.driver_cuda_version())
logging = ["runlytics"]
dev = ["pytest>=8", "ruff", "build>=1.0", "twine>=5.0"]

setup(extras_require={"gpu": gpu, "logging": logging, "all": gpu + logging, "dev": dev})
