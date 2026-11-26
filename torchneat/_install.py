import argparse
import importlib.metadata as metadata
import subprocess
import sys

from ._cuda import driver_cuda_version, gpu_requirements, torch_index


def _pip(*args: str, dry: bool):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(" ".join(cmd))
    if not dry:
        subprocess.check_call(cmd)


def _kind(tag: str | None):
    """('cpu', None) or ('cuda', major or None) for a torch local version tag ('cpu', 'cu132', '' for a PyPI build)."""
    if tag is None:
        return None
    if tag == "cpu":
        return "cpu", None
    if tag.startswith("cu"):
        return "cuda", int(tag[2:-1])
    return ("cuda", None) if sys.platform.startswith("linux") else ("cpu", None)


def _installed_tag() -> str | None:
    try:
        return metadata.version("torch").partition("+")[2]
    except metadata.PackageNotFoundError:
        return None


def _matches(installed, target) -> bool:
    if installed is None or target is None:
        return installed is target
    if installed[0] != target[0]:
        return False
    return installed[0] == "cpu" or installed[1] is None or target[1] is None or installed[1] == target[1]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="torchneat-install", description="Install PyTorch and CUDA dependencies matching this system")
    parser.add_argument("--cpu", action="store_true", help="CPU only")
    parser.add_argument("--cuda", metavar="X.Y", help="CUDA version to target instead of the detected one")
    parser.add_argument("--no-torch", action="store_true", help="skip PyTorch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.cpu:
        version = None
    elif args.cuda:
        major, minor = args.cuda.split(".")
        version = (int(major), int(minor))
    else:
        version = driver_cuda_version()

    if not args.no_torch:
        index = torch_index(version)
        index_args = ["--index-url", index] if index else []
        installed = _installed_tag()
        if installed is None:
            _pip("torch", *index_args, dry=args.dry_run)
        elif index and not _matches(_kind(installed), _kind(index.rsplit("/", 1)[-1])):
            _pip("--force-reinstall", "--no-deps", "torch", *index_args, dry=args.dry_run)
        else:
            print(f"torch {metadata.version('torch')} already matches, skipped")
    if version is not None:
        _pip(*gpu_requirements(version), dry=args.dry_run)


if __name__ == "__main__":
    main()
