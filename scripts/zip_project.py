"""Archive the project into a zip file, leaving out ignored files.

    python scripts/zip_project.py [--out PATH] [--git] [--ignored] [--exclude PATTERN ...]

By default the files listed by ``.gitignore`` (and ``.gitignore`` files themselves) are left out, together with
``.git``, caches and build output. ``--git`` adds ``.git``; ``--ignored`` adds the files ignored by ``.gitignore``.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALWAYS = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".DS_Store", "*.pyc", "*.zip")
SKIP_NAMES = (".gitignore", ".gitattributes")


def _git(*args: str) -> list[str] | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def _walk() -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()]


def _matches(rel: str, patterns) -> bool:
    parts = rel.split("/")
    return any(fnmatch.fnmatch(part, pat) or fnmatch.fnmatch(rel, pat) for pat in patterns for part in parts)


def collect(git: bool, ignored: bool, exclude) -> list[str]:
    if ignored:
        files = _walk()
    else:
        files = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
        if files is None:
            files = _walk()
    files = [f for f in files if (ROOT / f).is_file()]
    if not git:
        files = [f for f in files if not f.startswith(".git/")]
    else:
        tracked = {f for f in files}
        files += [f for f in _walk() if f.startswith(".git/") and f not in tracked]
    skip = [] if ignored else [*ALWAYS]
    skip += list(exclude)
    keep = []
    for f in sorted(set(files)):
        if f.startswith(".git/"):
            keep.append(f)
        elif Path(f).name in SKIP_NAMES and not ignored:
            continue
        elif _matches(f, skip):
            continue
        else:
            keep.append(f)
    return keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="output zip (default: ../<project>.zip)")
    ap.add_argument("--git", action="store_true", help="include the .git directory")
    ap.add_argument("--ignored", action="store_true", help="include files ignored by .gitignore")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="PATTERN", help="extra glob patterns to leave out")
    args = ap.parse_args(argv)

    out = (args.out or ROOT.parent / f"{ROOT.name}.zip").resolve()
    files = [f for f in collect(args.git, args.ignored, args.exclude) if (ROOT / f).resolve() != out]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(ROOT / f, f"{ROOT.name}/{f}")
    print(f"{out} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
