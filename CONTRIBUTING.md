# Contributing

```
pip install -e ".[dev]"
pytest -m "not gpu"      # CPU tests (what CI runs)
pytest -m gpu            # GPU tests, needs an NVIDIA GPU + `pip install -e ".[gpu]"`; skipped when none is usable
python scripts/benchmark_backends.py   # CPU vs GPU backend timings (needs an editable install)
ruff check .
```

- The package lives at `torchneat/` in the repository root; use relative imports inside it.
- Layout: `core/` (genome, species, population), `backends/` (CPU/GPU kernels), `nn/` (evolvable modules + wrapped
  PyTorch modules), `rl/` (trainers), `config/`, `optim/`, `reporting/`, `utils/`.
- Add a line under `## Unreleased` in `CHANGELOG.md` for user-visible changes.
- Releases: `python scripts/pypimgr/pypimgr.py release patch --push`.
