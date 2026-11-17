## What changed

## Why

## Checks

- [ ] `ruff check .` and `pytest -m "not gpu"` pass
- [ ] `pytest -m gpu` run locally if kernels or the GPU backend changed
- [ ] Added or updated a test for changed behaviour
- [ ] Added a line under `## Unreleased` in `CHANGELOG.md`
