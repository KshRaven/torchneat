# Benchmarking

Compare the CPU and GPU backends with the benchmark script. Install the package first (`pip install -e ".[gpu]"`).

```bash
python scripts/benchmark_backends.py                     # populations 10, 100, 1000; dims 16 to 64; 1 to 4 layers
python scripts/benchmark_backends.py --generations 5 --csv results.csv
python scripts/benchmark_backends.py --profile           # cProfile of one generation of the largest case
```

It times population creation and generations of reproduction and speciation with a trivial fitness function on each
backend. The module is created on the backend's device. Speed-ups here only reflect the backend; small cases are
dominated by fixed overhead.
