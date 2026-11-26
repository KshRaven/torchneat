"""CPU vs GPU backend benchmark.

Population sizes 10^(t+1), t=0..2; module dimension 16*(t+1) and layer count (t+1), t=0..3.
Reports population creation and per-generation (speciate + reproduce) time on each backend and the speed up.

    python scripts/benchmark_backends.py [--generations 3] [--csv results.csv]
"""

import argparse
import csv
import time

import numpy as np
import torch

import torchneat
from torchneat import Config, NeatModule, NeatParameter, Population

POPULATIONS = [10 ** (t + 1) for t in range(3)]
DIMENSIONS = [8 * (2 * (t + 1)) for t in range(4)]
LAYERS = [t + 1 for t in range(4)]


class BenchModule(NeatModule):
    def __init__(self, dim: int, layers: int, device: str = "cpu"):
        super().__init__()
        for index in range(layers):
            setattr(self, f"weight{index}", NeatParameter((dim, dim), device=device, dtype=torch.float32))
            setattr(self, f"bias{index}", NeatParameter((dim,), device=device, dtype=torch.float32))


def make_config() -> Config:
    config = Config("benchmark", "benchmark")
    config.general.seed = 1
    config.genome.init_type = "normal"
    config.genome.weight_init_std = 0.5
    config.genome.weight_min_value = -np.inf
    config.genome.weight_max_value = np.inf
    config.reproduction.min_species_size = 5
    config.reproduction.purge = 0
    config.species.compatibility_threshold = np.inf
    return config


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
        try:
            import cupy
            cupy.cuda.Stream.null.synchronize()
        except ImportError:
            pass


def run(device: str, size: int, dim: int, layers: int, generations: int) -> tuple[float, float]:
    torchneat.set_device(device)
    rng = np.random.default_rng(0)

    def evaluate(population, **_options):
        for genome in population.genomes.values():
            genome.fitness = float(rng.random())

    module = BenchModule(dim, layers, device)
    sync(device)
    start = time.perf_counter()
    population = Population(size, module, make_config(), init_reporter=False, verbose=False)
    sync(device)
    create = time.perf_counter() - start

    population.run(evaluate, generations=1, verbose=False)
    sync(device)
    start = time.perf_counter()
    population.run(evaluate, generations=generations, verbose=False)
    sync(device)
    return create, (time.perf_counter() - start) / generations


def profile(device: str, size: int, dim: int, layers: int, top: int):
    import cProfile
    import pstats

    torchneat.set_device(device)
    rng = np.random.default_rng(0)

    def evaluate(population, **_options):
        for genome in population.genomes.values():
            genome.fitness = float(rng.random())

    population = Population(size, BenchModule(dim, layers, device), make_config(), init_reporter=False, verbose=False)
    population.run(evaluate, generations=1, verbose=False)
    sync(device)
    profiler = cProfile.Profile()
    profiler.enable()
    population.run(evaluate, generations=1, verbose=False)
    sync(device)
    profiler.disable()
    print(f"\n===== {device}: population={size} dim={dim} layers={layers}, one generation =====")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="profile one generation of the largest case instead")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--csv")
    args = parser.parse_args()

    devices = ["cpu"] + (["cuda"] if torchneat.cuda_is_available() else [])
    if args.profile:
        for device in devices:
            profile(device, POPULATIONS[-1], DIMENSIONS[-1], LAYERS[-1], args.top)
        return
    for device in devices:
        run(device, 10, 16, 1, 1)

    header = ["population", "dim", "layers", "params", "cpu_create_s", "cpu_gen_s",
              "gpu_create_s", "gpu_gen_s", "create_speedup", "gen_speedup"]
    rows = []
    print(f"{'pop':>5} {'dim':>4} {'lay':>3} {'params':>10} | {'cpu new':>9} {'cpu gen':>9} | "
          f"{'gpu new':>9} {'gpu gen':>9} | {'new x':>7} {'gen x':>7}")
    for size in POPULATIONS:
        for dim in DIMENSIONS:
            for layers in LAYERS:
                params = layers * (dim * dim + dim)
                cpu_new, cpu_gen = run("cpu", size, dim, layers, args.generations)
                if "cuda" in devices:
                    gpu_new, gpu_gen = run("cuda", size, dim, layers, args.generations)
                    new_x, gen_x = cpu_new / gpu_new, cpu_gen / gpu_gen
                else:
                    gpu_new = gpu_gen = new_x = gen_x = float("nan")
                rows.append([size, dim, layers, params, cpu_new, cpu_gen, gpu_new, gpu_gen, new_x, gen_x])
                print(f"{size:>5} {dim:>4} {layers:>3} {params:>10} | {cpu_new:>9.4f} {cpu_gen:>9.4f} | "
                      f"{gpu_new:>9.4f} {gpu_gen:>9.4f} | {new_x:>6.2f}x {gen_x:>6.2f}x", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
