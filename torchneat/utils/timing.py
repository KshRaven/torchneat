
import datetime
import time as clock

import numpy as np


def unix_to_datetime(unix_time: int):
    dt = datetime.datetime.fromtimestamp(int(unix_time))
    formatted_datetime = dt.strftime("%A, %B %d, %Y - %H:%M:%S")
    return formatted_datetime


def unix_to_datetime_file(unix_time: int | float):
    dt = datetime.datetime.fromtimestamp(int(unix_time))
    formatted_datetime = dt.strftime("%Y_%m_%d-%H_%M_%S")
    return formatted_datetime





def eta(time_start, units_done, units_total, text=''):
    time_done = clock.perf_counter() - time_start
    progress = round(units_done / units_total * 100)
    rem = round(time_done / units_done * (units_total - units_done))
    print(f"\r{text}: progress={progress}% eta={rem}sec", end='')


def timer(func, epochs: int, steps: int = 1, **kwargs):
    """Benchmark ``func(**kwargs)``: returns (mean, std, max, min) seconds per step; the first epoch is a warm-up."""
    runs = []
    for _ in range(epochs):
        durations = []
        for _ in range(steps):
            start = clock.perf_counter()
            func(**kwargs)
            durations.append(clock.perf_counter() - start)
        runs.append(np.mean(durations))
    runs = runs[1:]
    return np.mean(runs), np.std(runs), np.max(runs), np.min(runs)
