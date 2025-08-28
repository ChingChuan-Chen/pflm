import sys
import gc
import time
import pprint
import numpy as np
from pflm.smooth import Polyfit1DModel, KernelType


def benchmark_polyfit1d(rng, n, x, w, x_new, bandwidth, kernel_type=KernelType.GAUSSIAN):
    """Benchmark the polyfit1d function."""
    gc.collect()  # Clear garbage collector to avoid interference
    y = x ** 2 - 3.0 * x + 1.0 + 0.5 * rng.standard_normal(n)
    start_time = time.time_ns()
    polyfit1d_model = Polyfit1DModel(kernel_type=kernel_type)
    polyfit1d_model.fit(x, y, w, bandwidth=bandwidth)
    y_new = polyfit1d_model.predict(x_new)
    elapsed_time = time.time_ns() - start_time
    del y, y_new  # Free memory
    return elapsed_time


if __name__ == "__main__":
    n = int(1e4)
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = np.ones_like(x)
    x_new = np.linspace(0.0, 1.0, n * 2, dtype=np.float64)
    bandwidth = 0.05
    rng = np.random.default_rng(42)
    kernel_types = [KernelType.GAUSSIAN, KernelType.EPANECHNIKOV]

    print("Python Information:\n", sys.version)
    np.show_config()

    num_replications = 30
    run_times = dict()
    for kernel in kernel_types:
        run_times[kernel] = []
        for i in range(num_replications):
            # Generate y with some noise each time to simulate different data
            run_times[kernel].append(benchmark_polyfit1d(rng, n, x, w, x_new, bandwidth, kernel_type=kernel))

    for kernel in kernel_types:
        # remove fastest and slowest
        run_times_remove = np.sort(run_times[kernel])[1:-1]

        print(
            f"Average time (remove fastest and slowest) for {num_replications} replications with sample size {n} on " +
            f"Polyfit1DModel with {kernel} kernel runs: {np.mean(run_times_remove) / 1e9:.6f} seconds"
        )
        print(f"Standard deviation of run times: {np.std(run_times_remove) / 1e9:.6f} seconds")

    for kernel, run_time in run_times.items():
        print(f"kernel - {kernel}, run_time:")
        pprint.pprint(np.array(run_time) / 1e9)
