import argparse
import gc
import pprint
import sys
import time

import numpy as np

from pflm.smooth import KernelType, Polyfit1DModel


def benchmark_polyfit1d(rng, n: int, x: np.ndarray, w: np.ndarray, x_new: np.ndarray, bandwidth: float, kernel_type: KernelType) -> int:
    """Return elapsed time (ns) for one Polyfit1D fit+predict run."""
    gc.collect()
    y = x**2 - 3.0 * x + 1.0 + 0.5 * rng.standard_normal(n)
    start_time = time.time_ns()
    model = Polyfit1DModel(kernel_type=kernel_type)
    model.fit(x, y, w, bandwidth=bandwidth)
    y_new = model.predict(x_new)
    elapsed_ns = time.time_ns() - start_time
    del y, y_new
    return elapsed_ns


def summarize_times(label: str, times_ns):
    times_ns = np.asarray(times_ns, dtype=np.int64)
    if times_ns.size > 2:
        times_trimmed = np.sort(times_ns)[1:-1]
    else:
        times_trimmed = times_ns
    print(
        f"Average time (remove fastest and slowest) for {times_ns.size} replications on {label}: "
        f"{np.mean(times_trimmed) / 1e9:.6f} seconds"
    )
    print(f"Standard deviation of run times: {np.std(times_trimmed) / 1e9:.6f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Polyfit1D in Python (pflm)")
    parser.add_argument("--n", type=int, default=int(1e4), help="Number of training points")
    parser.add_argument("--num-replications", type=int, default=30, help="Number of benchmark replications")
    parser.add_argument("--bandwidth", type=float, default=0.05, help="Smoothing bandwidth")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 1.0, args.n, dtype=np.float64)
    w = np.ones_like(x)
    x_new = np.linspace(0.0, 1.0, args.n * 2, dtype=np.float64)

    print("Python Information:\n", sys.version)
    np.show_config()
    print(f"n={args.n}, num_replications={args.num_replications}, bandwidth={args.bandwidth}")

    run_times = {"GAUSSIAN": [], "EPANECHNIKOV": []}
    kernel_map = {
        "GAUSSIAN": KernelType.GAUSSIAN,
        "EPANECHNIKOV": KernelType.EPANECHNIKOV,
    }
    for kernel_name, kernel_type in kernel_map.items():
        for _ in range(args.num_replications):
            run_times[kernel_name].append(
                benchmark_polyfit1d(rng, args.n, x, w, x_new, args.bandwidth, kernel_type)
            )

    summarize_times("Polyfit1DModel (GAUSSIAN)", run_times["GAUSSIAN"])
    summarize_times("Polyfit1DModel (EPANECHNIKOV)", run_times["EPANECHNIKOV"])

    for kernel_name, times_ns in run_times.items():
        print(f"kernel - {kernel_name}, run_time (seconds):")
        pprint.pprint(np.asarray(times_ns, dtype=np.float64) / 1e9)


if __name__ == "__main__":
    main()
