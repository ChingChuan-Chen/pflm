import argparse
import gc
import pprint
import sys
import time

import numpy as np

from pflm.smooth import KernelType, Polyfit2DModel


def _make_data_2d(rng, n_side: int):
    # Grid in [-1, 1] x [-1, 1]
    x1 = np.linspace(-1.0, 1.0, n_side, dtype=np.float64)
    x2 = np.linspace(-1.0, 1.0, n_side, dtype=np.float64)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    # Smooth target + noise
    Z = np.sin(np.pi * X1) * np.cos(np.pi * X2) + 0.1 * rng.standard_normal(X1.shape)
    X = np.column_stack([X1.ravel(), X2.ravel()])
    y = Z.ravel()
    w = np.ones_like(y)
    return (x1, x2, X, y, w)


def benchmark_polyfit2d(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x1_new: np.ndarray,
    x2_new: np.ndarray,
    bandwidth1: float,
    bandwidth2: float,
    kernel_type: KernelType,
) -> int:
    """Return elapsed time in ns for one fit+predict run."""
    gc.collect()
    start_time = time.time_ns()
    model = Polyfit2DModel(kernel_type=kernel_type, degree=1, interp_kind="linear")
    # Use fixed bandwidths for consistent benchmarking; avoid CV overhead
    model.fit(
        X,
        y,
        sample_weight=w,
        bandwidth1=bandwidth1,
        bandwidth2=bandwidth2,
        reg_grid1=x1_new,  # use prediction grid as model reg grid
        reg_grid2=x2_new,
    )
    _ = model.predict(x1_new, x2_new, use_model_interp=True)
    elapsed_ns = time.time_ns() - start_time
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
    parser = argparse.ArgumentParser(description="Benchmark Polyfit2D in Python (pflm)")
    parser.add_argument("--n-side", type=int, default=100, help="Grid side length (n = n_side^2)")
    parser.add_argument("--num-replications", type=int, default=30, help="Number of benchmark replications")
    parser.add_argument("--bandwidth1", type=float, default=0.1, help="Bandwidth for first dimension")
    parser.add_argument("--bandwidth2", type=float, default=0.1, help="Bandwidth for second dimension")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    x1, x2, X, y, w = _make_data_2d(rng, args.n_side)

    # Prediction grid (can be made denser to increase load, e.g., 150 or 200)
    x1_new = x1
    x2_new = x2

    print("Python Information:\n", sys.version)
    np.show_config()
    print(
        f"n_side={args.n_side}, num_replications={args.num_replications}, "
        f"bandwidth1={args.bandwidth1}, bandwidth2={args.bandwidth2}"
    )

    run_times = {"GAUSSIAN": [], "EPANECHNIKOV": []}
    kernel_map = {
        "GAUSSIAN": KernelType.GAUSSIAN,
        "EPANECHNIKOV": KernelType.EPANECHNIKOV,
    }
    for kernel_name, kernel_type in kernel_map.items():
        for _ in range(args.num_replications):
            run_times[kernel_name].append(
                benchmark_polyfit2d(
                    X,
                    y,
                    w,
                    x1_new,
                    x2_new,
                    args.bandwidth1,
                    args.bandwidth2,
                    kernel_type,
                )
            )

    summarize_times("Polyfit2DModel (GAUSSIAN)", run_times["GAUSSIAN"])
    summarize_times("Polyfit2DModel (EPANECHNIKOV)", run_times["EPANECHNIKOV"])

    for kernel_name, times_ns in run_times.items():
        print(f"kernel - {kernel_name}, run_time (seconds):")
        pprint.pprint(np.asarray(times_ns, dtype=np.float64) / 1e9)


if __name__ == "__main__":
    main()
