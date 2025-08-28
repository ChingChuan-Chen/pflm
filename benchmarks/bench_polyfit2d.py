import sys
import gc
import time
import pprint
import numpy as np
from pflm.smooth import Polyfit2DModel, KernelType


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
    rng,
    n_side: int,
    x1: np.ndarray,
    x2: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x1_new: np.ndarray,
    x2_new: np.ndarray,
    bandwidth1: float,
    bandwidth2: float,
    kernel_type: KernelType = KernelType.GAUSSIAN,
) -> int:
    """Return elapsed time in ns for one fit+predict run."""
    gc.collect()
    start = time.time_ns()
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
    elapsed = time.time_ns() - start
    return elapsed


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Training grid side length (n = n_side^2 to mirror 1D's n≈1e4)
    n_side = 100               # 100 x 100 => 10,000 samples
    x1, x2, X, y, w = _make_data_2d(rng, n_side)

    # Prediction grid (can be made denser to increase load, e.g., 150 or 200)
    x1_new = x1
    x2_new = x2

    # Bandwidths (tune as needed)
    bw1 = 0.1
    bw2 = 0.1

    kernel_types = [KernelType.GAUSSIAN, KernelType.EPANECHNIKOV]

    print("Python Information:\n", sys.version)
    np.show_config()

    num_replications = 30
    run_times = {k: [] for k in kernel_types}

    for kernel in kernel_types:
        for _ in range(num_replications):
            t_ns = benchmark_polyfit2d(
                rng,
                n_side,
                x1, x2, X, y, w,
                x1_new, x2_new,
                bandwidth1=bw1,
                bandwidth2=bw2,
                kernel_type=kernel,
            )
            run_times[kernel].append(t_ns)

    for kernel in kernel_types:
        arr = np.array(run_times[kernel], dtype=np.int64)
        arr_sorted = np.sort(arr)
        trimmed = arr_sorted[1:-1]  # drop fastest/slowest
        print(
            f"Average time (remove fastest and slowest) for {num_replications} replications "
            f"with grid {n_side}x{n_side} on Polyfit2DModel with {kernel} kernel runs: "
            f"{np.mean(trimmed) / 1e9:.6f} seconds"
        )
        print(f"Standard deviation of run times: {np.std(trimmed) / 1e9:.6f} seconds")

    for kernel, times in run_times.items():
        print(f"kernel - {kernel}, run_time (s):")
        pprint.pprint(np.array(times) / 1e9)
