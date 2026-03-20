import argparse
import gc
import pprint
import sys
import time

import numpy as np

from pflm.fpca.utils import get_fpca_ce_score, get_fpca_in_score
from pflm.utils.utility import flatten_and_sort_data_matrices


BASE_LY = [
    np.array([1.0, 2.0, 2.0], dtype=np.float64),
    np.array([3.0, 4.0], dtype=np.float64),
    np.array([4.0, 5.0], dtype=np.float64),
]
BASE_LT = [
    np.array([0.1, 0.2, 0.3], dtype=np.float64),
    np.array([0.2, 0.3], dtype=np.float64),
    np.array([0.1, 0.3], dtype=np.float64),
]
MU = np.array([2.5, 2.5, 11.0 / 3.0], dtype=np.float64)
FITTED_COV = np.array(
    [
        [7.33511649, 0.47405257, 5.63170887],
        [0.47405257, 0.52865747, 0.57323255],
        [5.63170887, 0.57323255, 4.41181136],
    ],
    dtype=np.float64,
)
FPCA_LAMBDA = np.array([0.58938792, 0.05082422], dtype=np.float64)
FPCA_PHI = np.array(
    [
        [3.52004852, -0.79550581],
        [0.28836234, 3.07203424],
        [2.72817755, 0.70169923],
    ],
    dtype=np.float64,
)
SIGMA2 = np.float64(0.0)
NUM_PCS = 2


def build_dataset(num_subjects: int):
    """Build repeated sparse trajectories so both Python and R benchmarks use the same base patterns."""
    y = []
    t = []
    for i in range(num_subjects):
        j = i % len(BASE_LY)
        y.append(BASE_LY[j].copy())
        t.append(BASE_LT[j].copy())
    return y, t


def benchmark_ce(flatten_func_data):
    gc.collect()
    start_time = time.time_ns()
    xi, xi_var, yhat_mat, yhat = get_fpca_ce_score(
        flatten_func_data,
        MU,
        NUM_PCS,
        FPCA_LAMBDA,
        FPCA_PHI,
        FITTED_COV,
        SIGMA2,
    )
    elapsed_ns = time.time_ns() - start_time
    del xi, xi_var, yhat_mat, yhat
    return elapsed_ns


def benchmark_in(flatten_func_data):
    gc.collect()
    start_time = time.time_ns()
    xi, xi_var, yhat_mat, yhat = get_fpca_in_score(
        flatten_func_data,
        MU,
        NUM_PCS,
        FPCA_LAMBDA,
        FPCA_PHI,
        SIGMA2,
        False,
    )
    elapsed_ns = time.time_ns() - start_time
    del xi, xi_var, yhat_mat, yhat
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
    parser = argparse.ArgumentParser(description="Benchmark FPCA score calculation in Python (pflm)")
    parser.add_argument("--num-subjects", type=int, default=3000, help="Number of repeated sparse subjects")
    parser.add_argument("--num-replications", type=int, default=30, help="Number of benchmark replications")
    args = parser.parse_args()

    y, t = build_dataset(args.num_subjects)
    flatten_func_data = flatten_and_sort_data_matrices(y, t, np.float64)

    print("Python Information:\n", sys.version)
    np.show_config()
    print(f"num_subjects={args.num_subjects}, num_replications={args.num_replications}")

    run_times = {"CE": [], "IN": []}
    for _ in range(args.num_replications):
        run_times["CE"].append(benchmark_ce(flatten_func_data))
        run_times["IN"].append(benchmark_in(flatten_func_data))

    summarize_times("get_fpca_ce_score", run_times["CE"])
    summarize_times("get_fpca_in_score", run_times["IN"])

    for method, times_ns in run_times.items():
        print(f"method - {method}, run_time (seconds):")
        pprint.pprint(np.asarray(times_ns, dtype=np.float64) / 1e9)


if __name__ == "__main__":
    main()
