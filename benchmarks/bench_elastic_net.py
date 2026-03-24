import argparse
import gc
import pprint
import sys
import time

import numpy as np
from sklearn.linear_model import ElasticNet as SKLearnElasticNet

from pflm.pflm.utils.linear_model import ElasticNet as ADMMElasticNet


def generate_gaussian_data(rng, n_samples, n_features, n_informative=10, noise_std=1.0):
    """Generate synthetic regression data with sparse true coefficients."""
    X = rng.standard_normal((n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = rng.standard_normal(n_informative)
    y = X @ true_coef + noise_std * rng.standard_normal(n_samples)
    return X, y, true_coef


def benchmark_admm(X, y, alpha, l1_ratio, max_iter, rho, abs_tol, rel_tol):
    """Return (elapsed_ns, n_iter) for one ADMM ElasticNet fit."""
    gc.collect()
    model = ADMMElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        max_iter=max_iter, rho=rho,
        abs_tol=abs_tol, rel_tol=rel_tol,
    )
    start_ns = time.time_ns()
    model.fit(X, y)
    elapsed_ns = time.time_ns() - start_ns
    return elapsed_ns, model.n_iter


def benchmark_sklearn(X, y, alpha, l1_ratio, max_iter, tol):
    """Return (elapsed_ns, n_iter) for one sklearn ElasticNet fit."""
    gc.collect()
    model = SKLearnElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        max_iter=max_iter, tol=tol,
        fit_intercept=True,
    )
    start_ns = time.time_ns()
    model.fit(X, y)
    elapsed_ns = time.time_ns() - start_ns
    return elapsed_ns, model.n_iter_


def summarize_times(label, times_ns):
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
    parser = argparse.ArgumentParser(description="Benchmark ADMM ElasticNet (pflm) vs sklearn ElasticNet")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--n-features", type=int, default=500, help="Number of features")
    parser.add_argument("--n-informative", type=int, default=10, help="Number of non-zero true coefficients")
    parser.add_argument("--alpha", type=float, default=0.1, help="Regularization strength")
    parser.add_argument("--l1-ratio", type=float, default=0.5, help="L1/L2 mixing ratio")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations")
    parser.add_argument("--num-replications", type=int, default=30, help="Number of benchmark replications")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    print("Python Information:\n", sys.version)
    np.show_config()
    print(
        f"n_samples={args.n_samples}, n_features={args.n_features}, "
        f"n_informative={args.n_informative}\n"
        f"alpha={args.alpha}, l1_ratio={args.l1_ratio}, max_iter={args.max_iter}, "
        f"num_replications={args.num_replications}\n"
    )

    admm_times = []
    admm_iters = []
    sklearn_times = []
    sklearn_iters = []

    for rep in range(args.num_replications):
        X, y, _ = generate_gaussian_data(rng, args.n_samples, args.n_features, args.n_informative)

        elapsed, n_iter = benchmark_admm(
            X, y, args.alpha, args.l1_ratio,
            max_iter=args.max_iter, rho=1.0, abs_tol=1e-4, rel_tol=1e-5,
        )
        admm_times.append(elapsed)
        admm_iters.append(n_iter)

        elapsed, n_iter = benchmark_sklearn(
            X, y, args.alpha, args.l1_ratio,
            max_iter=args.max_iter, tol=1e-4,
        )
        sklearn_times.append(elapsed)
        sklearn_iters.append(n_iter)

    # --- Summary ---
    print("=" * 70)
    summarize_times("ADMM ElasticNet (pflm)", admm_times)
    print(f"  Average ADMM iterations: {np.mean(admm_iters):.1f}")
    print()
    summarize_times("sklearn ElasticNet (coordinate descent)", sklearn_times)
    print(f"  Average sklearn iterations: {np.mean(sklearn_iters):.1f}")
    print("=" * 70)

    # --- Detailed per-replication times ---
    print("\nADMM ElasticNet run_time (seconds):")
    pprint.pprint(np.asarray(admm_times, dtype=np.float64) / 1e9)

    print("\nsklearn ElasticNet run_time (seconds):")
    pprint.pprint(np.asarray(sklearn_times, dtype=np.float64) / 1e9)


if __name__ == "__main__":
    main()
