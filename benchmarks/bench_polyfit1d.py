import sys
import gc
import time
import numpy as np
from pflm.smooth import polyfit1d, KernelType


def benchmark_polyfit1d(rng, n, x, w, x_new, bandwidth, kernel_type=KernelType.GAUSSIAN):
    """Benchmark the polyfit1d function."""
    gc.collect()  # Clear garbage collector to avoid interference
    y = x ** 2 - 3.0 * x + 1.0 + 0.5 * rng.standard_normal(n)
    start_time = time.time_ns()
    y_new = polyfit1d(x, y, w, x_new, bandwidth, kernel_type=kernel_type)
    elapsed_time = time.time_ns() - start_time
    del y, y_new  # Free memory
    return elapsed_time


if __name__ == "__main__":
    n = int(1e3)
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = np.ones_like(x)
    x_new = np.linspace(0.0, 1.0, n * 2, dtype=np.float64)
    bandwidth = 0.05
    rng = np.random.default_rng(42)

    print("Python Information:\n", sys.version)
    np.show_config()

    num_replications = 20
    for kernel in [KernelType.GAUSSIAN, KernelType.EPANECHNIKOV]:
        run_times = []
        for i in range(num_replications):
            # Generate y with some noise each time to simulate different data
            run_times.append(benchmark_polyfit1d(rng, n, x, w, x_new, bandwidth, kernel_type=kernel))

        print(f"Average time for {num_replications} replications on polyfit1d with {kernel} kernel runs: {np.mean(run_times) / 1e9:.6f} seconds")
        print(f"Standard deviation of run times: {np.std(run_times) / 1e9:.6f} seconds")

# Python Information:
# 3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)]
#
# Build Dependencies:
#   blas:
#     detection method: pkgconfig
#     found: true
#     include directory: C:/Users/zw123/Downloads/PACE_in_R/pflm/.conda/Library/include
#     lib directory: C:/Users/zw123/Downloads/PACE_in_R/pflm/.conda/Library/lib
#     name: mkl-sdl
#     openblas configuration: unknown
#     pc file directory: C:\b\abs_958d_utj4g\croot\numpy_and_numpy_base_1750883811830\_h_env\Library\lib\pkgconfig
#     version: '2023.1'
#   lapack:
#     detection method: pkgconfig
#     found: true
#     include directory: C:/Users/zw123/Downloads/PACE_in_R/pflm/.conda/Library/include
#     lib directory: C:/Users/zw123/Downloads/PACE_in_R/pflm/.conda/Library/lib
#     name: mkl-sdl
#     openblas configuration: unknown
#     pc file directory: C:\b\abs_958d_utj4g\croot\numpy_and_numpy_base_1750883811830\_h_env\Library\lib\pkgconfig
#     version: '2023.1'
# Compilers:
#   c:
#     commands: cl.exe
#     linker: link
#     name: msvc
#     version: 19.29.30159
#   c++:
#     commands: cl.exe
#     linker: link
#     name: msvc
#     version: 19.29.30159
#   cython:
#     commands: cython
#     linker: cython
#     name: cython
#     version: 3.0.11
# Machine Information:
#   build:
#     cpu: x86_64
#     endian: little
#     family: x86_64
#     system: windows
#   host:
#     cpu: x86_64
#     endian: little
#     family: x86_64
#     system: windows
# Python Information:
#   path: C:\b\abs_958d_utj4g\croot\numpy_and_numpy_base_1750883811830\_h_env\python.exe
#   version: '3.13'
# SIMD Extensions:
#   baseline:
#   - SSE
#   - SSE2
#   - SSE3
#   found:
#   - SSSE3
#   - SSE41
#   - POPCNT
#   - SSE42
#   - AVX
#   - F16C

# Average time for 20 replications on polyfit1d with GAUSSIAN kernel runs: 0.006722 seconds
# Standard deviation of run times: 0.008586 seconds
# Average time for 20 replications on polyfit1d with EPANECHNIKOV kernel runs: 0.000611 seconds
# Standard deviation of run times: 0.000058 seconds
