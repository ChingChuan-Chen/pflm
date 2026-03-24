pflm Benchmarks
***************

This benchmark compares the performance of the Python package ``pflm`` against the R package ``fdapace`` on simulated data,
and benchmarks the pflm ADMM-based ElasticNet solver against scikit-learn's coordinate descent ElasticNet.

Environment
===========

- CPU: Intel Core i9-13900K @ 5.3 GHz
- RAM: 128 GB @ 4800 MHz
- OS: Windows 11 Pro
- Python: ``3.13.5``
  - numpy
  - scipy
  - BLAS/LAPACK: mkl
- R: ``4.5.2``
  - vanilla BLAS/LAPACK

polyfit 1D in Python v.s. Lwls1D in R
=====================================

- Python benchmarks: see ``benchmarks/bench_polyfit1d.py``.
- R benchmarks: see ``benchmarks/bench_polyfit1d.R``.

Suggested steps:

.. code-block:: shell

    # Run Python benchmark script(s)
    python ./benchmarks/bench_polyfit1d.py

    # Run R
    Rscript ./benchmarks/bench_polyfit1d.R

polyfit 2D in Python v.s. Lwls2D in R
=====================================

- Python benchmark: see ``benchmarks/bench_polyfit2d.py``.
- R benchmark: see ``benchmarks/bench_polyfit2d.R``.

Suggested steps:

.. code-block:: shell

    # Run Python benchmark script(s)
    python ./benchmarks/bench_polyfit2d.py

    # Run R
    Rscript ./benchmarks/bench_polyfit2d.R


Polyfit 1D and 2D Benchmark
-------------------------------

Configuration:

- num_subjects = 3000
- num_replications = 30
- summary = remove fastest and slowest, then compute mean/std

For Polyfit 1D, we set up the 10000 points in 1D space and fit a local linear smoother with bandwidth 0.05.

.. table:: Polyfit 1D (LWLS 1D)
   :widths: 20 20 20 20
   :align: left

   +----------------+------------------+------------------+--------------+
   | Implementation | Kernel           | Average time (s) | Std dev (s)  |
   +================+==================+==================+==============+
   | polyfit1d      | Gaussian         | 0.004472         | 0.000275     |
   +----------------+------------------+------------------+--------------+
   | polyfit1d      | Epanechnikov     | 0.001175         | 0.000176     |
   +----------------+------------------+------------------+--------------+
   | lwls1d (R)     | gauss            | 8.143571         | 0.324165     |
   +----------------+------------------+------------------+--------------+
   | lwls1d (R)     | epan             | 0.236429         | 0.010616     |
   +----------------+------------------+------------------+--------------+

For Polyfit 2D, we set up the 10000 points in 2D space (100 x 100 grid) and fit a local linear smoother with bandwidth 0.1.

.. table:: Polyfit 2D (LWLS 2D)
   :widths: 20 20 20 20
   :align: left

   +----------------+------------------+------------------+--------------+
   | Implementation | Kernel           | Average time (s) | Std dev (s)  |
   +================+==================+==================+==============+
   | polyfit2d      | Gaussian         |  0.358458        | 0.019532     |
   +----------------+------------------+------------------+--------------+
   | polyfit2d      | Epanechnikov     |  0.015823        | 0.000508     |
   +----------------+------------------+------------------+--------------+
   | lwls2d (R)     | gauss            | 18.065714        | 1.727727     |
   +----------------+------------------+------------------+--------------+
   | lwls2d (R)     | epan             |  0.028214        | 0.003900     |
   +----------------+------------------+------------------+--------------+

FPCA score in Python v.s. R
===========================

- Python benchmark: see ``benchmarks/bench_fpca_score.py``.
- R benchmark: see ``benchmarks/bench_fpca_score.R``.

Both scripts use the same fixed base trajectories and FPCA parameters (from ``fdapace/test_scores.R``),
repeat them to ``num_subjects`` subjects, and then benchmark:

- CE score path
- IN score path

Suggested steps:

.. code-block:: shell

    # Python (defaults: num_subjects=3000, num_replications=30)
    python ./benchmarks/bench_fpca_score.py

    # R (defaults: num_subjects=3000, num_replications=30)
    # Pass positional args if needed: <num_subjects> <num_replications>
    Rscript ./benchmarks/bench_fpca_score.R
    Rscript ./benchmarks/bench_fpca_score.R 3000 30

FPCA Score Benchmark
---------------------------------

Configuration:

- num_subjects = 3000
- num_replications = 30
- summary = remove fastest and slowest, then compute mean/std

.. table:: FPCA score (CE/IN)
  :widths: 28 24 24 24
  :align: left

  +------------------------+-------------------------+------------------+--------------+
  | Implementation         | Method                  | Average time (s) | Std dev (s)  |
  +========================+=========================+==================+==============+
  | get_fpca_ce_score      | CE (Python, pflm)       | 0.006906         | 0.001192     |
  +------------------------+-------------------------+------------------+--------------+
  | get_fpca_in_score      | IN (Python, pflm)       | 0.002761         | 0.000157     |
  +------------------------+-------------------------+------------------+--------------+
  | GetCEScores            | CE (R, fdapace)         | 0.358929         | 0.011001     |
  +------------------------+-------------------------+------------------+--------------+
  | GetINScores (mapply)   | IN (R, fdapace)         | 0.192143         | 0.006862     |
  +------------------------+-------------------------+------------------+--------------+

ADMM ElasticNet (pflm) v.s. sklearn ElasticNet
==============================================

- Python benchmark: see ``benchmarks/bench_elastic_net.py``.

This benchmark compares the pflm ADMM-based ElasticNet solver (Cython) against
scikit-learn's coordinate descent ElasticNet on synthetic Gaussian regression data
with sparse true coefficients.

Suggested steps:

.. code-block:: shell

    # Run with default settings (n_samples=5000, n_features=500)
    python ./benchmarks/bench_elastic_net.py

    # Customize problem size
    python ./benchmarks/bench_elastic_net.py --n-samples 10000 --n-features 1000

ElasticNet Benchmark
---------------------------------

Configuration:

- n_samples = 5000
- n_features = 500
- n_informative = 10 (sparse true coefficients)
- alpha = 0.1, l1_ratio = 0.5
- num_replications = 30
- summary = remove fastest and slowest, then compute mean/std

.. table:: ElasticNet (Gaussian, 5000 x 500)
   :widths: 28 24 24 24
   :align: left

   +----------------------------+------------------+------------------+--------------+
   | Implementation             | Solver           | Average time (s) | Std dev (s)  |
   +============================+==================+==================+==============+
   | ElasticNet (pflm)          | ADMM (Cython)    | 0.033099         | 0.002353     |
   +----------------------------+------------------+------------------+--------------+
   | ElasticNet (sklearn)       | Coordinate desc. | 0.070434         | 0.003825     |
   +----------------------------+------------------+------------------+--------------+
