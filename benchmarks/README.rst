pflm vs fdapace
***************

This benchmark compares the performance of the Python package ``pflm`` against the R package ``fdapace`` on simulated data.

Environment
===========

- CPU: Intel Core i9-13900K @ 5.3 GHz
- RAM: 128 GB @ 4800 MHz
- OS: Windows 11 Pro
- Python: ``3.13.5``
  - numpy
  - scipy
  - BLAS/LAPACK: mkl
- R: ``4.5.1``
  - vanilla BLAS/LAPACK

polyfit 1D in Python v.s. Lwls1D in R
=====================================

- Python benchmarks: see ``benchmarks/bench_polyfit1d.py``.
- R benchmarks: see ``benchmarks/bench_polyfit1d.R``.

Suggested steps:

.. code-block:: shell

    # From repo root: build and install pflm in editable mode
    python -m pip install -U pip wheel
    python -m pip install -e .

    # Run Python benchmark script(s)
    python ./benchmarks/bench_polyfit1d.py

    # Run R
    Rscript ./benchmarks/bench_polyfit1d.R

Experiment Setup
----------------

- Number of replications: 30 (remove fastest and slowest)
- The averages and standard deviations are computed after dropping the fastest and slowest run out of 30 replications.

Results
-------

For Polyfit 1D, we set up the 10000 points in 1D space and fit a local linear smoother with bandwidth 0.1.

.. table:: Polyfit 1D (LWLS 1D)
   :widths: 20 20 20 20
   :align: left

   +----------------+------------------+------------------+--------------+
   | Implementation | Kernel           | Average time (s) | Std dev (s)  |
   +================+==================+==================+==============+
   | polyfit1d      | Gaussian         | 0.004530         | 0.000353     |
   +----------------+------------------+------------------+--------------+
   | polyfit1d      | Epanechnikov     | 0.001545         | 0.000126     |
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
   | polyfit2d      | Gaussian         |  0.327259        | 0.017827     |
   +----------------+------------------+------------------+--------------+
   | polyfit2d      | Epanechnikov     |  0.014651        | 0.000324     |
   +----------------+------------------+------------------+--------------+
   | lwls2d (R)     | gauss            | 18.065714        | 1.727727     |
   +----------------+------------------+------------------+--------------+
   | lwls2d (R)     | epan             |  0.028214        | 0.003900     |
   +----------------+------------------+------------------+--------------+
