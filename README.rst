pflm
-----

**pflm** is a Python package to help users fit partial function linear models with L1/L2 regularization via Alternating Direction Method of Multipliers (ADMM).
**PACE** is a popular package written in MATLAB for functional data analysis, and **fdapace** is an R package which is based on the PACE methodology.
This package is designed to be a Python alternative to these packages, providing similar functionality for users familiar with Python.
It supports various types of functional data analysis, including functional linear models, functional principal component analysis (FPCA), and functional partial least squares (FPLS).
Also, it provides user to use the similar interface as `scikit-learn <https://scikit-learn.org/stable/>`_ for fitting models and making predictions.
Note that this package is not a complete replacement for PACE or fdapace, but rather a complementary tool for users who prefer Python.
And it is under active development, so please feel free to contribute if you have any suggestions or issues.

Installation
~~~~~~~~~~~~

You can install **pflm** via `pip` command:

.. code-block:: shell

  pip install git+https://github.com:ChingChuan-Chen/pflm.git


Performance vs R and MATLAB packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This benchmark compares the performance of the **pflm** package with the R package **fdapace** and the MATLAB package **PACE**.
The benchmark is run on a machine with the following specifications:

- CPU: Intel Core i9-13900K@5.3GHz
- RAM: 128GB@4800MHz
- OS: Windows 11 Pro
- Python version: 3.13.5
- R version: 4.5.1
- MATLAB version: R2023b

The benchmark is run on a dataset with 1000 replications on the simulated data base on our data generating function, `FunctionalDataGenerator`.
The parameters used for the benchmark are as follows:

- n: 300
- t: np.linspace(0, 1, 100)
- mean_func: lambda t: np.sin(2 * np.pi * t)
- var_func: lambda t: 0.1 * np.ones_like(t)
- corr_func: scipy.special.j0
- variation_prop_thresh: 0.95
- error_var: 1

.. code-block:: python

    import numpy as np
    from pflm import FunctionalDataGenerator
    fdg = FunctionalDataGenerator(
        np.linspace(0, 1, 100),
        lambda t: np.sin(2 * np.pi * t),
        lambda t: 0.1 * np.ones_like(t),
        scipy.special.j0,
        0.95, 1
    )
    regular_data = fdg.generate(300, data_type='regular')
    regular_data_with_missing = fdg.make_missing(regular_data, 10)

Reference
~~~~~~~~~

1. fdapace: Functional Data Analysis and Empirical Dynamics, https://cran.r-project.org/web/packages/fdapace/index.html.
2. PACE: Principal Analysis by Conditional Expectation, https://www.stat.ucdavis.edu/PACE/
