##################
pflm documentation
##################

**Version**: |version|

**pflm** is a Python package to help users fit partial function linear models with L1/L2 regularization via Alternating Direction Method of Multipliers (ADMM).
**PACE** is a popular package written in MATLAB for functional data analysis, and **fdapace** is an R package which is based on the PACE methodology.
This package is designed to be a Python alternative to these packages, providing similar functionality for users familiar with Python.
It supports various types of functional data analysis, including functional linear models, functional principal component analysis (FPCA), and functional partial least squares (FPLS).
Also, it provides user to use the similar interface as `scikit-learn <https://scikit-learn.org/stable/>`_ for fitting models and making predictions.
Note that this package is not a complete replacement for PACE or fdapace, but rather a complementary tool for users who prefer Python.
And it is under active development, so please feel free to contribute if you have any suggestions or issues.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: API reference

   reference/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
