Installation
============

Requirements
------------

- Python ≥ 3.10
- A C/C++ compiler (for Cython extensions)
- Core dependencies: ``numpy ≥ 1.22``, ``scipy ≥ 1.14``, ``scikit-learn ≥ 1.4``

From GitHub (recommended)
---------------------------

Install the latest version directly from GitHub. pip will build the
Cython extensions automatically via meson-python.

.. code-block:: shell

   pip install git+https://github.com/ChingChuan-Chen/pflm.git

Development install
-------------------

For local development, clone the repository and install in editable mode.
You will need the build tool-chain installed first.

.. code-block:: shell

   git clone https://github.com/ChingChuan-Chen/pflm.git
   cd pflm
   pip install "meson-python>=0.17.1" "meson>=1.3" "ninja" "Cython>=3.0.10"
   pip install -e .

Documentation dependencies
--------------------------

To build the Sphinx documentation locally:

.. code-block:: shell

   pip install .[docs]
   cd doc
   python -m sphinx -b html source build/html

Verify the installation
-----------------------

.. code-block:: python

   from pflm.fpca import FunctionalPCA
   from pflm.smooth import Polyfit1DModel
   print("pflm imported successfully")