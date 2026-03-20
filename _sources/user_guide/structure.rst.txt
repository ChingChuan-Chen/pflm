Structure of pflm
=================

High-level layout

- ``pflm.fpca``: FPCA and helpers
  - Classes: ``FunctionalPCA``, ``FunctionalDataGenerator``, ``FpcaModelParams``, ``SmoothedModelResult``
  - Functions: ``get_covariance_matrix``, ``get_eigen_analysis_results``, ``estimate_rho``, ...
- ``pflm.smooth``: Smoothing models
  - Classes: ``Polyfit1DModel``, ``Polyfit2DModel``
  - Enum: ``KernelType``
- ``pflm.interp``: Lightweight interpolation
  - Functions: ``interp1d``, ``interp2d``
- ``pflm.utils``: Utilities
  - ``FlattenFunctionalData``, ``flatten_and_sort_data_matrices``, ``trapz``

Typical imports

.. code-block:: python

   from pflm.fpca import FunctionalPCA, FunctionalDataGenerator
   from pflm.smooth import Polyfit1DModel, Polyfit2DModel, KernelType
   from pflm.interp import interp1d, interp2d
   from pflm.utils import FlattenFunctionalData, flatten_and_sort_data_matrices, trapz