Interpolation 1D
================

``interp1d`` interpolates 1D scattered data using linear or cubic spline
methods.  It is backed by Cython for speed and handles duplicate
x-coordinates gracefully (first occurrence is used).

Linear interpolation
--------------------

.. code-block:: python

   import numpy as np
   from pflm.interp import interp1d

   x = np.linspace(0, 1, 11)
   y = np.sin(2 * np.pi * x)

   xq = np.linspace(0, 1, 51)
   yq = interp1d(x, y, xq, method="linear")
   print("Query shape:", yq.shape)  # (51,)

Spline interpolation
--------------------

Use ``method="spline"`` for smoother results when the underlying
function is expected to be smooth.

.. code-block:: python

   yq_spline = interp1d(x, y, xq, method="spline")

   # Compare max absolute error
   y_true = np.sin(2 * np.pi * xq)
   print("Linear  max-err:", np.max(np.abs(interp1d(x, y, xq, method='linear') - y_true)))
   print("Spline  max-err:", np.max(np.abs(yq_spline - y_true)))
