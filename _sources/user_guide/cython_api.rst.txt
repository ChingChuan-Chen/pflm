Access pflm Cython API
=================

Available extensions
-------------------------------

- ``pflm.smooth.polyfit``: low-level kernels and polynomial fitters
  - ``calculate_kernel_value_f32``, ``calculate_kernel_value_f64``
  - ``polyfit1d_helper``, ``polyfit2d_helper``
- ``pflm.interp.interp``: internal interpolation routines
  - ``interp1d_linear``, ``interp1d_spline_small``, ``interp1d_spline``
  - ``interp2d_linear``, ``interp2d_spline``
- ``pflm.utils.lapack_helper``: LAPACK wrappers
  - ``_gels``, ``_gelss``, ``_gtsv``, ``_posv`` and ``_syevd``
- ``pflm.utils.blas_helper``: BLAS wrappers
  - ``_gemv`` and ``gemm``
- ``pflm.utils.trapz``: numeric helpers
  - ``trapz``

Example: call GEMV
------------------

.. code-block:: cython

    from pflm.utils.blas_helper cimport BLAS_Order, BLAS_Trans, _gemv
    cimport numpy as np
    import numpy as np

    # y := alpha * op(A) @ x + beta * y
    def call_gemv_f64(
        np.float64_t[:, :] A, np.float64_t[:] x,
        BLAS_Trans trans = BLAS_Trans.NoTrans,
        double alpha = 1.0, double beta = 0.0,
    ):
        cdef int m = A.shape[0]
        cdef int n = A.shape[1]
        cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
        cdef int lda = m if order == BLAS_Order.ColMajor else n
        cdef int out_size = m if trans == BLAS_Trans.NoTrans else n
        cdef np.ndarray[np.float64_t, ndim=1] y = np.zeros(out_size, dtype=np.float64)
        _gemv(order, trans, m, n, alpha, &A[0, 0], lda, &x[0], 1, beta, &y[0], 1)
        return y

Usage in Python:

.. code-block:: python

    import numpy as np

    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    y = call_gemv_f64(A, b)
    print(y)
