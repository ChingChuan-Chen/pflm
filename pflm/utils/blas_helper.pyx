# BLAS (Basic Linear Algebra Subprograms) helper functions
# This is a copy from sklearn 1.6.1. (sklearn.utils._cython_blas)

from cython cimport floating
import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport sgemv, dgemv, sgemm, dgemm


cdef void _gemv(
    BLAS_Order order, BLAS_Trans trans, int m, int n, floating alpha,
    const floating *A, int lda, const floating *x, int incx,
    floating beta, floating *y, int incy
) noexcept nogil:
    """y := alpha * op(A) @ x + beta * y"""
    cdef char trans_ = <char> trans
    if order == BLAS_Order.RowMajor:
        trans_ = BLAS_Trans.NoTrans if trans == BLAS_Trans.Trans else BLAS_Trans.Trans
        if floating is float:
            sgemv(&trans_, &n, &m, &alpha, <float*> A, &lda, <float*> x, &incx, &beta, y, &incy)
        else:
            dgemv(&trans_, &n, &m, &alpha, <double*> A, &lda, <double*> x, &incx, &beta, y, &incy)
    else:
        if floating is float:
            sgemv(&trans_, &m, &n, &alpha, <float*> A, &lda, <float*> x, &incx, &beta, y, &incy)
        else:
            dgemv(&trans_, &m, &n, &alpha, <double*> A, &lda, <double*> x, &incx, &beta, y, &incy)


def _gemv_memview_f64(BLAS_Trans trans, np.float64_t[:, :] A, np.float64_t[:] x):
    cdef int m = A.shape[0], n = A.shape[1]
    cdef int output_size = m if trans == BLAS_Trans.NoTrans else n
    cdef np.ndarray[np.float64_t, ndim=1] y = np.zeros(output_size, dtype=np.float64)
    cdef np.float64_t[:] y_view = y
    cdef BLAS_Order order = A.strides[0] == A.itemsize
    cdef int lda = m if order == BLAS_Order.ColMajor else n
    _gemv(order, trans, m, n, 1.0, &A[0, 0], lda, &x[0], 1, 0.0, &y[0], 1)
    return y


def _gemv_memview_f32(BLAS_Trans trans, np.float32_t[:, :] A, np.float32_t[:] x):
    cdef int m = A.shape[0], n = A.shape[1]
    cdef int output_size = m if trans == BLAS_Trans.NoTrans else n
    cdef np.ndarray[np.float32_t, ndim=1] y = np.zeros(output_size, dtype=np.float32)
    cdef np.float32_t[:] y_view = y
    cdef BLAS_Order order = A.strides[0] == A.itemsize
    cdef int lda = m if order == BLAS_Order.ColMajor else n
    _gemv(order, trans, m, n, 1.0, &A[0, 0], lda, &x[0], 1, 0.0, &y[0], 1)
    return y


cdef void _gemm(
    BLAS_Order order, BLAS_Trans ta, BLAS_Trans tb, int m, int n,
    int k, floating alpha, const floating *A, int lda, const floating *B,
    int ldb, floating beta, floating *C, int ldc
) noexcept nogil:
    """C := alpha * op(A).op(B) + beta * C"""
    cdef:
        char ta_ = ta
        char tb_ = tb
    if order == BLAS_Order.RowMajor:
        if floating is float:
            sgemm(&tb_, &ta_, &n, &m, &k, &alpha, <float*> B, &ldb, <float*> A, &lda, &beta, C, &ldc)
        else:
            dgemm(&tb_, &ta_, &n, &m, &k, &alpha, <double*> B, &ldb, <double*> A, &lda, &beta, C, &ldc)
    else:
        if floating is float:
            sgemm(&ta_, &tb_, &m, &n, &k, &alpha, <float*> A, &lda, <float*> B, &ldb, &beta, C, &ldc)
        else:
            dgemm(&ta_, &tb_, &m, &n, &k, &alpha, <double*> A, &lda, <double*> B, &ldb, &beta, C, &ldc)


cdef inline void _gemm_memview(
    BLAS_Order order, BLAS_Trans ta, BLAS_Trans tb,
    floating[:, :] A, floating[:, :] B, int m, int n, int k, floating[:, :] C
):
    """
    Compute C = op(A) @ op(B) with fixed alpha=1 and beta=0.
    A and B are the underlying (pre-op) matrices flattened to 1-D with layout
    matching `order`:
      - If order == RowMajor: flatten with C-order
      - If order == ColMajor: flatten with F-order
    Shapes before op are:
      A: (m, k) if ta == NoTrans else (k, m)
      B: (k, n) if tb == NoTrans else (n, k)
    Returns C as an (m, n) ndarray with memory order matching `order`.
    """
    cdef int lda, ldb, ldc
    if order == BLAS_Order.ColMajor:
        lda = m if ta == BLAS_Trans.NoTrans else k
        ldb = k if tb == BLAS_Trans.NoTrans else n
        ldc = m
    else:
        lda = k if ta == BLAS_Trans.NoTrans else m
        ldb = n if tb == BLAS_Trans.NoTrans else k
        ldc = n

    _gemm(order, ta, tb, m, n, k, 1.0, &A[0, 0], lda, &B[0, 0], ldb, 0.0, &C[0, 0], ldc)


def _gemm_memview_f64(
    BLAS_Trans ta, BLAS_Trans tb,
    np.float64_t[:, :] A, np.float64_t[:, :] B,
    int m, int n, int k
):
    """
    Float64 version of GEMM memview wrapper.
    Delegates to the fused-typed `_gemm_memview`.
    """
    cdef BLAS_Order order = A.strides[0] == A.itemsize
    cdef object output_order = 'F' if order == BLAS_Order.ColMajor else 'C'
    cdef np.ndarray[np.float64_t, ndim=2] C = np.zeros((m, n), dtype=np.float64, order=output_order)
    cdef np.float64_t[:, :] C_view = C
    _gemm_memview(order, ta, tb, A, B, m, n, k, C_view)
    return C


def _gemm_memview_f32(
    BLAS_Trans ta, BLAS_Trans tb,
    np.float32_t[:, :] A, np.float32_t[:, :] B,
    int m, int n, int k
):
    """
    Float32 version of GEMM memview wrapper.
    Delegates to the fused-typed `_gemm_memview`.
    """
    cdef BLAS_Order order = A.strides[0] == A.itemsize
    cdef object output_order = 'F' if order == BLAS_Order.ColMajor else 'C'
    cdef np.ndarray[np.float32_t, ndim=2] C = np.zeros((m, n), dtype=np.float32, order=output_order)
    cdef np.float32_t[:, :] C_view = C
    _gemm_memview(order, ta, tb, A, B, m, n, k, C_view)
    return C
