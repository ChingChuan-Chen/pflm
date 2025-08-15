import numpy as np
cimport numpy as np
from cython cimport floating
from sklearn.utils._cython_blas cimport _dot, _gemv
from sklearn.utils._cython_blas cimport BLAS_Order, ColMajor, RowMajor, NoTrans

# core: works for m >= 1 and any layout (row/col major)
cdef inline void _trapz_mat_blas(
    floating[:, :] y,        # shape (m, n)
    floating[:] dx,          # shape (n-1)
    floating[:] out,         # shape (m,)
    int m,
    int n,
    BLAS_Order order
) nogil:
    cdef int lda = m if order == ColMajor else n   # FULL leading dim (not n-1)
    cdef floating alpha = <floating>0.5
    cdef floating beta0  = <floating>0.0
    cdef floating beta1  = <floating>1.0

    # out = 0.5 * Y[:, :n-1] * dx
    _gemv(order, NoTrans, m, n-1, alpha,
          &y[0, 0], lda,
          &dx[0], 1,
          beta0, &out[0], 1)

    # out += 0.5 * Y[:, 1:n] * dx   (shift by one COLUMN)
    _gemv(order, NoTrans, m, n-1, alpha,
          &y[0, 1], lda,
          &dx[0], 1,
          beta1, &out[0], 1)


def trapz_f64(
    np.ndarray[np.float64_t, ndim=2] y,
    np.ndarray[np.float64_t] x
) -> np.ndarray[np.float64_t]:
    cdef np.ndarray[np.float64_t, ndim=2] Y
    cdef BLAS_Order order
    if y.flags.f_contiguous:
        order, Y = ColMajor, y
    elif y.flags.c_contiguous:
        order, Y = RowMajor, y
    else:
        # choose a contiguous layout; RowMajor is fine
        order, Y = RowMajor, np.ascontiguousarray(y)

    cdef int m = Y.shape[0]
    cdef int n = Y.shape[1]
    if n < 2 or m == 0:
        return np.zeros(m, dtype=np.float64)

    cdef np.ndarray[np.float64_t] dx = np.diff(x).astype(np.float64, copy=False)
    cdef np.ndarray[np.float64_t] out = np.zeros(m, dtype=np.float64)

    cdef np.float64_t[:, :] y_view = Y
    cdef np.float64_t[:] dx_view = dx
    cdef np.float64_t[:] out_view = out

    with nogil:
        _trapz_mat_blas(y_view, dx_view, out_view, m, n, order)
    return out


def trapz_f32(np.ndarray[np.float32_t, ndim=2] y,
              np.ndarray[np.float32_t] x) -> np.ndarray[np.float32_t]:
    cdef np.ndarray[np.float32_t, ndim=2] Y
    cdef BLAS_Order order
    if y.flags.f_contiguous:
        order, Y = ColMajor, y
    elif y.flags.c_contiguous:
        order, Y = RowMajor, y
    else:
        order, Y = RowMajor, np.ascontiguousarray(y)

    cdef int m = Y.shape[0]
    cdef int n = Y.shape[1]
    if n < 2 or m == 0:
        return np.zeros(m, dtype=np.float32)

    cdef np.ndarray[np.float32_t] dx = np.diff(x).astype(np.float32, copy=False)
    cdef np.ndarray[np.float32_t] out = np.zeros(m, dtype=np.float32)

    cdef np.float32_t[:, :] y_view = Y
    cdef np.float32_t[:] dx_view = dx
    cdef np.float32_t[:] out_view = out

    with nogil:
        _trapz_mat_blas(y_view, dx_view, out_view, m, n, order)
    return out
