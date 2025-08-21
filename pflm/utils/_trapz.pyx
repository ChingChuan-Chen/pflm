import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.stdint cimport int64_t, uint64_t
from libc.stdlib cimport malloc, free
from sklearn.utils._cython_blas cimport _gemv
from sklearn.utils._cython_blas cimport BLAS_Order, ColMajor, RowMajor, NoTrans

cdef void trapz_mat_blas(
    floating* y,             # shape (m, n)
    floating* x,             # shape (n)
    floating* out,           # shape (m,)
    uint64_t m,
    uint64_t n,
    BLAS_Order order
) noexcept nogil:
    cdef int64_t i
    cdef floating *dx = <floating*> malloc((n - 1) * sizeof(floating))
    for i in prange(<int64_t> n - 1, nogil=True):
        dx[i] = x[i + 1] - x[i]

    cdef int y_shift = m if order == ColMajor else 1
    cdef uint64_t lda = m if order == ColMajor else n   # FULL leading dim (not n-1)
    cdef floating alpha = <floating> 0.5
    cdef floating beta0  = <floating> 0.0
    cdef floating beta1  = <floating> 1.0

    # out = 0.5 * Y[:, :n-1] * dx
    _gemv(
        order, NoTrans,
        <int> m, <int> n-1, alpha, # m, n, alpha
        y, <int> lda,
        dx, 1, beta0, # X, inc_X, beta
        out, 1 # y, inc_y
    )

    # out += 0.5 * Y[:, 1:n] * dx   (shift by one COLUMN)
    _gemv(
        order, NoTrans,
        <int> m, <int> n-1, alpha, # m, n, alpha
        y + y_shift, <int> lda,
        dx, 1, beta1, # X, inc_X, beta
        out, 1 # y, inc_y
    )
    free(dx)


cdef void trapz_memview(
    floating[:, :] y,
    floating[:] x,
    floating[:] out,
    uint64_t m,
    uint64_t n,
    BLAS_Order order
) noexcept nogil:
    trapz_mat_blas(&y[0, 0], &x[0], &out[0], m, n, order)


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

    cdef uint64_t m = Y.shape[0], n = Y.shape[1]
    if n < 2 or m == 0:
        return np.zeros(m, dtype=np.float64)

    cdef np.ndarray[np.float64_t] out = np.zeros(m, dtype=np.float64)
    cdef np.float64_t[:, :] y_view = Y
    cdef np.float64_t[:] x_view = x
    cdef np.float64_t[:] out_view = out
    with nogil:
        trapz_mat_blas(&y_view[0, 0], &x_view[0], &out_view[0], m, n, order)
    return out


def trapz_f32(
    np.ndarray[np.float32_t, ndim=2] y,
    np.ndarray[np.float32_t] x
) -> np.ndarray[np.float32_t]:
    cdef np.ndarray[np.float32_t, ndim=2] Y
    cdef BLAS_Order order
    if y.flags.f_contiguous:
        order, Y = ColMajor, y
    elif y.flags.c_contiguous:
        order, Y = RowMajor, y
    else:
        order, Y = RowMajor, np.ascontiguousarray(y)

    cdef uint64_t m = Y.shape[0], n = Y.shape[1]
    if n < 2 or m == 0:
        return np.zeros(m, dtype=np.float32)

    cdef np.ndarray[np.float32_t] out = np.zeros(m, dtype=np.float32)
    cdef np.float32_t[:, :] y_view = Y
    cdef np.float32_t[:] x_view = x
    cdef np.float32_t[:] out_view = out
    with nogil:
        trapz_mat_blas(&y_view[0, 0], &x_view[0], &out_view[0], m, n, order)
    return out
