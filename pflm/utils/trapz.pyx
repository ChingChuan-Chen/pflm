import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport NAN
from libc.stdint cimport int64_t, uint64_t
from libc.stdlib cimport malloc, free
from pflm.utils.blas_helper cimport BLAS_Order, ColMajor, RowMajor, BLAS_Trans, NoTrans, Trans, _gemv


cdef inline BLAS_Order opposite_order(BLAS_Order Order) noexcept nogil:
    return RowMajor if Order == ColMajor else ColMajor


cdef void trapz_mat_blas(
    BLAS_Order order_store,
    uint64_t y_rows,
    uint64_t y_cols,
    floating* y,
    uint64_t x_size,
    floating* x,
    uint64_t out_size,
    floating* out,
    int64_t inc_out
) noexcept nogil:
    cdef int64_t i
    if (y_cols != x_size) and (y_rows != x_size):
        for i in prange(<int64_t> out_size - 1, nogil=True):
            out[i] = NAN
        return

    cdef floating *dx = <floating*> malloc((x_size - 1) * sizeof(floating))
    if dx is NULL:
        for i in prange(<int64_t> out_size - 1, nogil=True):
            out[i] = NAN
        return

    for i in prange(<int64_t> x_size - 1, nogil=True):
        dx[i] = x[i + 1] - x[i]

    cdef uint64_t lda_store = y_cols if order_store == ColMajor else y_rows
    cdef bint need_transpose = (y_cols != x_size)
    cdef uint64_t m = y_cols if need_transpose else y_rows
    cdef uint64_t n = y_rows if need_transpose else y_cols
    cdef BLAS_Order order = opposite_order(order_store) if need_transpose else order_store
    cdef int64_t lda, y_offset
    if need_transpose:
        lda = y_cols if order == ColMajor else y_rows
        y_offset = m if order == ColMajor else 1
    else:
        lda = y_rows if order == ColMajor else y_cols
        y_offset = y_rows if order == ColMajor else 1

    cdef floating alpha = <floating> 0.5
    cdef floating beta0 = <floating> 0.0
    cdef floating beta1 = <floating> 1.0

    # out = 0.5 * Y[:, :n-1] * dx
    _gemv(
        order, NoTrans,
        <int> m, <int> n-1, alpha, # m, n, alpha
        y, <int> lda,
        dx, 1, beta0, # X, inc_X, beta
        out, <int> inc_out # y, inc_y
    )

    # out += 0.5 * Y[:, 1:n] * dx   (shift by one COLUMN)
    _gemv(
        order, NoTrans,
        <int> m, <int> n-1, alpha, # m, n, alpha
        y + y_offset, <int> lda,
        dx, 1, beta1, # X, inc_X, beta
        out, <int> inc_out # y, inc_y
    )
    free(dx)


cdef void trapz_memview(
    BLAS_Order order,
    floating[:, :] y,
    floating[:] x,
    floating[:] out,
    int64_t inc_out
) noexcept nogil:
    cdef uint64_t y_rows = y.shape[0], y_cols = y.shape[1], x_size = x.shape[0], out_size = out.shape[0]
    trapz_mat_blas(order, y_rows, y_cols, &y[0, 0], x_size, &x[0], out_size, &out[0], inc_out)


def trapz_f64(
    np.ndarray[np.float64_t, ndim=2] y,
    np.ndarray[np.float64_t] x
) -> np.ndarray[np.float64_t]:
    if not (y.flags.c_contiguous or y.flags.f_contiguous):
        raise ValueError("Input array must be C or F contiguous")
    cdef BLAS_Order order = ColMajor if y.flags.f_contiguous else RowMajor

    cdef uint64_t y_rows = y.shape[0], y_cols = y.shape[1], x_size = x.size
    cdef uint64_t out_size = y_rows if y_cols == x_size else y_cols
    cdef np.ndarray[np.float64_t] out = np.zeros(out_size, dtype=np.float64)
    cdef np.float64_t[:, :] y_view = y
    cdef np.float64_t[:] x_view = x
    cdef np.float64_t[:] out_view = out

    with nogil:
        trapz_mat_blas(order, y_rows, y_cols, &y_view[0, 0], x_size, &x_view[0], out_size, &out_view[0], 1)

    if not np.isfinite(out).all():
        raise ValueError("Non-finite values encountered in the integration result.")
    return out


def trapz_f32(
    np.ndarray[np.float32_t, ndim=2] y,
    np.ndarray[np.float32_t] x
) -> np.ndarray[np.float32_t]:
    if not (y.flags.c_contiguous or y.flags.f_contiguous):
        raise ValueError("Input array must be C or F contiguous")
    cdef BLAS_Order order = ColMajor if y.flags.f_contiguous else RowMajor

    cdef uint64_t y_rows = y.shape[0], y_cols = y.shape[1], x_size = x.size
    cdef uint64_t out_size = y_rows if y_cols == x_size else y_cols
    cdef np.ndarray[np.float32_t] out = np.zeros(out_size, dtype=np.float32)
    cdef np.float32_t[:, :] y_view = y
    cdef np.float32_t[:] x_view = x
    cdef np.float32_t[:] out_view = out

    with nogil:
        trapz_mat_blas(order, y_rows, y_cols, &y_view[0, 0], x_size, &x_view[0], out_size, &out_view[0], 1)

    if not np.isfinite(out).all():
        raise ValueError("Non-finite values encountered in the integration result.")
    return out
