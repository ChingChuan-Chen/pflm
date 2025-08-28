import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t


def find_le_indices_memview_f64(
    np.float64_t[:] a, np.float64_t[:] b
) -> np.ndarray[np.int64_t]:
    """find_le_indices_memview_f64(a, b) -> np.ndarray[np.int64_t] (test only)"""
    cdef uint64_t n = a.size, m = b.size
    cdef np.ndarray[np.int64_t] result = np.empty(m, dtype=np.int64)
    cdef int64_t[:] result_ptr = result
    find_le_indices[np.float64_t](&a[0], n, &b[0], m, &result_ptr[0])
    return result


def find_le_indices_memview_f32(np.float32_t[:] a, np.float32_t[:] b) -> np.ndarray[np.int64_t]:
    """find_le_indices_memview_f32(a, b) -> np.ndarray[np.int64_t] (test only)"""
    cdef uint64_t n = a.size, m = b.size
    cdef np.ndarray[np.int64_t] result = np.empty(m, dtype=np.int64)
    cdef int64_t[:] result_ptr = result
    find_le_indices[np.float32_t](&a[0], n, &b[0], m, &result_ptr[0])
    return result


cdef void interp1d_memview_f64(
    np.float64_t[:] x,
    np.float64_t[:] y,
    np.float64_t[:] x_new,
    np.float64_t[:] y_new,
    int method = 0
) noexcept nogil:
    cdef uint64_t x_size = x.shape[0], x_new_size = x_new.shape[0]
    if method == 0:
        interp1d_linear[np.float64_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
    elif method == 1:
        if x_size <= 3:
            interp1d_spline_small[np.float64_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
        else:
            interp1d_spline[np.float64_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)


def interp1d_f64(
    np.ndarray[np.float64_t] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] x_new,
    int method = 0
) -> np.ndarray[np.float64_t]:
    cdef np.ndarray[np.float64_t] y_new = np.empty(x_new.size, dtype=np.float64)
    interp1d_memview_f64(x, y, x_new, y_new, method)
    return y_new


cdef void interp1d_memview_f32(
    np.float32_t[:] x,
    np.float32_t[:] y,
    np.float32_t[:] x_new,
    np.float32_t[:] y_new,
    int method = 0
) noexcept nogil:
    cdef uint64_t x_size = x.shape[0], x_new_size = x_new.shape[0]
    if method == 0:
        interp1d_linear[np.float32_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
    elif method == 1:
        if x_size <= 3:
            interp1d_spline_small[np.float32_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
        else:
            interp1d_spline[np.float32_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)


def interp1d_f32(
    np.ndarray[np.float32_t] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] x_new,
    int method = 0
) -> np.ndarray[np.float32_t]:
    cdef np.ndarray[np.float32_t] y_new = np.empty(x_new.size, dtype=np.float32)
    interp1d_memview_f32(x, y, x_new, y_new, method)
    return y_new


def interp2d_f64(
  np.ndarray[np.float64_t] x,
  np.ndarray[np.float64_t] y,
  np.ndarray[np.float64_t, ndim=2] v,
  np.ndarray[np.float64_t] x_new,
  np.ndarray[np.float64_t] y_new,
  int method = 0
):
    cdef uint64_t x_size = x.size, y_size = y.size, x_new_size = x_new.size, y_new_size = y_new.size
    cdef np.ndarray[np.float64_t, ndim=2] v_new = np.empty((y_new_size, x_new_size), order='C', dtype=np.float64)
    if method == 0:
        interp2d_linear[np.float64_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)
    elif method == 1:
        interp2d_spline[np.float64_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)
    return v_new


def interp2d_f32(
    np.ndarray[np.float32_t] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t, ndim=2] v,
    np.ndarray[np.float32_t] x_new,
    np.ndarray[np.float32_t] y_new,
    int method = 0
):
    cdef uint64_t x_size = x.size, y_size = y.size, x_new_size = x_new.size, y_new_size = y_new.size
    cdef np.ndarray[np.float32_t, ndim=2] v_new = np.empty((y_new_size, x_new_size), order='C', dtype=np.float32)
    if method == 0:
        interp2d_linear[np.float32_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)
    elif method == 1:
        interp2d_spline[np.float32_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)
    return v_new
