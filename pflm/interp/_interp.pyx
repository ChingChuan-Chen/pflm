import numpy as np
cimport numpy as np
from libc.stddef cimport ptrdiff_t

cdef extern from "src/interp.cpp" nogil:
    pass

cdef extern from "src/interp.h" nogil:
    void _find_le_indices "find_le_indices"[T](T*, size_t, T*, size_t, ptrdiff_t*)
    void _interp1d_linear "interp1d_linear"[T](T*, T*, T*, T*, ptrdiff_t, ptrdiff_t)
    void _interp1d_spline_small "interp1d_spline_small"[T](T*, T*, T*, T*, ptrdiff_t, ptrdiff_t)
    void _interp1d_spline "interp1d_spline"[T](T*, T*, T*, T*, ptrdiff_t, ptrdiff_t)
    void _interp2d_linear "interp2d_linear"[T](T*, T*, T*, T*, T*, T*, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t)
    void _interp2d_spline "interp2d_spline"[T](T*, T*, T*, T*, T*, T*, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t)

def find_le_indices_memview_f64(
    np.float64_t[:] a, np.float64_t[:] b
) -> np.ndarray[np.int64_t]:
    """find_le_indices_memview_f64(a, b) -> np.ndarray[np.int64_t] (test only)"""
    cdef ptrdiff_t n = a.shape[0], m = b.shape[0]
    cdef np.ndarray[np.int64_t] result = np.empty(m, dtype=np.int64)
    cdef ptrdiff_t[:] result_ptr = result
    _find_le_indices[np.float64_t](&a[0], n, &b[0], m, &result_ptr[0])
    return result

def find_le_indices_memview_f32(
    np.float32_t[:] a, np.float32_t[:] b
) -> np.ndarray[np.int64_t]:
    """find_le_indices_memview_f32(a, b) -> np.ndarray[np.intp_t] (test only)"""
    cdef ptrdiff_t n = a.shape[0], m = b.shape[0]
    cdef np.ndarray[np.int64_t] result = np.empty(m, dtype=np.int64)
    cdef ptrdiff_t[:] result_ptr = result
    _find_le_indices[np.float32_t](&a[0], n, &b[0], m, &result_ptr[0])
    return result

cdef void interp1d_memview_f64(
    np.float64_t[:] x,
    np.float64_t[:] y,
    np.float64_t[:] x_new,
    np.float64_t[:] y_new,
    int method = 0
) noexcept nogil:
    cdef ptrdiff_t x_size = x.shape[0], x_new_size = x_new.shape[0]
    if method == 0:
      _interp1d_linear[np.float64_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
    elif method == 1:
        if x_size <= 3:
            _interp1d_spline_small[np.float64_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
        else:
            _interp1d_spline[np.float64_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)

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
    cdef ptrdiff_t x_size = x.shape[0], x_new_size = x_new.shape[0]
    if method == 0:
        _interp1d_linear[np.float32_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
    elif method == 1:
        if x_size <= 3:
            _interp1d_spline_small[np.float32_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)
        else:
            _interp1d_spline[np.float32_t](&x[0], &y[0], &x_new[0], &y_new[0], x_size, x_new_size)

def interp1d_f32(
    np.ndarray[np.float32_t] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] x_new,
    int method = 0
) -> np.ndarray[np.float32_t]:
    cdef np.ndarray[np.float32_t] y_new = np.empty(x_new.size, dtype=np.float32)
    interp1d_memview_f32(x, y, x_new, y_new, method)
    return y_new

cdef void interp2d_memview_f64(
    np.float64_t[:] x,
    np.float64_t[:] y,
    np.float64_t[:, ::1] v,
    np.float64_t[:] x_new,
    np.float64_t[:] y_new,
    np.float64_t[:, ::1] v_new,
    int method = 0
) noexcept nogil:
    cdef ptrdiff_t x_size = x.shape[0], y_size = y.shape[0], x_new_size = x_new.shape[0], y_new_size = y_new.shape[0], i
    if method == 0:
        _interp2d_linear[np.float64_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)
    elif method == 1:
        _interp2d_spline[np.float64_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)

def interp2d_f64(
  np.ndarray[np.float64_t] x,
  np.ndarray[np.float64_t] y,
  np.ndarray[np.float64_t, ndim=2] v,
  np.ndarray[np.float64_t] x_new,
  np.ndarray[np.float64_t] y_new,
  int method = 0
):
    cdef ptrdiff_t x_new_size = x_new.size, y_new_size = y_new.size
    cdef np.ndarray[np.float64_t, ndim=2] v_new = np.empty((y_new_size, x_new_size), order='C', dtype=np.float64)
    interp2d_memview_f64(x, y, v, x_new, y_new, v_new, method)
    return v_new

cdef void interp2d_memview_f32(
    np.float32_t[:] x,
    np.float32_t[:] y,
    np.float32_t[:, ::1] v,
    np.float32_t[:] x_new,
    np.float32_t[:] y_new,
    np.float32_t[:, ::1] v_new,
    int method = 0
) noexcept nogil:
    cdef ptrdiff_t x_size = x.shape[0], y_size = y.shape[0], x_new_size = x_new.shape[0], y_new_size = y_new.shape[0], i
    if method == 0:
        _interp2d_linear[np.float32_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)
    elif method == 1:
        _interp2d_spline[np.float32_t](&x[0], &y[0], &v[0, 0], &x_new[0], &y_new[0], &v_new[0, 0], x_size, y_size, x_new_size, y_new_size)

def interp2d_f32(
    np.ndarray[np.float32_t] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t, ndim=2] v,
    np.ndarray[np.float32_t] x_new,
    np.ndarray[np.float32_t] y_new,
    int method = 0
):
    cdef ptrdiff_t x_new_size = x_new.size, y_new_size = y_new.size
    cdef np.ndarray[np.float32_t, ndim=2] v_new = np.empty((y_new_size, x_new_size), order='C', dtype=np.float32)
    interp2d_memview_f32(x, y, v, x_new, y_new, v_new, method)
    return v_new
