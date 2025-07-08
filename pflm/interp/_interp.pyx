import numpy as np
cimport numpy as np
from libc.stddef cimport ptrdiff_t

cdef extern from "src/interp_helper.cpp" nogil:
    pass

cdef extern from "src/interp_helper.h" nogil:
    ptrdiff_t _search_sorted "search_sorted"[T](T*, ptrdiff_t)

cdef extern from "src/interp.cpp" nogil:
    pass

cdef extern from "src/interp.h" nogil:
    void _interp1d_linear "interp1d_linear"[T](T*, T*, T*, T*, ptrdiff_t, ptrdiff_t)
    void _interp1d_spline_small "interp1d_spline_small"[T](T*, T*, T*, T*, ptrdiff_t, ptrdiff_t)
    void _interp1d_spline "interp1d_spline"[T](T*, T*, T*, T*, ptrdiff_t, ptrdiff_t)
    void _interp2d_linear "interp2d_linear"[T](T*, T*, T*, T*, T*, T*, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t)
    void _interp2d_spline "interp2d_spline"[T](T*, T*, T*, T*, T*, T*, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t)

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
):
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
):
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
