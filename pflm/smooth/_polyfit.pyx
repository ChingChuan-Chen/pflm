import numpy as np
cimport numpy as np
# from cython.parallel cimport prange
from libc.math cimport abs, exp, isnan, pow, sqrt, NAN
from libc.stddef cimport ptrdiff_t, size_t
# from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from pflm.utils._lapack_helper cimport _gels_helper

cdef extern from "src/polyfit_helper.h" nogil:
    ptrdiff_t _search_lower_bound "search_lower_bound"[T](T[], ptrdiff_t, T, bint)
    ptrdiff_t _search_location "search_location"[T](T[], ptrdiff_t, T)
    np.float64_t _calculate_sqrt_kernel_value_f64 "calculate_sqrt_kernel_value_f64"(np.float64_t, int, np.float64_t)
    np.float32_t _calculate_sqrt_kernel_value_f32 "calculate_sqrt_kernel_value_f32"(np.float32_t, int, np.float32_t)
    void _polyfit1d_prepare "polyfit1d_prepare"[T](
        T, T, T*, T*, T*, size_t, int, int, int,
        vector[T]&, vector[T]&, ptrdiff_t* , ptrdiff_t*, int*
    )
    double[11] factorials

def search_lower_bound_f64(
    np.float64_t[:] a,
    np.float64_t target,
    bint right_inclusive
) -> ptrdiff_t:
    cdef ptrdiff_t n = a.shape[0]
    cdef np.float64_t[:] a_ptr = a
    cdef ptrdiff_t result = _search_lower_bound[np.float64_t](&a_ptr[0], n, target, right_inclusive)
    return result

def search_lower_bound_f32(
    np.float32_t[:] a,
    np.float32_t target,
    bint right_inclusive
) -> ptrdiff_t:
    cdef ptrdiff_t n = a.shape[0]
    cdef np.float32_t[:] a_ptr = a
    cdef ptrdiff_t result = _search_lower_bound[np.float32_t](&a_ptr[0], n, target, right_inclusive)
    return result

def search_location_f64(
    np.float64_t[:] a,
    np.float64_t target
) -> ptrdiff_t:
    cdef ptrdiff_t n = a.shape[0]
    cdef np.float64_t[:] a_ptr = a
    cdef ptrdiff_t result = _search_location[np.float64_t](&a_ptr[0], n, target)
    return result

def search_location_f32(
    np.float32_t[:] a,
    np.float32_t target
) -> ptrdiff_t:
    cdef ptrdiff_t n = a.shape[0]
    cdef np.float32_t[:] a_ptr = a
    cdef ptrdiff_t result = _search_location[np.float32_t](&a_ptr[0], n, target)
    return result

def calculate_sqrt_kernel_value_f64(
    np.float64_t u,
    int kernel_type,
    np.float64_t wj
) -> np.float64_t:
    if kernel_type >= 5 and abs(u) > 1.0:
        return 0.0
    if kernel_type == 11 and abs(abs(u) - 1.0) <= 1e-7:  # Cosine kernel is zero at the boundaries
        return 0.0
    return _calculate_sqrt_kernel_value_f64(u, kernel_type, wj)

def calculate_sqrt_kernel_value_f32(
    np.float32_t u,
    int kernel_type,
    np.float32_t wj
) -> np.float32_t:
    if kernel_type >= 5 and abs(u) > 1.0:
        return 0.0
    if kernel_type == 11 and abs(abs(u) - 1.0) <= 1e-7:  # Cosine kernel is zero at the boundaries
        return 0.0
    return _calculate_sqrt_kernel_value_f32(u, kernel_type, wj)


cdef void polyfit1d_memview_f64(
    np.float64_t[:] x,
    np.float64_t[:] y,
    np.float64_t[:] w,
    np.float64_t[:] x_new,
    np.float64_t[:] mu,
    np.float64_t bandwidth,
    int kernel,
    int degree,
    int deriv
) noexcept nogil:
    cdef ptrdiff_t i, x_new_size = x_new.shape[0], n = x.shape[0], left = 0, right = x.shape[0]
    cdef int info = 0
    cdef vector[np.float64_t] lx, ly
    cdef np.float64_t inv_deriv = 1.0 / deriv
    for i in range(x_new_size):
        _polyfit1d_prepare[np.float64_t](
            bandwidth, x_new[i],
            &x[0], &y[0], &w[0], n, degree, deriv, kernel,
            lx, ly, &left, &right, &info
        )
        if info != 0:
            mu[i] = NAN
            continue
        _gels_helper(
            110, n, degree + 1, 1,
            &lx[0], n, &ly[0], 1, &info
        )
        if info == 0:
            mu[i] = ly[deriv] * factorials[deriv] * inv_deriv
        else:
            mu[i] = NAN


def polyfit1d_f64(
    np.ndarray[np.float64_t] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] w,
    np.ndarray[np.float64_t] x_new,
    np.float64_t bandwidth,
    int kernel,
    int degree,
    int deriv
) -> np.ndarray[np.float64_t]:
    cdef np.ndarray[np.float64_t] mu = np.empty(x_new.size, dtype=np.float64)
    polyfit1d_memview_f64(x, y, w, x_new, mu, bandwidth, kernel, degree, deriv)
    return mu


cdef void polyfit1d_memview_f32(
    np.float32_t[:] x,
    np.float32_t[:] y,
    np.float32_t[:] w,
    np.float32_t[:] x_new,
    np.float32_t[:] mu,
    np.float32_t bandwidth,
    int kernel,
    int degree,
    int deriv
) noexcept nogil:
    cdef ptrdiff_t i, x_new_size = x_new.shape[0], n = x.shape[0], left = 0, right = x.shape[0]
    cdef int info = 0
    cdef vector[np.float32_t] lx, ly
    cdef np.float32_t inv_deriv = 1.0 / deriv
    for i in range(x_new_size):
        _polyfit1d_prepare[np.float32_t](
            bandwidth, x_new[i],
            &x[0], &y[0], &w[0], n, degree, deriv, kernel,
            lx, ly, &left, &right, &info
        )
        if info != 0:
            mu[i] = NAN
            continue
        _gels_helper(
            110, n, degree + 1, 1,
            &lx[0], n, &ly[0], 1, &info
        )
        if info == 0:
            mu[i] = ly[deriv] * factorials[deriv] * inv_deriv
        else:
            mu[i] = NAN


def polyfit1d_f32(
    np.ndarray[np.float32_t] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] w,
    np.ndarray[np.float32_t] x_new,
    np.float32_t bandwidth,
    int kernel,
    int degree,
    int deriv
) -> np.ndarray[np.float32_t]:
    cdef np.ndarray[np.float32_t] mu = np.empty(x_new.size, dtype=np.float32)
    polyfit1d_memview_f32(x, y, w, x_new, mu, bandwidth, kernel, degree, deriv)
    return mu
