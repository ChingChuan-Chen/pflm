import numpy as np
cimport numpy as np
# from cython.parallel cimport prange
from libc.math cimport abs, exp, isnan, pow, sqrt, NAN
from libc.stddef cimport ptrdiff_t, size_t
# from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
# from pflm.utils._lapack_helper cimport _gels_helper

cdef extern from "src/polyfit_helper.h":
    ptrdiff_t _search_lower_bound "search_lower_bound"[T](T[], ptrdiff_t, T, bint)
    ptrdiff_t _search_location "search_location"[T](T[], ptrdiff_t, T)
    np.float64_t _calculate_sqrt_kernel_value_f64 "calculate_sqrt_kernel_value_f64"(np.float64_t, int, np.float64_t)
    np.float32_t _calculate_sqrt_kernel_value_f32 "calculate_sqrt_kernel_value_f32"(np.float32_t, int, np.float32_t)
    void _polyfit1d_prepare "polyfit1d_prepare"[T](
        T, T, const T*, const T*, const T*, size_t, int, int, int,
        vector[T]&, vector[T]&, ptrdiff_t* , ptrdiff_t*, int*
    )

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
    return _calculate_sqrt_kernel_value_f64(u, kernel_type, wj)

def calculate_sqrt_kernel_value_f32(
    np.float32_t u,
    int kernel_type,
    np.float32_t wj
) -> np.float32_t:
    if kernel_type >= 5 and abs(u) > 1.0:
        return 0.0
    return _calculate_sqrt_kernel_value_f32(u, kernel_type, wj)
