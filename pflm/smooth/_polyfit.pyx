import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport abs, pow, sqrt, NAN
from libc.stddef cimport ptrdiff_t
from libcpp.iterator cimport distance
from libcpp.algorithm cimport lower_bound
from libc.stdlib cimport malloc, free
from pflm.utils._lapack_helper cimport _gels_helper


cdef extern from "src/polyfit_helper.h" nogil:
    np.float64_t _calculate_kernel_value_f64 "calculate_kernel_value_f64"(np.float64_t, int)
    np.float32_t _calculate_kernel_value_f32 "calculate_kernel_value_f32"(np.float32_t, int)
    double[11] factorials


def search_lower_bound_f64(
    np.float64_t[:] a,
    np.float64_t target
) -> ptrdiff_t:
    """search_lower_bound_f64(a, target) -> ptrdiff_t (test only)"""
    cdef ptrdiff_t n = a.shape[0]
    cdef np.float64_t[:] a_ptr = a
    cdef np.float64_t *it = lower_bound(&a_ptr[0], &a_ptr[0] + n, target)
    if it == &a_ptr[0] + n:
        return -1
    else:
        return distance(&a_ptr[0], it)


def search_location_f64(
    np.float64_t[:] a,
    np.float64_t target
) -> ptrdiff_t:
    """search_location_f64(a, target) -> ptrdiff_t (test only)"""
    cdef ptrdiff_t n = a.shape[0]
    cdef np.float64_t[:] a_ptr = a
    cdef np.float64_t *it = lower_bound(&a_ptr[0], &a_ptr[0] + n, target)
    if it != &a_ptr[0] + n and it[0] == target:
        return distance(&a_ptr[0], it)
    else:
        return -1


def calculate_kernel_value_f64(
    np.float64_t u,
    int kernel_type
) -> np.float64_t:
    if kernel_type >= 100 and kernel_type <= 106 and abs(u) > 1.0:
        return 0.0
    if kernel_type == 106 and abs(abs(u) - 1.0) <= 1e-7:
        return 0.0
    return _calculate_kernel_value_f64(u, kernel_type)


def calculate_kernel_value_f32(
    np.float32_t u,
    int kernel_type
) -> np.float32_t:
    if kernel_type >= 100 and kernel_type <= 106 and abs(u) > 1.0:
        return 0.0
    if kernel_type == 106 and abs(abs(u) - 1.0) <= 1e-7:
        return 0.0
    return _calculate_kernel_value_f32(u, kernel_type)


cdef void polyfit1d_helper_f64(
    np.float64_t bw,
    np.float64_t center,
    np.float64_t* mu,
    np.float64_t[:] x,
    np.float64_t[:] y,
    np.float64_t[:] w,
    int degree,
    int deriv,
    int kernel_type
) noexcept nogil:
    cdef ptrdiff_t n = x.shape[0], left = 0, right = n
    cdef np.float64_t *left_it
    cdef np.float64_t *right_it
    if kernel_type >= 4:
        left_it = lower_bound(&x[0], &x[0] + n, center - bw)
        if left_it != &x[0] + n:
            left = distance(&x[0], left_it)
        right_it = lower_bound(&x[0], &x[0] + n, center + bw)
        if right_it == &x[0] + n:
            right = n
        else:
            right = distance(&x[0], right_it)
        if (left >= right) or (right - left - 1 <= deriv):
            mu[0] = NAN
            return

    cdef ptrdiff_t n_rows = right - left, i = 0, j, k
    cdef np.float64_t inv_deriv = 1.0 if deriv == 0.0 else 1.0 / deriv
    cdef np.float64_t xj_minus_center = 0.0, u = 0.0, sqrt_wj = 0.0
    cdef np.float64_t *lx = <np.float64_t*> malloc(n_rows * (degree+1) * sizeof(np.float64_t))
    cdef np.float64_t *ly = <np.float64_t*> malloc(n_rows * sizeof(np.float64_t))
    for j in range(left, right):
        xj_minus_center = x[j] - center
        u = xj_minus_center / bw
        sqrt_wj = sqrt(_calculate_kernel_value_f32(u, kernel_type) * w[j])

        ly[i] = y[j] * sqrt_wj
        lx[i] = sqrt_wj
        for k in range(1, degree + 1):
            lx[i + n_rows * k] = pow(xj_minus_center, k) * sqrt_wj
        i += 1
    cdef int info = 0
    _gels_helper(
        110, n_rows, degree + 1, 1,
        lx, n_rows,
        ly, n_rows,
        &info
    )
    if info == 0:
        mu[0] = ly[deriv] * factorials[deriv] * inv_deriv
    else:
        mu[0] = NAN
    free(lx)
    free(ly)


cdef void polyfit1d_memview_f64(
    np.float64_t[:] x,
    np.float64_t[:] y,
    np.float64_t[:] w,
    np.float64_t[:] x_new,
    np.float64_t[:] mu,
    np.float64_t bandwidth,
    int kernel_type,
    int degree,
    int deriv
) noexcept nogil:
    cdef ptrdiff_t i, x_new_size = x_new.shape[0]
    for i in prange(x_new_size, nogil=True):
        polyfit1d_helper_f64(bandwidth, x_new[i], &mu[i], x, y, w, degree, deriv, kernel_type)


def polyfit1d_f64(
    np.ndarray[np.float64_t] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] w,
    np.ndarray[np.float64_t] x_new,
    np.float64_t bandwidth,
    int kernel_type,
    int degree,
    int deriv
) -> np.ndarray[np.float64_t]:
    cdef np.ndarray[np.float64_t] mu = np.empty(x_new.size, dtype=np.float64)
    polyfit1d_memview_f64(x, y, w, x_new, mu, bandwidth, kernel_type, degree, deriv)
    return mu


cdef void polyfit1d_helper_f32(
    np.float32_t bw,
    np.float32_t center,
    np.float32_t* mu,
    np.float32_t[:] x,
    np.float32_t[:] y,
    np.float32_t[:] w,
    int degree,
    int deriv,
    int kernel_type
) noexcept nogil:
    cdef ptrdiff_t n = x.shape[0], left = 0, right = n
    cdef np.float32_t *left_it
    cdef np.float32_t *right_it
    if kernel_type >= 4:
        left_it = lower_bound(&x[0], &x[0] + n, center - bw)
        if left_it != &x[0] + n:
            left = distance(&x[0], left_it)
        right_it = lower_bound(&x[0], &x[0] + n, center + bw)
        if right_it == &x[0] + n:
            right = n
        else:
            right = distance(&x[0], right_it)
        if (left >= right) or (right - left - 1 <= deriv):
            mu[0] = NAN
            return

    cdef ptrdiff_t n_rows = right - left, i = 0, j, k
    cdef np.float32_t inv_deriv = 1.0 if deriv == 0.0 else 1.0 / deriv
    cdef np.float32_t xj_minus_center = 0.0, u = 0.0, sqrt_wj = 0.0
    cdef np.float32_t *lx = <np.float32_t*> malloc(n_rows * (degree+1) * sizeof(np.float32_t))
    cdef np.float32_t *ly = <np.float32_t*> malloc(n_rows * sizeof(np.float32_t))
    for j in range(left, right):
        xj_minus_center = x[j] - center
        u = xj_minus_center / bw
        sqrt_wj = sqrt(_calculate_kernel_value_f32(u, kernel_type) * w[j])

        ly[i] = y[j] * sqrt_wj
        lx[i] = sqrt_wj
        for k in range(1, degree + 1):
            lx[i + n_rows * k] = pow(xj_minus_center, k) * sqrt_wj
        i += 1
    cdef int info = 0
    _gels_helper(
        110, n_rows, degree + 1, 1,
        lx, n_rows,
        ly, n_rows,
        &info
    )
    if info == 0:
        mu[0] = ly[deriv] * factorials[deriv] * inv_deriv
    else:
        mu[0] = NAN
    free(lx)
    free(ly)


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
    cdef ptrdiff_t i, x_new_size = x_new.shape[0]
    for i in prange(x_new_size, nogil=True):
        polyfit1d_helper_f32(bandwidth, x_new[i], &mu[i], x, y, w, degree, deriv, kernel)


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


cdef void polyfit2d_helper_f64(
    np.float64_t bw1,
    np.float64_t bw2,
    np.float64_t center1,
    np.float64_t center2,
    np.float64_t* mu,
    np.float64_t[:, ::1] x_grid,
    np.float64_t[:] y,
    np.float64_t[:] w,
    int kernel_type,
    int degree = 1,
    int deriv1 = 0,
    int deriv2 = 0
) noexcept nogil:
    cdef ptrdiff_t n = x_grid.shape[0], left = 0, right = n
    cdef np.float64_t *left_it
    cdef np.float64_t *right_it
    if kernel_type >= 4:
        left_it = lower_bound(&x_grid[0, 0], &x_grid[0, 0] + n, center1 - bw1)
        if left_it != &x_grid[0, 0] + n:
            left = distance(&x_grid[0, 0], left_it)
        right_it = lower_bound(&x_grid[0, 0], &x_grid[0, 0] + n, center1 + bw1)
        if right_it == &x_grid[0, 0] + n:
            right = n
        else:
            right = distance(&x_grid[0, 0], right_it)
        if (left >= right) or (right - left - 1 <= 0):
            mu[0] = NAN
            return


cdef void polyfit2d_memview_f64(
    np.float64_t[:, ::1] x_grid,
    np.float64_t[:] y,
    np.float64_t[:] w,
    np.float64_t[:] x_new1,
    np.float64_t[:] x_new2,
    np.float64_t[:] mu,
    np.float64_t bandwidth1,
    np.float64_t bandwidth2,
    int kernel_type,
    int degree = 1,
    int deriv1 = 0,
    int deriv2 = 0
) noexcept nogil:
    cdef ptrdiff_t n = x_grid.shape[0], n1 = x_new1.shape[0], n2 = x_new2.shape[0]
    # Implement the logic for 2D polynomial fitting here
    pass


def polyfit2d_f64(
    np.ndarray[np.float64_t, ndim=2] x_grid,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] w,
    np.ndarray[np.float64_t] x_new1,
    np.ndarray[np.float64_t] x_new2,
    np.float64_t bandwidth1,
    np.float64_t bandwidth2,
    int kernel_type,
    int degree = 1,
    int deriv1 = 0,
    int deriv2 = 0
) -> np.ndarray[np.float64_t]:
    cdef ptrdiff_t n = x_grid.shape[0], n1 = x_new1.shape[0], n2 = x_new2.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] mu = np.empty((n2, n1), dtype=np.float64)
    polyfit2d_memview_f64(x_grid, y, w, x_new1, x_new2, mu, bandwidth1, bandwidth2, kernel_type, degree, deriv1, deriv2)
    return mu


cdef void polyfit2d_memview_f32(
    np.float32_t[:, ::1] x_grid,
    np.float32_t[:] y,
    np.float32_t[:] w,
    np.float32_t[:] x_new1,
    np.float32_t[:] x_new2,
    np.float32_t[:] mu,
    np.float32_t bandwidth1,
    np.float32_t bandwidth2,
    int kernel_type,
    int degree = 1,
    int deriv1 = 0,
    int deriv2 = 0
) noexcept nogil:
    cdef ptrdiff_t n = x_grid.shape[0], n1 = x_new1.shape[0], n2 = x_new2.shape[0]
    # Implement the logic for 2D polynomial fitting here
    pass


def polyfit2d_f32(
    np.ndarray[np.float32_t, ndim=2] x_grid,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] w,
    np.ndarray[np.float32_t] x_new1,
    np.ndarray[np.float32_t] x_new2,
    np.float32_t bandwidth1,
    np.float32_t bandwidth2,
    int kernel_type,
    int degree = 1,
    int deriv1 = 0,
    int deriv2 = 0
) -> np.ndarray[np.float32_t]:
    cdef ptrdiff_t n = x_grid.shape[0], n1 = x_new1.shape[0], n2 = x_new2.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] mu = np.empty((n2, n1), dtype=np.float32)
    polyfit2d_memview_f32(x_grid, y, w, x_new1, x_new2, mu, bandwidth1, bandwidth2, kernel_type, degree, deriv1, deriv2)
    return mu
