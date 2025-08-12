import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport abs, pow, sqrt, NAN, acos, exp, cosh, cos
from libc.stdint cimport int64_t
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.utility cimport pair
from libcpp.iterator cimport distance
from libcpp.algorithm cimport lower_bound
from libc.stdlib cimport malloc, free
from pflm.utils._lapack_helper cimport _gels_helper, _gelss_helper

cdef double half_pi = acos(0.0)
cdef double quarter_pi = half_pi / 2.0
cdef double inv_sqrt_2 = 1.0 / sqrt(2.0)
cdef double inv_2pi = 1.0 / (2.0 * acos(-1.0))
cdef double inv_sqrt_2pi = sqrt(inv_2pi)
cdef double[11] factorials = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0]


cdef floating calculate_kernel_value(
    floating u,
    int kernel_type
) noexcept nogil:
    if kernel_type == 0: # GAUSSIAN
        return inv_sqrt_2pi * exp(-0.5 * u * u)
    elif kernel_type == 1: # LOGISTIC
        return 0.5 / (cosh(u) + 1.0)
    elif kernel_type == 2: # SIGMOID
        return 1.0 / cosh(u) / acos(-1.0)
    # Shifted Gaussian kernel is not included since it might produce negative weights that are not supported in our implementation.
    # elif kernel_type == 3: # GAUSSIAN_VAR
    #     cdef floating u_sq = u * u;
    #     return inv_sqrt_2pi * exp(-0.5 * u_sq) * (1.25 - 0.25 * u_sq);
    # Silverman kernel is not included since it might produce negative weights that are not supported in our implementation.
    # elif kernel_type == 4:  # SILVERMAN
    #     cdef floating temp = abs(u) * inv_sqrt_2;
    #     return 0.5 * exp(-0.5 * temp) * sin(temp + quarter_pi);
    elif kernel_type == 100:  # RECTANGULAR
        return 0.5
    elif kernel_type == 101:  # TRIANGULAR
        return (1.0 - abs(u))
    elif kernel_type == 102:  # EPANECHNIKOV
        return 0.75 * (1.0 - u * u)
    elif kernel_type == 103:  # BIWEIGHT
        return 15.0 / 16.0 * pow(1.0 - u * u, 2.0)
    elif kernel_type == 104:  # TRIWEIGHT
        return 35.0 / 32.0 * pow(1.0 - u * u, 3.0)
    elif kernel_type == 105:  # TRICUBE
        return 70.0 / 81.0 * pow(1.0 - pow(abs(u), 3.0), 3.0)
    elif kernel_type == 106:  # COSINE
        return quarter_pi * cos(half_pi * u)
    else:
        return 0.0


def calculate_kernel_value_f64(
    np.float64_t u,
    int kernel_type
) -> np.float64_t:
    if kernel_type >= 100 and kernel_type <= 106 and abs(u) > 1.0:
        return 0.0
    if kernel_type == 106 and abs(abs(u) - 1.0) <= 1e-7:
        return 0.0
    return calculate_kernel_value(u, kernel_type)


def calculate_kernel_value_f32(
    np.float32_t u,
    int kernel_type
) -> np.float32_t:
    if kernel_type >= 100 and kernel_type <= 106 and abs(u) > 1.0:
        return 0.0
    if kernel_type == 106 and abs(abs(u) - 1.0) <= 1e-7:
        return 0.0
    return calculate_kernel_value(u, kernel_type)


cdef void polyfit1d_helper(
    floating bw,
    floating center,
    floating* mu,
    floating[:] x,
    floating[:] y,
    floating[:] w,
    int degree,
    int deriv,
    int kernel_type
) noexcept nogil:
    cdef int64_t n = x.shape[0], left = 0, right = n
    cdef floating *left_it
    cdef floating *right_it
    cdef floating *x_start = &x[0]
    cdef floating *x_end = &x[0] + n
    cdef floating lb = center - bw, ub = center + bw
    if kernel_type >= 100:
        left_it = lower_bound(x_start, x_end, lb)
        right_it = lower_bound(x_start, x_end, ub)
        left = distance(x_start, left_it)
        right = distance(x_start, right_it) if right_it != x_end else n
        if (left > right) or (right - left - 1 <= deriv):
            mu[0] = NAN
            return

    cdef int64_t n_rows = right - left, i = 0, j, k
    cdef floating inv_deriv = 1.0 if deriv == 0.0 else 1.0 / deriv
    cdef floating xj_minus_center = 0.0, u = 0.0, sqrt_wj = 0.0
    cdef floating *lx = <floating*> malloc(n_rows * (degree+1) * sizeof(floating))
    cdef floating *ly = <floating*> malloc(n_rows * sizeof(floating))
    for j in range(left, right):
        xj_minus_center = x[j] - center
        u = xj_minus_center / bw
        sqrt_wj = sqrt(calculate_kernel_value(u, kernel_type) * w[j])

        ly[i] = y[j] * sqrt_wj
        for k in range(degree + 1):
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


cdef void polyfit1d_memview(
    floating[:] x,
    floating[:] y,
    floating[:] w,
    floating[:] x_new,
    floating[:] mu,
    floating bandwidth,
    int kernel_type,
    int degree,
    int deriv
) noexcept nogil:
    cdef int64_t i, x_new_size = x_new.shape[0]
    for i in prange(x_new_size, nogil=True):
        polyfit1d_helper(bandwidth, x_new[i], &mu[i], x, y, w, degree, deriv, kernel_type)


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
    cdef np.float64_t[:] x_view = x
    cdef np.float64_t[:] y_view = y
    cdef np.float64_t[:] w_view = w
    cdef np.float64_t[:] x_new_view = x_new
    cdef np.float64_t[:] mu_view = mu
    polyfit1d_memview(x_view, y_view, w_view, x_new_view, mu_view, bandwidth, kernel_type, degree, deriv)
    return mu

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
    cdef np.float32_t[:] x_view = x
    cdef np.float32_t[:] y_view = y
    cdef np.float32_t[:] w_view = w
    cdef np.float32_t[:] x_new_view = x_new
    cdef np.float32_t[:] mu_view = mu
    polyfit1d_memview(x_view, y_view, w_view, x_new_view, mu_view, bandwidth, kernel, degree, deriv)
    return mu


cdef void polyfit2d_helper(
    floating bw1,
    floating bw2,
    floating center1,
    floating center2,
    floating* mu,
    floating[:, ::1] x_grid,
    floating[:] y,
    floating[:] w,
    int kernel_type,
    int degree,
    int deriv1,
    int deriv2
) noexcept nogil:
    cdef int64_t n = x_grid.shape[1], left = 0, right = n, i
    cdef floating *left_it
    cdef floating *right_it
    cdef floating *x1_start = &x_grid[0, 0]
    cdef floating *x1_end = &x_grid[0, 0] + n
    cdef floating lb1, ub1, lb2, ub2
    cdef int64_t num_lx_cols = (degree + 1) * (degree + 2) // 2
    cdef vector[int64_t] idx = vector[int64_t]()
    cdef bint check_rank = 1, use_svd = 0
    if kernel_type >= 100:
        lb1 = center1 - bw1 - 1e-6
        ub1 = center1 + bw1 + 1e-6
        left_it = lower_bound(x1_start, x1_end, lb1)
        right_it = lower_bound(x1_start, x1_end, ub1)
        left = distance(x1_start, left_it)
        right = distance(x1_start, right_it) if right_it != x1_end else n
        if left >= right:
            mu[0] = NAN
            return

        lb2 = center2 - bw2 - 1e-6
        ub2 = center2 + bw2 + 1e-6
        for i in range(left, right):
            if x_grid[1, i] > lb2 and x_grid[1, i] < ub2:
                idx.push_back(i)


        if idx.empty():
            mu[0] = NAN
            return

        if idx.size() < num_lx_cols:
            check_rank = 0
            use_svd = 1
    else:
        idx.resize(n)
        for i in range(n):
            idx[i] = i
        check_rank = 0
        use_svd = 0

    cdef int64_t n_rows = <int64_t> idx.size(), j
    cdef set[pair[floating, floating]] unique_grid_points
    if check_rank == 1:
        for i in range(n_rows):
            j = idx[i]
            unique_grid_points.insert(pair[floating, floating](x_grid[0, j], x_grid[1, j]))
        if unique_grid_points.size() < num_lx_cols:
            use_svd = 1
        else:
            use_svd = 0

    cdef int64_t total_deg, py, col_idx
    cdef floating *lx = <floating*> malloc(n_rows * num_lx_cols * sizeof(floating))
    cdef floating *ly = <floating*> malloc(n_rows * sizeof(floating))
    cdef floating x1j_minus_center1, x2j_minus_center2, u1, u2, sqrt_wj
    for i in range(n_rows):
        j = idx[i]
        x1j_minus_center1 = x_grid[0, j] - center1
        x2j_minus_center2 = x_grid[1, j] - center2
        u1 = x1j_minus_center1 / bw1
        u2 = x2j_minus_center2 / bw2
        sqrt_wj = sqrt(calculate_kernel_value(u1, kernel_type) * calculate_kernel_value(u2, kernel_type) * w[j])

        ly[i] = y[j] * sqrt_wj
        for total_deg in range(degree + 1):
            for py in range(total_deg + 1):
                col_idx = total_deg * (total_deg + 1) // 2 + py
                lx[i + n_rows * col_idx] = pow(x1j_minus_center1, total_deg - py) * pow(x2j_minus_center2, py) * sqrt_wj

    cdef int info = 0, rank = 0
    cdef floating rcond = -1.0
    if use_svd == 1:
        _gelss_helper(
            n_rows, num_lx_cols, 1,
            lx, n_rows,
            ly, n_rows,
            &rcond, &rank, &info
        )
    else:
        _gels_helper(
            110, n_rows, num_lx_cols, 1,
            lx, n_rows,
            ly, n_rows,
            &info
        )

    cdef int64_t total_deriv = <int64_t> deriv1 + <int64_t> deriv2
    cdef int64_t mu_idx = total_deriv * (total_deriv + 1) // 2 + deriv2
    if info == 0:
        mu[0] = ly[mu_idx] * factorials[deriv1] * factorials[deriv2]
    else:
        mu[0] = NAN
    free(lx)
    free(ly)


cdef void polyfit2d_memview(
    floating[:, ::1] x_grid,
    floating[:] y,
    floating[:] w,
    floating[:] x_new1,
    floating[:] x_new2,
    floating[:, ::1] mu,
    floating bandwidth1,
    floating bandwidth2,
    int kernel_type,
    int degree,
    int deriv1,
    int deriv2
) noexcept nogil:
    cdef int64_t n = x_grid.shape[1], n1 = x_new1.shape[0], n2 = x_new2.shape[0], i, j, l
    for l in prange(n1 * n2, nogil=True):
        i = l // n2
        j = l % n2
        polyfit2d_helper(
            bandwidth1, bandwidth2,
            x_new1[i], x_new2[j],
            &mu[j, i],
            x_grid, y, w,
            kernel_type, degree, deriv1, deriv2
        )


def polyfit2d_f64(
    np.ndarray[np.float64_t, ndim=2] x_grid,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] w,
    np.ndarray[np.float64_t] x_new1,
    np.ndarray[np.float64_t] x_new2,
    np.float64_t bandwidth1,
    np.float64_t bandwidth2,
    int kernel_type,
    int degree,
    int deriv1,
    int deriv2
) -> np.ndarray[np.float64_t]:
    cdef int64_t n1 = x_new1.shape[0], n2 = x_new2.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] mu = np.empty((n2, n1), order='C', dtype=np.float64)
    cdef np.float64_t[:, ::1] x_grid_view = x_grid
    cdef np.float64_t[:] y_view = y
    cdef np.float64_t[:] w_view = w
    cdef np.float64_t[:] x_new1_view = x_new1
    cdef np.float64_t[:] x_new2_view = x_new2
    cdef np.float64_t[:, ::1] mu_view = mu
    polyfit2d_memview(x_grid_view, y_view, w_view, x_new1_view, x_new2_view, mu_view, bandwidth1, bandwidth2, kernel_type, degree, deriv1, deriv2)
    return mu.T


def polyfit2d_f32(
    np.ndarray[np.float32_t, ndim=2] x_grid,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] w,
    np.ndarray[np.float32_t] x_new1,
    np.ndarray[np.float32_t] x_new2,
    np.float32_t bandwidth1,
    np.float32_t bandwidth2,
    int kernel_type,
    int degree,
    int deriv1,
    int deriv2
) -> np.ndarray[np.float32_t]:
    cdef int64_t n1 = x_new1.shape[0], n2 = x_new2.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] mu = np.empty((n2, n1), order='C', dtype=np.float32)
    cdef np.float32_t[:, ::1] x_grid_view = x_grid
    cdef np.float32_t[:] y_view = y
    cdef np.float32_t[:] w_view = w
    cdef np.float32_t[:] x_new1_view = x_new1
    cdef np.float32_t[:] x_new2_view = x_new2
    cdef np.float32_t[:, ::1] mu_view = mu
    polyfit2d_memview(x_grid_view, y_view, w_view, x_new1_view, x_new2_view, mu_view, bandwidth1, bandwidth2, kernel_type, degree, deriv1, deriv2)
    return mu.T
