import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport pow, sqrt, NAN
from libc.stdint cimport int64_t, uint64_t
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.utility cimport pair
from libcpp.iterator cimport distance
from libcpp.algorithm cimport lower_bound
from libc.stdlib cimport malloc, free
from pflm.smooth._polyfit cimport calculate_kernel_value
from pflm.utils._lapack_helper cimport _gels_helper, _gelss_helper


cdef void rotate_polyfit2d_helper(
    floating bw,
    floating center1,
    floating center2,
    floating* mu,
    floating[:, ::1] x_grid,
    floating[:] y,
    floating[:] w,
    int kernel_type
):
    cdef uint64_t n = x_grid.shape[1]
    cdef int64_t left = 0, right = <int64_t> n, i
    cdef floating *left_it
    cdef floating *right_it
    cdef floating *x1_start = &x_grid[0, 0]
    cdef floating *x1_end = &x_grid[0, 0] + n
    cdef floating lb1, ub1, lb2, ub2, epsilon = <floating> 1e-6
    cdef uint64_t num_lx_cols = 3
    cdef vector[int64_t] idx = vector[int64_t]()
    cdef bint check_rank = 1, use_svd = 0
    if kernel_type >= 100:
        lb1 = center1 - bw - epsilon
        ub1 = center1 + bw + epsilon
        left_it = lower_bound(x1_start, x1_end, lb1)
        right_it = lower_bound(x1_start, x1_end, ub1)
        left = distance(x1_start, left_it)
        right = distance(x1_start, right_it) if right_it != x1_end else <int64_t> n
        if left >= right:
            mu[0] = NAN
            return

        lb2 = center2 - bw - epsilon
        ub2 = center2 + bw + epsilon
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

    cdef uint64_t n_rows = idx.size()
    cdef set[pair[floating, floating]] unique_grid_points
    if check_rank == 1:
        for i in range(n_rows):
            unique_grid_points.insert(pair[floating, floating](x_grid[0, idx[i]], x_grid[1, idx[i]]))
        if unique_grid_points.size() < num_lx_cols:
            use_svd = 1
        else:
            use_svd = 0

    cdef int total_deg, py, col_idx
    cdef int64_t j
    cdef floating *lx = <floating*> malloc(n_rows * 3 * sizeof(floating))
    cdef floating *ly = <floating*> malloc(n_rows * sizeof(floating))
    cdef floating x1j_minus_center1, x2j_minus_center2, u1, u2, sqrt_wj
    for i in range(<int64_t> n_rows):
        j = idx[i]
        x1j_minus_center1 = x_grid[0, j] - center1
        x2j_minus_center2 = x_grid[1, j] - center2
        u1 = x1j_minus_center1 / bw
        u2 = x2j_minus_center2 / bw
        sqrt_wj = sqrt(calculate_kernel_value(u1, kernel_type) * calculate_kernel_value(u2, kernel_type) * w[j])

        ly[i] = y[j] * sqrt_wj
        lx[i] = sqrt_wj
        lx[i + n_rows] = pow(x1j_minus_center1, <floating> 2.0) * sqrt_wj
        lx[i + n_rows * 2] = x2j_minus_center2 * sqrt_wj

    cdef int info = 0, rank = 0
    cdef floating rcond = -1.0
    if use_svd == 1:
        _gelss_helper(
            <int> n_rows, 3, 1,
            lx, <int> n_rows,
            ly, <int> n_rows,
            &rcond, &rank, &info
        )
    else:
        _gels_helper(
            110, <int> n_rows, 3, 1,
            lx, <int> n_rows,
            ly, <int> n_rows,
            &info
        )

    if info == 0:
        mu[0] = ly[0]
    else:
        mu[0] = NAN
    free(lx)
    free(ly)


def rotate_polyfit2d_f64(
    np.ndarray[np.float64_t, ndim=2] x_grid,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] w,
    np.ndarray[np.float64_t, ndim=2] new_grid,
    np.float64_t bandwidth,
    int kernel_type
):
    cdef uint64_t n_new = new_grid.shape[1]
    cdef np.ndarray[np.float64_t] mu = np.empty(n_new, dtype=np.float64)
    cdef np.float64_t[:, ::1] x_grid_view = x_grid
    cdef np.float64_t[:] y_view = y
    cdef np.float64_t[:] w_view = w
    cdef np.float64_t[:, ::1] new_grid_view = new_grid
    cdef np.float64_t[:] mu_view = mu

    cdef int64_t i
    for i in range(<int64_t> n_new):
        rotate_polyfit2d_helper(bandwidth, new_grid_view[0, i], new_grid_view[1, i], &mu_view[i], x_grid_view, y_view, w_view, kernel_type)
    return mu


def rotate_polyfit2d_f32(
    np.ndarray[np.float32_t, ndim=2] x_grid,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] w,
    np.ndarray[np.float32_t, ndim=2] new_grid,
    np.float32_t bandwidth,
    int kernel_type
):
    cdef uint64_t n_new = new_grid.shape[1]
    cdef np.ndarray[np.float32_t] mu = np.empty(n_new, dtype=np.float32)
    cdef np.float32_t[:, ::1] x_grid_view = x_grid
    cdef np.float32_t[:] y_view = y
    cdef np.float32_t[:] w_view = w
    cdef np.float32_t[:, ::1] new_grid_view = new_grid
    cdef np.float32_t[:] mu_view = mu

    cdef int64_t i
    for i in range(<int64_t> n_new):
        rotate_polyfit2d_helper(bandwidth, new_grid_view[0, i], new_grid_view[1, i], &mu_view[i], x_grid_view, y_view, w_view, kernel_type)
    return mu
