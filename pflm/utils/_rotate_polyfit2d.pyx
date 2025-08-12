import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport pow, sqrt, NAN
from libc.stdint cimport int64_t
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.utility cimport pair
from libcpp.iterator cimport distance
from libcpp.algorithm cimport lower_bound
from libc.stdlib cimport malloc, free
from pflm.smooth._polyfit cimport inv_sqrt_2, calculate_kernel_value
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
) noexcept nogil:
    cdef int64_t n = x_grid.shape[1], left = 0, right = n, i
    cdef floating *left_it
    cdef floating *right_it
    cdef floating *x1_start = &x_grid[0, 0]
    cdef floating *x1_end = &x_grid[0, 0] + n
    cdef floating lb1, ub1, lb2, ub2
    cdef vector[int64_t] idx = vector[int64_t]()
    cdef bint check_rank = 1, use_svd = 0
    if kernel_type >= 100:
        lb1 = center1 - bw - 1e-6
        ub1 = center1 + bw + 1e-6
        left_it = lower_bound(x1_start, x1_end, lb1)
        right_it = lower_bound(x1_start, x1_end, ub1)
        left = distance(x1_start, left_it)
        right = distance(x1_start, right_it) if right_it != x1_end else n
        if left >= right:
            mu[0] = NAN
            return

        lb2 = center2 - bw - 1e-6
        ub2 = center2 + bw + 1e-6
        for i in range(left, right):
            if x_grid[1, i] > lb2 and x_grid[1, i] < ub2:
                idx.push_back(i)

        if idx.empty():
            mu[0] = NAN
            return

        if idx.size() < 3:
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
        if unique_grid_points.size() < 3:
            use_svd = 1
        else:
            use_svd = 0

    cdef int64_t total_deg, py, col_idx
    cdef floating *lx = <floating*> malloc(n_rows * 3 * sizeof(floating))
    cdef floating *ly = <floating*> malloc(n_rows * sizeof(floating))
    cdef floating x1j_minus_center1, x2j_minus_center2, u1, u2, sqrt_wj
    for i in range(n_rows):
        j = idx[i]
        x1j_minus_center1 = x_grid[0, j] - center1
        x2j_minus_center2 = x_grid[1, j] - center2
        u1 = x1j_minus_center1 / bw
        u2 = x2j_minus_center2 / bw
        sqrt_wj = sqrt(calculate_kernel_value(u1, kernel_type) * calculate_kernel_value(u2, kernel_type) * w[j])

        ly[i] = y[j] * sqrt_wj
        lx[i] = sqrt_wj
        lx[i + n_rows] = pow(x1j_minus_center1 * sqrt_wj, 2.0)
        lx[i + n_rows * 2] = x2j_minus_center2 * sqrt_wj


    cdef int info = 0, rank = 0
    cdef floating rcond = -1.0
    if use_svd == 1:
        _gelss_helper(
            n_rows, 3, 1,
            lx, n_rows,
            ly, n_rows,
            &rcond, &rank, &info
        )
    else:
        _gels_helper(
            110, n_rows, 3, 1,
            lx, n_rows,
            ly, n_rows,
            &info
        )

    if info == 0:
        mu[0] = ly[0]
    else:
        mu[0] = NAN
    free(lx)
    free(ly)


cdef void rotate_polyfit2d_memview(
    floating[:, ::1] x_grid,
    floating[:] y,
    floating[:] w,
    floating[:, ::1] new_grid,
    floating[:] mu,
    floating bandwidth,
    int kernel_type
) noexcept nogil:
    cdef int64_t i, x_new_size = new_grid.shape[1]
    for i in prange(x_new_size, nogil=True):
        rotate_polyfit2d_helper(bandwidth, new_grid[0, i], new_grid[1, i], &mu[i], x_grid, y, w, kernel_type)


def rotate_polyfit2d_f64(
    np.ndarray[np.float64_t, ndim=2] x_grid,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] w,
    np.ndarray[np.float64_t, ndim=2] new_grid,
    np.float64_t bandwidth,
    int kernel_type
):
    cdef int64_t n_new = new_grid.shape[1]
    cdef np.ndarray[np.float64_t] mu = np.empty(n_new, dtype=np.float64)
    cdef np.float64_t[:, ::1] x_grid_view = x_grid
    cdef np.float64_t[:] y_view = y
    cdef np.float64_t[:] w_view = w
    cdef np.float64_t[:, ::1] new_grid_view = new_grid
    cdef np.float64_t[:] mu_view = mu
    rotate_polyfit2d_memview(x_grid_view, y_view, w_view, new_grid_view, mu_view, bandwidth, kernel_type)
    return mu


def rotate_polyfit2d_f32(
    np.ndarray[np.float32_t, ndim=2] x_grid,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] w,
    np.ndarray[np.float32_t, ndim=2] new_grid,
    np.float32_t bandwidth,
    int kernel_type
):
    cdef int64_t n_new = new_grid.shape[1]
    cdef np.ndarray[np.float32_t] mu = np.empty(n_new, dtype=np.float32)
    cdef np.float32_t[:, ::1] x_grid_view = x_grid
    cdef np.float32_t[:] y_view = y
    cdef np.float32_t[:] w_view = w
    cdef np.float32_t[:, ::1] new_grid_view = new_grid
    cdef np.float32_t[:] mu_view = mu
    rotate_polyfit2d_memview(x_grid_view, y_view, w_view, new_grid_view, mu_view, bandwidth, kernel_type)
    return mu
