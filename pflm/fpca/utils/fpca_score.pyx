import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel import prange
from libc.stdint cimport int64_t, uint64_t
from libc.math cimport NAN
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp.vector cimport vector
from pflm.utils.blas_helper cimport BLAS_Order, ColMajor, RowMajor, NoTrans, Trans, _gemv, _gemm
from pflm.utils.lapack_helper cimport _gels_helper, _gelss_helper
from pflm.utils.trapz cimport trapz


cdef void get_fitted_y_mat(
    BLAS_Order order,
    uint64_t nt,
    uint64_t num_unique_sid,
    uint64_t num_pcs,
    floating* fitted_y_mat,   # shape: (nt, num_unique_sid)
    floating* mu,             # shape: (nt, )
    floating* xi,             # shape: (num_unique_sid, num_pcs)
    floating* fpca_phi        # shape: (nt, num_pcs)
) noexcept nogil:
    # perform fitted_y = mu + xi @ fpca_phi.T
    cdef uint64_t phi_ld = nt if order == ColMajor else num_pcs
    cdef uint64_t xi_ld = num_unique_sid if order == ColMajor else num_pcs
    cdef uint64_t fitted_y_ld = nt if order == ColMajor else num_unique_sid
    _gemm(
        order, NoTrans, Trans,
        <int> nt, <int> num_unique_sid, <int> num_pcs, # m, n, k
        1.0, # alpha
        fpca_phi, <int> phi_ld, # A, lda
        xi, <int> xi_ld, # B, ldb
        0.0, # beta
        fitted_y_mat, <int> fitted_y_ld # C, ldc
    )

    cdef int64_t i, j
    if order == ColMajor:
        for j in prange(<int64_t> num_unique_sid, nogil=True):
            for i in range(<int64_t> nt):
                fitted_y_mat[j * nt + i] += mu[i]
    elif order == RowMajor:
        for i in prange(<int64_t> nt, nogil=True):
            for j in range(<int64_t> num_unique_sid):
                fitted_y_mat[i * num_unique_sid + j] += mu[i]


cdef void get_fitted_y_list(
    BLAS_Order order,
    uint64_t nt,
    uint64_t num_unique_sid,
    floating* fitted_y_mat,   # shape: (nt, num_unique_sid)
    uint64_t sid,
    uint64_t data_cnt,
    floating* fitted_y,       # shape: (data_cnt,)
    int64_t* tid
) noexcept nogil:
    cdef uint64_t i
    for i in range(data_cnt):
        if order == ColMajor:
            fitted_y[i] = fitted_y_mat[sid * num_unique_sid + tid[i]]
        elif order == RowMajor:
            fitted_y[i] = fitted_y_mat[tid[i] * nt + sid]


cdef void fpca_ce_score_helper(
    BLAS_Order order,
    uint64_t num_unique_sid,
    uint64_t nt,
    uint64_t num_pcs,
    uint64_t data_cnt,
    floating* xi,         # shape: (num_unique_sid, num_pcs) (pointer to xi[idx, 0])
    floating* xi_var,     # shape: (num_pcs, num_pcs)
    floating* yy,
    floating* tt,
    int64_t* tid,
    floating* mu,         # shape: (nt, )
    floating* sigma_y,    # shape: (nt, nt) symmetric
    floating* lambda_phi  # shape: (nt, num_pcs)
) noexcept nogil:
    cdef int64_t i, j
    cdef int64_t inc_xi = num_unique_sid if order == ColMajor else 1

    cdef floating* sub_lambda_phi = <floating*> malloc(num_pcs * data_cnt * sizeof(floating))        # shape: (data_cnt, num_pcs) col-major
    cdef floating* sub_sigma_y = <floating*> malloc(data_cnt * data_cnt * sizeof(floating))          # shape: (data_cnt, data_cnt) col-major, symmetric
    cdef floating* sub_sigma_y_copy = <floating*> malloc(data_cnt * data_cnt * sizeof(floating))     # shape: (data_cnt, data_cnt) col-major, symmetric
    cdef floating* sub_y_minus_mu = <floating*> malloc(data_cnt * sizeof(floating))                  # shape: (data_cnt, )
    if (not sub_lambda_phi) or (not sub_sigma_y) or (not sub_sigma_y_copy) or (not sub_y_minus_mu):
        free(sub_lambda_phi); free(sub_sigma_y); free(sub_sigma_y_copy); free(sub_y_minus_mu)
        for i in prange(<int64_t> num_pcs, nogil=True):
            xi[i*inc_xi] = NAN
            for j in range(<int64_t> num_pcs):
                xi_var[i * num_pcs + j] = NAN
        return
    for i in range(<int64_t> data_cnt):
        sub_y_minus_mu[i] = yy[i] - mu[tid[i]]
        for j in range(<int64_t> data_cnt):
            sub_sigma_y[i * data_cnt + j] = sigma_y[tid[i] * nt + tid[j]]

    # make a copy of sub_sigma_y since _sysv/_gelss will modify the input matrix
    memcpy(sub_sigma_y_copy, sub_sigma_y, data_cnt * data_cnt * sizeof(floating))

    if order == ColMajor:
        for j in range(<int64_t> num_pcs):
            for i in range(<int64_t> data_cnt):
                sub_lambda_phi[i + j * data_cnt] = lambda_phi[tid[i] + j * nt]
    elif order == RowMajor:
        for i in range(<int64_t> data_cnt):
            for j in range(<int64_t> num_pcs):
                sub_lambda_phi[i + j * data_cnt] = lambda_phi[tid[i] * num_pcs + j]

    cdef int info = 0, rank = 0
    cdef floating rcond = -1.0
    if num_pcs >= data_cnt:
        _gels_helper(ColMajor, NoTrans, <int> data_cnt, <int> data_cnt, 1, sub_sigma_y, <int> data_cnt, sub_y_minus_mu, <int> data_cnt, &info)
    else:
        _gelss_helper(ColMajor, <int> data_cnt, <int> data_cnt, 1, sub_sigma_y, <int> data_cnt, sub_y_minus_mu, <int> data_cnt, &rcond, &rank, &info)

    if info != 0:
        free(sub_lambda_phi); free(sub_sigma_y); free(sub_sigma_y_copy); free(sub_y_minus_mu)
        for i in prange(<int64_t> num_pcs, nogil=True):
            xi[i*inc_xi] = NAN
            for j in range(<int64_t> num_pcs):
                xi_var[i * num_pcs + j] = NAN
        return

    # compute xi
    _gemv(
        ColMajor, Trans,
        <int> data_cnt, <int> num_pcs, # m, n
        1.0, # alpha
        sub_lambda_phi, <int> data_cnt, # A, lda
        sub_y_minus_mu, 1, # X, inc_x
        0.0, # beta
        xi, <int> inc_xi # y, inc_y
    )
    free(sub_sigma_y)
    free(sub_y_minus_mu)

    # make a copy of sub_lambda_phi since _sysv/_gelss will modify the input matrix
    cdef floating* sub_lambda_phi_copy = <floating*> malloc(num_pcs * data_cnt * sizeof(floating))   # shape: (data_cnt, num_pcs) col-major
    memcpy(sub_lambda_phi_copy, sub_lambda_phi, num_pcs * data_cnt * sizeof(floating))

    info = 0; rank = 0; rcond = -1.0
    if num_pcs >= data_cnt:
        _gels_helper(ColMajor, NoTrans, <int> data_cnt, <int> data_cnt, num_pcs, sub_sigma_y_copy, <int> data_cnt, sub_lambda_phi_copy, <int> data_cnt, &info)
    else:
        _gelss_helper(ColMajor, <int> data_cnt, <int> data_cnt, num_pcs, sub_sigma_y_copy, <int> data_cnt, sub_lambda_phi_copy, <int> data_cnt, &rcond, &rank, &info)

    if info != 0:
        free(sub_lambda_phi); free(sub_sigma_y_copy)
        for i in prange(<int64_t> num_pcs, nogil=True):
            for j in range(<int64_t> num_pcs):
                xi_var[i * num_pcs + j] = NAN
        return

    # compute xi_var
    _gemm(
        ColMajor, Trans, NoTrans,
        <int> num_pcs, <int> num_pcs, <int> data_cnt, # m, n, k
        -1.0, # alpha
        sub_lambda_phi, <int> data_cnt, # A, lda
        sub_lambda_phi_copy, <int> data_cnt, # B, ldb
        1.0, # beta
        xi_var, <int> num_pcs # C, ldc
    )

    free(sub_lambda_phi)
    free(sub_lambda_phi_copy)
    free(sub_sigma_y_copy)


def fpca_ce_score_f64(
    np.ndarray[np.float64_t] yy,
    np.ndarray[np.float64_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float64_t] mu,
    np.ndarray[np.float64_t, ndim=2] sigma_y,
    np.ndarray[np.float64_t] fpca_lambda,
    np.ndarray[np.float64_t, ndim=2] fpca_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
):
    cdef uint64_t num_unique_sid = unique_sid.size, num_pcs = fpca_phi.shape[1], nt = mu.size
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef BLAS_Order order = ColMajor if fpca_phi.flags.f_contiguous else RowMajor
    cdef str order_char = "F" if order == ColMajor else "C"

    cdef np.float64_t[:] yy_view = yy
    cdef np.float64_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef np.float64_t[:] mu_view = mu
    cdef np.float64_t[:, :] sigma_y_view = sigma_y
    cdef np.float64_t[:] fpca_lambda_view = fpca_lambda
    cdef np.float64_t[:, :] fpca_phi_view = fpca_phi

    cdef list xi_var = []
    cdef list fitted_y = []
    cdef vector[np.float64_t*] xi_var_ptrs
    cdef vector[np.float64_t*] fitted_y_ptrs
    cdef np.ndarray[np.float64_t, ndim=2] temp_xi_var
    cdef np.ndarray[np.float64_t] temp_fitted_y
    cdef int64_t i, j
    cdef uint64_t data_start_idx, data_cnt
    for i in range(<int64_t> num_unique_sid):
        temp_xi_var = np.diag(fpca_lambda)
        xi_var.append(temp_xi_var)
        xi_var_ptrs.push_back(<np.float64_t*> &temp_xi_var.data[0])

        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        temp_fitted_y = np.zeros((data_cnt,), dtype=np.float64)
        fitted_y_ptrs.push_back(<np.float64_t*> &temp_fitted_y.data[0])
        fitted_y.append(temp_fitted_y)

    cdef np.ndarray[np.float64_t, ndim=2] lambda_phi = np.zeros((nt, num_pcs), order=order_char, dtype=np.float64)
    cdef np.float64_t[:, :] lambda_phi_view = lambda_phi
    for j in range(<int64_t> num_pcs):
        for i in prange(<int64_t> nt, nogil=True):
            lambda_phi_view[i, j] = fpca_phi_view[i, j] * fpca_lambda_view[j]

    cdef np.ndarray[np.float64_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order=order_char, dtype=np.float64)
    cdef np.float64_t[:, :] xi_view = xi
    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        fpca_ce_score_helper(
            order, num_unique_sid, nt, num_pcs, data_cnt,
            &xi_view[i, 0], xi_var_ptrs[i], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &sigma_y_view[0, 0], &lambda_phi_view[0, 0]
        )

    cdef np.ndarray[np.float64_t, ndim=2] fitted_y_mat = np.zeros((nt, num_unique_sid), order=order_char, dtype=np.float64)
    cdef np.float64_t[:, :] fitted_y_mat_view = fitted_y_mat
    get_fitted_y_mat(
        order, nt, num_unique_sid, num_pcs,
        &fitted_y_mat_view[0, 0], &mu_view[0], &xi_view[0, 0], &fpca_phi_view[0, 0]
    )

    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        get_fitted_y_list(
            order, nt, num_unique_sid, &fitted_y_mat_view[0, 0], i,
            data_cnt, fitted_y_ptrs[i], &tid_view[data_start_idx]
        )
    return xi, xi_var, fitted_y_mat, fitted_y


def fpca_ce_score_f32(
    np.ndarray[np.float32_t] yy,
    np.ndarray[np.float32_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float32_t] mu,
    np.ndarray[np.float32_t, ndim=2] sigma_y,
    np.ndarray[np.float32_t] fpca_lambda,
    np.ndarray[np.float32_t, ndim=2] fpca_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
):
    cdef uint64_t num_unique_sid = unique_sid.size, num_pcs = fpca_phi.shape[1], nt = mu.size
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef BLAS_Order order = ColMajor if fpca_phi.flags.f_contiguous else RowMajor
    cdef str order_char = "F" if order == ColMajor else "C"

    cdef np.float32_t[:] yy_view = yy
    cdef np.float32_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef np.float32_t[:] mu_view = mu
    cdef np.float32_t[:, :] sigma_y_view = sigma_y
    cdef np.float32_t[:] fpca_lambda_view = fpca_lambda
    cdef np.float32_t[:, :] fpca_phi_view = fpca_phi

    cdef list xi_var = []
    cdef list fitted_y = []
    cdef vector[np.float32_t*] xi_var_ptrs
    cdef vector[np.float32_t*] fitted_y_ptrs
    cdef np.ndarray[np.float32_t, ndim=2] temp_xi_var
    cdef np.ndarray[np.float32_t] temp_fitted_y
    cdef int64_t i, j
    cdef uint64_t data_start_idx, data_cnt
    for i in range(<int64_t> num_unique_sid):
        temp_xi_var = np.diag(fpca_lambda)
        xi_var.append(temp_xi_var)
        xi_var_ptrs.push_back(<np.float32_t*> &temp_xi_var.data[0])

        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        temp_fitted_y = np.zeros((data_cnt,), dtype=np.float32)
        fitted_y_ptrs.push_back(<np.float32_t*> &temp_fitted_y.data[0])
        fitted_y.append(temp_fitted_y)

    cdef np.ndarray[np.float32_t, ndim=2] lambda_phi = np.zeros((nt, num_pcs), order=order_char, dtype=np.float32)
    cdef np.float32_t[:, :] lambda_phi_view = lambda_phi
    for j in range(<int64_t> num_pcs):
        for i in prange(<int64_t> nt, nogil=True):
            lambda_phi_view[i, j] = fpca_phi_view[i, j] * fpca_lambda_view[j]

    cdef np.ndarray[np.float32_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order=order_char, dtype=np.float32)
    cdef np.float32_t[:, :] xi_view = xi
    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        fpca_ce_score_helper(
            order, num_unique_sid, nt, num_pcs, data_cnt,
            &xi_view[i, 0], xi_var_ptrs[i], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &sigma_y_view[0, 0], &lambda_phi_view[0, 0]
        )

    cdef np.ndarray[np.float32_t, ndim=2] fitted_y_mat = np.zeros((nt, num_unique_sid), order=order_char, dtype=np.float32)
    cdef np.float32_t[:, :] fitted_y_mat_view = fitted_y_mat
    get_fitted_y_mat(
        order, nt, num_unique_sid, num_pcs,
        &fitted_y_mat_view[0, 0], &mu_view[0], &xi_view[0, 0], &fpca_phi_view[0, 0]
    )

    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        get_fitted_y_list(
            order, nt, num_unique_sid, &fitted_y_mat_view[0, 0], i,
            data_cnt, fitted_y_ptrs[i], &tid_view[data_start_idx]
        )
    return xi, xi_var, fitted_y_mat, fitted_y


cdef void fpca_in_score_helper(
    BLAS_Order order,
    uint64_t num_unique_sid,
    uint64_t nt,
    uint64_t num_pcs,
    uint64_t data_cnt,
    floating* xi,          # shape: (num_unique_sid, num_pcs) (pointer to xi[idx, 0])
    floating* yy,
    floating* tt,
    int64_t* tid,
    floating* mu,          # shape: (nt, )
    floating* fpca_lambda, # shape: (num_pcs, )
    floating* fpca_phi,    # shape: (nt, num_pcs)
    floating sigma2,
    floating t_range,
    bint if_shrinkage
) noexcept nogil:
    cdef int64_t i, j
    cdef floating* temp = <floating*> malloc(num_pcs * data_cnt * sizeof(floating))       # shape: (num_pcs, data_cnt) fixed to use ColMajor
    cdef floating* sub_y_minus_mu = <floating*> malloc(data_cnt * sizeof(floating))       # shape: (data_cnt, )
    cdef floating* sub_t = <floating*> malloc(data_cnt * sizeof(floating))                # shape: (data_cnt, )
    if (not temp) or (not sub_y_minus_mu) or (not sub_t):
        free(temp); free(sub_y_minus_mu); free(sub_t)
        for j in range(<int64_t> num_pcs):
            xi[j] = NAN
        return

    for i in range(<int64_t> data_cnt):
        sub_y_minus_mu[i] = yy[i] - mu[tid[i]]
        sub_t[i] = tt[i]

    if order == ColMajor:
        for j in range(<int64_t> data_cnt):
            for i in range(<int64_t> num_pcs):
                temp[i + j * num_pcs] = fpca_phi[tid[j] + i * nt] * sub_y_minus_mu[j]
    elif order == RowMajor:
        for j in range(<int64_t> data_cnt):
            for i in range(<int64_t> num_pcs):
                temp[i + j * num_pcs] = fpca_phi[tid[j] * num_pcs + i] * sub_y_minus_mu[j]

    cdef int64_t inc_xi = num_unique_sid if order == ColMajor else 1
    trapz(ColMajor, num_pcs, data_cnt, temp, data_cnt, sub_t, num_pcs, xi, inc_xi)

    if if_shrinkage:
        if order == ColMajor:
            for i in range(<int64_t> num_pcs):
                xi[i * inc_xi] *= fpca_lambda[i] / (fpca_lambda[i] + t_range * sigma2 / data_cnt)
        elif order == RowMajor:
            for i in range(<int64_t> num_pcs):
                xi[i] *= fpca_lambda[i] / (fpca_lambda[i] + t_range * sigma2 / data_cnt)

    free(temp)
    free(sub_y_minus_mu)
    free(sub_t)


def fpca_in_score_f64(
    np.ndarray[np.float64_t] yy,
    np.ndarray[np.float64_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float64_t] mu,
    np.ndarray[np.float64_t] fpca_lambda,
    np.ndarray[np.float64_t, ndim=2] fpca_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt,
    np.float64_t sigma2,
    np.float64_t t_range,
    bint if_shrinkage
):
    cdef uint64_t num_unique_sid = unique_sid.size, num_pcs = fpca_phi.shape[1], nt = mu.size
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef BLAS_Order order = ColMajor if fpca_phi.flags.f_contiguous else RowMajor
    cdef str order_char = "F" if order == ColMajor else "C"

    cdef np.float64_t[:] yy_view = yy
    cdef np.float64_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef np.float64_t[:] mu_view = mu
    cdef np.float64_t[:] fpca_lambda_view = fpca_lambda
    cdef np.float64_t[:, :] fpca_phi_view = fpca_phi
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt

    cdef list fitted_y = []
    cdef vector[np.float64_t*] fitted_y_ptrs
    cdef np.ndarray[np.float64_t] temp_fitted_y
    cdef int64_t i, j
    cdef uint64_t data_start_idx, data_cnt
    for i in range(<int64_t> num_unique_sid):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        temp_fitted_y = np.zeros((data_cnt,), dtype=np.float64)
        fitted_y_ptrs.push_back(<np.float64_t*> &temp_fitted_y.data[0])
        fitted_y.append(temp_fitted_y)

    cdef np.ndarray[np.float64_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order=order_char, dtype=np.float64)
    cdef np.float64_t[:, :] xi_view = xi
    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        fpca_in_score_helper(
            order, num_unique_sid, nt, num_pcs, data_cnt,
            &xi_view[i, 0], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &fpca_lambda_view[0], &fpca_phi_view[0, 0],
            sigma2, t_range, if_shrinkage
        )

    cdef np.ndarray[np.float64_t, ndim=2] fitted_y_mat = np.zeros((nt, num_unique_sid), order=order_char, dtype=np.float64)
    cdef np.float64_t[:, :] fitted_y_mat_view = fitted_y_mat
    get_fitted_y_mat(
        order, nt, num_unique_sid, num_pcs,
        &fitted_y_mat_view[0, 0], &mu_view[0], &xi_view[0, 0], &fpca_phi_view[0, 0]
    )

    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        get_fitted_y_list(
            order, nt, num_unique_sid, &fitted_y_mat_view[0, 0], i,
            data_cnt, fitted_y_ptrs[i], &tid_view[data_start_idx]
        )
    return xi, fitted_y_mat, fitted_y


def fpca_in_score_f32(
    np.ndarray[np.float32_t] yy,
    np.ndarray[np.float32_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float32_t] mu,
    np.ndarray[np.float32_t] fpca_lambda,
    np.ndarray[np.float32_t, ndim=2] fpca_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt,
    np.float32_t sigma2,
    np.float32_t t_range,
    bint if_shrinkage
):
    cdef uint64_t num_unique_sid = unique_sid.size, num_pcs = fpca_phi.shape[1], nt = mu.size
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef BLAS_Order order = ColMajor if fpca_phi.flags.f_contiguous else RowMajor
    cdef str order_char = "F" if order == ColMajor else "C"

    cdef np.float32_t[:] yy_view = yy
    cdef np.float32_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef np.float32_t[:] mu_view = mu
    cdef np.float32_t[:] fpca_lambda_view = fpca_lambda
    cdef np.float32_t[:, :] fpca_phi_view = fpca_phi
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt

    cdef list fitted_y = []
    cdef vector[np.float32_t*] fitted_y_ptrs
    cdef np.ndarray[np.float32_t] temp_fitted_y
    cdef int64_t i, j
    cdef uint64_t data_start_idx, data_cnt
    for i in range(<int64_t> num_unique_sid):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        temp_fitted_y = np.zeros((data_cnt,), dtype=np.float32)
        fitted_y_ptrs.push_back(<np.float32_t*> &temp_fitted_y.data[0])
        fitted_y.append(temp_fitted_y)

    cdef np.ndarray[np.float32_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order=order_char, dtype=np.float32)
    cdef np.float32_t[:, :] xi_view = xi
    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        fpca_in_score_helper(
            order, num_unique_sid, nt, num_pcs, data_cnt,
            &xi_view[i, 0], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &fpca_lambda_view[0], &fpca_phi_view[0, 0],
            sigma2, t_range, if_shrinkage
        )

    cdef np.ndarray[np.float32_t, ndim=2] fitted_y_mat = np.zeros((nt, num_unique_sid), order=order_char, dtype=np.float32)
    cdef np.float32_t[:, :] fitted_y_mat_view = fitted_y_mat
    get_fitted_y_mat(
        order, nt, num_unique_sid, num_pcs,
        &fitted_y_mat_view[0, 0], &mu_view[0], &xi_view[0, 0], &fpca_phi_view[0, 0]
    )

    for i in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        get_fitted_y_list(
            order, nt, num_unique_sid, &fitted_y_mat_view[0, 0], i,
            data_cnt, fitted_y_ptrs[i], &tid_view[data_start_idx]
        )
    return xi, fitted_y_mat, fitted_y
