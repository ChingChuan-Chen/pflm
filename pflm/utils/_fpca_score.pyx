import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel import prange
from libc.stdint cimport int64_t, uint64_t
from libcpp.vector cimport vector
from libc.math cimport NAN
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from sklearn.utils._cython_blas cimport _gemv, _gemm
from sklearn.utils._cython_blas cimport ColMajor, NoTrans, Trans
from pflm.utils._lapack_helper cimport _posv
from pflm.utils._trapz cimport trapz_mat_blas


cdef void fpca_ce_score_helper(
    uint64_t nt,
    uint64_t data_cnt,
    uint64_t num_pcs,
    floating* xi,         # shape: (num_samples, num_pcs) row-major (pointer to xi[idx, 0])
    floating* xi_var,     # shape: (data_cnt, data_cnt) row-major
    floating* yy,
    floating* tt,
    int64_t* tid,
    floating* mu,         # shape: (nt, )
    floating* sigma_y,    # shape: (nt, nt) symmetric
    floating* lambda_phi  # shape: (nt, num_pcs) row-major
) noexcept nogil:
    cdef int64_t i, j
    cdef floating* sub_lambda_phi = <floating*> malloc(num_pcs * data_cnt * sizeof(floating)) # shape: (data_cnt, num_pcs) col-major
    cdef floating* sub_sigma_lambda_phi = <floating*> malloc(num_pcs * data_cnt * sizeof(floating)) # shape: (data_cnt, num_pcs) col-major
    cdef floating* sub_sigma_y = <floating*> malloc(data_cnt * data_cnt * sizeof(floating))   # shape: (data_cnt, data_cnt) col-major, upper-triangular
    cdef floating* sub_y_minus_mu = <floating*> malloc(data_cnt * sizeof(floating))           # shape: (data_cnt, )
    for i in range(<int64_t> data_cnt):
        sub_y_minus_mu[i] = yy[i] - mu[tid[i]]
        for j in range(i, <int64_t> data_cnt):
            sub_sigma_y[i * data_cnt + j] = sigma_y[tid[i] * nt + tid[j]]


    for j in range(<int64_t> num_pcs):
        for i in range(<int64_t> data_cnt):
            sub_lambda_phi[i + j * data_cnt] = lambda_phi[tid[i] * num_pcs + j]
    memcpy(sub_sigma_lambda_phi, sub_lambda_phi, num_pcs * data_cnt * sizeof(floating))

    # perform A = inv(sub_sigma_y) * sub_lambda_phi
    cdef int info = 0
    _posv(108, <int> data_cnt, <int> num_pcs, sub_sigma_y, <int> data_cnt, sub_sigma_lambda_phi, <int> data_cnt, &info)
    if info != 0:
        for j in range(<int64_t> num_pcs):
            xi[j] = NAN

    # perform A * (y - mu) with _gemv: y = alpha * A^T * x + beta * y
    _gemv(
        ColMajor, Trans,
        <int> data_cnt, <int> num_pcs, # m, n
        1.0, # alpha
        sub_sigma_lambda_phi, <int> data_cnt, # A, lda
        sub_y_minus_mu, 1, # X, inc_x
        0.0, # beta
        xi, 1 # y, inc_y
    )

    # perform C = sub_lambda_phi * inv(sub_sigma_y) * sub_lambda_phi
    _gemm(
        ColMajor, Trans, NoTrans,
        <int> num_pcs, <int> num_pcs, <int> data_cnt, # m, n, k
        -1.0, # alpha
        sub_lambda_phi, <int> data_cnt, # A, lda
        sub_sigma_lambda_phi, <int> data_cnt, # B, ldb
        1.0, # beta
        xi_var, <int> num_pcs # C, ldc
    )
    free(sub_lambda_phi)
    free(sub_sigma_lambda_phi)
    free(sub_sigma_y)
    free(sub_y_minus_mu)


def fpca_ce_score_f64(
    np.ndarray[np.float64_t] yy,
    np.ndarray[np.float64_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float64_t] mu,
    np.ndarray[np.float64_t, ndim=2] sigma_y,
    np.ndarray[np.float64_t] fpca_lambda,
    np.ndarray[np.float64_t, ndim=2] lambda_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
):
    cdef uint64_t num_unique_sid = unique_sid.size, num_pcs = lambda_phi.shape[1], nt = mu.size
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef np.ndarray[np.float64_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order='C', dtype=np.float64)
    cdef list xi_var = []
    cdef vector[np.float64_t*] xi_var_ptrs
    cdef np.ndarray[np.float64_t, ndim=2] temp_xi_var
    cdef int64_t i
    for i in range(<int64_t> num_unique_sid):
        temp_xi_var = np.diag(fpca_lambda)
        xi_var.append(temp_xi_var)
        xi_var_ptrs.push_back(<np.float64_t*> &temp_xi_var.data[0])

    cdef np.float64_t[:] yy_view = yy
    cdef np.float64_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef np.float64_t[:] mu_view = mu
    cdef np.float64_t[:, :] sigma_y_view = sigma_y
    cdef np.float64_t[:, ::1] lambda_phi_view = lambda_phi
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef np.float64_t[:, ::1] xi_view = xi

    cdef uint64_t data_start_idx, data_cnt
    cdef int64_t idx
    for idx in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[idx - 1] if idx > 0 else 0
        data_cnt = sid_cum_cnt_view[idx] - data_start_idx
        fpca_ce_score_helper(
            nt, data_cnt, num_pcs,
            &xi_view[idx, 0], xi_var_ptrs[idx], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &sigma_y_view[0, 0], &lambda_phi_view[0, 0]
        )
    return xi, xi_var


def fpca_ce_score_f32(
    np.ndarray[np.float32_t] yy,
    np.ndarray[np.float32_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float32_t] mu,
    np.ndarray[np.float32_t, ndim=2] sigma_y,
    np.ndarray[np.float32_t] fpca_lambda,
    np.ndarray[np.float32_t, ndim=2] lambda_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
):
    cdef uint64_t num_unique_sid = unique_sid.size, num_pcs = lambda_phi.shape[1], nt = mu.size
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef np.ndarray[np.float32_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order='C', dtype=np.float32)
    cdef list xi_var = []
    cdef vector[np.float32_t*] xi_var_ptrs
    cdef np.ndarray[np.float32_t, ndim=2] temp_xi_var
    cdef int64_t i
    for i in range(<int64_t> num_unique_sid):
        temp_xi_var = np.diag(fpca_lambda)
        xi_var.append(temp_xi_var)
        xi_var_ptrs.push_back(<np.float32_t*> &temp_xi_var.data[0])

    cdef np.float32_t[:] yy_view = yy
    cdef np.float32_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef np.float32_t[:] mu_view = mu
    cdef np.float32_t[:, :] sigma_y_view = sigma_y
    cdef np.float32_t[:, ::1] lambda_phi_view = lambda_phi
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef np.float32_t[:, ::1] xi_view = xi

    cdef uint64_t data_start_idx, data_cnt
    cdef int64_t idx
    for idx in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[idx - 1] if idx > 0 else 0
        data_cnt = sid_cum_cnt_view[idx] - data_start_idx
        fpca_ce_score_helper(
            nt, data_cnt, num_pcs,
            &xi_view[idx, 0], xi_var_ptrs[idx], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &sigma_y_view[0, 0], &lambda_phi_view[0, 0]
        )
    return xi, xi_var


cdef void fpca_in_score_helper(
    uint64_t nt,
    uint64_t data_cnt,
    uint64_t num_pcs,
    floating* xi,          # shape: (num_samples, num_pcs) row-major (pointer to xi[idx, 0])
    floating* yy,
    floating* tt,
    int64_t* tid,
    floating* mu,          # shape: (nt, )
    floating* fpca_lambda, # shape: (num_pcs, )
    floating* fpca_phi,    # shape: (nt, num_pcs) row-major
    floating sigma2,
    floating t_range,
    bint if_shrinkage
) noexcept nogil:
    cdef int64_t i, j
    cdef floating* temp = <floating*> malloc(num_pcs * data_cnt * sizeof(floating))       # shape: (num_pcs, data_cnt) col-major
    cdef floating* sub_y_minus_mu = <floating*> malloc(data_cnt * sizeof(floating))       # shape: (data_cnt, )
    cdef floating* sub_t = <floating*> malloc(data_cnt * sizeof(floating))                # shape: (data_cnt, )
    for i in range(<int64_t> data_cnt):
        sub_y_minus_mu[i] = yy[i] - mu[tid[i]]
        sub_t[i] = tt[i]
        for j in range(<int64_t> num_pcs):
            temp[j + i * num_pcs] = fpca_phi[tid[i] * num_pcs + j] * sub_y_minus_mu[i]

    trapz_mat_blas(temp, sub_t, xi, num_pcs, data_cnt, ColMajor)
    if if_shrinkage:
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
    cdef np.ndarray[np.float64_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order='C', dtype=np.float64)

    cdef np.float64_t[:] yy_view = yy
    cdef np.float64_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef np.float64_t[:] mu_view = mu
    cdef np.float64_t[:] fpca_lambda_view = fpca_lambda
    cdef np.float64_t[:, ::1] fpca_phi_view = fpca_phi
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef np.float64_t[:, ::1] xi_view = xi

    cdef uint64_t data_start_idx, data_cnt
    cdef int64_t idx
    for idx in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[idx - 1] if idx > 0 else 0
        data_cnt = sid_cum_cnt_view[idx] - data_start_idx
        fpca_in_score_helper(
            nt, data_cnt, num_pcs,
            &xi_view[idx, 0], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &fpca_lambda_view[0], &fpca_phi_view[0, 0],
            sigma2, t_range, if_shrinkage
        )
    return xi


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
    cdef np.ndarray[np.float32_t, ndim=2] xi = np.zeros((num_unique_sid, num_pcs), order='C', dtype=np.float32)

    cdef np.float32_t[:] yy_view = yy
    cdef np.float32_t[:] tt_view = tt
    cdef int64_t[:] tid_view = tid
    cdef np.float32_t[:] mu_view = mu
    cdef np.float32_t[:] fpca_lambda_view = fpca_lambda
    cdef np.float32_t[:, ::1] fpca_phi_view = fpca_phi
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef np.float32_t[:, ::1] xi_view = xi

    cdef uint64_t data_start_idx, data_cnt
    cdef int64_t idx
    for idx in prange(<int64_t> num_unique_sid, nogil=True):
        data_start_idx = sid_cum_cnt_view[idx - 1] if idx > 0 else 0
        data_cnt = sid_cum_cnt_view[idx] - data_start_idx
        fpca_in_score_helper(
            nt, data_cnt, num_pcs,
            &xi_view[idx, 0], &yy_view[data_start_idx], &tt_view[data_start_idx],
            &tid_view[data_start_idx], &mu_view[0], &fpca_lambda_view[0], &fpca_phi_view[0, 0],
            sigma2, t_range, if_shrinkage
        )
    return xi
