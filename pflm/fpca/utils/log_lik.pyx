import numpy as np
cimport numpy as np
from cython cimport floating
from libc.stdint cimport int64_t, uint64_t
from libc.math cimport NAN, log
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from pflm.utils.blas_helper cimport ColMajor, Lower, NoTrans, Trans, _gemv, _gemm
from pflm.utils.lapack_helper cimport _posv


cdef floating log_likelihood_helper(
    uint64_t nt,
    uint64_t num_unique_sid,
    uint64_t data_cnt,
    uint64_t num_pcs,
    floating* xi,         # shape: (num_unique_sid, num_pcs) pointer to xi[sid, 0]
    floating* xi_var,     # shape: (num_pcs, num_pcs)
    floating* yy,
    floating* tt,
    int64_t* tid,
    floating* mu,         # shape: (nt, )
    floating* sigma_y,    # shape: (nt, nt) symmetric row-major
    floating* lambda_phi  # shape: (nt, num_pcs), row/col major by `order`
):
    cdef int64_t i, j
    cdef int64_t inc_xi = 1
    cdef floating quad = 0.0
    cdef floating logdet = 0.0

    cdef floating* sub_lambda_phi = <floating*> malloc(num_pcs * data_cnt * sizeof(floating))
    cdef floating* sub_sigma_y = <floating*> malloc(data_cnt * data_cnt * sizeof(floating))
    cdef floating* sub_sigma_y_copy = <floating*> malloc(data_cnt * data_cnt * sizeof(floating))
    cdef floating* sub_y_minus_mu = <floating*> malloc(data_cnt * sizeof(floating))
    cdef floating* sub_y_solved = <floating*> malloc(data_cnt * sizeof(floating))
    cdef floating* sub_lambda_phi_copy = <floating*> malloc(num_pcs * data_cnt * sizeof(floating))
    if (not sub_lambda_phi) or (not sub_sigma_y) or (not sub_sigma_y_copy) or (not sub_y_minus_mu) or (not sub_y_solved) or (not sub_lambda_phi_copy):
        if sub_lambda_phi: free(sub_lambda_phi)
        if sub_sigma_y: free(sub_sigma_y)
        if sub_sigma_y_copy: free(sub_sigma_y_copy)
        if sub_y_minus_mu: free(sub_y_minus_mu)
        if sub_y_solved: free(sub_y_solved)
        if sub_lambda_phi_copy: free(sub_lambda_phi_copy)
        for i in range(<int64_t> num_pcs):
            xi[i * inc_xi] = NAN
            for j in range(<int64_t> num_pcs):
                xi_var[i * num_pcs + j] = NAN
        return NAN

    for i in range(<int64_t> data_cnt):
        sub_y_minus_mu[i] = yy[i] - mu[tid[i]]
        for j in range(<int64_t> data_cnt):
            sub_sigma_y[i * data_cnt + j] = sigma_y[tid[i] * nt + tid[j]]
        for j in range(<int64_t> num_pcs):
            sub_lambda_phi[i + j * data_cnt] = lambda_phi[tid[i] * num_pcs + j]

    memcpy(sub_sigma_y_copy, sub_sigma_y, data_cnt * data_cnt * sizeof(floating))
    memcpy(sub_y_solved, sub_y_minus_mu, data_cnt * sizeof(floating))

    cdef int info = 0
    _posv(ColMajor, Lower, <int> data_cnt, 1, sub_sigma_y, <int> data_cnt, sub_y_solved, <int> data_cnt, &info)
    if info != 0:
        free(sub_lambda_phi); free(sub_sigma_y); free(sub_sigma_y_copy)
        free(sub_y_minus_mu); free(sub_y_solved); free(sub_lambda_phi_copy)
        for i in range(<int64_t> num_pcs):
            xi[i * inc_xi] = NAN
            for j in range(<int64_t> num_pcs):
                xi_var[i * num_pcs + j] = NAN
        return NAN

    for i in range(<int64_t> data_cnt):
        if sub_sigma_y[i * data_cnt + i] <= 0.0:
            free(sub_lambda_phi); free(sub_sigma_y); free(sub_sigma_y_copy)
            free(sub_y_minus_mu); free(sub_y_solved); free(sub_lambda_phi_copy)
            for j in range(<int64_t> num_pcs):
                xi[j * inc_xi] = NAN
                for info in range(<int> num_pcs):
                    xi_var[j * num_pcs + info] = NAN
            return NAN
        logdet += 2.0 * log(sub_sigma_y[i * data_cnt + i])
        quad += sub_y_minus_mu[i] * sub_y_solved[i]

    _gemv(
        ColMajor, Trans,
        <int> data_cnt, <int> num_pcs,
        1.0,
        sub_lambda_phi, <int> data_cnt,
        sub_y_solved, 1,
        0.0,
        xi, <int> inc_xi
    )

    memcpy(sub_lambda_phi_copy, sub_lambda_phi, num_pcs * data_cnt * sizeof(floating))
    _posv(
        ColMajor, Lower,
        <int> data_cnt, <int> num_pcs,
        sub_sigma_y_copy, <int> data_cnt,
        sub_lambda_phi_copy, <int> data_cnt,
        &info
    )
    if info != 0:
        free(sub_lambda_phi); free(sub_sigma_y); free(sub_sigma_y_copy)
        free(sub_y_minus_mu); free(sub_y_solved); free(sub_lambda_phi_copy)
        for i in range(<int64_t> num_pcs):
            for j in range(<int64_t> num_pcs):
                xi_var[i * num_pcs + j] = NAN
        return NAN

    _gemm(
        ColMajor, Trans, NoTrans,
        <int> num_pcs, <int> num_pcs, <int> data_cnt,
        -1.0,
        sub_lambda_phi, <int> data_cnt,
        sub_lambda_phi_copy, <int> data_cnt,
        1.0,
        xi_var, <int> num_pcs
    )

    free(sub_lambda_phi)
    free(sub_sigma_y)
    free(sub_sigma_y_copy)
    free(sub_y_minus_mu)
    free(sub_y_solved)
    free(sub_lambda_phi_copy)
    return logdet + quad


def get_log_likelihood_f64(
    np.ndarray[np.float64_t] yy,
    np.ndarray[np.float64_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float64_t] mu,
    np.ndarray[np.float64_t, ndim=2] sigma_y,
    np.ndarray[np.float64_t] fpca_lambda,
    np.ndarray[np.float64_t, ndim=2] lambda_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
) -> float:
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
    cdef np.float64_t loglik = 0.0
    for i in range(<int64_t> num_unique_sid):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        loglik += log_likelihood_helper(
            nt, num_unique_sid, data_cnt, num_pcs,
            &xi_view[i, 0], xi_var_ptrs[i],
            &yy_view[data_start_idx], &tt_view[data_start_idx], &tid_view[data_start_idx],
            &mu_view[0], &sigma_y_view[0, 0], &lambda_phi_view[0, 0]
        )
    return float(loglik)


def get_log_likelihood_f32(
    np.ndarray[np.float32_t] yy,
    np.ndarray[np.float32_t] tt,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.float32_t] mu,
    np.ndarray[np.float32_t, ndim=2] sigma_y,
    np.ndarray[np.float32_t] fpca_lambda,
    np.ndarray[np.float32_t, ndim=2] lambda_phi,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
) -> float:
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
    cdef np.float32_t loglik = 0.0
    for i in range(<int64_t> num_unique_sid):
        data_start_idx = sid_cum_cnt_view[i - 1] if i > 0 else 0
        data_cnt = sid_cum_cnt_view[i] - data_start_idx
        loglik += log_likelihood_helper(
            nt, num_unique_sid, data_cnt, num_pcs,
            &xi_view[i, 0], xi_var_ptrs[i],
            &yy_view[data_start_idx], &tt_view[data_start_idx], &tid_view[data_start_idx],
            &mu_view[0], &sigma_y_view[0, 0], &lambda_phi_view[0, 0]
        )
    return float(loglik)
