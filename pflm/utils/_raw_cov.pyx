import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel import prange
from libc.stdint cimport int64_t
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef inline vector[pair[int64_t, int64_t]] _all_pairs_indices_vec(int64_t n) noexcept nogil:
    cdef int64_t i, j
    cdef vector[pair[int64_t, int64_t]] out
    for i in range(n):
        for j in range(i, n):
            out.push_back(pair[int64_t, int64_t](i, j))
    return out


cdef inline void _set_raw_cov(
    floating* raw_cov,
    int64_t sid,
    int64_t data_cnt,
    floating* yy,
    floating* tt,
    floating* ww,
    int64_t* tid,
    floating* mu
) noexcept nogil:
    cdef int64_t i, j, k, idx = 0
    cdef vector[pair[int64_t, int64_t]] pair = _all_pairs_indices_vec(data_cnt)
    for k in range(pair.size()):
        i = pair[k].first
        j = pair[k].second
        raw_cov[idx] = <floating> sid
        raw_cov[idx + 1] = tt[i]
        raw_cov[idx + 2] = tt[j]
        raw_cov[idx + 3] = ww[i]
        raw_cov[idx + 4] = (yy[i] - mu[tid[i]]) * (yy[j] - mu[tid[j]])
        idx += 5


def get_raw_cov_f64(
    np.ndarray[np.float64_t] yy,
    np.ndarray[np.float64_t] tt,
    np.ndarray[np.float64_t] ww,
    np.ndarray[np.float64_t] mu,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
) -> np.ndarray[np.float64_t]:
    cdef int64_t num_unique_sid = unique_sid.shape[0]
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef np.ndarray[np.int64_t] pairs_cum_cnt = np.cumsum(sid_cnt * (sid_cnt + 1) // 2)
    cdef int64_t total_pairs = pairs_cum_cnt[pairs_cum_cnt.size-1]
    if total_pairs <= 0:
        return np.empty((0, 5), dtype=np.float64)
    # create an empty array to hold the pairwise covariances (sid, t1, t2, w, cov)
    cdef np.ndarray[np.float64_t, ndim=2] raw_cov = np.empty((total_pairs, 5), dtype=np.float64)
    cdef np.float64_t[:] yy_view = yy
    cdef np.float64_t[:] tt_view = tt
    cdef np.float64_t[:] ww_view = ww
    cdef np.float64_t[:] mu_view = mu
    cdef int64_t[:] tid_view = tid
    cdef int64_t[:] unique_sid_view = unique_sid
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef int64_t[:] pairs_cum_cnt_view = pairs_cum_cnt
    cdef np.float64_t[:, ::1] raw_cov_view = raw_cov

    cdef int64_t s, idx, data_start_idx, data_cnt, pair_start_idx
    for idx in prange(num_unique_sid, nogil=True):
        s = unique_sid_view[idx]
        data_start_idx = sid_cum_cnt_view[idx - 1] if idx > 0 else 0
        data_cnt = sid_cum_cnt_view[idx] - data_start_idx
        pair_start_idx = pairs_cum_cnt_view[idx - 1] if idx > 0 else 0
        _set_raw_cov(
            &raw_cov_view[pair_start_idx, 0], s, data_cnt, &yy_view[data_start_idx],
            &tt_view[data_start_idx], &ww_view[data_start_idx], &tid_view[data_start_idx], &mu_view[0]
        )
    return raw_cov


def get_raw_cov_f32(
    np.ndarray[np.float32_t] yy,
    np.ndarray[np.float32_t] tt,
    np.ndarray[np.float32_t] ww,
    np.ndarray[np.float32_t] mu,
    np.ndarray[np.int64_t] tid,
    np.ndarray[np.int64_t] unique_sid,
    np.ndarray[np.int64_t] sid_cnt
) -> np.ndarray[np.float32_t]:
    cdef int64_t num_unique_sid = unique_sid.shape[0]
    cdef np.ndarray[np.int64_t] sid_cum_cnt = np.cumsum(sid_cnt)
    cdef np.ndarray[np.int64_t] pairs_cum_cnt = np.cumsum((sid_cnt + 1) * sid_cnt // 2)
    cdef int64_t total_pairs = pairs_cum_cnt[pairs_cum_cnt.size-1]
    if total_pairs <= 0:
        return np.empty((0, 5), dtype=np.float32)
    # create an empty array to hold the pairwise covariances (sid, t1, t2, w, cov)
    cdef np.ndarray[np.float32_t, ndim=2] raw_cov = np.empty((total_pairs, 5), dtype=np.float32)
    cdef np.float32_t[:] yy_view = yy
    cdef np.float32_t[:] tt_view = tt
    cdef np.float32_t[:] ww_view = ww
    cdef np.float32_t[:] mu_view = mu
    cdef int64_t[:] tid_view = tid
    cdef int64_t[:] unique_sid_view = unique_sid
    cdef int64_t[:] sid_cum_cnt_view = sid_cum_cnt
    cdef int64_t[:] pairs_cum_cnt_view = pairs_cum_cnt
    cdef np.float32_t[:, ::1] raw_cov_view = raw_cov

    cdef int64_t s, idx, data_start_idx, data_cnt, pair_start_idx
    for idx in prange(num_unique_sid, nogil=True):
        s = unique_sid_view[idx]
        data_start_idx = sid_cum_cnt_view[idx - 1] if idx > 0 else 0
        data_cnt = sid_cum_cnt_view[idx] - data_start_idx
        pair_start_idx = pairs_cum_cnt_view[idx - 1] if idx > 0 else 0
        _set_raw_cov(
            &raw_cov_view[pair_start_idx, 0], s, data_cnt, &yy_view[data_start_idx],
            &tt_view[data_start_idx], &ww_view[data_start_idx], &tid_view[data_start_idx], &mu_view[0]
        )
    return raw_cov
