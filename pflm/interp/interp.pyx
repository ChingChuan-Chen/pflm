import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport NAN
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t, uint64_t
from pflm.utils.blas_helper cimport BLAS_Order, ColMajor, RowMajor
from pflm.utils.lapack_helper cimport _gtsv


cdef void find_le_indices(floating* a, uint64_t n, floating* b, uint64_t m, int64_t* result) noexcept nogil:
    """
    For each b[j], return the largest i such that a[i] <= b[j] < a[i+1].
    Special case: if b[j] == a[n-1], return n-2 to keep j+1 valid downstream.
    Returns -1 if b[j] < a[0].
    """
    cdef uint64_t i = 0, j
    for j in range(m):
        while i < n and a[i] <= b[j]:
            i += 1
        if i < n:
            result[j] = i - 1
        elif i == n and a[i - 1] == b[j]:
            result[j] = <int64_t> n - 2
        else:
            result[j] = -1


cdef void interp1d_linear(uint64_t x_size, floating* x, floating* y, uint64_t x_new_size, floating* x_new, floating* y_new) noexcept nogil:
    """
    Linear interpolation: y_new = y[j] + (x_new - x[j]) * (y[j+1]-y[j]) / (x[j+1]-x[j])
    where j = find_le_indices(x, x_new).
    y_new[i] = NaN if j < 0.
    """
    cdef int64_t* idx = <int64_t*> malloc(x_new_size * sizeof(int64_t))
    cdef int64_t i, j

    if idx == NULL:
        for i in prange(<int64_t> x_new_size, nogil=True):
            y_new[i] = NAN
        return

    find_le_indices(x, x_size, x_new, x_new_size, idx)

    for i in prange(<int64_t> x_new_size, nogil=True):
        j = idx[i]
        if j >= 0:
            y_new[i] = y[j] + (x_new[i] - x[j]) * (y[j + 1] - y[j]) / (x[j + 1] - x[j])
        else:
            y_new[i] = NAN

    free(idx)


cdef void interp1d_spline_small(
    uint64_t x_size,
    floating* x,
    floating* y,
    uint64_t x_new_size,
    floating* x_new,
    floating* y_new
) noexcept nogil:
    """
    Small-size spline interpolation:
    - x_size == 2: linear through (x0,y0),(x1,y1)
    - x_size == 3: quadratic through (x0,y0),(x1,y1),(x2,y2)
    Out-of-domain -> NaN. No search; uses polynomial in s = x_new - x[0].
    """
    cdef floating ca, cb, cc, s
    cdef int64_t i

    if x_size == 2:
        cc = <floating>0.0
        cb = (y[1] - y[0]) / (x[1] - x[0])
        ca = y[0]
    elif x_size == 3:
        cc = (y[2] - y[0]) / (x[2] - x[0]) / (x[2] - x[1]) \
           - (y[1] - y[0]) / (x[1] - x[0]) / (x[2] - x[1])
        cb = (y[1] - y[0]) * (x[2] - x[0]) / (x[1] - x[0]) / (x[2] - x[1]) \
           - (y[2] - y[0]) * (x[1] - x[0]) / (x[2] - x[0]) / (x[2] - x[1])
        ca = y[0]
    else:
        for i in prange(<int64_t> x_new_size, nogil=True):
            y_new[i] = NAN
        return

    for i in prange(<int64_t> x_new_size, nogil=True):
        if x_new[i] < x[0] or x_new[i] > x[x_size - 1]:
            y_new[i] = NAN
        else:
            s = x_new[i] - x[0]
            y_new[i] = ca + s * cb + (s * s) * cc


cdef void interp1d_spline(uint64_t x_size, floating* x, floating* y, uint64_t x_new_size, floating* x_new, floating* y_new) noexcept nogil:
    """
    Cubic spline interpolation
    n := x_size >= 4. Solves the interior second-derivatives system via LAPACK xGTSV.
    For out-of-domain x_new, returns NaN.
    """
    cdef uint64_t n = x_size
    cdef uint64_t n1 = n - 1          # number of intervals
    cdef uint64_t n2 = n - 2          # system size for interior second derivatives

    cdef floating *h = <floating*> malloc(n1 * sizeof(floating))
    cdef floating *ca = <floating*> malloc(n1 * sizeof(floating))
    cdef floating *cb = <floating*> malloc(n1 * sizeof(floating))
    cdef floating *cc = <floating*> malloc(n  * sizeof(floating))
    cdef floating *cd = <floating*> malloc(n1 * sizeof(floating))

    # Tri-diagonal system: dl (n2-1), d (n2), du (n2-1), rhs (n2)
    cdef floating *dl = <floating*> malloc((n2 - 1) * sizeof(floating)) if n2 > 1 else NULL
    cdef floating *d_ = <floating*> malloc( n2       * sizeof(floating)) if n2 > 0 else NULL
    cdef floating *du = <floating*> malloc((n2 - 1) * sizeof(floating)) if n2 > 1 else NULL
    cdef floating *rhs = <floating*> malloc( n2      * sizeof(floating)) if n2 > 0 else NULL

    cdef int64_t i
    if (not h) or (not ca) or (not cb) or (not cc) or (not cd) or (n2 > 0 and (not d_ or not rhs)) or (n2 > 1 and (not dl or not du)):
        # allocation failure -> fill NaN and clean up
        for i in prange(<int64_t> x_new_size, nogil=True):
            y_new[i] = NAN
        if h: free(h)
        if ca: free(ca)
        if cb: free(cb)
        if cc: free(cc)
        if cd: free(cd)
        if dl: free(dl)
        if d_: free(d_)
        if du: free(du)
        if rhs: free(rhs)
        return

    # Prepare intervals and simple diffs
    for i in prange(<int64_t> n1, nogil=True):
        h[i] = x[i + 1] - x[i]
        cb[i] = y[i + 1] - y[i]
        ca[i] = y[i]

    rhs[0] = (<floating>3.0) / (h[0] + h[1]) * (cb[1] - h[1] / h[0] * cb[0])
    # rhs[x_size-3] (which is index n-3)
    rhs[n2 - 1] = (<floating>3.0) / (h[n - 3] + h[n - 2]) * (h[n - 3] / h[n - 2] * cb[n - 2] - cb[n - 3])

    cdef int info
    cdef floating s
    if x_size > 4:
        for i in prange(1, <int64_t> n2 - 1, nogil=True):
            # rhs[i] = 3*cb[i+1]/h[i+1] - 3*cb[i]/h[i];
            rhs[i] = (<floating>3.0) * (cb[i + 1] / h[i + 1] - cb[i] / h[i])

        # Build tri-diagonal bands for size n2
        for i in prange(<int64_t> n2 - 1, nogil=True):
            d_[i] = (<floating>2.0) * (h[i] + h[i + 1])
            dl[i] = h[i + 1]
            du[i] = h[i + 1]
        d_[0] -= h[0]
        d_[n2 - 1] = (<floating>2.0) * h[n - 3] + h[n - 2]
        du[0] -= h[0]
        dl[n2 - 2] -= h[n - 2]

        # Solve tri-diagonal: place solution in rhs (used as B with nrhs=1)
        info = 0
        _gtsv(ColMajor, <int> n2, 1, dl, d_, du, rhs, <int> n2, &info)
        if info != 0:
            # fall back: if solver failed, return NaNs
            for i in prange(<int64_t> x_new_size, nogil=True):
                y_new[i] = NAN
            free(h); free(ca); free(cb); free(cc); free(cd); free(dl); free(d_); free(du); free(rhs)
            return

        # Fill cc (second derivatives): interior from solution
        for i in prange(<int64_t> n2, nogil=True):
            cc[i + 1] = rhs[i]
    else:
        # special case for x_size = 4
        s = 3 * h[0] * h[1] + 3 * h[1] * h[2] + 3 * h[1] * h[1]
        cc[1] = (2 * rhs[0] * h[1] + rhs[0] * h[2] + rhs[1] * h[0] - rhs[1] * h[1]) / s
        cc[2] = (2 * rhs[1] * h[1] + rhs[1] * h[0] + rhs[0] * h[2] - rhs[0] * h[1]) / s

    # Extrapolate boundary second derivatives as in reference
    # cc[0] = cc[1] + h0/h1 * (cc[1] - cc[2])
    # cc[n-1] = cc[n-2] + h[n-2]/h[n-3] * (cc[n-2] - cc[n-3])
    cc[0] = cc[1] + h[0] / h[1] * (cc[1] - cc[2])
    cc[n - 1] = cc[n - 2] + h[n - 2] / h[n - 3] * (cc[n - 2] - cc[n - 3])

    # Compute piecewise cubic coefficients (reuse cb as in the C++ code)
    for i in prange(<int64_t> n1, nogil=True):
        cb[i] = cb[i] / h[i]
        cb[i] -= h[i] * (cc[i + 1] + (<floating>2.0) * cc[i]) / (<floating>3.0)
        cd[i] = (cc[i + 1] - cc[i]) / ((<floating>3.0) * h[i])

    # Interpolate x_new
    cdef int64_t *idx = <int64_t*> malloc(x_new_size * sizeof(int64_t))
    if idx == NULL:
        for i in prange(<int64_t> x_new_size, nogil=True):
            y_new[i] = NAN
        free(h); free(ca); free(cb); free(cc); free(cd); free(dl); free(d_); free(du); free(rhs)
        return

    find_le_indices(x, x_size, x_new, x_new_size, idx)
    cdef int64_t j
    for i in prange(<int64_t> x_new_size, nogil=True):
        j = idx[i]
        if 0 <= j < <int64_t> x_size - 1:
            s = x_new[i] - x[j]
            # y = ca + s*cb + s^2*cc + s^3*cd
            y_new[i] = ca[j] + s * cb[j] + (s * s) * cc[j] + (s * s * s) * cd[j]
        else:
            y_new[i] = NAN

    free(idx)
    free(h); free(ca); free(cb); free(cc); free(cd); free(dl); free(d_); free(du); free(rhs)


cdef void interp2d_linear(
    uint64_t x_size,
    floating* x,
    uint64_t y_size,
    floating* y,
    floating* v,           # shape (x_size, y_size) row-major: v[i, j] = v[i*y_size + j]
    uint64_t x_new_size,
    floating* x_new,
    uint64_t y_new_size,
    floating* y_new,
    floating* v_new        # shape (x_new_size, y_new_size) column-major: v_new[i, j] = v_new[j*x_new_size + i]
) noexcept nogil:
    """
    2D bilinear interpolation on a rectilinear grid.
    - x: shape (x_size,)
    - y: shape (y_size,)
    - v: shape (x_size, y_size) in row-major (v[i, j] = v[i * y_size + j])
    - x_new: shape (x_new_size,)
    - y_new: shape (y_new_size,)
    - v_new: shape (x_new_size, y_new_size) in column-major (v_new[i, j] = v_new[j * x_new_size + i])
    Out-of-domain -> NaN. Indices found via find_le_indices_.
    """
    cdef int64_t *x_idx = <int64_t*> malloc(x_new_size * sizeof(int64_t))
    cdef int64_t *y_idx = <int64_t*> malloc(y_new_size * sizeof(int64_t))
    cdef int64_t i, j
    cdef floating dx, dy, current_v, current_v_right, current_v_down

    if x_idx == NULL or y_idx == NULL:
        # allocation failure -> fill NaN
        if x_idx: free(x_idx)
        if y_idx: free(y_idx)
        for j in prange(<int64_t> y_new_size, nogil=True):
            for i in range(<int64_t> x_new_size):
                v_new[j * x_new_size + i] = NAN
        return

    find_le_indices(x, x_size, x_new, x_new_size, x_idx)
    find_le_indices(y, y_size, y_new, y_new_size, y_idx)

    for j in prange(<int64_t> y_new_size, nogil=True):
        for i in range(<int64_t> x_new_size):
            if x_idx[i] < 0 or y_idx[j] < 0:
                v_new[j * x_new_size + i] = NAN
            else:
                dx = (x_new[i] - x[x_idx[i]]) / (x[x_idx[i] + 1] - x[x_idx[i]])
                dy = (y_new[j] - y[y_idx[j]]) / (y[y_idx[j] + 1] - y[y_idx[j]])

                current_v = v[x_idx[i] * y_size + y_idx[j]]
                current_v_right = v[(x_idx[i] + 1) * y_size + y_idx[j]]
                current_v_down = v[x_idx[i] * y_size + y_idx[j] + 1]

                # Bilinear form
                v_new[j * x_new_size + i] = (
                    current_v
                    + dx * (current_v_right - current_v) + dy * (current_v_down - current_v)
                    + dx * dy * (v[(x_idx[i] + 1) * y_size + (y_idx[j] + 1)] - current_v_right - current_v_down + current_v)
                )

    free(x_idx)
    free(y_idx)


cdef void interp2d_spline(
    uint64_t x_size,
    floating* x,
    uint64_t y_size,
    floating* y,
    floating* v,            # shape (x_size, y_size) row-major: v[i, j] = v[i*y_size + j]
    uint64_t x_new_size,
    floating* x_new,
    uint64_t y_new_size,
    floating* y_new,
    floating* v_new         # shape (x_new_size, y_new_size) column-major: v_new[i, j] = v_new[j*x_new_size + i]
) noexcept nogil:
    """
    2D cubic spline interpolation on a rectilinear grid, separable:
    1) along y (inner, contiguous) for each fixed x -> temp(x_size, y_new_size)
    2) along x for each fixed y_new -> v_new(x_new_size, y_new_size)

    Layout matches interp2d_linear:
      - v: (x_size, y_size) row-major
      - v_new: (x_new_size, y_new_size) column-major
    """
    cdef int64_t i, j

    # create temp space for a row-major matrix with shape (x_size, y_new_size)
    cdef floating* temp = <floating*> malloc(x_size * y_new_size * sizeof(floating))
    if temp == NULL:
        # allocation failure -> fill v_new with NaN and return
        for j in prange(<int64_t> y_new_size, nogil=True):
            for i in range(<int64_t> x_new_size):
                v_new[j * x_new_size + i] = NAN
        return

    # Step 1: for each x[i], spline along y to y_new; input rows v[i, :]
    for i in prange(<int64_t> x_size, nogil=True):
        if y_size <= 3:
            interp1d_spline_small(y_size, y, &v[i * y_size], y_new_size, y_new, &temp[i * y_new_size])
        else:
            interp1d_spline(y_size, y, &v[i * y_size], y_new_size, y_new, &temp[i * y_new_size])

    # create a column-major matrix for temp with shape (x_size, y_new_size)
    cdef floating* temp_trans = <floating*> malloc(x_size * y_new_size * sizeof(floating))
    if temp_trans == NULL:
        free(temp)
        # allocation failure -> fill v_new with NaN and return
        for j in prange(<int64_t> y_new_size, nogil=True):
            for i in range(<int64_t> x_new_size):
                v_new[j * x_new_size + i] = NAN
        return

    # Step 2: transpose, temp_trans will be column-major matrix with shape (x_size, y_new_size)
    for i in prange(<int64_t> x_size, nogil=True):
        for j in range(<int64_t> y_new_size):
            temp_trans[j * x_size + i] = temp[i * y_new_size + j]

    # Step 3: for each y_new[j], spline along x from temp[:, j] -> v_new[:, j]
    for j in prange(<int64_t> y_new_size, nogil=True):
        if x_size <= 3:
            interp1d_spline_small(x_size, x, &temp_trans[j * x_size], x_new_size, x_new, &v_new[j * x_new_size])
        else:
            interp1d_spline(x_size, x, &temp_trans[j * x_size], x_new_size, x_new, &v_new[j * x_new_size])

    free(temp)
    free(temp_trans)


def find_le_indices_memview_f64(np.float64_t[:] a, np.float64_t[:] b) -> np.ndarray[np.int64_t]:
    """find_le_indices_memview_f64(a, b) -> np.ndarray[np.int64_t] (test only)"""
    cdef uint64_t n = a.size, m = b.size
    cdef np.ndarray[np.int64_t] result = np.empty(m, dtype=np.int64)
    cdef int64_t[:] result_ptr = result
    find_le_indices(&a[0], n, &b[0], m, &result_ptr[0])
    return result


def find_le_indices_memview_f32(np.float32_t[:] a, np.float32_t[:] b) -> np.ndarray[np.int64_t]:
    """find_le_indices_memview_f32(a, b) -> np.ndarray[np.int64_t] (test only)"""
    cdef uint64_t n = a.size, m = b.size
    cdef np.ndarray[np.int64_t] result = np.empty(m, dtype=np.int64)
    cdef int64_t[:] result_ptr = result
    find_le_indices(&a[0], n, &b[0], m, &result_ptr[0])
    return result


def interp1d_f64(
    np.ndarray[np.float64_t] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] x_new,
    int method = 0
) -> np.ndarray[np.float64_t]:
    cdef uint64_t x_size = x.shape[0], x_new_size = x_new.shape[0]
    cdef np.float64_t[:] x_view = x
    cdef np.float64_t[:] y_view = y
    cdef np.float64_t[:] x_new_view = x_new

    cdef np.ndarray[np.float64_t] y_new = np.empty(x_new.size, dtype=np.float64)
    cdef np.float64_t[:] y_new_view = y_new

    if method == 0:
        interp1d_linear(x_size, &x_view[0], &y_view[0], x_new_size, &x_new_view[0], &y_new_view[0])
    elif method == 1:
        if x_size <= 3:
            interp1d_spline_small(x_size, &x_view[0], &y_view[0], x_new_size, &x_new_view[0], &y_new_view[0])
        else:
            interp1d_spline(x_size, &x[0], &y[0], x_new_size, &x_new[0], &y_new[0])

    return y_new


def interp1d_f32(
    np.ndarray[np.float32_t] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] x_new,
    int method = 0
) -> np.ndarray[np.float32_t]:
    cdef uint64_t x_size = x.shape[0], x_new_size = x_new.shape[0]
    cdef np.float32_t[:] x_view = x
    cdef np.float32_t[:] y_view = y
    cdef np.float32_t[:] x_new_view = x_new

    cdef np.ndarray[np.float32_t] y_new = np.empty(x_new.size, dtype=np.float32)
    cdef np.float32_t[:] y_new_view = y_new

    if method == 0:
        interp1d_linear(x_size, &x_view[0], &y_view[0], x_new_size, &x_new_view[0], &y_new_view[0])
    elif method == 1:
        if x_size <= 3:
            interp1d_spline_small(x_size, &x_view[0], &y_view[0], x_new_size, &x_new_view[0], &y_new_view[0])
        else:
            interp1d_spline(x_size, &x[0], &y[0], x_new_size, &x_new[0], &y_new[0])
    return y_new


def interp2d_f64(
  np.ndarray[np.float64_t] x,
  np.ndarray[np.float64_t] y,
  np.ndarray[np.float64_t, ndim=2] v,
  np.ndarray[np.float64_t] x_new,
  np.ndarray[np.float64_t] y_new,
  int method = 0
):
    cdef uint64_t x_size = x.size, y_size = y.size, x_new_size = x_new.size, y_new_size = y_new.size
    cdef np.ndarray[np.float64_t, ndim=2] v_new = np.empty((x_new_size, y_new_size), order='F', dtype=np.float64)
    if method == 0:
        interp2d_linear(x_size, &x[0], y_size, &y[0], &v[0, 0], x_new_size, &x_new[0], y_new_size, &y_new[0], &v_new[0, 0])
    elif method == 1:
        interp2d_spline(x_size, &x[0], y_size, &y[0], &v[0, 0], x_new_size, &x_new[0], y_new_size, &y_new[0], &v_new[0, 0])
    return v_new


def interp2d_f32(
    np.ndarray[np.float32_t] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t, ndim=2] v,
    np.ndarray[np.float32_t] x_new,
    np.ndarray[np.float32_t] y_new,
    int method = 0
):
    cdef uint64_t x_size = x.size, y_size = y.size, x_new_size = x_new.size, y_new_size = y_new.size
    cdef np.ndarray[np.float32_t, ndim=2] v_new = np.empty((x_new_size, y_new_size), order='F', dtype=np.float32)
    if method == 0:
        interp2d_linear(x_size, &x[0], y_size, &y[0], &v[0, 0], x_new_size, &x_new[0], y_new_size, &y_new[0], &v_new[0, 0])
    elif method == 1:
        interp2d_spline(x_size, &x[0], y_size, &y[0], &v[0, 0], x_new_size, &x_new[0], y_new_size, &y_new[0], &v_new[0, 0])
    return v_new
