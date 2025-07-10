import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport sqrt, pow, exp, NAN, isnan
from libc.stddef cimport ptrdiff_t
from libc.stdlib cimport malloc, free
from pflm.utils._lapack_helper cimport _gels_helper

cdef void gaussian_lwls1d_helper(
    double bw,
    double center,
    np.float64_t *mu,
    np.float64_t[:] x,
    np.float64_t[:] y,
    np.float64_t[:] w,
    int npoly,
    int nder
)  noexcept nogil:
    cdef int info = -1
    cdef ptrdiff_t n = x.shape[0], i, j
    cdef np.float64_t *lx = <np.float64_t*> malloc(n * (npoly+1) * sizeof(np.float64_t))
    cdef np.float64_t *ly = <np.float64_t*> malloc(n * sizeof(np.float64_t))
    cdef double center_minus_xj, sqrt_wj
    for i in range(n):
        center_minus_xj = center - x[i]
        sqrt_wj = sqrt(w[i] * exp(-0.5 * pow(center_minus_xj / bw, 2.0)))

        lx[i] = sqrt_wj
        for j in range(0, npoly):
            lx[i + n*(j+1)] = pow(center_minus_xj, j+1) * sqrt_wj
            ly[i] = y[i] * sqrt_wj
    # dgels_helper(lx, ly, <int> n, npoly+1, &info)
    if info == 0:
        mu[0] = ly[nder] * pow(-1.0, <np.float64_t> nder)
    else:
        mu[0] = NAN
    free(lx)
    free(ly)

def polyfit1d_gaussian_kernel_f64(
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y,
    np.ndarray[np.float64_t, ndim=1] w,
    np.ndarray[np.float64_t, ndim=1] x_new,
    double bandwidth,
    int kernel,
    int degree,
    int deriv,
):
  cdef np.ndarray[np.float64_t] mu = np.zeros(x_new.shape[0])
  gaussian_lwls1d_helper(bandwidth, x_new[0], &mu[0], x, y, w, degree, deriv)
  return mu
