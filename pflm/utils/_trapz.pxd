from cython cimport floating
from libc.stdint cimport int64_t
from sklearn.utils._cython_blas cimport BLAS_Order

cdef void _trapz_memview(
    floating[:, :], floating[:], floating[:], int64_t, int64_t, BLAS_Order
) noexcept nogil
