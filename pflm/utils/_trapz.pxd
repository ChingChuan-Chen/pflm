from cython cimport floating
from libc.stdint cimport uint64_t
from sklearn.utils._cython_blas cimport BLAS_Order

cdef void _trapz_memview(
    floating[:, :], floating[:], floating[:], uint64_t, uint64_t, BLAS_Order
) noexcept nogil
