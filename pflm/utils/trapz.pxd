from cython cimport floating
from libc.stdint cimport int64_t, uint64_t
from pflm.utils.blas_helper cimport BLAS_Order, BLAS_Trans

cdef void trapz_mat_blas(
    BLAS_Order, uint64_t, uint64_t, floating*, uint64_t, floating*, uint64_t, floating*, int64_t
) noexcept nogil

cdef void trapz_memview(
    BLAS_Order, floating[:, :], floating[:], floating[:], int64_t
) noexcept nogil
