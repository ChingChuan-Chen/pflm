from cython cimport floating
from pflm.utils.blas_helper cimport BLAS_Jobz, BLAS_Uplo, BLAS_Trans, BLAS_Order

cdef void _gels(BLAS_Trans, int, int, int, floating*, int, floating*, int, floating*, int, int*) noexcept nogil
cdef void _gels_helper(BLAS_Order, BLAS_Trans, int, int, int, floating*, int, floating*, int, int*) noexcept nogil

cdef void _gelss(
    int, int, int, floating*, int, floating*, int, floating*, floating*, int*, floating*, int, int*
) noexcept nogil
cdef void _gelss_helper(BLAS_Order, int, int, int, floating*, int, floating*, int, floating*, int*, int*) noexcept nogil

cdef void _syevd(
    BLAS_Jobz, BLAS_Uplo, int,
    floating*, int, floating*, floating*, int, int*, int, int*
) noexcept nogil

cdef void _syevd_helper(
    BLAS_Order, BLAS_Jobz, BLAS_Uplo, int,
    floating*, int, floating*, int*
) noexcept nogil

cdef void _posv(BLAS_Order, BLAS_Uplo, int, int, floating*, int, floating*, int, int*) noexcept nogil

cdef void _sysv(BLAS_Order, BLAS_Uplo, int, int, floating*, int, int*, floating*, int, int*) noexcept nogil

cdef void _gtsv(BLAS_Order, int, int, floating*, floating*, floating*, floating*, int, int*) noexcept nogil
