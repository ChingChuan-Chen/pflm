from cython cimport floating

cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous

cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'

cpdef enum BLAS_Jobz:
    NoVec = 110  # correspond to 'n'
    Vec = 118    # correspond to 'v'

cpdef enum BLAS_Uplo:
    Upper = 117  # correspond to 'u'
    Lower = 108  # correspond to 'l'

cdef void _gemv(
    BLAS_Order, BLAS_Trans, int, int, floating, const floating*, int,
    const floating*, int, floating, floating*, int
) noexcept nogil

cdef void _gemm(
    BLAS_Order, BLAS_Trans, BLAS_Trans, int, int, int, floating,
    const floating*, int, const floating*, int, floating, floating*, int
) noexcept nogil
