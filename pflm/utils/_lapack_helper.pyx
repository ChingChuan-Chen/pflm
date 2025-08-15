from cython cimport floating
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport sgels, dgels, sgelss, dgelss, ssyevd, dsyevd

cdef void _gels(
    char trans, int m, int n, int nrhs, floating *a, int lda, floating *b, int ldb, floating *work, int lwork, int *info
) noexcept nogil:
    """Solve the linear equation or least squares problem with QR or LU decomposition."""
    if floating is float:
        sgels(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info)
    else:
        dgels(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info)

cdef void _gels_helper(
    char trans, int m, int n, int nrhs, floating *a, int lda, floating *b, int ldb, int *info
) noexcept nogil:
    cdef int lwork = -1
    cdef floating *work_query = <floating *> malloc(1 * sizeof(floating))

    if not work_query:
        info[0] = -1
        free(work_query)
        return  # Memory allocation failure

    _gels(trans, m, n, nrhs, a, lda, b, ldb, work_query, lwork, info)
    lwork = int(work_query[0])
    free(work_query)

    cdef floating* work = <floating*> malloc(lwork * sizeof(floating))
    if not work:
        info[0] = -1
        return  # Memory allocation failure

    _gels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info)
    free(work)


cdef void _gelss(
    int m, int n, int nrhs, floating *a, int lda, floating *b, int ldb, floating *s, floating *rcond, int *rank, floating *work, int lwork, int *info
) noexcept nogil:
    """Solve the least squares problem with singular value decomposition."""
    if floating is float:
        sgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond, rank, work, &lwork, info)
    else:
        dgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond, rank, work, &lwork, info)


cdef void _gelss_helper(
    int m, int n, int nrhs, floating *a, int lda, floating *b, int ldb, floating *rcond, int *rank, int *info
) noexcept nogil:
    cdef int lwork = -1, s_size = min(m, n)
    cdef floating *s = <floating *>malloc(s_size * sizeof(floating))
    cdef floating *work_query = <floating *>malloc(1 * sizeof(floating))

    if not s or not work_query:
        info[0] = -1
        free(s)
        free(work_query)
        return  # Memory allocation failure

    _gelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work_query, lwork, info)
    lwork = <int> work_query[0]
    free(work_query)

    cdef floating *work = <floating *>malloc(lwork * sizeof(floating))
    if not work:
        info[0] = -1
        free(s)
        free(work)
        return  # Memory allocation failure

    _gelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, info)
    free(s)
    free(work)


cdef void _syevd(char jobz, char uplo, int n, floating *a, int lda, floating *w, floating *work, int lwork, int* iwork, int liwork, int *info) noexcept nogil:
    if floating is float:
        ssyevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info)
    else:
        dsyevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info)


cdef void _syevd_helper(
    char jobz, char uplo, int n, floating *a, int lda, floating *w, int *info
) noexcept nogil:
    cdef int lwork = -1, liwork = -1
    cdef floating *work_query = <floating *> malloc(1 * sizeof(floating))
    cdef int *iwork_query = <int *> malloc(1 * sizeof(int))

    if not work_query or not iwork_query:
        info[0] = -1
        free(work_query)
        free(iwork_query)
        return  # Memory allocation failure

    _syevd(jobz, uplo, n, a, lda, w, work_query, lwork, iwork_query, liwork, info)
    lwork = <int> work_query[0]
    liwork = iwork_query[0]
    free(work_query)
    free(iwork_query)

    cdef floating *work = <floating *> malloc(lwork * sizeof(floating))
    cdef int *iwork = <int *> malloc(liwork * sizeof(int))
    if not work or not iwork:
        info[0] = -1
        free(work)
        free(iwork)
        return  # Memory allocation failure

    _syevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
    free(work)
    free(iwork)


def _gels_memview_f64(np.float64_t[:] A, np.float64_t[:] b, int m, int n, int nrhs, int lda, int ldb):
    cdef int info = 0
    _gels_helper(
        110,  # No transpose
        m,  # Number of rows
        n,  # Number of columns
        nrhs,  # Number of right-hand sides
        &A[0],  # Pointer to the data of A
        lda,  # Leading dimension of A
        &b[0],  # Pointer to the data of b
        ldb,  # Leading dimension of b
        &info  # Info variable to capture LAPACK status
    )
    return info


def _gels_memview_f32(np.float32_t[:] A, np.float32_t[:] b, int m, int n, int nrhs, int lda, int ldb):
    cdef int info = 0
    _gels_helper(
        110,  # No transpose
        m,  # Number of rows
        n,  # Number of columns
        nrhs,  # Number of right-hand sides
        &A[0],  # Pointer to the data of A
        lda,  # Leading dimension of A
        &b[0],  # Pointer to the data of b
        ldb,  # Leading dimension of b
        &info  # Info variable to capture LAPACK status
    )
    return info


def _gelss_memview_f64(np.float64_t[:] A, np.float64_t[:] b, int m, int n, int nrhs, int lda, int ldb):
    cdef int info = 0, rank = 0
    cdef np.float64_t rcond = -1.0
    _gelss_helper(
        m,  # Number of rows
        n,  # Number of columns
        nrhs,  # Number of right-hand sides
        &A[0],  # Pointer to the data of A
        lda,  # Leading dimension of A
        &b[0],  # Pointer to the data of b
        ldb,  # Leading dimension of b
        &rcond,  # Pointer to the reciprocal condition number
        &rank,  # Pointer to the rank
        &info  # Info variable to capture LAPACK status
    )
    return info, rcond, rank

def _gelss_memview_f32(np.float32_t[:] A, np.float32_t[:] b, int m, int n, int nrhs, int lda, int ldb):
    cdef int info = 0, rank = 0
    cdef np.float32_t rcond = -1.0
    _gelss_helper(
        m,  # Number of rows
        n,  # Number of columns
        nrhs,  # Number of right-hand sides
        &A[0],  # Pointer to the data of A
        lda,  # Leading dimension of A
        &b[0],  # Pointer to the data of b
        ldb,  # Leading dimension of b
        &rcond,  # Pointer to the reciprocal condition number
        &rank,  # Pointer to the rank
        &info  # Info variable to capture LAPACK status
    )
    return info, rcond, rank


def _syevd_memview_f64(np.float64_t[:] A, np.float64_t[:] w, int n, int lda):
    cdef int info = 0
    _syevd_helper(
        118,  # Compute eigenvalues and eigenvectors
        108,  # Lower triangle of the matrix is stored
        n,  # Order of the matrix
        &A[0],  # Pointer to the data of A
        lda,  # Leading dimension of A
        &w[0],  # Pointer to the array for eigenvalues
        &info  # Info variable to capture LAPACK status
    )
    return info

def _syevd_memview_f32(np.float32_t[:] A, np.float32_t[:] w, int n, int lda):
    cdef int info = 0
    _syevd_helper(
        118,  # Compute eigenvalues and eigenvectors
        108,  # Lower triangle of the matrix is stored
        n,  # Order of the matrix
        &A[0],  # Pointer to the data of A
        lda,  # Leading dimension of A
        &w[0],  # Pointer to the array for eigenvalues
        &info  # Info variable to capture LAPACK status
    )
    return info
