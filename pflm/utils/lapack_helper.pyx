from cython cimport floating
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from pflm.utils.blas_helper cimport BLAS_Jobz, BLAS_Uplo, BLAS_Trans, BLAS_Order, ColMajor
from scipy.linalg.cython_lapack cimport sgels, dgels, sgelss, dgelss, ssyevd, dsyevd, sposv, dposv, sgtsv, dgtsv, ssysv, dsysv


cdef void _gels(
    BLAS_Trans trans, int m, int n, int nrhs,
    floating *a, int lda, floating *b, int ldb,
    floating *work, int lwork, int *info
) noexcept nogil:
    """
    Least squares solver (gels), assumes inputs are ColMajor-compatible.
    RowMajor handling is delegated to _gels_helper.
    """
    cdef char trans_ = <char> trans
    if floating is float:
        sgels(&trans_, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info)
    else:
        dgels(&trans_, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info)
    if info[0] < 0:
        info[0] -= 1


cdef void _gels_helper(
    BLAS_Order order, BLAS_Trans trans, int m, int n, int nrhs,
    floating *a, int lda, floating *b, int ldb, int *info
) noexcept nogil:
    """
    LAPACK GELS (least-squares) helper with RowMajor/ColMajor support and workspace query.

    Parameters
    ----------
    order : BLAS_Order
        Input/output memory layout (RowMajor or ColMajor).
    trans : BLAS_Trans
        'N' (NoTrans) solves min || A*X - B ||, 'T' solves min || A^T*X - B ||.
    m, n, nrhs : int
        A is m-by-n; B has nrhs right-hand sides.
    a : floating*
        On entry: A, stored according to `order` and `lda`.
        On exit: overwritten by the QR/LQ factors as in LAPACK xGELS.
    lda : int
        Leading dimension of A.
        - ColMajor: lda >= max(1, m)
        - RowMajor: lda >= max(1, n)
    b : floating*
        On entry:
        - ColMajor: B with leading dimension ldb >= max(m, n)
        - RowMajor: B with shape (m, nrhs) and ldb >= nrhs
        On exit: first min(m, n) rows contain the solution X; if m > n the
        remaining rows may contain residual information (per LAPACK).
    ldb : int
        Leading dimension of B.
        - ColMajor: ldb >= max(1, max(m, n))
        - RowMajor: ldb >= max(1, nrhs)
    info : int*
        Output status (C-style). 0 on success.
        < 0: illegal argument at index -info (C indexing).
        > 0: algorithm-specific failure.

    Behavior
    --------
    - ColMajor: perform lwork=-1 workspace query, then compute in-place on (a, b).
    - RowMajor: check lda >= n (info = -7), ldb >= nrhs (info = -9); transpose A,B
      into temporary ColMajor buffers (Ac with ldac=m, Bc with ldbc=max(m,n)),
      pad Bc rows if m < ldbc, query then compute on temporaries, and copy back
      to RowMajor buffers on success.

    Notes
    -----
    - Negative info from the Fortran routine is adjusted earlier to C-style.
    - This routine is noexcept and safe to call under the GIL released.
    """
    cdef int lwork = -1
    cdef floating *work_query = <floating *> malloc(1 * sizeof(floating))
    if not work_query:
        info[0] = -1
        return

    cdef floating *work
    if order == BLAS_Order.ColMajor:
        if not work_query:
            info[0] = -1
            if work_query: free(work_query)
            return
        _gels(trans, m, n, nrhs, a, lda, b, ldb, work_query, lwork, info)
        lwork = <int> work_query[0]
        free(work_query)

        work = <floating*> malloc(lwork * sizeof(floating))
        if not work:
            info[0] = -1
            return
        _gels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info)
        free(work)
        return

    # RowMajor path (single transpose, reused for query+compute)
    if lda < n:
        info[0] = -7
        return
    if ldb < nrhs:
        info[0] = -9
        return

    cdef int ldac = m
    cdef int ldbc = m if m >= n else n
    cdef floating *a_cm = <floating*> malloc(ldac * n * sizeof(floating))
    cdef floating *b_cm = <floating*> malloc(ldbc * nrhs * sizeof(floating))
    if not a_cm or not b_cm:
        if a_cm: free(a_cm)
        if b_cm: free(b_cm)
        info[0] = -1
        return

    cdef int i, j
    for i in range(m):
        for j in range(n):
            a_cm[i + j * ldac] = a[i * lda + j]
    for j in range(nrhs):
        for i in range(m):
            b_cm[i + j * ldbc] = b[i * ldb + j]
        for i in range(m, ldbc):
            b_cm[i + j * ldbc] = 0

    _gels(trans, m, n, nrhs, a_cm, ldac, b_cm, ldbc, work_query, lwork, info)
    lwork = <int> work_query[0]
    free(work_query)

    work = <floating*> malloc(lwork * sizeof(floating))
    if not work:
        info[0] = -1
        free(a_cm); free(b_cm)
        return

    _gels(trans, m, n, nrhs, a_cm, ldac, b_cm, ldbc, work, lwork, info)
    free(work)

    if info[0] == 0:
        for i in range(m):
            for j in range(n):
                a[i * lda + j] = a_cm[i + j * ldac]
        for j in range(nrhs):
            for i in range(m):
                b[i * ldb + j] = b_cm[i + j * ldbc]
    free(a_cm)
    free(b_cm)


cdef void _gelss(
    int m, int n, int nrhs, floating *a, int lda, floating *b, int ldb,
    floating *s, floating *rcond, int *rank, floating *work, int lwork, int *info
) noexcept nogil:
    """Least squares (SVD) direct LAPACK call; helper handles RowMajor."""
    if floating is float:
        sgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond, rank, work, &lwork, info)
    else:
        dgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond, rank, work, &lwork, info)
    if info[0] < 0:
        info[0] -= 1


cdef void _gelss_helper(
    BLAS_Order order, int m, int n, int nrhs,
    floating *a, int lda, floating *b, int ldb, floating *rcond, int *rank, int *info
) noexcept nogil:
    """
    LAPACK GELSS (SVD-based least-squares) helper with RowMajor/ColMajor support and workspace query.

    Parameters
    ----------
    order : BLAS_Order
        Input/output memory layout (RowMajor or ColMajor).
    m, n, nrhs : int
        A is m-by-n; B has nrhs right-hand sides.
    a : floating*
        On entry: A, stored according to `order` and `lda`.
        On exit: overwritten; contents are LAPACK-internal (SVD factors).
    lda : int
        Leading dimension of A.
        - ColMajor: lda >= max(1, m)
        - RowMajor: lda >= max(1, n)
    b : floating*
        On entry:
        - ColMajor: ldb >= max(1, max(m, n))
        - RowMajor: shape (m, nrhs) and ldb >= nrhs
        On exit: the first n rows contain the solution X (for each RHS),
        consistent with LAPACK xGELSS contract.
    ldb : int
        Leading dimension of B.
        - ColMajor: ldb >= max(1, max(m, n))
        - RowMajor: ldb >= max(1, nrhs)
    rcond : floating*
        RCOND for effective-rank determination. Use negative to trigger default.
    rank : int*
        On exit: effective numerical rank of A as determined by SVD and rcond.
    info : int*
        Output status (C-style). 0 on success.
        < 0: illegal argument at index -info (C indexing).
        > 0: DBDSQR did not converge; INFO = number of unconverged off-diagonals.

    Behavior
    --------
    - ColMajor: allocate S, query with lwork=-1, then compute in-place on (a, b).
    - RowMajor: check lda >= n (info = -6), ldb >= nrhs (info = -8); transpose
      to temporary Ac (ldac=m) and Bc (ldbc=max(m,n)), pad Bc rows if m < ldbc,
      query then compute on temporaries, and copy back on success.

    Notes
    -----
    - Negative info from the Fortran routine is adjusted to C-style in _gelss.
    - This routine is noexcept and safe to call under the GIL released.
    """
    cdef int lwork = -1
    cdef int s_size = min(m, n)
    cdef floating *s = <floating *> malloc(s_size * sizeof(floating))
    cdef floating *work_query = <floating *> malloc(1 * sizeof(floating))
    if not s or not work_query:
        if s: free(s)
        if work_query: free(work_query)
        info[0] = -1
        return

    cdef floating *work
    if order == BLAS_Order.ColMajor:
        _gelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work_query, lwork, info)
        lwork = <int> work_query[0]
        free(work_query)

        work = <floating *> malloc(lwork * sizeof(floating))
        if not work:
            free(s)
            info[0] = -1
            return

        _gelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, info)
        free(work)
        free(s)
        return

    # RowMajor path: basic checks (argument indices with BLAS_Order first)
    # 1=order,2=m,3=n,4=nrhs,5=a,6=lda,7=b,8=ldb,9=rcond,10=rank,11=info
    if lda < n:
        free(s); free(work_query)
        info[0] = -6
        return
    if ldb < nrhs:
        free(s); free(work_query)
        info[0] = -8
        return

    cdef int ldac = m
    cdef int ldbc = m if m >= n else n  # LAPACK requires ldb >= max(m,n) for robust storage
    cdef floating *a_cm = <floating*> malloc(ldac * n * sizeof(floating))
    cdef floating *b_cm = <floating*> malloc(ldbc * nrhs * sizeof(floating))
    if not a_cm or not b_cm:
        if a_cm: free(a_cm)
        if b_cm: free(b_cm)
        free(s); free(work_query)
        info[0] = -1
        return

    cdef int i, j
    # RowMajor -> ColMajor
    for i in range(m):
        for j in range(n):
            a_cm[i + j * ldac] = a[i * lda + j]
    for j in range(nrhs):
        for i in range(m):
            b_cm[i + j * ldbc] = b[i * ldb + j]
        for i in range(m, ldbc):
            b_cm[i + j * ldbc] = 0

    # Query
    _gelss(m, n, nrhs, a_cm, ldac, b_cm, ldbc, s, rcond, rank, work_query, lwork, info)
    lwork = <int> work_query[0]
    free(work_query)

    # Compute
    work = <floating *> malloc(lwork * sizeof(floating))
    if not work:
        free(a_cm); free(b_cm); free(s)
        info[0] = -1
        return

    _gelss(m, n, nrhs, a_cm, ldac, b_cm, ldbc, s, rcond, rank, work, lwork, info)
    free(work)
    free(s)

    if info[0] == 0:
        # Copy back to RowMajor buffers (first m rows; solution typically in first n rows)
        for i in range(m):
            for j in range(n):
                a[i * lda + j] = a_cm[i + j * ldac]
        for j in range(nrhs):
            for i in range(m):
                b[i * ldb + j] = b_cm[i + j * ldbc]

    free(a_cm)
    free(b_cm)


cdef void _syevd(
    BLAS_Jobz jobz, BLAS_Uplo uplo, int n,
    floating *a, int lda, floating *w, floating *work, int lwork,
    int* iwork, int liwork, int *info
) noexcept nogil:
    """
    Symmetric EVD (syevd), assumes inputs are ColMajor-compatible.
    RowMajor handling is delegated to _syevd_helper.
    """
    cdef char jobz_ = <char> jobz
    cdef char uplo_ = <char> uplo
    if floating is float:
        ssyevd(&jobz_, &uplo_, &n, a, &lda, w, work, &lwork, iwork, &liwork, info)
    else:
        dsyevd(&jobz_, &uplo_, &n, a, &lda, w, work, &lwork, iwork, &liwork, info)
    if info[0] < 0:
        info[0] -= 1

cdef void _syevd_helper(
    BLAS_Order order, BLAS_Jobz jobz, BLAS_Uplo uplo, int n,
    floating *a, int lda, floating *w, int *info
) noexcept nogil:
    """
    LAPACK SYEVD (symmetric eigen-decomposition) helper with RowMajor/ColMajor support and workspace query.

    Parameters
    ----------
    order : BLAS_Order
        Input/output memory layout (RowMajor or ColMajor).
    jobz : BLAS_Jobz
        'N' (NoVec) for eigenvalues only, 'V' (Vec) for eigenvalues and eigenvectors.
    uplo : BLAS_Uplo
        Triangle of A to reference: 'U' (Upper) or 'L' (Lower).
    n : int
        Order of the matrix A (n-by-n).
    a : floating*
        On entry: symmetric matrix A stored per `order` and `lda`.
        On exit: if jobz='V', contains orthonormal eigenvectors; otherwise overwritten.
    lda : int
        Leading dimension of A.
        - ColMajor: lda >= max(1, n)
        - RowMajor: lda >= max(1, n) (number of columns)
    w : floating*
        On exit: eigenvalues in ascending order (length n).
    info : int*
        Output status (C-style). 0 on success.
        < 0: illegal argument at index -info (C indexing).
        > 0: algorithm failed to converge.

    Behavior
    --------
    - ColMajor: query lwork/liwork with lwork=liwork=-1, then compute in-place on A.
    - RowMajor: check lda >= n (info = -6); transpose A into temporary ColMajor
      buffer Ac (ldac=n), query then compute on Ac, and copy back to RowMajor on success.

    Notes
    -----
    - Negative info from the Fortran routine is adjusted earlier to C-style.
    - This routine is noexcept and safe to call under the GIL released.
    """
    cdef int lwork = -1, liwork = -1
    cdef floating *work_query = <floating *> malloc(1 * sizeof(floating))
    cdef int *iwork_query = <int *> malloc(1 * sizeof(int))
    cdef floating *work
    cdef int *iwork
    if not work_query or not iwork_query:
        info[0] = -1
        if work_query: free(work_query)
        if iwork_query: free(iwork_query)
        return
    if order == BLAS_Order.ColMajor:
        _syevd(jobz, uplo, n, a, lda, w, work_query, lwork, iwork_query, liwork, info)
        lwork = <int> work_query[0]
        liwork = iwork_query[0]
        free(work_query)
        free(iwork_query)

        work = <floating *> malloc(lwork * sizeof(floating))
        iwork = <int *> malloc(liwork * sizeof(int))
        if not work or not iwork:
            info[0] = -1
            if work: free(work)
            if iwork: free(iwork)
            return

        _syevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
        free(work)
        free(iwork)
        return

    # RowMajor path (single transpose)
    if lda < n:
        info[0] = -6
        return

    cdef int ldac = n
    cdef floating *a_cm = <floating*> malloc(n * n * sizeof(floating))
    if not a_cm:
        info[0] = -1
        return

    cdef int i, j
    for i in range(n):
        for j in range(n):
            a_cm[i + j * ldac] = a[i * lda + j]

    _syevd(jobz, uplo, n, a_cm, ldac, w, work_query, lwork, iwork_query, liwork, info)
    lwork = <int> work_query[0]
    liwork = iwork_query[0]
    free(work_query)
    free(iwork_query)

    work = <floating *> malloc(lwork * sizeof(floating))
    iwork = <int *> malloc(liwork * sizeof(int))
    if not work or not iwork:
        info[0] = -1
        if work: free(work)
        if iwork: free(iwork)
        free(a_cm)
        return

    _syevd(jobz, uplo, n, a_cm, ldac, w, work, lwork, iwork, liwork, info)

    if info[0] == 0:
        for i in range(n):
            for j in range(n):
                a[i * lda + j] = a_cm[i + j * ldac]

    free(work)
    free(iwork)
    free(a_cm)


cdef void _posv(
    BLAS_Order order, BLAS_Uplo uplo, int n, int nrhs,
    floating *a, int lda, floating *b, int ldb, int *info
) noexcept nogil:
    """
    Solve SPD system A * X = B via Cholesky (POSV), LAPACKE-like API.

    Parameters
    ----------
    order : BLAS_Order
        RowMajor (C-order) or ColMajor (Fortran-order) layout of input/output.
    uplo : BLAS_Uplo
        Which triangle of A is referenced.
    n : int
        The order of A (A is n-by-n).
    nrhs : int
        Number of right-hand sides (columns of B and X).
    a : pointer to floating
        Pointer to A data. Overwritten by the factorization.
        Layout according to `order` and leading dimension `lda`.
    lda : int
        Leading dimension of A. For ColMajor: lda >= n. For RowMajor: lda >= n (number of columns).
    b : pointer to floating
        Pointer to B data. Overwritten by the solution X.
        Layout according to `order` and leading dimension `ldb`.
    ldb : int
        Leading dimension of B. For ColMajor: ldb >= n. For RowMajor: ldb >= nrhs.
    info : int*
        Output status (0=success). Negative on illegal argument (C-style index).
    """
    cdef char uplo_ = <char> uplo

    if order == BLAS_Order.ColMajor:
        # Direct call on column-major buffers
        if floating is float:
            sposv(&uplo_, &n, &nrhs, a, &lda, b, &ldb, info)
        else:
            dposv(&uplo_, &n, &nrhs, a, &lda, b, &ldb, info)
        # LAPACKE adjusts negative info by -1 (Fortran 1-based -> C 0-based)
        if info[0] < 0:
            info[0] -= 1
        return

    # RowMajor path (LAPACKE-like): dimension checks first.
    # LAPACKE returns -6 if lda < n, -8 if ldb < nrhs.
    if lda < n:
        info[0] = -6
        return
    if ldb < nrhs:
        info[0] = -8
        return

    # Copy to temporary column-major buffers, call, then copy back.
    cdef int ldac = n
    cdef int ldbc = n
    cdef floating *a_cm = <floating*> malloc(n * n * sizeof(floating))
    cdef floating *b_cm = <floating*> malloc(n * nrhs * sizeof(floating))
    if not a_cm or not b_cm:
        if a_cm: free(a_cm)
        if b_cm: free(b_cm)
        info[0] = -1  # memory allocation failure
        return

    cdef int i, j
    # Transpose A: RowMajor -> ColMajor by element (logical indices preserved)
    for i in range(n):
        for j in range(n):
            # a[i, j] with RowMajor leading dimension lda (num of columns)
            a_cm[i + j * ldac] = a[i * lda + j]

    # Transpose B: RowMajor -> ColMajor (n x nrhs)
    for i in range(n):
        for j in range(nrhs):
            b_cm[i + j * ldbc] = b[i * ldb + j]

    # Call Fortran LAPACK on column-major temporaries
    if floating is float:
        sposv(&uplo_, &n, &nrhs, a_cm, &ldac, b_cm, &ldbc, info)
    else:
        dposv(&uplo_, &n, &nrhs, a_cm, &ldac, b_cm, &ldbc, info)

    # Adjust negative info as LAPACKE does
    if info[0] < 0:
        info[0] -= 1

    if info[0] == 0:
        # Transpose back outputs
        for i in range(n):
            for j in range(n):
                a[i * lda + j] = a_cm[i + j * ldac]
        for i in range(n):
            for j in range(nrhs):
                b[i * ldb + j] = b_cm[i + j * ldbc]

    free(a_cm)
    free(b_cm)


cdef void _sysv(
    BLAS_Order order, BLAS_Uplo uplo, int n, int nrhs,
    floating *a, int lda, int *ipiv, floating *b, int ldb, int *info
) noexcept nogil:
    """
    Solve symmetric (possibly indefinite) system A * X = B via LAPACK xSYSV
    (Bunch–Kaufman: factorization + solve), LAPACKE-like API with Row/Col support.

    Parameters
    ----------
    order : BLAS_Order
        RowMajor (C-order) or ColMajor (Fortran-order) for input/output buffers.
    uplo : BLAS_Uplo
        'U' or 'L' triangle of A to reference.
    n : int
        Order of A (n-by-n).
    nrhs : int
        Number of RHS columns in B/X.
    a : floating*
        On entry: symmetric matrix A, per `order` with leading dimension `lda`.
        On exit: overwritten by factorization (L/U and D blocks).
    lda : int
        Leading dimension of A. For both layouts require lda >= n.
    ipiv : int*
        Pivot indices (length >= n). On exit: 1-based pivoting info.
    b : floating*
        On entry: RHS B per `order` with leading dimension `ldb`.
        On exit: solution X in place of B.
    ldb : int
        Leading dimension of B. ColMajor: ldb >= n; RowMajor: ldb >= nrhs.
    info : int*
        0 success; <0 illegal arg (C-style index); >0 singularity/zero pivot.
    """
    cdef char uplo_ = <char> uplo
    cdef int lwork = -1
    cdef floating *work_query = <floating *> malloc(1 * sizeof(floating))
    if not work_query:
        info[0] = -1
        return

    cdef floating *work
    if order == BLAS_Order.ColMajor:
        # Query workspace
        if floating is float:
            ssysv(&uplo_, &n, &nrhs, a, &lda, ipiv, b, &ldb, work_query, &lwork, info)
        else:
            dsysv(&uplo_, &n, &nrhs, a, &lda, ipiv, b, &ldb, work_query, &lwork, info)
        if info[0] < 0:
            info[0] -= 1
        lwork = <int> work_query[0]
        free(work_query)

        work = <floating *> malloc(lwork * sizeof(floating))
        if not work:
            info[0] = -1
            return

        if floating is float:
            ssysv(&uplo_, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, info)
        else:
            dsysv(&uplo_, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, info)
        if info[0] < 0:
            info[0] -= 1
        free(work)
        return

    # RowMajor path: LAPACKE-like checks: -6 for lda, -9 for ldb
    if lda < n:
        free(work_query)
        info[0] = -6
        return
    if ldb < nrhs:
        free(work_query)
        info[0] = -9
        return

    # Transpose into column-major temporaries
    cdef int ldac = n
    cdef int ldbc = n
    cdef floating *a_cm = <floating*> malloc(n * n * sizeof(floating))
    cdef floating *b_cm = <floating*> malloc(n * nrhs * sizeof(floating))
    cdef int *ipiv_tmp = <int*> malloc(n * sizeof(int))
    if not a_cm or not b_cm or not ipiv_tmp:
        if a_cm: free(a_cm)
        if b_cm: free(b_cm)
        if ipiv_tmp: free(ipiv_tmp)
        free(work_query)
        info[0] = -1
        return

    cdef int i, j
    for i in range(n):
        for j in range(n):
            a_cm[i + j * ldac] = a[i * lda + j]
    for i in range(n):
        for j in range(nrhs):
            b_cm[i + j * ldbc] = b[i * ldb + j]

    # Query
    if floating is float:
        ssysv(&uplo_, &n, &nrhs, a_cm, &ldac, ipiv_tmp, b_cm, &ldbc, work_query, &lwork, info)
    else:
        dsysv(&uplo_, &n, &nrhs, a_cm, &ldac, ipiv_tmp, b_cm, &ldbc, work_query, &lwork, info)
    if info[0] < 0:
        info[0] -= 1
    lwork = <int> work_query[0]
    free(work_query)

    work = <floating *> malloc(lwork * sizeof(floating))
    if not work:
        free(a_cm); free(b_cm); free(ipiv_tmp)
        info[0] = -1
        return

    # Compute
    if floating is float:
        ssysv(&uplo_, &n, &nrhs, a_cm, &ldac, ipiv_tmp, b_cm, &ldbc, work, &lwork, info)
    else:
        dsysv(&uplo_, &n, &nrhs, a_cm, &ldac, ipiv_tmp, b_cm, &ldbc, work, &lwork, info)
    if info[0] < 0:
        info[0] -= 1

    if info[0] == 0:
        # Copy back factors/solution and pivots
        for i in range(n):
            for j in range(n):
                a[i * lda + j] = a_cm[i + j * ldac]
        for i in range(n):
            for j in range(nrhs):
                b[i * ldb + j] = b_cm[i + j * ldbc]
        for i in range(n):
            ipiv[i] = ipiv_tmp[i]

    free(work)
    free(a_cm)
    free(b_cm)
    free(ipiv_tmp)


cdef void _gtsv(
    BLAS_Order order, int n, int nrhs,
    floating *dl, floating *d, floating *du,
    floating *b, int ldb, int *info
) noexcept nogil:
    """
    Solve a general tridiagonal system A * X = B via LAPACK xGTSV (no pivoting).

    This routine wraps SGTSV/DGTSV depending on `floating` (float32/float64).
    It overwrites the tridiagonal bands (dl, d, du) with the LU factors and
    overwrites `b` in-place with the solution `X`.

    Parameters
    ----------
    order : BLAS_Order
        Input/output memory layout for `b`: RowMajor (C-order) or ColMajor (Fortran-order).
        The tridiagonal bands `dl, d, du` are always supplied as 1D arrays independent of layout.
    n : int
        Order of the system (A is n-by-n).
    nrhs : int
        Number of right-hand sides (columns of B and X).
    dl : floating*, shape (n-1,)
        Sub-diagonal of A. Overwritten on exit by multipliers for the LU factorization.
    d : floating*, shape (n,)
        Main diagonal of A. Overwritten on exit by the diagonal of U.
    du : floating*, shape (n-1,)
        Super-diagonal of A. Overwritten on exit by the super-diagonal of U.
    b : floating*
        Right-hand side matrix B. On exit, contains the solution X.
        - ColMajor: leading dimension `ldb >= max(1, n)`, stored n-by-nrhs.
        - RowMajor: treated as shape (n, nrhs) with `ldb >= max(1, nrhs)`.
    ldb : int
        Leading dimension of B with respect to `order` (see above).
    info : int*
        Output status (C-style):
        - 0  : success.
        - <0 : illegal argument at index -info.
        - >0 : factorization produced a zero pivot at U(info, info); no solution.

    Notes
    -----
    - xGTSV performs Gaussian elimination without pivoting; it may be unstable
      for ill-conditioned problems. For more robust behavior use xGTTRF + xGTTRS.
    - RowMajor path copies only B to a temporary ColMajor buffer; the bands
      (dl, d, du) are used directly.

    See Also
    --------
    dgttrf, dgttrs : LU factorization and solve for tridiagonal with partial pivoting.
    dptsv : SPD tridiagonal solver (if A is symmetric positive definite).
    """
    if order == BLAS_Order.ColMajor:
        # Direct call on column-major buffers
        if floating is float:
            sgtsv(&n, &nrhs, dl, d, du, b, &ldb, info)
        else:
            dgtsv(&n, &nrhs, dl, d, du, b, &ldb, info)
        if info[0] < 0:
            info[0] -= 1
        return

    # RowMajor path (LAPACKE-like): dimension checks first.
    # LAPACKE returns -8 if ldb < nrhs.
    if ldb < nrhs:
        info[0] = -8
        return

    # Copy to temporary column-major buffers, call, then copy back.
    cdef int ldbc = n
    cdef floating *b_cm = <floating*> malloc(n * nrhs * sizeof(floating))
    if not b_cm:
        info[0] = -1  # memory allocation failure
        return

    cdef int i, j
    # Transpose B: RowMajor -> ColMajor (n x nrhs)
    for i in range(n):
        for j in range(nrhs):
            b_cm[i + j * ldbc] = b[i * ldb + j]

    # Call Fortran LAPACK on column-major temporaries
    if floating is float:
        sgtsv(&n, &nrhs, dl, d, du, b_cm, &ldbc, info)
    else:
        dgtsv(&n, &nrhs, dl, d, du, b_cm, &ldbc, info)

    # Adjust negative info as LAPACKE does
    if info[0] < 0:
        info[0] -= 1

    if info[0] == 0:
        # Transpose back outputs
        for i in range(n):
            for j in range(nrhs):
                b[i * ldb + j] = b_cm[i + j * ldbc]

    free(b_cm)


def _gels_memview_f64(
    BLAS_Trans trans,
    np.float64_t[:, :] A, np.float64_t[:, :] B,
    int m, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = m if order == ColMajor else n
    cdef int ldb = m if order == ColMajor else nrhs
    cdef int info = 0
    _gels_helper(order, trans, m, n, nrhs, &A[0, 0], lda, &B[0, 0], ldb, &info)
    return info


def _gels_memview_f32(
    BLAS_Trans trans,
    np.float32_t[:, :] A, np.float32_t[:, :] B,
    int m, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = m if order == ColMajor else n
    cdef int ldb = m if order == ColMajor else nrhs
    cdef int info = 0
    _gels_helper(order, trans, m, n, nrhs, &A[0, 0], lda, &B[0, 0], ldb, &info)
    return info


def _gelss_memview_f64(
    np.float64_t[:, :] A, np.float64_t[:, :] B,
    int m, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = m if order == ColMajor else n
    cdef int ldb = m if order == ColMajor else nrhs
    cdef int info = 0, rank = 0
    cdef np.float64_t rcond = -1.0
    _gelss_helper(order, m, n, nrhs, &A[0, 0], lda, &B[0, 0], ldb, &rcond, &rank, &info)
    return info, rcond, rank

def _gelss_memview_f32(
    np.float32_t[:, :] A, np.float32_t[:, :] B,
    int m, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = m if order == ColMajor else n
    cdef int ldb = m if order == ColMajor else nrhs
    cdef int info = 0, rank = 0
    cdef np.float32_t rcond = -1.0
    _gelss_helper(order, m, n, nrhs, &A[0, 0], lda, &B[0, 0], ldb, &rcond, &rank, &info)
    return info, rcond, rank


def _syevd_memview_f64(
    BLAS_Jobz jobz, BLAS_Uplo uplo,
    np.float64_t[:, :] A, np.float64_t[:] w, int n
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int info = 0
    _syevd_helper(order, jobz, uplo, n, &A[0, 0], n, &w[0], &info)
    return info


def _syevd_memview_f32(
    BLAS_Jobz jobz, BLAS_Uplo uplo,
    np.float32_t[:, :] A, np.float32_t[:] w, int n
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int info = 0
    _syevd_helper(order, jobz, uplo, n, &A[0, 0], n, &w[0], &info)
    return info


def _posv_memview_f64(
    BLAS_Uplo uplo,
    np.float64_t[:, :] A, np.float64_t[:, :] b,
    int m, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = m if order == BLAS_Order.ColMajor else n
    cdef int ldb = m if order == BLAS_Order.ColMajor else nrhs
    cdef int info = 0
    _posv(order, uplo, n, nrhs, &A[0, 0], lda, &b[0, 0], ldb, &info)
    return info


def _posv_memview_f32(
    BLAS_Uplo uplo,
    np.float32_t[:, :] A, np.float32_t[:, :] b,
    int m, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = m if order == BLAS_Order.ColMajor else n
    cdef int ldb = m if order == BLAS_Order.ColMajor else nrhs
    cdef int info = 0
    _posv(order, uplo, n, nrhs, &A[0, 0], lda, &b[0, 0], ldb, &info)
    return info


def _sysv_memview_f64(
    BLAS_Uplo uplo,
    np.float64_t[:, :] A,  # n x n
    np.int32_t[:] ipiv,    # length n
    np.float64_t[:, :] B,  # n x nrhs
    int n, int nrhs
):
    if sizeof(int) != sizeof(np.int32_t):
        raise RuntimeError("C int is not 32-bit.")
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = n   # square; both layouts require >= n
    cdef int ldb = n if order == BLAS_Order.ColMajor else nrhs
    cdef int *ipiv_ptr = <int *> &ipiv[0]
    cdef int info = 0
    _sysv(order, uplo, n, nrhs, &A[0, 0], lda, ipiv_ptr, &B[0, 0], ldb, &info)
    return info


def _sysv_memview_f32(
    BLAS_Uplo uplo,
    np.float32_t[:, :] A,  # n x n
    np.int32_t[:] ipiv,    # length n
    np.float32_t[:, :] B,  # n x nrhs
    int n, int nrhs
):
    if sizeof(int) != sizeof(np.int32_t):
        raise RuntimeError("C int is not 32-bit.")
    cdef BLAS_Order order = BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
    cdef int lda = n
    cdef int ldb = n if order == BLAS_Order.ColMajor else nrhs
    cdef int *ipiv_ptr = <int *> &ipiv[0]
    cdef int info = 0
    _sysv(order, uplo, n, nrhs, &A[0, 0], lda, ipiv_ptr, &B[0, 0], ldb, &info)
    return info


def _gtsv_memview_f64(
    np.float64_t[:] dl, np.float64_t[:] d, np.float64_t[:] du,
    np.float64_t[:, :] b, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if b.strides[0] == b.itemsize else BLAS_Order.RowMajor
    cdef int ldb = n if order == BLAS_Order.ColMajor else nrhs
    cdef int info = 0
    _gtsv(order, n, nrhs, &dl[0], &d[0], &du[0], &b[0, 0], ldb, &info)
    return info


def _gtsv_memview_f32(
    np.float32_t[:] dl, np.float32_t[:] d, np.float32_t[:] du,
    np.float32_t[:, :] b, int n, int nrhs
):
    cdef BLAS_Order order = BLAS_Order.ColMajor if b.strides[0] == b.itemsize else BLAS_Order.RowMajor
    cdef int ldb = n if order == BLAS_Order.ColMajor else nrhs
    cdef int info = 0
    _gtsv(order, n, nrhs, &dl[0], &d[0], &du[0], &b[0, 0], ldb, &info)
    return info
