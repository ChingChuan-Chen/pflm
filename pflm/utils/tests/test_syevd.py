import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.utils.blas_helper import BLAS_Jobz, BLAS_Uplo
from pflm.utils.lapack_helper import _syevd_memview_f32, _syevd_memview_f64


@pytest.mark.parametrize(
    "dtype, syevd_func",
    [
        (np.float32, _syevd_memview_f32),
        (np.float64, _syevd_memview_f64),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_syevd_lower_triangular_matrix_memview(dtype, syevd_func, order):
    A = np.array(
        [
            [6.39, 0.0, 0.0, 0.0, 0.0],
            [0.13, 8.37, 0.0, 0.0, 0.0],
            [-8.23, -4.46, -9.58, 0.0, 0.0],
            [5.71, -6.1, -9.25, 3.72, 0.0],
            [-3.18, 7.21, -7.42, 8.54, 2.51],
        ],
        dtype=dtype,
        order=order
    )
    w = np.zeros(A.shape[0], dtype=dtype)

    # Also check the solution is close to numpy.linalg.eigh
    eig_val, eig_vec = np.linalg.eigh(A, "L")
    # convert A to column-major order
    info = syevd_func(BLAS_Jobz.Vec, BLAS_Uplo.Lower, A, w, A.shape[0])  # 108 = 'l'
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK syevd failed with info={info}"
    # check eigenvalues
    assert_allclose(w, eig_val, rtol=1e-4, atol=1e-4)
    # check eigenvectors
    assert_allclose(eig_vec, A, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype, syevd_func",
    [
        (np.float32, _syevd_memview_f32),
        (np.float64, _syevd_memview_f64),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_syevd_upper_triangular_matrix_memview(dtype, syevd_func, order):
    A = np.array(
        [
            [6.39, 0.13, -8.23, 5.71, -3.18],
            [0.0, 8.37, -4.46, -6.1, 7.21],
            [0.0, -4.46, -9.58, -9.25, -7.42],
            [0.0, 0.0, 0.0, 3.72, 8.54],
            [0.0, 0.0, 0.0, 0.0, 2.51],
        ],
        dtype=dtype,
        order=order
    )
    w = np.zeros(A.shape[0], dtype=dtype)

    # Also check the solution is close to numpy.linalg.eigh
    eig_val, eig_vec = np.linalg.eigh(A, "U")
    info = syevd_func(BLAS_Jobz.Vec, BLAS_Uplo.Upper, A, w, A.shape[0])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK syevd failed with info={info}"
    # check eigenvalues
    assert_allclose(w, eig_val, rtol=1e-5, atol=1e-4)
    # check eigenvectors
    assert_allclose(eig_vec, A, rtol=1e-4, atol=1e-4)
