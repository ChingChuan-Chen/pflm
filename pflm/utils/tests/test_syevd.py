import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.utils._lapack_helper import _syevd_memview_f32, _syevd_memview_f64


@pytest.mark.parametrize(
    "dtype, func",
    [
        (np.float32, _syevd_memview_f32),
        (np.float64, _syevd_memview_f64),
    ],
)
def test_syevd_lower_triangular_matrix_memview(dtype, func):
    A = np.array(
        [
            [6.39, 0.0, 0.0, 0.0, 0.0],
            [0.13, 8.37, 0.0, 0.0, 0.0],
            [-8.23, -4.46, -9.58, 0.0, 0.0],
            [5.71, -6.1, -9.25, 3.72, 0.0],
            [-3.18, 7.21, -7.42, 8.54, 2.51],
        ],
        dtype=dtype,
    )
    w = np.zeros(A.shape[0], dtype=dtype)

    # Also check the solution is close to numpy.linalg.eigh
    eig_val, eig_vec = np.linalg.eigh(A, "L")
    # convert A to column-major order
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    info = func(A_c, w, 108, A.shape[0], A.shape[0])  # 108 = 'l'
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK syevd failed with info={info}"
    # check eigenvalues
    assert_allclose(w, eig_val, rtol=1e-5, atol=0.0)
    # check eigenvectors
    assert_allclose(eig_vec, A_c.reshape(A.shape[0], -1).T, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize(
    "dtype, func",
    [
        (np.float32, _syevd_memview_f32),
        (np.float64, _syevd_memview_f64),
    ],
)
def test_syevd_upper_triangular_matrix_memview(dtype, func):
    A = np.array(
        [
            [6.39, 0.13, -8.23, 5.71, -3.18],
            [0.0, 8.37, -4.46, -6.1, 7.21],
            [0.0, -4.46, -9.58, -9.25, -7.42],
            [0.0, 0.0, 0.0, 3.72, 8.54],
            [0.0, 0.0, 0.0, 0.0, 2.51],
        ],
        dtype=dtype,
    )
    w = np.zeros(A.shape[0], dtype=dtype)

    # Also check the solution is close to numpy.linalg.eigh
    eig_val, eig_vec = np.linalg.eigh(A, "U")
    # convert A to column-major order
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    info = func(A_c, w, 117, A.shape[0], A.shape[0])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK syevd failed with info={info}"
    # check eigenvalues
    assert_allclose(w, eig_val, rtol=1e-5, atol=0.0)
    # check eigenvectors
    assert_allclose(eig_vec, A_c.reshape(A.shape[0], -1).T, rtol=1e-5, atol=0.0)
