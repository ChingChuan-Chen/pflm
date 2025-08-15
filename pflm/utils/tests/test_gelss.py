import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.utils._lapack_helper import _gelss_memview_f32, _gelss_memview_f64


@pytest.mark.parametrize(
    "dtype, func",
    [
        (np.float32, _gelss_memview_f32),
        (np.float64, _gelss_memview_f64),
    ],
)
def test_gelss_memview(dtype, func):
    # Create a simple overdetermined system Ax = b
    # A = [[1, 1], [1, 2], [1, 3]], b = [6, 0, 0]
    A = np.array(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        dtype=dtype,
    )
    b = np.array([6.0, 0.0, 0.0], dtype=dtype)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np, *_ = np.linalg.lstsq(A, b, rcond=None)
    # convert A and b to column-major order
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    b_c = np.ascontiguousarray(b.ravel(order="F"))
    info, _, rank = func(A_c, b_c, A.shape[0], A.shape[1], 1, A.shape[0], b.shape[0])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gelss failed with info={info}"
    # check rank
    assert rank == 2
    # check if the solution is correct
    x = b_c[: A.shape[1]]
    assert_allclose(x, x_np, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize(
    "dtype, func",
    [
        (np.float32, _gelss_memview_f32),
        (np.float64, _gelss_memview_f64),
    ],
)
def test_gelss_memview_multiple_rhs(dtype, func):
    # Create a simple overdetermined system Ax = b2 with multiple RHS
    # A = [[1, 1], [1, 2], [1, 3]], b2 = [[6, 6], [0, 2], [0, -2]]
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=dtype)
    b = np.array([[6, 6], [0, 2], [0, -2]], dtype=dtype)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np2, *_ = np.linalg.lstsq(A, b, rcond=None)
    # convert A and b2 to column-major order
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    b_c = np.ascontiguousarray(b.ravel(order="F"))
    info, _, rank = func(A_c, b_c, A.shape[0], A.shape[1], b.shape[1], A.shape[0], b.shape[0])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gelss failed with info={info}"
    # check rank
    assert rank == 2
    # the solution length is A.shape[1] * b.shape[1]
    # the dimensions of x2 should be (A.shape[1], b.shape[1])
    x2 = np.array(b_c, dtype=dtype, order="F").reshape(b.shape[0], b.shape[1], order="F")[: b.shape[1], :]
    assert_allclose(x2, x_np2, rtol=1e-5, atol=0.0)
