import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.utils.lapack_helper import _gelss_memview_f32, _gelss_memview_f64


@pytest.mark.parametrize(
    "dtype, func",
    [
        (np.float32, _gelss_memview_f32),
        (np.float64, _gelss_memview_f64),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_gelss_memview(dtype, func, order):
    # Create a simple overdetermined system Ax = b
    # A = [[1, 1], [1, 2], [1, 3]], b = [6, 0, 0]
    A = np.array(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        dtype=dtype, order=order
    )
    B = np.array([[6.0], [0.0], [0.0]], dtype=dtype, order=order)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np, *_ = np.linalg.lstsq(A, B, rcond=None)
    info, _, rank = func(A, B, A.shape[0], A.shape[1], B.shape[1])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gelss failed with info={info}"
    # check rank
    assert rank == 2
    # check if the solution is correct
    x = B[: A.shape[1]]
    assert_allclose(x, x_np, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize(
    "dtype, func",
    [
        (np.float32, _gelss_memview_f32),
        (np.float64, _gelss_memview_f64),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_gelss_memview_multiple_rhs(dtype, func, order):
    # Create a simple overdetermined system Ax = b2 with multiple RHS
    # A = [[1, 1], [1, 2], [1, 3]], b2 = [[6, 6], [0, 2], [0, -2]]
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=dtype, order=order)
    B = np.array([[6, 6], [0, 2], [0, -2]], dtype=dtype, order=order)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np2, *_ = np.linalg.lstsq(A, B, rcond=None)
    info, _, rank = func(A, B, A.shape[0], A.shape[1], B.shape[1])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gelss failed with info={info}"
    # check rank
    assert rank == 2
    # the solution length is A.shape[1] * B.shape[1]
    # the dimensions of x2 should be (A.shape[1], B.shape[1])
    x2 = B[:B.shape[1], :]
    assert_allclose(x2, x_np2, rtol=1e-5, atol=0.0)
