import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.utils.blas_helper import BLAS_Trans
from pflm.utils.lapack_helper import _gels_memview_f32, _gels_memview_f64


@pytest.mark.parametrize("dtype, gels_func", [(np.float32, _gels_memview_f32), (np.float64, _gels_memview_f64)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_gels(dtype, gels_func, order):
    # Create a simple overdetermined system Ax = B
    # A = [[1, 1], [1, 2], [1, 3]], B = [6, 0, 0]
    A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=dtype, order=order)
    B = np.array([[6.0], [0.0], [0.0]], dtype=dtype, order=order)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np, *_ = np.linalg.lstsq(A, B, rcond=None)
    info = gels_func(BLAS_Trans.NoTrans, A, B, A.shape[0], A.shape[1], B.shape[1])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gels failed with info={info}"
    # Optionally, check that B is modified (solution in-place)
    # For least squares, solution is in B[:A.shape[1]]
    x = B[: A.shape[1]]
    # The system is overdetermined, so check Ax ≈ b (first two rows)
    assert_allclose(x, x_np, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype, gels_func", [(np.float32, _gels_memview_f32), (np.float64, _gels_memview_f64)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_gels_multiple_rhs(dtype, gels_func, order):
    # Create a simple overdetermined system Ax = B with multiple RHS
    # A = [[1, 1], [1, 2], [1, 3]], B = [[6, 6], [0, 2], [0, -2]]
    A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=dtype, order=order)
    B = np.array([[6.0, 6.0], [0.0, 2.0], [0.0, -2.0]], dtype=dtype, order=order)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np2, *_ = np.linalg.lstsq(A, B, rcond=None)
    info = gels_func(BLAS_Trans.NoTrans, A, B, A.shape[0], A.shape[1], B.shape[1])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gels failed with info={info}"
    # the solution length is A.shape[1] * b2.shape[1]
    # the dimensions of x2 should be (A.shape[1], b2.shape[1])
    x2 = B[: B.shape[1], :]
    assert_allclose(x2, x_np2, rtol=1e-5, atol=0.0)
