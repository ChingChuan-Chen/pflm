import numpy as np
import pytest

from pflm.utils._lapack_helper import _gels_memview_f32, _gels_memview_f64


@pytest.mark.parametrize("dtype, func", [(np.float32, _gels_memview_f32), (np.float64, _gels_memview_f64)])
def test_gels(dtype, func):
    # Create a simple overdetermined system Ax = b
    # A = [[1, 1], [1, 2], [1, 3]], b = [6, 0, 0]
    A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=dtype)
    b = np.array([6.0, 0.0, 0.0], dtype=dtype)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Convert A and b to Fortran-contiguous arrays
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    b_c = np.ascontiguousarray(b)
    info = func(A_c, b_c, A.shape[0], A.shape[1], 1, A.shape[0], b.shape[0])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gels failed with info={info}"
    # Optionally, check that b_c is modified (solution in-place)
    # For least squares, solution is in b_c[:A.shape[1]]
    x = b_c[: A.shape[1]]
    # The system is overdetermined, so check Ax ≈ b (first two rows)
    assert np.allclose(x, x_np, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype, func", [(np.float32, _gels_memview_f32), (np.float64, _gels_memview_f64)])
def test_gels_multiple_rhs(dtype, func):
    # Create a simple overdetermined system Ax = b2 with multiple RHS
    # A = [[1, 1], [1, 2], [1, 3]], b2 = [[6, 6], [0, 2], [0, -2]]
    A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=dtype)
    b2 = np.array([[6.0, 6.0], [0.0, 2.0], [0.0, -2.0]], dtype=dtype)
    # Also check the solution is close to numpy.linalg.lstsq
    x_np2, *_ = np.linalg.lstsq(A, b2, rcond=None)
    # Convert A and b to Fortran-contiguous arrays
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    b2_c = np.ascontiguousarray(b2.ravel(order="F"))
    info = func(A_c, b2_c, A.shape[0], A.shape[1], b2.shape[1], A.shape[0], b2.shape[0])
    # check info
    # info = 0 means successful exit
    # info < 0 means illegal value
    # info > 0 means the algorithm failed to converge
    assert info == 0, f"LAPACK gels failed with info={info}"
    # the solution length is A.shape[1] * b2.shape[1]
    # the dimensions of x2 should be (A.shape[1], b2.shape[1])
    x2 = np.array(b2_c, dtype=dtype, order="F").reshape(b2.shape[0], b2.shape[1], order="F")[: b2.shape[1], :]
    assert np.allclose(x2, x_np2, rtol=1e-5, atol=0.0)
