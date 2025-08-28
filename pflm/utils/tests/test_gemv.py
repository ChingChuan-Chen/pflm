import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.utils.blas_helper import BLAS_Trans, _gemv_memview_f32, _gemv_memview_f64


@pytest.mark.parametrize(("dtype", "memview_fn"), [(np.float64, _gemv_memview_f64), (np.float32, _gemv_memview_f32)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("trans", [BLAS_Trans.NoTrans, BLAS_Trans.Trans])
def test_gemv_random(dtype, memview_fn, order, trans):
    rng = np.random.default_rng(0)
    m, n = 3, 4

    A = rng.standard_normal((m, n)).astype(dtype, order=order)
    x = rng.standard_normal(n if trans == BLAS_Trans.NoTrans else m, dtype=dtype)
    y = memview_fn(trans, A, x)

    expected = (A @ x) if trans == BLAS_Trans.NoTrans else (A.T @ x)
    assert_allclose(y, expected, rtol=1e-6, atol=1e-8)


def test_gemv_known_values():
    # m=2, n=3
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")

    # RowMajor + NoTrans: y = A @ x
    x = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    y_rm_nt = _gemv_memview_f64(BLAS_Trans.NoTrans, A, x)
    assert_allclose(np.asarray(y_rm_nt), A @ x, rtol=1e-6, atol=1e-8)

    # ColMajor + Trans: y = A.T @ x2
    x2 = np.array([1.0, 1.0], dtype=np.float64)
    y_cm_t = _gemv_memview_f64(BLAS_Trans.Trans, A, x2)
    assert_allclose(np.asarray(y_cm_t), A.T @ x2, rtol=1e-6, atol=1e-8)
