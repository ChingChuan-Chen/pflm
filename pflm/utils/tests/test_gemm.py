import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.utils.blas_helper import BLAS_Trans, _gemm_memview_f32, _gemm_memview_f64


@pytest.mark.parametrize(
    ("dtype", "fn"),
    [(np.float64, _gemm_memview_f64), (np.float32, _gemm_memview_f32)],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_gemm_random_typed_wrappers(dtype, fn, order):
    rng = np.random.default_rng(0)
    m, n, k = 3, 4, 5

    # Underlying (pre-op) matrices; order determines memory layout.
    A = rng.standard_normal((m, k)).astype(dtype, order=order)
    B = rng.standard_normal((k, n)).astype(dtype, order=order)

    C = fn(BLAS_Trans.NoTrans, BLAS_Trans.NoTrans, A, B, m, n, k)
    assert C.shape == (m, n)
    assert_allclose(np.asarray(C), A @ B, rtol=1e-4, atol=1e-6)


def test_gemm_known_values():
    m, n, k = 2, 3, 4
    A0 = np.arange(m * k, dtype=np.float64).reshape(m, k, order="C")
    B0 = np.arange(k * n, dtype=np.float64).reshape(k, n, order="C")

    # NoTrans, NoTrans
    C_nn = _gemm_memview_f64(BLAS_Trans.NoTrans, BLAS_Trans.NoTrans, A0, B0, m, n, k)
    assert_allclose(C_nn, A0 @ B0, rtol=1e-4, atol=1e-6)

    # Trans, Trans with Fortran order buffers
    A1 = np.asfortranarray(A0.T)
    B1 = np.asfortranarray(B0.T)
    C_tt = _gemm_memview_f64(BLAS_Trans.Trans, BLAS_Trans.Trans, A1, B1, m, n, k)
    assert_allclose(C_tt, A0 @ B0, rtol=1e-4, atol=1e-6)
