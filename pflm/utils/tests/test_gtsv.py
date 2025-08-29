import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.utils.lapack_helper import _gtsv_memview_f32, _gtsv_memview_f64


def tri_to_full(dl, d, du):
    """Build full tridiagonal matrix A from bands dl, d, du."""
    n = d.size
    A = np.zeros((n, n), dtype=d.dtype)
    A[np.arange(n), np.arange(n)] = d
    A[np.arange(1, n), np.arange(n - 1)] = dl  # sub-diagonal
    A[np.arange(n - 1), np.arange(1, n)] = du  # super-diagonal
    return A


@pytest.mark.parametrize("func, dtype", [
    (_gtsv_memview_f64, np.float64),
    (_gtsv_memview_f32, np.float32),
])
@pytest.mark.parametrize("order", ["C", "F"])
def test_gtsv_f64_single_rhs(func, dtype, order):
    rng = np.random.default_rng(0)
    n, nrhs = 7, 1
    # Make it diagonally dominant to avoid zero pivots (xGTSV has no pivoting)
    d = (2.0 + rng.random(n)).astype(dtype)
    dl = (0.1 * rng.random(n - 1)).astype(dtype)
    du = (0.1 * rng.random(n - 1)).astype(dtype)

    A = tri_to_full(dl, d, du)
    x_true = rng.standard_normal(n).astype(dtype)
    B = (A @ x_true).reshape(n, 1)

    b = np.array(B, order=order, copy=True)
    info = func(dl, d, du, b, n, nrhs)
    assert info == 0
    x_cmp = b.ravel()
    assert_allclose(x_cmp, x_true, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("func, dtype", [
    (_gtsv_memview_f64, np.float64),
    (_gtsv_memview_f32, np.float32),
])
def test_gtsv_zero_pivot_info(func, dtype):
    # Construct a singular tridiagonal: first diagonal element is zero
    n, nrhs = 3, 1
    d = np.array([0, 0, 1], dtype=dtype)
    dl = np.zeros(n - 1, dtype=dtype)
    du = np.zeros(n - 1, dtype=dtype)
    b = np.zeros((n, nrhs), dtype=dtype)  # any RHS; solution undefined

    info = func(dl.copy(), d.copy(), du.copy(), b, n, nrhs)
    # xGTSV returns >0 when a zero pivot is found; 1-based index of pivot
    assert info > 0
