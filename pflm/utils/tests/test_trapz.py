import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.utils import trapz


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_trapz_1d(dtype):
    x = np.linspace(0, 1, 5).astype(dtype)
    y = x**2
    val = trapz(y, x)
    assert np.isclose(val, 0.34375)


@pytest.mark.parametrize("dtype, order", [(np.float64, "F"), (np.float64, "C"), (np.float32, "F"), (np.float32, "C")])
def test_trapz_2d(dtype, order):
    if order == "F":
        x = np.asfortranarray(np.linspace(0, 1, 5))
        y = np.asfortranarray(np.vstack([x**2, x**3]), dtype=dtype)
    elif order == "C":
        x = np.ascontiguousarray(np.linspace(0, 1, 5), dtype=dtype)
        y = np.ascontiguousarray(np.vstack([x**2, x**3]), dtype=dtype)
    val = trapz(y, x)
    assert val.shape == (2,)
    assert_allclose(val, np.array([0.34375, 0.265625], dtype=dtype))


def test_trapz_mismatch_dtype():
    x = np.linspace(0, 1, 5)
    y = x**2
    val = trapz(y.astype(np.float32), x)
    assert np.isclose(val, 0.34375)


def test_trapz_int_dtype():
    x = np.array([1, 2, 3, 4, 5])
    y = x**2
    val = trapz(y, x)
    assert np.isclose(val, 42.0)


def test_trapz_not_enough_points():
    x = np.array([1.0])
    y = np.array([2.0])
    val = trapz(y, x)
    assert np.isclose(val, 0.0)


def test_trapz_exceptions():
    x = np.linspace(0, 1, 5)
    # shape mismatch
    with pytest.raises(ValueError):
        trapz(np.ones(4), x)
    with pytest.raises(ValueError):
        trapz(np.ones((2, 4)), x)
    # wrong dimension of x
    with pytest.raises(ValueError):
        trapz(np.ones((2, 2, 2)), x)
