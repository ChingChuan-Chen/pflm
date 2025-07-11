import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.interp import interp1d


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp1d_happy_case(dtype):
    x = np.array([0.1, 0.35, 0.5, 0.75, 0.8], dtype=dtype)
    y = x**2

    x_new = np.linspace(0.1, 0.8, 5, dtype=dtype)
    expected_linear = np.array([0.01, 0.08875, 0.2075, 0.40625, 0.64], dtype=dtype)
    expected_spline = np.array([0.01, 0.075625, 0.2025, 0.390625, 0.64], dtype=dtype)
    assert_allclose(interp1d(x, y, x_new, "linear"), expected_linear, rtol=1e-5, atol=0.0)
    assert_allclose(interp1d(x, y, x_new, "spline"), expected_spline, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp1d_unsorted(dtype):
    x = np.array([0.1, 0.35, 0.1, 0.6, 0.3, 0.8, 0.55], dtype=dtype)
    y = x**2

    x_new = np.linspace(0.1, 0.8, 5, dtype=dtype)
    expected_linear = np.array([0.01, 0.08, 0.2125, 0.395, 0.64], dtype=dtype)
    expected_spline = np.array([0.01, 0.075625, 0.2025, 0.390625, 0.64], dtype=dtype)
    assert_allclose(interp1d(x, y, x_new, "linear"), expected_linear, rtol=1e-5, atol=0.0)
    assert_allclose(interp1d(x[1:], y[1:], x_new, "linear"), expected_linear, rtol=1e-5, atol=0.0)

    assert_allclose(interp1d(x, y, x_new, "spline"), expected_spline, rtol=1e-5, atol=0.0)
    assert_allclose(interp1d(x[1:], y[1:], x_new, "spline"), expected_spline, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp1d_big(dtype):
    x = np.array([0.0, 0.8, 0.3, 0.05, 0.6, 0.9, 0.5, 0.2, 0.0, 0.7, 1.0, 0.4, 1.1], dtype=dtype)
    y = x**2

    x_new = np.linspace(0.1, 1.1, 21, dtype=dtype)
    expected_linear = np.array(
        [0.015, 0.0275, 0.04, 0.065, 0.09, 0.125, 0.16, 0.205, 0.25, 0.305, 0.36, 0.425, 0.49, 0.565, 0.64, 0.725, 0.81, 0.905, 1.0, 1.105, 1.21],
        dtype=dtype,
    )
    expected_spline = np.array(
        [
            0.01,
            0.0225,
            0.04,
            0.0625,
            0.09,
            0.1225,
            0.16,
            0.2025,
            0.25,
            0.3025,
            0.36,
            0.4225,
            0.49,
            0.5625,
            0.64,
            0.7225,
            0.81,
            0.9025,
            1.0,
            1.1025,
            1.21,
        ],
        dtype=dtype,
    )
    assert_allclose(interp1d(x, y, x_new, "linear"), expected_linear, rtol=1e-5, atol=0.0)
    assert_allclose(interp1d(x, y, x_new, "spline"), expected_spline, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp1d_two_points(dtype):
    x = np.array([0.1, 0.8], dtype=dtype)
    y = x**2

    x_new = np.linspace(0.0, 1.0, 3, dtype=dtype)
    expected = np.array([np.nan, 0.37, np.nan], dtype=dtype)
    assert_allclose(interp1d(x, y, x_new, "linear"), expected, rtol=1e-5, atol=0.0)
    assert_allclose(interp1d(x, y, x_new, "spline"), expected, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp1d_spline_small_size(dtype):
    x_new = np.linspace(0.1, 0.8, 5, dtype=dtype)

    x1 = np.array([0.1, 0.8], dtype=dtype)
    y1 = x1**2
    expected1 = np.array([0.01, 0.1675, 0.325, 0.4825, 0.64], dtype=dtype)
    assert_allclose(interp1d(x1, y1, x_new, "spline"), expected1, rtol=1e-5, atol=0.0)

    x2 = np.array([0.1, 0.5, 0.8], dtype=dtype)
    y2 = x2**2
    expected2 = np.array([0.01, 0.075625, 0.2025, 0.390625, 0.64], dtype=dtype)
    assert_allclose(interp1d(x2, y2, x_new, "spline"), expected2, rtol=1e-5, atol=0.0)

    x3 = np.array([0.1, 0.5, 0.6, 0.8], dtype=dtype)
    y3 = x3**2
    assert_allclose(interp1d(x3, y3, x_new, "spline"), expected2, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp1d_exception(dtype):
    x_new = np.linspace(0.1, 0.8, 5, dtype=dtype)
    x = np.array([0.1, 0.8], dtype=dtype)
    y = x**2

    with pytest.raises(ValueError, match="x, y, and x_new must be 1-dimensional arrays."):
        interp1d(np.array([x, x]), y, x_new, -1)
    with pytest.raises(ValueError, match="x, y, and x_new must not be empty."):
        interp1d(np.array([], dtype=dtype), y, x_new, -1)
    with pytest.raises(ValueError, match="Invalid method. Use 'linear' or 'spline'."):
        interp1d(x, y, x_new, -1)
    with pytest.raises(ValueError, match="Invalid method. Use 'linear' or 'spline'."):
        interp1d(x, y, x_new, 3)
    with pytest.raises(ValueError, match="x must have the same size as y."):
        interp1d(x[:-1], y, x_new, "linear")
    with pytest.raises(ValueError, match="x must have the same size as y."):
        interp1d(x, y[:-1], x_new, "linear")


def test_interp1d_type_mismatch():
    x_f64 = np.array([0.1, 0.8], dtype=np.float64)
    x_f32 = np.array([0.1, 0.8], dtype=np.float32)
    y = np.array([0.01, 0.64], dtype=np.float64)
    x_new = np.linspace(0.1, 0.8, 5, dtype=np.float32)
    y_new_f32 = interp1d(x_f32, y, x_new, "linear")
    assert y_new_f32.dtype == np.float32
    y_new_f64 = interp1d(x_f64, y, x_new, "linear")
    assert y_new_f64.dtype == np.float64

    y_new_f32_spline = interp1d(x_f32, y, x_new, "spline")
    assert y_new_f32_spline.dtype == np.float32
    y_new_f64_spline = interp1d(x_f64, y, x_new, "spline")
    assert y_new_f64_spline.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp1d_nan_inputs(dtype):
    # x contains NaN
    x = np.array([0.1, np.nan, 0.8], dtype=dtype)
    y = np.array([0.01, 0.04, 0.64], dtype=dtype)
    x_new = np.linspace(0.1, 0.8, 5, dtype=dtype)
    with pytest.raises(ValueError, match="Input array x contains NaN values."):
        interp1d(x, y, x_new, "linear")
    with pytest.raises(ValueError, match="Input array x contains NaN values."):
        interp1d(x, y, x_new, "spline")

    # y contains NaN
    x = np.array([0.1, 0.5, 0.8], dtype=dtype)
    y = np.array([0.01, np.nan, 0.64], dtype=dtype)
    with pytest.raises(ValueError, match="Input array y contains NaN values."):
        interp1d(x, y, x_new, "linear")
    with pytest.raises(ValueError, match="Input array y contains NaN values."):
        interp1d(x, y, x_new, "spline")

    # x_new contains NaN
    x = np.array([0.1, 0.5, 0.8], dtype=dtype)
    y = x**2
    x_new_nan = x_new.copy()
    x_new_nan[2] = np.nan
    with pytest.raises(ValueError, match="Input array x_new contains NaN values."):
        interp1d(x, y, x_new_nan, "linear")
    with pytest.raises(ValueError, match="Input array x_new contains NaN values."):
        interp1d(x, y, x_new_nan, "spline")
