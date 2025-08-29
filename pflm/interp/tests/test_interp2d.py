import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.interp.interp import interp2d_f32, interp2d_f64

from pflm.interp.interp_model import interp2d


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp2d_pyx_happy_case(dtype):
    A = np.array([[13., 8., 5., 1.], [-1., 2., 4., 6.], [12., 7., 3., 2.], [-5., -4., 1., 4.], [8., 4., 2., -3.]], dtype=dtype)
    x = np.array([0.0, 1.0, 3.0, 4.0, 5.0], dtype=dtype)
    y = np.array([10.0, 11.0, 12.0, 13.0], dtype=dtype)

    x_new = np.linspace(0.0, 5.0, 9, dtype=dtype)
    y_new = np.linspace(10.0, 13.0, 6, dtype=dtype)
    expected_linear = np.array(
        [

            [13., 10., 7.4, 5.6, 3.4, 1.],
            [4.25, 4.25, 4.275, 4.35, 4.275, 4.125],
            [0.625, 1.825, 2.875, 3.625, 4.525, 5.5],
            [4.6875, 4.3875, 4.0625, 3.6875, 3.8375, 4.25],
            [8.75, 6.95, 5.25, 3.75, 3.15, 3.],
            [9.875, 7.325, 5.05, 3.325, 2.55, 2.25],
            [-0.75, -1.05, -0.7, 0.95, 2.3, 3.5],
            [-0.125, -0.65, -0.525, 0.9, 1.375, 1.375],
            [8., 5.6, 3.6, 2.4, -0., -3.],
        ],
        dtype=dtype,
    )
    expected_spline = np.array(
        [
            [13., 9.592, 7.336, 5.584, 3.688, 1.],
            [-0.13807896, 1.22153544, 2.58993136, 3.8294269, 4.80234012, 5.37098912],
            [0.37304688, 2.10114062, 3.07084375, 3.72259375, 4.49682813, 5.83398438],
            [7.20043945, 6.96972461, 5.70437891, 4.25722266, 3.48107617, 4.22875977],
            [13.01116071, 10.56619643, 7.41617857, 4.42703571, 2.46469643, 2.39508929],
            [10.50366211, 7.6534082, 5.14654297, 3.23091797, 2.15438477, 2.16479492],
            [-0.96902902, -1.92985491, -1.05662946, 0.76229911, 2.63858259, 3.68387277],
            [-7.2066476, -7.01973075, -4.21197824, -0.44331752, 2.62632394, 3.33701869],
            [8., 5.08, 3.6, 2.48, 0.64, -3.]
        ],
        dtype=dtype,
    )

    interp_func = interp2d_f32 if x.dtype == np.float32 else interp2d_f64
    print("linear:", np.abs(interp_func(x, y, A, x_new, y_new, 0) - expected_linear))
    print("spline:", np.abs(interp_func(x, y, A, x_new, y_new, 1) - expected_spline))
    assert_allclose(interp_func(x, y, A, x_new, y_new, 0), expected_linear, rtol=0, atol=1e-4)
    assert_allclose(interp_func(x, y, A, x_new, y_new, 1), expected_spline, rtol=0, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp2d_happy_case(dtype):
    A = np.array(
        [
            [13.0, 8.0, 5.0, 1.0],
            [-1.0, 2.0, 4.0, 6.0],
            [12.0, 7.0, 3.0, 2.0],
            [-5.0, -4.0, 1.0, 4.0],
            [8.0, 4.0, 2.0, -3.0],
        ],
        dtype=dtype,
    )
    x = np.array([0.0, 1.0, 3.0, 4.0, 5.0], dtype=dtype)
    y = np.array([10.0, 11.0, 12.0, 13.0], dtype=dtype)

    x_new = np.linspace(0.0, 5.0, 7, dtype=dtype)
    y_new = np.linspace(10.0, 13.0, 7, dtype=dtype)

    expected_linear = np.array(
        [
            [13.00000000, 10.50000000, 8.00000000, 6.50000000, 5.00000000, 3.00000000, 1.00000000],
            [1.33333333, 2.16666667, 3.00000000, 3.58333333, 4.16666667, 4.66666667, 5.16666667],
            [3.33333333, 3.50000000, 3.66666667, 3.66666667, 3.66666667, 4.16666667, 4.66666667],
            [8.75000000, 7.25000000, 5.75000000, 4.50000000, 3.25000000, 3.12500000, 3.00000000],
            [6.33333333, 4.83333333, 3.33333333, 2.83333333, 2.33333333, 2.50000000, 2.66666667],
            [-2.83333333, -2.75000000, -2.66666667, -0.75000000, 1.16666667, 2.00000000, 2.83333333],
            [8.00000000, 6.00000000, 4.00000000, 3.00000000, 2.00000000, -0.50000000, -3.00000000],
        ],
        dtype=dtype,
    )
    expected_spline = np.array(
        [
            [13.00000000, 10.06250000, 8.00000000, 6.43750000, 5.00000000, 3.31250000, 1.00000000],
            [-1.12216160, 0.45731027, 1.82550705, 3.01658606, 4.06470459, 5.00401992, 5.86868937],
            [4.67548501, 5.22619048, 4.99294533, 4.40873016, 3.90652557, 3.91931217, 4.88007055],
            [13.01116071, 11.04603795, 8.49107143, 5.84737723, 3.61607143, 2.29827009, 2.39508929],
            [7.09832451, 5.07457011, 3.64726631, 2.75248016, 2.32627866, 2.30472884, 2.62389771],
            [-6.70968364, -6.83861400, -5.04706790, -2.20551215, 0.81558642, 3.14576100, 3.91454475],
            [8.00000000, 5.43750000, 4.00000000, 3.06250000, 2.00000000, 0.18750000, -3.00000000],
        ],
        dtype=dtype,
    )

    assert_allclose(interp2d(x, y, A, x_new, y_new, "linear"), expected_linear, rtol=1e-5, atol=1e-6)
    assert_allclose(interp2d(x, y, A, x_new, y_new, "spline"), expected_spline, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp2d_unsorted(dtype):
    A = np.array([[13.0, 5.0, 1.0], [-1.0, 4.0, 6.0], [12.0, 3.0, 2.0]], dtype=dtype)
    x = np.array([0.0, 1.0, 4.0], dtype=dtype)
    y = np.array([10.0, 11.0, 12.0], dtype=dtype)
    x_new = np.linspace(0.0, 4.0, 4, dtype=dtype)
    y_new = np.linspace(10.0, 12.0, 4, dtype=dtype)

    x_ord = np.array([0, 2, 1], dtype=np.int64)
    y_ord = np.array([2, 1, 0], dtype=np.int64)

    sorted_linear_result = interp2d(x, y, A, x_new, y_new, "linear")
    unsorted_linear_result = interp2d(x[x_ord], y[y_ord], A[x_ord, :][:, y_ord], x_new, y_new, "linear")
    assert_allclose(unsorted_linear_result, sorted_linear_result, rtol=1e-5, atol=1e-6)
    sorted_spline_result = interp2d(x, y, A, x_new, y_new, "spline")
    unsorted_spline_result = interp2d(x[x_ord], y[y_ord], A[x_ord, :][:, y_ord], x_new, y_new, "spline")
    assert_allclose(unsorted_spline_result, sorted_spline_result, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp2d_two_points(dtype):
    A = np.array([[13.0, 5.0], [-1.0, 4.0]], dtype=dtype)
    x = np.array([0.0, 1.0], dtype=dtype)
    y = np.array([10.0, 11.0], dtype=dtype)
    expected = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 5.25, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=dtype,
    )

    x_new = np.linspace(-0.5, 1.5, 3, dtype=dtype)
    y_new = np.linspace(9.5, 11.5, 3, dtype=dtype)
    assert_allclose(interp2d(x, y, A, x_new, y_new, "linear"), expected, rtol=1e-5, atol=1e-6)
    assert_allclose(interp2d(x, y, A, x_new, y_new, "spline"), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp2d_spline_small_size(dtype):
    expected1 = np.array(
        [
            [13.00000000, 9.59200000, 7.33600000, 5.58400000, 3.68800000, 1.00000000],
            [-0.58400000, 1.12128000, 2.55008000, 3.75424000, 4.78560000, 5.69600000],
            [2.08800000, 3.30016000, 3.63680000, 3.71136000, 4.13728000, 5.52800000],
            [9.75200000, 8.23360000, 6.07008000, 4.02176000, 2.84896000, 3.31200000],
            [11.14400000, 8.02656000, 5.32384000, 3.25184000, 2.02656000, 1.86400000],
            [-5.00000000, -5.21600000, -3.12800000, -0.03200000, 2.77600000, 4.00000000],
        ],
        dtype=dtype,
    )
    expected2 = np.array(
        [
            [13.00000000, 7.22222222, 3.22222222, 1.00000000],
            [-3.62962963, 1.74485597, 5.27572016, 6.96296296],
            [-3.96296296, 1.04115226, 4.79423868, 7.29629630],
            [12.00000000, 5.11111111, 1.77777778, 2.00000000],
        ],
        dtype=dtype,
    )
    expected3 = np.array(
        [
            [13.00000000, 10.33333333, 7.66666667, 5.00000000],
            [8.33333333, 7.11111111, 5.88888889, 4.66666667],
            [3.66666667, 3.88888889, 4.11111111, 4.33333333],
            [-1.00000000, 0.66666667, 2.33333333, 4.00000000],
        ],
        dtype=dtype,
    )

    A1 = np.array(
        [
            [13.00000000, 8.00000000, 5.00000000, 1.00000000],
            [-1.00000000, 2.00000000, 4.00000000, 6.00000000],
            [12.00000000, 7.00000000, 3.00000000, 2.00000000],
            [-5.00000000, -4.00000000, 1.00000000, 4.00000000],
        ],
        dtype=dtype,
    )
    x1 = np.array([0.0, 1.0, 3.0, 4.0], dtype=dtype)
    y1 = np.array([10.0, 11.0, 12.0, 13.0], dtype=dtype)
    x_new1 = np.linspace(0.0, 4.0, 6, dtype=dtype)
    y_new1 = np.linspace(10.0, 13.0, 6, dtype=dtype)
    assert_allclose(interp2d(x1, y1, A1, x_new1, y_new1, "spline"), expected1, rtol=1e-5, atol=1e-6)

    A2 = np.array(
        [[13.00000000, 5.00000000, 1.00000000], [-1.00000000, 4.00000000, 6.00000000], [12.00000000, 3.00000000, 2.00000000]], dtype=dtype
    )
    x2 = np.array([0.0, 1.0, 4.0], dtype=dtype)
    y2 = np.array([10.0, 11.0, 12.0], dtype=dtype)
    x_new2 = np.linspace(0.0, 4.0, 4, dtype=dtype)
    y_new2 = np.linspace(10.0, 12.0, 4, dtype=dtype)
    assert_allclose(interp2d(x2, y2, A2, x_new2, y_new2, "spline"), expected2, rtol=1e-5, atol=1e-6)

    A3 = np.array([[13.0, 5.0], [-1.0, 4.0]], dtype=dtype)
    x3 = np.array([0.0, 1.0], dtype=dtype)
    y3 = np.array([10.0, 11.0], dtype=dtype)
    x_new3 = np.linspace(0.0, 1.0, 4, dtype=dtype)
    y_new3 = np.linspace(10.0, 11.0, 4, dtype=dtype)
    assert_allclose(interp2d(x3, y3, A3, x_new3, y_new3, "spline"), expected3, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp2d_raise_exception(dtype):
    A = np.array([[13.0, -1.0, 12.0], [5.0, 4.0, 3.0], [1.0, 6.0, 2.0]], dtype=dtype)
    x = np.array([0.0, 1.0, 4.0], dtype=dtype)
    y = np.array([10.0, 11.0, 12.0], dtype=dtype)
    x_new = np.linspace(0.0, 4.0, 4, dtype=dtype)
    y_new = np.linspace(10.0, 12.0, 4, dtype=dtype)

    # x is not 1D
    with pytest.raises(ValueError, match="x, y, and v must be 1D and 2D arrays respectively."):
        interp2d(np.array([x, x]), y, A, x_new, y_new, "linear")
    # y is not 1D
    with pytest.raises(ValueError, match="x, y, and v must be 1D and 2D arrays respectively."):
        interp2d(x, np.array([y, y]), A, x_new, y_new, "linear")
    # x is empty
    with pytest.raises(ValueError, match="x, y, v, x_new, and y_new must not be empty."):
        interp2d(np.array([], dtype=dtype), y, A, x_new, y_new, "linear")
    # y shorter than v.shape[0]
    with pytest.raises(ValueError, match="y must have the same length as the second dimension of v"):
        interp2d(x, y[:-1], A, x_new, y_new, "linear")
    # x shorter than v.shape[1]
    with pytest.raises(ValueError, match="x must have the same length as the first dimension of v"):
        interp2d(x[:-1], y, A, x_new, y_new, "linear")
    # invalid method string
    with pytest.raises(ValueError, match="Invalid method. Use 'linear' or 'spline'."):
        interp2d(x, y, A, x_new, y_new, "bad_method")
    # invalid method type
    with pytest.raises(ValueError, match="Invalid method. Use 'linear' or 'spline'."):
        interp2d(x, y, A, x_new, y_new, -1)


def test_interp2d_type_mismatch():
    x_f64 = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    y_f32 = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    v_f32 = np.array([[13.0, -1.0, 12.0], [5.0, 4.0, 3.0], [1.0, 6.0, 2.0]], dtype=np.float32)
    x_new_f32 = np.linspace(0.0, 4.0, 4, dtype=np.float32)
    y_new_f64 = np.linspace(10.0, 12.0, 4, dtype=np.float64)

    y_out1 = interp2d(x_f64, y_f32, v_f32, x_new_f32, y_new_f64, "linear")
    assert y_out1.dtype == np.float64

    y_out2 = interp2d(x_f64, y_f32, v_f32, x_new_f32, y_new_f64, "spline")
    assert y_out2.dtype == np.float64

    y_out3 = interp2d(x_f64.astype(np.float32), y_f32, v_f32, x_new_f32, y_new_f64.astype(np.float32), "linear")
    assert y_out3.dtype == np.float32

    y_out4 = interp2d(x_f64.astype(np.float32), y_f32, v_f32, x_new_f32, y_new_f64.astype(np.float32), "spline")
    assert y_out4.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interp2d_nan_inputs(dtype):
    # x contains NaN
    x = np.array([0.0, np.nan, 4.0], dtype=dtype)
    y = np.array([10.0, 11.0, 12.0], dtype=dtype)
    v = np.ones((3, 3), dtype=dtype)
    x_new = np.linspace(0.0, 4.0, 4, dtype=dtype)
    y_new = np.linspace(10.0, 12.0, 4, dtype=dtype)
    with pytest.raises(ValueError, match="Input array x contains NaN values."):
        interp2d(x, y, v, x_new, y_new, "linear")
    with pytest.raises(ValueError, match="Input array x contains NaN values."):
        interp2d(x, y, v, x_new, y_new, "spline")

    # y contains NaN
    x = np.array([0.0, 1.0, 4.0], dtype=dtype)
    y = np.array([10.0, np.nan, 12.0], dtype=dtype)
    v = np.ones((3, 3), dtype=dtype)
    with pytest.raises(ValueError, match="Input array y contains NaN values."):
        interp2d(x, y, v, x_new, y_new, "linear")
    with pytest.raises(ValueError, match="Input array y contains NaN values."):
        interp2d(x, y, v, x_new, y_new, "spline")

    # v contains NaN
    x = np.array([0.0, 1.0, 4.0], dtype=dtype)
    y = np.array([10.0, 11.0, 12.0], dtype=dtype)
    v = np.ones((3, 3), dtype=dtype)
    v[1, 1] = np.nan
    with pytest.raises(ValueError, match="Input array v contains NaN values."):
        interp2d(x, y, v, x_new, y_new, "linear")
    with pytest.raises(ValueError, match="Input array v contains NaN values."):
        interp2d(x, y, v, x_new, y_new, "spline")

    # x_new contains NaN
    x_new_nan = x_new.copy()
    x_new_nan[2] = np.nan
    v = np.ones((3, 3), dtype=dtype)
    with pytest.raises(ValueError, match="Input array x_new contains NaN values."):
        interp2d(x, y, v, x_new_nan, y_new, "linear")
    with pytest.raises(ValueError, match="Input array x_new contains NaN values."):
        interp2d(x, y, v, x_new_nan, y_new, "spline")

    # y_new contains NaN
    y_new_nan = y_new.copy()
    y_new_nan[1] = np.nan
    with pytest.raises(ValueError, match="Input array y_new contains NaN values."):
        interp2d(x, y, v, x_new, y_new_nan, "linear")
    with pytest.raises(ValueError, match="Input array y_new contains NaN values."):
        interp2d(x, y, v, x_new, y_new_nan, "spline")
