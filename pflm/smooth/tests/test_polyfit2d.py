# ruff: noqa: E501
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.smooth.kernel import KernelType
from pflm.smooth.polyfit import polyfit2d


@pytest.mark.parametrize("dtype", [np.float64])
def test_polyfit2d(dtype):
    bw1 = 0.15
    bw2 = 0.15
    x = np.linspace(0.0, 1.0, 11, dtype=dtype)
    x1v, x2v = np.meshgrid(x, x)
    x_grid = np.hstack((x2v.ravel(), x1v.ravel())).reshape(2, -1).T
    y = x_grid[:, 0] ** 2 + x_grid[:, 1] ** 2 + 6.0 * x_grid[:, 0] + 6.0 * x_grid[:, 1] + 2.0
    w = np.ones_like(y, dtype=dtype)

    x_new1 = np.linspace(0.0, 1.0, 11, dtype=dtype)
    x_new2 = np.linspace(0.0, 1.0, 11, dtype=dtype)

    # fmt: off
    expected_gaussian = np.array(
        [
            [1.98496411535527, 2.61137618850057, 3.24991383719667, 3.90356223760462, 4.57470940409128, 5.26491695522127, 5.97470940409128, 6.70356223760462, 7.44991383719667, 8.21137618850057, 8.98496411535525],
            [2.61137618850057, 3.23778826164587, 3.87632591034198, 4.52997431074992, 5.20112147723659, 5.89132902836657, 6.60112147723659, 7.32997431074993, 8.07632591034198, 8.83778826164587, 9.61137618850056],
            [3.24991383719667, 3.87632591034198, 4.51486355903809, 5.16851195944603, 5.8396591259327, 6.52986667706269, 7.2396591259327, 7.96851195944604, 8.71486355903809, 9.47632591034198, 10.2499138371967],
            [3.90356223760462, 4.52997431074993, 5.16851195944603, 5.82216035985398, 6.49330752634064, 7.18351507747063, 7.89330752634065, 8.62216035985398, 9.36851195944604, 10.1299743107499, 10.9035622376046],
            [4.57470940409128, 5.20112147723659, 5.8396591259327, 6.49330752634064, 7.1644546928273, 7.85466224395729, 8.56445469282731, 9.29330752634064, 10.0396591259327, 10.8011214772366, 11.5747094040913],
            [5.26491695522127, 5.89132902836657, 6.52986667706268, 7.18351507747063, 7.85466224395729, 8.54486979508727, 9.25466224395729, 9.98351507747063, 10.7298666770627, 11.4913290283666, 12.2649169552213],
            [5.97470940409128, 6.60112147723659, 7.2396591259327, 7.89330752634064, 8.5644546928273, 9.25466224395729, 9.96445469282731, 10.6933075263406, 11.4396591259327, 12.2011214772366, 12.9747094040913],
            [6.70356223760462, 7.32997431074993, 7.96851195944603, 8.62216035985398, 9.29330752634064, 9.98351507747063, 10.6933075263406, 11.422160359854, 12.168511959446, 12.9299743107499, 13.7035622376046],
            [7.44991383719668, 8.07632591034198, 8.71486355903808, 9.36851195944603, 10.0396591259327, 10.7298666770627, 11.4396591259327, 12.168511959446, 12.9148635590381, 13.676325910342, 14.4499138371967],
            [8.21137618850057, 8.83778826164588, 9.47632591034198, 10.1299743107499, 10.8011214772366, 11.4913290283666, 12.2011214772366, 12.9299743107499, 13.676325910342, 14.4377882616459, 15.2113761885006],
            [8.98496411535527, 9.61137618850058, 10.2499138371967, 10.9035622376046, 11.5747094040913, 12.2649169552213, 12.9747094040913, 13.7035622376046, 14.4499138371967, 15.2113761885006, 15.9849641153553],
        ]
    )
    expected_epanechnikov = np.array(
        [
            [2, 2.61526315789474, 3.24526315789474, 3.89526315789474, 4.56526315789474, 5.25526315789474, 5.96526315789474, 6.69526315789474, 7.44526315789474, 8.21526315789474, 9],
            [2.61526315789474, 3.23052631578947, 3.86052631578947, 4.51052631578948, 5.18052631578947, 5.87052631578947, 6.58052631578947, 7.31052631578947, 8.06052631578947, 8.83052631578948, 9.61526315789474],
            [3.24526315789474, 3.86052631578947, 4.49052631578947, 5.14052631578947, 5.81052631578947, 6.50052631578948, 7.21052631578947, 7.94052631578948, 8.69052631578947, 9.46052631578948, 10.2452631578947],
            [3.89526315789474, 4.51052631578948, 5.14052631578947, 5.79052631578948, 6.46052631578948, 7.15052631578947, 7.86052631578948, 8.59052631578948, 9.34052631578948, 10.1105263157895, 10.8952631578947],
            [4.56526315789474, 5.18052631578947, 5.81052631578947, 6.46052631578948, 7.13052631578947, 7.82052631578948, 8.53052631578948, 9.26052631578947, 10.0105263157895, 10.7805263157895, 11.5652631578947],
            [5.25526315789474, 5.87052631578947, 6.50052631578947, 7.15052631578947, 7.82052631578947, 8.51052631578948, 9.22052631578948, 9.95052631578948, 10.7005263157895, 11.4705263157895, 12.2552631578947],
            [5.96526315789474, 6.58052631578947, 7.21052631578948, 7.86052631578947, 8.53052631578948, 9.22052631578948, 9.93052631578948, 10.6605263157895, 11.4105263157895, 12.1805263157895, 12.9652631578947],
            [6.69526315789474, 7.31052631578947, 7.94052631578947, 8.59052631578948, 9.26052631578947, 9.95052631578948, 10.6605263157895, 11.3905263157895, 12.1405263157895, 12.9105263157895, 13.6952631578947],
            [7.44526315789474, 8.06052631578947, 8.69052631578947, 9.34052631578948, 10.0105263157895, 10.7005263157895, 11.4105263157895, 12.1405263157895, 12.8905263157895, 13.6605263157895, 14.4452631578947],
            [8.21526315789474, 8.83052631578948, 9.46052631578947, 10.1105263157895, 10.7805263157895, 11.4705263157895, 12.1805263157895, 12.9105263157895, 13.6605263157895, 14.4305263157895, 15.2152631578947],
            [9, 9.61526315789474, 10.2452631578947, 10.8952631578947, 11.5652631578947, 12.2552631578947, 12.9652631578947, 13.6952631578947, 14.4452631578947, 15.2152631578947, 16],
        ],
        dtype=dtype
    )

    # fmt: on
    assert_allclose(polyfit2d(x_grid, y, w, x_new1, x_new2, bw1, bw2, KernelType.GAUSSIAN), expected_gaussian, rtol=1e-5, atol=1e-6)
    assert_allclose(polyfit2d(x_grid, y, w, x_new1, x_new2, bw1, bw2, KernelType.EPANECHNIKOV), expected_epanechnikov, rtol=1e-5, atol=1e-6)


def make_valid_inputs(dtype=np.float64):
    x_grid = np.zeros((2, 2), dtype=dtype)
    y = np.zeros(2, dtype=dtype)
    w = np.zeros(2, dtype=dtype)
    x_new1 = np.array([0.1, 0.2], dtype=dtype)
    x_new2 = np.array([0.1, 0.2], dtype=dtype)
    return x_grid, y, w, x_new1, x_new2


def test_polyfit2d_x_grid_not_2d():
    x_grid = np.zeros(2)
    y = np.zeros(2)
    w = np.zeros(2)
    x_new1 = np.array([0.1, 0.2])
    x_new2 = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="x_grid must be a 2D array."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_y_not_1d():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    y = np.zeros((2, 2))
    with pytest.raises(ValueError, match="y must be a 1D array."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_w_not_1d():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    w = np.zeros((2, 2))
    with pytest.raises(ValueError, match="w must be a 1D array."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_x_grid_y_size_mismatch():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    y = np.zeros(3)
    with pytest.raises(ValueError, match="y must have the same size as the first dimension of x_grid."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_y_size_w_size_mismatch():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    w = np.zeros(3)
    with pytest.raises(ValueError, match="w must have the same size as y."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_w_negative():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    w = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="All weights in w must be greater than 0."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_x_new_not_1d():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    x_new1 = np.zeros((2, 2))
    with pytest.raises(ValueError, match="x_new1 and x_new2 must be 1D arrays."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_x_new_empty():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    x_new1 = np.array([])
    with pytest.raises(ValueError, match="x_new1 and x_new2 must not be empty."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_bandwidth_non_positive():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    with pytest.raises(ValueError, match="Bandwidths, bandwidth1 and bandwidth2, should be positive."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 0.0, 1.0)
    with pytest.raises(ValueError, match="Bandwidths, bandwidth1 and bandwidth2, should be positive."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 0.0)


def test_polyfit2d_kernel_type_invalid():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()

    class Dummy:
        value = 999

    with pytest.raises(ValueError, match="kernel must be one of"):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0, Dummy())


def test_polyfit2d_degree_non_positive():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    with pytest.raises(ValueError, match="Degree of polynomial, degree, should be positive."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 0.1, 1.0, KernelType.GAUSSIAN, 0)


def test_polyfit2d_deriv1_negative():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    with pytest.raises(ValueError, match="Order of derivative, deriv1, should be positive."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 0.1, 1.0, KernelType.GAUSSIAN, 1, -1)


def test_polyfit2d_deriv2_negative():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    with pytest.raises(ValueError, match="Order of derivative, deriv2, should be positive."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 0.1, 1.0, KernelType.GAUSSIAN, 1, 0, -1)


def test_polyfit2d_degree_less_than_sum_of_deriv1_deriv2():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    with pytest.raises(ValueError, match="Degree of polynomial, degree, should be greater than or equal to the sum of orders of derivatives, deriv1 and deriv2."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 0.1, 1.0, KernelType.GAUSSIAN, 1, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit2d_nan_inputs(dtype):
    from pflm.smooth.polyfit import polyfit2d

    x_grid = np.zeros((2, 2), dtype=dtype)
    y = np.zeros(2, dtype=dtype)
    w = np.ones(2, dtype=dtype)
    x_new1 = np.array([0.1, 0.2], dtype=dtype)
    x_new2 = np.array([0.1, 0.2], dtype=dtype)

    # x_grid contains NaN
    x_grid_nan = x_grid.copy()
    x_grid_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="Input array x_grid contains NaN values."):
        polyfit2d(x_grid_nan, y, w, x_new1, x_new2, 1.0, 1.0, KernelType.GAUSSIAN)

    # y contains NaN
    y_nan = y.copy()
    y_nan[1] = np.nan
    with pytest.raises(ValueError, match="Input array y contains NaN values."):
        polyfit2d(x_grid, y_nan, w, x_new1, x_new2, 1.0, 1.0, KernelType.GAUSSIAN)

    # w contains NaN
    w_nan = w.copy()
    w_nan[0] = np.nan
    with pytest.raises(ValueError, match="Input array w contains NaN values."):
        polyfit2d(x_grid, y, w_nan, x_new1, x_new2, 1.0, 1.0, KernelType.GAUSSIAN)

    # x_new1 contains NaN
    x_new1_nan = x_new1.copy()
    x_new1_nan[1] = np.nan
    with pytest.raises(ValueError, match="Input array x_new1 contains NaN values."):
        polyfit2d(x_grid, y, w, x_new1_nan, x_new2, 1.0, 1.0, KernelType.GAUSSIAN)

    # x_new2 contains NaN
    x_new2_nan = x_new2.copy()
    x_new2_nan[0] = np.nan
    with pytest.raises(ValueError, match="Input array x_new2 contains NaN values."):
        polyfit2d(x_grid, y, w, x_new1, x_new2_nan, 1.0, 1.0, KernelType.GAUSSIAN)


@pytest.mark.parametrize("bad_type", [float('nan'), np.nan])
def test_polyfit2d_bandwidths_nan(bad_type):
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    with pytest.raises(ValueError, match="Bandwidths, bandwidth1 and bandwidth2, should not be NaN."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, bad_type, 1.0, KernelType.GAUSSIAN)

    with pytest.raises(ValueError, match="Bandwidths, bandwidth1 and bandwidth2, should not be NaN."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, bad_type, KernelType.GAUSSIAN)


@pytest.mark.parametrize("bad_type", ["2", None, [1], (2,), {"a": 1}])
def test_polyfit2d_bandwidth_non_int_type(bad_type):
    x_grid = np.zeros((2, 2))
    y = np.zeros(2)
    w = np.ones(2)
    x_new1 = np.array([0.1, 0.2])
    x_new2 = np.array([0.1, 0.2])
    with pytest.raises(TypeError, match="Bandwidth, bandwidth1, should not be None and must be a float or int."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, bad_type, 1.0, KernelType.GAUSSIAN)

    with pytest.raises(TypeError, match="Bandwidth, bandwidth2, should not be None and must be a float or int."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, bad_type, KernelType.GAUSSIAN)


@pytest.mark.parametrize("bad_type", [1.5, "2", None, float('nan'), np.nan, [1], (2,), {"a": 1}])
def test_polyfit2d_degree_non_int_type(bad_type):
    x_grid = np.zeros((2, 2))
    y = np.zeros(2)
    w = np.ones(2)
    x_new1 = np.array([0.1, 0.2])
    x_new2 = np.array([0.1, 0.2])
    with pytest.raises(TypeError, match="Degree of polynomial, degree, should not be None and must be an integer."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0, KernelType.GAUSSIAN, bad_type, 0, 0)


@pytest.mark.parametrize("bad_type", [1.5, "2", None, float('nan'), np.nan, [1], (2,), {"a": 1}])
def test_polyfit2d_deriv1_non_int_type(bad_type):
    x_grid = np.zeros((2, 2))
    y = np.zeros(2)
    w = np.ones(2)
    x_new1 = np.array([0.1, 0.2])
    x_new2 = np.array([0.1, 0.2])
    with pytest.raises(TypeError, match="Order of derivative, deriv1, should not be None and must be an integer."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0, KernelType.GAUSSIAN, 2, bad_type, 0)


@pytest.mark.parametrize("bad_type", [1.5, "2", None, float('nan'), np.nan, [1], (2,), {"a": 1}])
def test_polyfit2d_deriv2_non_int_type(bad_type):
    x_grid = np.zeros((2, 2))
    y = np.zeros(2)
    w = np.ones(2)
    x_new1 = np.array([0.1, 0.2])
    x_new2 = np.array([0.1, 0.2])
    with pytest.raises(TypeError, match="Order of derivative, deriv2, should not be None and must be an integer."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0, KernelType.GAUSSIAN, 2, 0, bad_type)
