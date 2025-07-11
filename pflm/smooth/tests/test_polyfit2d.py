# ruff: noqa: E501
import numpy as np
import pytest

from pflm.smooth.kernel import KernelType
from pflm.smooth.polyfit import polyfit2d


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit2d(dtype):
    bw1 = 0.25
    bw2 = 0.25
    x = np.linspace(0.0, 1.0, 11, dtype=dtype)
    x1v, x2v = np.meshgrid(x, x)
    x_grid = np.vstack((x1v.ravel(), x2v.ravel())).T
    y = x1v.ravel() ** 2 - 3.0 * x2v.ravel() ** 2 + 6.0 * x1v.ravel() + 4.0 * x2v.ravel()
    w = np.ones_like(y, dtype=dtype)

    x_new0 = np.linspace(0.0, 1.0, 11, dtype=dtype)
    x_new1 = np.linspace(0.0, 1.0, 11, dtype=dtype)

    # fmt: off
    expected_guassian = np.array(
        [
            [0.0591629392161743, 0.332863362626855, 0.573896163997318, 0.777778746944114, 0.941272980082079, 1.06261194763721, 1.14127298008208, 1.17777874694411, 1.17389616399732, 1.13286336262685, 1.05916293921617],
            [0.70126279807928, 0.974963221489959, 1.21599602286042, 1.41987860580722, 1.58337283894518, 1.70471180650031, 1.78337283894519, 1.81987860580722, 1.81599602286042, 1.77496322148996, 1.70126279807928],
            [1.35425186428912, 1.6279522876998, 1.86898508907027, 2.07286767201706, 2.23636190515503, 2.35770087271016, 2.43636190515503, 2.47286767201707, 2.46898508907027, 2.4279522876998, 2.35425186428913],
            [2.01962433664019, 2.29332476005087, 2.53435756142133, 2.73824014436813, 2.9017343775061, 3.02307334506123, 3.1017343775061, 3.13824014436813, 3.13435756142134, 3.09332476005087, 3.01962433664019],
            [2.69845959226087, 2.97216001567155, 3.21319281704201, 3.41707539998881, 3.58056963312678, 3.7019086006819, 3.78056963312678, 3.81707539998881, 3.81319281704201, 3.77216001567155, 3.69845959226087],
            [3.39134660307583, 3.6650470264865, 3.90607982785697, 4.10996241080377, 4.27345664394174, 4.39479561149686, 4.47345664394174, 4.50996241080377, 4.50607982785697, 4.46504702648651, 4.39134660307583],
            [4.09845959226087, 4.37216001567155, 4.61319281704201, 4.81707539998881, 4.98056963312678, 5.1019086006819, 5.18056963312678, 5.21707539998881, 5.21319281704202, 5.17216001567155, 5.09845959226087],
            [4.81962433664019, 5.09332476005087, 5.33435756142133, 5.53824014436813, 5.7017343775061, 5.82307334506123, 5.9017343775061, 5.93824014436813, 5.93435756142133, 5.89332476005087, 5.81962433664019],
            [5.55425186428913, 5.82795228769981, 6.06898508907026, 6.27286767201707, 6.43636190515503, 6.55770087271016, 6.63636190515503, 6.67286767201707, 6.66898508907027, 6.62795228769981, 6.55425186428912],
            [6.30126279807928, 6.57496322148996, 6.81599602286042, 7.01987860580722, 7.18337283894518, 7.30471180650031, 7.38337283894519, 7.41987860580722, 7.41599602286042, 7.37496322148996, 7.30126279807928],
            [7.05916293921618, 7.33286336262685, 7.57389616399731, 7.77777874694412, 7.94127298008208, 8.0626119476372, 8.14127298008208, 8.17777874694411, 8.17389616399731, 8.13286336262685, 8.05916293921617],
        ]
    )
    expected_epanechnikov = np.array(
        [
            [0.0046840148698886, 0.342026300916233, 0.637422698447409, 0.887422698447409, 1.07742269844741, 1.20742269844741, 1.27742269844741, 1.28742269844741, 1.23742269844741, 1.14202630091623, 1.00468401486989],
            [0.625569919521107, 0.962912205567452, 1.25830860309863, 1.50830860309863, 1.69830860309863, 1.82830860309863, 1.89830860309863, 1.90830860309863, 1.85830860309863, 1.76291220556745, 1.62556991952111],
            [1.26043778701072, 1.59778007305706, 1.89317647058824, 2.14317647058824, 2.33317647058824, 2.46317647058824, 2.53317647058824, 2.54317647058824, 2.49317647058824, 2.39778007305706, 2.26043778701072],
            [1.91043778701072, 2.24778007305706, 2.54317647058824, 2.79317647058824, 2.98317647058824, 3.11317647058824, 3.18317647058824, 3.19317647058824, 3.14317647058824, 3.04778007305706, 2.91043778701071],
            [2.58043778701072, 2.91778007305706, 3.21317647058823, 3.46317647058824, 3.65317647058824, 3.78317647058824, 3.85317647058823, 3.86317647058824, 3.81317647058824, 3.71778007305706, 3.58043778701072],
            [3.27043778701071, 3.60778007305706, 3.90317647058824, 4.15317647058824, 4.34317647058823, 4.47317647058824, 4.54317647058823, 4.55317647058824, 4.50317647058823, 4.40778007305706, 4.27043778701071],
            [3.98043778701072, 4.31778007305706, 4.61317647058824, 4.86317647058824, 5.05317647058824, 5.18317647058824, 5.25317647058824, 5.26317647058824, 5.21317647058824, 5.11778007305706, 4.98043778701072],
            [4.71043778701072, 5.04778007305706, 5.34317647058824, 5.59317647058823, 5.78317647058824, 5.91317647058824, 5.98317647058824, 5.99317647058824, 5.94317647058824, 5.84778007305706, 5.71043778701071],
            [5.46043778701072, 5.79778007305706, 6.09317647058824, 6.34317647058824, 6.53317647058824, 6.66317647058824, 6.73317647058824, 6.74317647058824, 6.69317647058824, 6.59778007305706, 6.46043778701071],
            [6.22556991952111, 6.56291220556745, 6.85830860309863, 7.10830860309863, 7.29830860309863, 7.42830860309863, 7.49830860309863, 7.50830860309863, 7.45830860309863, 7.36291220556745, 7.22556991952111],
            [7.00468401486989, 7.34202630091624, 7.6374226984474, 7.88742269844741, 8.07742269844741, 8.20742269844741, 8.2774226984474, 8.28742269844741, 8.23742269844741, 8.14202630091623, 8.0046840148699],
        ],
        dtype=dtype
    )


    # fmt: on
    # assert_allclose(polyfit2d(x_grid, y, w, x_new0, x_new1, bw1, bw2, KernelType.GAUSSIAN), expected_guassian, rtol=1e-5, atol=1e-6)
    # assert_allclose(polyfit2d(x_grid, y, w, x_new0, x_new1, bw1, bw2, KernelType.EPANECHNIKOV), expected_epanechnikov, rtol=1e-5, atol=1e-6)


def make_valid_inputs(dtype=np.float64):
    x_grid = np.zeros((2, 2), dtype=dtype)
    y = np.zeros(2, dtype=dtype)
    w = np.zeros(2, dtype=dtype)
    x_new1 = np.array([0.1, 0.2], dtype=dtype)
    x_new2 = np.array([0.1, 0.2], dtype=dtype)
    return x_grid, y, w, x_new1, x_new2


def test_polyfit2d_xgrid_not_2d():
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


def test_polyfit2d_xgrid_y_size_mismatch():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    y = np.zeros(3)
    with pytest.raises(ValueError, match="y must have the same size as the first dimension of x_grid."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0)


def test_polyfit2d_xgrid_w_size_mismatch():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    w = np.zeros(3)
    with pytest.raises(ValueError, match="w must have the same size as the second dimension of x_grid."):
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


def test_polyfit2d_bandwidth_nonpositive():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()
    with pytest.raises(ValueError, match="Bandwidths, bandwidth1 and bandwidth2, should be positive."):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 0.0, 1.0)


def test_polyfit2d_kernel_type_invalid():
    x_grid, y, w, x_new1, x_new2 = make_valid_inputs()

    class Dummy:
        value = 999

    with pytest.raises(ValueError, match="kernel must be one of"):
        polyfit2d(x_grid, y, w, x_new1, x_new2, 1.0, 1.0, Dummy())


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
