import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from pflm.fpca.utils import get_measurement_error_variance
from pflm.smooth import KernelType


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=["raw_cov"])
def test_get_measurement_error_variance_happy_path(raw_cov):
    reg_grid = np.array([0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float64)

    expected_results = {
        KernelType.GAUSSIAN: (
            np.array([1.839379960077, 1.507852730004, 1.270683688169, 1.128430353743, 1.082323414423], dtype=np.float64),
            np.array([0.398148147778, 0.217592592778, 0.037037037778, -0.143518517222, -0.324074072222], dtype=np.float64),
            np.float64(1.304917577014),
        ),
        KernelType.LOGISTIC: (
            np.array([1.769429258922, 1.516548945730, 1.308391229755, 1.147666712377, 1.031790031024], dtype=np.float64),
            np.array([0.398148147778, 0.217592592778, 0.037037037778, -0.143518517222, -0.324074072222], dtype=np.float64),
            np.float64(1.306267095431),
        ),
        KernelType.SIGMOID: (
            np.array([1.821241837309, 1.510445144242, 1.275683251257, 1.134082092759, 1.067874121223], dtype=np.float64),
            np.array([0.398148147778, 0.217592592778, 0.037037037778, -0.143518517222, -0.324074072222], dtype=np.float64),
            np.float64(1.304155079103),
        ),
        KernelType.RECTANGULAR: (
            np.array([2.25, 1.25, 1.343253968571, 1.163194445, 1.311111112], dtype=np.float64),
            np.array([-0.625936911938, 0.217592592778, 0.037037037778, -0.143518517222, -1.337724192199], dtype=np.float64),
            np.float64(1.601930852077),
        ),
        KernelType.TRIANGULAR: (
            np.array([2.25, 1.25, 0.945707070909, 0.780555556, 1.311111112], dtype=np.float64),
            np.array([-0.625936911938, 0.217592592778, 0.037037037778, -0.143518517222, -1.337724192199], dtype=np.float64),
            np.float64(1.406884405411),
        ),
        KernelType.EPANECHNIKOV: (
            np.array([2.25, 1.25, 1.139857881395, 0.780555556, 1.311111112], dtype=np.float64),
            np.array([-0.625936911938, 0.217592592778, 0.037037037778, -0.143518517222, -1.337724192199], dtype=np.float64),
            np.float64(1.455422108033),
        ),
        KernelType.BIWEIGHT: (
            np.array([2.25, 1.25, 0.916618273519, 0.780555556, 1.311111112], dtype=np.float64),
            np.array([-0.625936911938, 0.217592592778, 0.037037037778, -0.143518517221, -1.337724192178], dtype=np.float64),
            np.float64(1.399612206061),
        ),
        KernelType.TRIWEIGHT: (
            np.array([2.25, 1.25, 0.709240145223, 0.780555556, 1.311111112], dtype=np.float64),
            np.array([-0.625936911945, 0.217592592778, 0.037037037778, -0.143518517222, -1.337724192210], dtype=np.float64),
            np.float64(1.347767673992),
        ),
        KernelType.TRICUBE: (
            np.array([2.25, 1.25, 0.962594219370, 0.780555556, 1.311111112], dtype=np.float64),
            np.array([-0.625936911932, 0.217592592778, 0.037037037778, -0.143518517222, -1.337724192203], dtype=np.float64),
            np.float64(1.411106192526),
        ),
        KernelType.COSINE: (
            np.array([2.25, 1.25, 1.100308642222, 0.780555556, 1.311111112], dtype=np.float64),
            np.array([-0.625936911938, 0.217592592778, 0.037037037778, -0.143518517222, -1.337724192199], dtype=np.float64),
            np.float64(1.445534798239),
        ),
    }
    for kernel_type, (expected_cov_diag, expected_diag_cov_surface, expected_sigma2) in expected_results.items():
        sigma2, smoothed_cov_diag, diag_smoothed_cov_surface = get_measurement_error_variance(raw_cov, reg_grid, 0.15, kernel_type)
        assert_allclose(smoothed_cov_diag, expected_cov_diag, rtol=1e-5, atol=1e-4, err_msg=f"Covariance diagonal mismatch for {kernel_type}")
        assert_allclose(
            diag_smoothed_cov_surface,
            expected_diag_cov_surface,
            rtol=1e-5,
            atol=1e-4,
            err_msg=f"Diagonal smoothed covariance surface mismatch for {kernel_type}",
        )
        assert_almost_equal(sigma2, expected_sigma2, decimal=5, err_msg=f"Sigma2 mismatch for {kernel_type}")


def test_get_measurement_error_variance_raw_cov_wrong_num_cols():
    raw_cov_bad = np.array([[0.0, 0.1, 0.1, 1.0]], dtype=np.float64)  # should have 5 columns
    reg_grid = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="raw_cov must have 5 columns"):
        get_measurement_error_variance(raw_cov_bad, reg_grid, 0.15, KernelType.GAUSSIAN)


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=["raw_cov"])
def test_get_measurement_error_variance_reg_grid_not_1d(raw_cov):
    reg_grid_bad = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)  # 2D
    with pytest.raises(ValueError, match="reg_grid must be a 1D array"):
        get_measurement_error_variance(raw_cov, reg_grid_bad, 0.15, KernelType.GAUSSIAN)


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=["raw_cov"])
def test_get_measurement_error_variance_reg_grid_not_increasing(raw_cov):
    reg_grid_bad = np.array([0.1, 0.1, 0.2], dtype=np.float64)  # non-increasing
    with pytest.raises(ValueError, match="reg_grid must be a 1D array with increasing values"):
        get_measurement_error_variance(raw_cov, reg_grid_bad, 0.15, KernelType.GAUSSIAN)


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=["raw_cov"])
@pytest.mark.parametrize("bad_bw", ["0.15", [0.15], (0.15,), {"bw": 0.15}])
def test_get_measurement_error_variance_invalid_bandwidth_type(raw_cov, bad_bw):
    reg_grid = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="bandwidth must be a numeric value"):
        get_measurement_error_variance(raw_cov, reg_grid, bad_bw, KernelType.GAUSSIAN)


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=["raw_cov"])
def test_get_measurement_error_variance_non_positive_bandwidth(raw_cov):
    reg_grid = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="bandwidth must be a positive number"):
        get_measurement_error_variance(raw_cov, reg_grid, 0.0, KernelType.GAUSSIAN)


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=["raw_cov"])
@pytest.mark.parametrize("nan_val", [float("nan"), np.nan])
def test_get_measurement_error_variance_bandwidth_nan(raw_cov, nan_val):
    reg_grid = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="bandwidth must not be NaN"):
        get_measurement_error_variance(raw_cov, reg_grid, nan_val, KernelType.GAUSSIAN)


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=["raw_cov"])
def test_get_measurement_error_variance_invalid_kernel_type(raw_cov):
    reg_grid = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="kernel must be one of"):
        get_measurement_error_variance(raw_cov, reg_grid, 0.15, kernel_type=999)
