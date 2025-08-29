import warnings

import numpy as np
import pytest

from pflm.fpca import FunctionalPCA, FunctionalPCAMuCovParams
from pflm.smooth import KernelType


@pytest.mark.parametrize("basic_func_data", [np.float32, np.float64], indirect=["basic_func_data"])
@pytest.mark.parametrize("method_pcs", ["CE", "IN"])
@pytest.mark.parametrize("assume_measurement_error", [True, False])
@pytest.mark.parametrize("method_rho", ["truncated", "ridge", "vanilla"])
@pytest.mark.parametrize("if_shrinkage", [True, False])
def test_functional_pca_basic(basic_func_data, method_pcs, assume_measurement_error, method_rho, if_shrinkage):
    y, t, w = basic_func_data
    fpca = FunctionalPCA(
        assume_measurement_error=assume_measurement_error, mu_cov_params=FunctionalPCAMuCovParams(kernel_type=KernelType.EPANECHNIKOV)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpca.fit(y, t, w, method_pcs=method_pcs, method_rho=method_rho, if_shrinkage=if_shrinkage)

    expected_output_attrs = [
        "y_",
        "t_",
        "flatten_func_data_",
        "raw_cov_",
        "smoothed_model_result_obs_",
        "smoothed_model_result_reg_",
        "fpca_model_params_",
        "xi_",
        "xi_var_",
        "fitted_y_mat_",
        "fitted_y_",
        "elapsed_time_",
    ]
    assert all(hasattr(fpca, attr) for attr in expected_output_attrs)
    assert fpca.fitted_y_mat_.shape == (fpca.smoothed_model_result_obs_.grid.shape[0], len(y))
    assert len(fpca.fitted_y_) == len(y)
    assert fpca.num_pcs_ is not None
    assert fpca.flatten_func_data_ is not None
    assert fpca.raw_cov_ is not None
    assert fpca.fpca_model_params_ is not None
    assert fpca.fpca_model_params_.fpca_phi.get("obs") is not None
    assert fpca.fpca_model_params_.fpca_phi.get("reg") is not None
    assert fpca.xi_ is not None
