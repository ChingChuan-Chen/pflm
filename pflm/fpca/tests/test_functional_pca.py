import numpy as np
import pytest
from pflm.fpca import FunctionalPCA


@pytest.mark.parametrize("basic_func_data, dtype", [(np.float32, np.float32), (np.float64, np.float64)], indirect=["basic_func_data"])
def test_functional_pca_basic(basic_func_data, dtype):
    y, t, w = basic_func_data
    fpca = FunctionalPCA()
    # fpca.fit(y, t, w)
    # expected_output_attrs = [
    #     "y_", "t_", "flatten_func_data_", "raw_cov_", "smoothed_model_result_obs_", "smoothed_model_result_reg_",
    #     "fpca_model_params_", "xi_", "xi_var_", "fitted_y_mat_", "fitted_y_", "elapsed_time_"
    # ]
    # assert all(hasattr(fpca, attr) for attr in expected_output_attrs)
