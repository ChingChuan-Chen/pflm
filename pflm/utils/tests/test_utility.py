import numpy as np
import pytest

from pflm.utils.utility import flatten_data_matrix, get_eigen_results


def test_flatten_data_matrix_basic_and_nan():
    y = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    t = np.array([0, 1, 2])
    # no weights
    yy, tt, ww = flatten_data_matrix(y, t)
    assert np.allclose(tt, np.sort(tt))
    assert not np.isnan(yy).any()
    assert not np.isnan(ww).any()
    # with weights
    w = np.array([0.5, 2.0])
    yy2, tt2, ww2 = flatten_data_matrix(y, t, w)
    assert np.allclose(tt2, np.sort(tt2))
    assert not np.isnan(yy2).any()
    assert not np.isnan(ww2).any()


def test_flatten_data_matrix_shape_errors():
    y = np.ones((2, 3))
    t = np.ones(2)
    with pytest.raises(ValueError):
        flatten_data_matrix(y, t)
    t = np.ones(3)
    w = np.ones(3)
    with pytest.raises(ValueError):
        flatten_data_matrix(y, t, w)
    with pytest.raises(ValueError):
        flatten_data_matrix(y, t.reshape(-1, 1), w)
    with pytest.raises(ValueError):
        flatten_data_matrix(np.ones((2, 3, 1)), t)
    with pytest.raises(ValueError):
        flatten_data_matrix(np.empty((0, 3)), t)
    with pytest.raises(ValueError):
        flatten_data_matrix(y, np.empty(0))


def test_get_eigen_results_all_branches():
    t = np.linspace(0, 1, 5)
    mean_func = np.sin(t)
    cov_func = np.eye(5)
    # happy path
    num_fpca, fpca_lambda, fpca_phi, cumu_fve = get_eigen_results(t, mean_func, cov_func, 0.8, 3)
    assert num_fpca > 0
    assert fpca_lambda.shape[0] == num_fpca
    assert fpca_phi.shape[1] == num_fpca
    assert np.allclose(cumu_fve[-1], 1.0)
    # wrong dimension of mean_func
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func.reshape(1, -1), cov_func, 0.8)
    # cov_func is not square matrix
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func, np.eye(5)[:4], 0.8)
    # t dimension is wrong
    with pytest.raises(ValueError):
        get_eigen_results(t.reshape(1, -1), mean_func, cov_func, 0.8)
    # empty array
    with pytest.raises(ValueError):
        get_eigen_results(np.array([]), np.array([]), np.array([[]]), 0.8)
    # length mismatch
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func[:-1], cov_func, 0.8)
    # cov_func shape mismatch
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func, np.eye(4), 0.8)
    # fve_thresh is wrong
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func, cov_func, 1.0)
    # max_principal is wrong
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func, cov_func, 0.8, 0)
    # NaN
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func, np.full((5, 5), np.nan), 0.8)
    # no positive eigenvalues
    with pytest.raises(ValueError):
        get_eigen_results(t, mean_func, -np.eye(5), 0.8)
