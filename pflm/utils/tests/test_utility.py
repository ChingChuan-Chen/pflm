import numpy as np
import pytest

from pflm.utils.utility import flatten_and_sort_data_matrices, get_eigen_results


def test_flatten_and_sort_data_matrices_happy_path():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    w = [np.array([1.0, 2.0]), np.array([3.0])]
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t, np.float64, w)
    # 檢查長度
    assert len(sid) == 3
    assert len(yy) == 3
    assert len(tt) == 3
    assert len(ww) == 3
    # 檢查 sid 對應
    assert set(sid) == {0, 1}


def test_flatten_and_sort_data_matrices_default_weights():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t)
    assert np.allclose(ww, [1.0, 1.0, 1.0])


def test_flatten_and_sort_data_matrices_empty_sample():
    y = [np.array([]), np.array([3.0])]
    t = [np.array([]), np.array([0.15])]
    sid, yy, tt, ww = flatten_and_sort_data_matrices(y, t)
    assert len(sid) == 1
    assert len(yy) == 1
    assert len(tt) == 1
    assert len(ww) == 1


def test_flatten_and_sort_data_matrices_nan_handling():
    y = [np.array([1.0, np.nan]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t)
    # should only include non-NaN values
    assert np.all(~np.isnan(yy))


def test_flatten_and_sort_data_matrices_all_nan():
    y = [np.array([np.nan]), np.array([np.nan])]
    t = [np.array([0.1]), np.array([0.15])]
    with pytest.raises(ValueError, match="All values in y are NaN. Cannot flatten data matrices."):
        flatten_and_sort_data_matrices(y, t)


def test_flatten_and_sort_data_matrices_shape_mismatch():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1]), np.array([0.15])]
    with pytest.raises(ValueError, match="Each element of y and t must have the same length."):
        flatten_and_sort_data_matrices(y, t)

    t2 = [np.array([0.1, 0.2])]
    with pytest.raises(ValueError, match="The length of y and t must be the same."):
        flatten_and_sort_data_matrices(y, t2)

    t3 = [np.array([0.1, 0.2]), np.array([[0.15]])]
    with pytest.raises(ValueError, match="Each element of y and t must be a 1D array."):
        flatten_and_sort_data_matrices(y, t3)


def test_flatten_and_sort_data_matrices_wrong_type():
    y = np.array([1.0, 2.0])
    t = [np.array([0.1, 0.2])]
    with pytest.raises(ValueError, match="y must be a list of arrays."):
        flatten_and_sort_data_matrices(y, t)

    y = [np.array([1.0, 2.0])]
    t = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="t must be a list of arrays."):
        flatten_and_sort_data_matrices(y, t)

    t = [np.array([0.1, 0.2])]
    w = np.array([1.0])
    with pytest.raises(ValueError, match="Weights w must be a list of 1D arrays."):
        flatten_and_sort_data_matrices(y, t, w=w)


def test_flatten_and_sort_data_matrices_weight_shape_mismatch():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    w = [np.array([1.0]), np.array([3.0])]
    with pytest.raises(ValueError, match="Each element of t and w must have the same length."):
        flatten_and_sort_data_matrices(y, t, w=w)

    w2 = [np.array([1.0, 0.2])]
    with pytest.raises(ValueError, match="The length of y and w must be the same."):
        flatten_and_sort_data_matrices(y, t, w=w2)

    w3 = [np.array([[1.0, 0.2]]), np.array([3.0])]
    with pytest.raises(ValueError, match="Each element of w must be a 1D array."):
        flatten_and_sort_data_matrices(y, t, w=w3)


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
