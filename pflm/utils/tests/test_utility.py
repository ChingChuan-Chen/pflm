import numpy as np
import pytest

from pflm.utils.utility import flatten_and_sort_data_matrices, get_covariance_matrix, get_eigen_results, get_raw_cov


def test_flatten_and_sort_data_matrices_happy_path():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    w = np.array([1.0, 2.0])
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
    w = [1.0]
    with pytest.raises(ValueError, match="Weights w must be a 1D array."):
        flatten_and_sort_data_matrices(y, t, w=w)


def test_flatten_and_sort_data_matrices_weight_shape_mismatch():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    w = np.array([1.0, 0.2, 3.0])
    with pytest.raises(ValueError, match="The length of y and w must be the same."):
        flatten_and_sort_data_matrices(y, t, w=w)

    w2 = np.array([[1.0, 0.2]])
    with pytest.raises(ValueError, match="Each element of w must be a 1D array."):
        flatten_and_sort_data_matrices(y, t, w=w2)


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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_get_raw_cov_happy_path(dtype):
    # prepare simple data
    y = [np.array([1.0, 2.0, 2.0], dtype=dtype), np.array([3.0, 4.0], dtype=dtype), np.array([4.0, 5.0], dtype=dtype)]
    t = [np.array([0.1, 0.2, 0.3], dtype=dtype), np.array([0.2, 0.3], dtype=dtype), np.array([0.1, 0.3], dtype=dtype)]
    w = np.array([1.0, 2.0, 2.0], dtype=dtype)
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t, dtype, w)
    obs_grid = np.unique(tt, sorted=True)
    tid = np.digitize(tt, obs_grid, right=True)
    mu = (np.bincount(tid, yy) / np.bincount(tid)).astype(dtype, copy=False)
    raw_cov = get_raw_cov(yy, tt, ww, mu, tid, sid)

    unique_sid, sid_cnt = np.unique(sid, return_counts=True, sorted=True)
    expected_num_pairs = np.sum(sid_cnt * (sid_cnt + 1) // 2)

    # check shape
    assert raw_cov.shape[0] == expected_num_pairs
    assert raw_cov.shape[1] == 5  # sid, t1, t2, w, cov
    # check type
    assert isinstance(raw_cov, np.ndarray)
    assert raw_cov.dtype == dtype

    # check result
    expected_raw_cov = np.array(
        [
            [0.0, 0.1, 0.1, 1.0, 2.25],
            [0.0, 0.1, 0.2, 1.0, 0.75],
            [0.0, 0.1, 0.3, 1.0, 2.5],
            [0.0, 0.2, 0.2, 1.0, 0.25],
            [0.0, 0.2, 0.3, 1.0, 0.83333333],
            [0.0, 0.3, 0.3, 1.0, 2.77777778],
            [1.0, 0.2, 0.2, 2.0, 0.25],
            [1.0, 0.2, 0.3, 2.0, 0.16666667],
            [1.0, 0.3, 0.3, 2.0, 0.11111111],
            [2.0, 0.1, 0.1, 2.0, 2.25],
            [2.0, 0.1, 0.3, 2.0, 2.0],
            [2.0, 0.3, 0.3, 2.0, 1.77777778],
        ],
        dtype=dtype,
    )
    assert np.allclose(raw_cov, expected_raw_cov)


def test_get_raw_cov_wrong_shape():
    # prepare simple data
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    w = np.array([1.0, 2.0])
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t, np.float64, w)
    obs_grid = np.unique(tt)
    tid = np.digitize(tt, obs_grid, right=True)
    mu = (np.bincount(tid, yy) / np.bincount(tid)).astype(np.float64, copy=False)
    with pytest.raises(ValueError, match="yy, tt, and ww must be 1D arrays."):
        get_raw_cov(yy.reshape((3, 1)), tt, ww, mu, tid, sid)
    with pytest.raises(ValueError, match="yy, tt, and ww must have the same length."):
        get_raw_cov(yy[:-1], tt, ww, mu, tid, sid)
    with pytest.raises(ValueError, match="yy, tt, and ww must have the same length."):
        get_raw_cov(yy, tt, ww[:-1], mu, tid, sid)
    with pytest.raises(ValueError, match="The length of mu must match the number of unique time indices in tid."):
        get_raw_cov(yy, tt, ww, mu[:-1], tid, sid)
    with pytest.raises(ValueError, match="sid must be a 1D array with the same length as yy."):
        get_raw_cov(yy, tt, ww, mu, tid, sid[:-1])
    with pytest.raises(ValueError, match="The sample indices, sid, must be sorted in ascending order."):
        get_raw_cov(yy, tt, ww, mu, tid, np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="Each sample must have at least two observations for covariance calculation."):
        get_raw_cov(yy, tt, ww, mu, tid, sid)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_get_covariance_matrix_happy_path(dtype):
    raw_cov = np.array(
        [
            [0.0, 0.1, 0.1, 1.0, 2.25],
            [0.0, 0.1, 0.2, 1.0, 0.75],
            [0.0, 0.1, 0.3, 1.0, 2.5],
            [0.0, 0.2, 0.2, 1.0, 0.25],
            [0.0, 0.2, 0.3, 1.0, 0.83333333],
            [0.0, 0.3, 0.3, 1.0, 2.77777778],
            [1.0, 0.2, 0.2, 2.0, 0.25],
            [1.0, 0.2, 0.3, 2.0, 0.16666667],
            [1.0, 0.3, 0.3, 2.0, 0.11111111],
            [2.0, 0.1, 0.1, 2.0, 2.25],
            [2.0, 0.1, 0.3, 2.0, 2.0],
            [2.0, 0.3, 0.3, 2.0, 1.77777778],
        ],
        dtype=dtype,
    )
    obs_grid = np.array([0.1, 0.2, 0.3], dtype=dtype)
    cov_matrix = get_covariance_matrix(raw_cov, obs_grid)
    # check shape
    assert cov_matrix.shape == (obs_grid.size, obs_grid.size)
    # check type
    assert isinstance(cov_matrix, np.ndarray)
    # check symmetry
    assert np.allclose(cov_matrix, cov_matrix.T)
    # check values
    expected_cov_matrix = np.array(
        [
            [3.375, 0., 3.25],
            [0., 0.375, 0.58333333],
            [3.25, 0.5833333, 1.6388888]
        ],
        dtype=dtype,
    )
    assert np.allclose(cov_matrix, expected_cov_matrix)


def test_get_covariance_matrix_empty():
    # empty raw_cov
    raw_cov = np.empty((0, 5), dtype=np.float64)
    obs_grid = np.array([0.1, 0.2])
    dense_cov = get_covariance_matrix(raw_cov, obs_grid)
    assert np.allclose(dense_cov, dense_cov.T)
    assert dense_cov.shape == (obs_grid.size, obs_grid.size)
    assert np.all(dense_cov == 0)
