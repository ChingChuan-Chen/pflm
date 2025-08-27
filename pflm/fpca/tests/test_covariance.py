import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.fpca import get_covariance_matrix, get_raw_cov


@pytest.mark.parametrize(
    "raw_cov,flatten_data,dtype",
    [(np.float64, np.float64, np.float64), (np.float32, np.float32, np.float32)],
    indirect=["raw_cov", "flatten_data"],
)
def test_get_raw_cov_happy_path(raw_cov, flatten_data, dtype):
    ffd, mu = flatten_data
    mu = (np.bincount(ffd.tid, ffd.y) / np.bincount(ffd.tid)).astype(dtype, copy=False)
    raw_cov = get_raw_cov(ffd, mu)
    expected_num_pairs = np.sum(ffd.sid_cnt * (ffd.sid_cnt + 1) // 2)

    # check shape
    assert raw_cov.shape[0] == expected_num_pairs
    assert raw_cov.shape[1] == 5  # sid, t1, t2, w, cov
    # check type
    assert isinstance(raw_cov, np.ndarray)
    assert raw_cov.dtype == dtype

    # check result
    expected_raw_cov = raw_cov.astype(dtype)
    assert_allclose(raw_cov, expected_raw_cov, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("flatten_data", [np.float64], indirect=True)
def test_get_raw_cov_wrong_shape(flatten_data):
    ffd, mu = flatten_data
    mu = (np.bincount(ffd.tid, ffd.y) / np.bincount(ffd.tid)).astype(np.float64, copy=False)
    with pytest.raises(ValueError, match="The length of mu must match the number of unique time indices"):
        get_raw_cov(ffd, mu[:-1])


@pytest.mark.parametrize("raw_cov,dtype", [(np.float64, np.float64), (np.float32, np.float32)], indirect=["raw_cov"])
def test_get_covariance_matrix_happy_path(raw_cov, dtype):
    obs_grid = np.array([0.1, 0.2, 0.3], dtype=dtype)
    cov_matrix = get_covariance_matrix(raw_cov, obs_grid)
    # check shape
    assert cov_matrix.shape == (obs_grid.size, obs_grid.size)
    # check type
    assert isinstance(cov_matrix, np.ndarray)
    # check symmetry
    assert_allclose(cov_matrix, cov_matrix.T)
    # check values
    expected_cov_matrix = np.array(
        [[3.375, 0.0, 3.25], [0.0, 0.375, 0.58333333], [3.25, 0.5833333, 1.6388888]],
        dtype=dtype,
    )
    assert_allclose(cov_matrix, expected_cov_matrix)


def test_get_covariance_matrix_empty():
    # empty raw_cov
    raw_cov = np.empty((0, 5), dtype=np.float64)
    obs_grid = np.array([0.1, 0.2])
    dense_cov = get_covariance_matrix(raw_cov, obs_grid)
    assert_allclose(dense_cov, dense_cov.T)
    assert dense_cov.shape == (obs_grid.size, obs_grid.size)
    assert np.all(dense_cov == 0)
