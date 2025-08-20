import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.utils import flatten_and_sort_data_matrices


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_flatten_and_sort_data_matrices_happy_path(dtype):
    y = [np.array([1.0, 2.0, 2.0], dtype=dtype), np.array([3.0, 4.0], dtype=dtype), np.array([4.0, 5.0], dtype=dtype)]
    t = [np.array([0.1, 0.2, 0.3], dtype=dtype), np.array([0.2, 0.3], dtype=dtype), np.array([0.1, 0.3], dtype=dtype)]
    w = np.array([1.0, 2.0, 2.0], dtype=dtype)
    ffd = flatten_and_sort_data_matrices(y, t, dtype, w)

    # check shapes
    assert ffd.y.shape == (7,)
    assert ffd.t.shape == (7,)
    assert ffd.w.shape == (7,)
    assert ffd.sid.shape == (7,)
    assert ffd.unique_tid.shape == (3,)
    assert ffd.inverse_tid_idx.shape == (7,)
    assert ffd.unique_sid.shape == (3,)
    assert ffd.sid_cnt.shape == (3,)

    # check types
    assert isinstance(ffd.y, np.ndarray)
    assert isinstance(ffd.t, np.ndarray)
    assert isinstance(ffd.w, np.ndarray)
    assert isinstance(ffd.sid, np.ndarray)
    assert isinstance(ffd.unique_tid, np.ndarray)
    assert isinstance(ffd.inverse_tid_idx, np.ndarray)
    assert isinstance(ffd.unique_sid, np.ndarray)
    assert isinstance(ffd.sid_cnt, np.ndarray)

    # check sid correspondence
    assert set(ffd.sid) == {0, 1, 2}
    assert_allclose(ffd.unique_tid, np.array([0.1, 0.2, 0.3]))
    assert_allclose(ffd.inverse_tid_idx, np.array([0, 1, 2, 1, 2, 0, 2]))
    assert_allclose(ffd.unique_sid, np.array([0, 1, 2]))
    assert_allclose(ffd.sid_cnt, np.array([3, 2, 2]))


def test_flatten_and_sort_data_matrices_default_weights():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    ffd = flatten_and_sort_data_matrices(y, t)
    assert_allclose(ffd.w, [1.0, 1.0, 1.0])


def test_flatten_and_sort_data_matrices_empty_sample():
    y = [np.array([]), np.array([3.0])]
    t = [np.array([]), np.array([0.15])]
    ffd = flatten_and_sort_data_matrices(y, t)
    assert len(ffd.sid) == 1
    assert len(ffd.y) == 1
    assert len(ffd.t) == 1
    assert len(ffd.w) == 1


def test_flatten_and_sort_data_matrices_nan_handling():
    y = [np.array([1.0, np.nan]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    ffd = flatten_and_sort_data_matrices(y, t)
    # should only include non-NaN values
    assert np.all(~np.isnan(ffd.y))


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
