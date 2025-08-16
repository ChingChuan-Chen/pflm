import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.utils import flatten_and_sort_data_matrices


def test_flatten_and_sort_data_matrices_happy_path():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    w = np.array([1.0, 2.0])
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t, np.float64, w)
    # check shapes
    assert yy.shape == (3,)
    assert tt.shape == (3,)
    assert ww.shape == (3,)
    assert sid.shape == (3,)
    # check types
    assert isinstance(yy, np.ndarray)
    assert isinstance(tt, np.ndarray)
    assert isinstance(ww, np.ndarray)
    assert isinstance(sid, np.ndarray)
    # check sid correspondence
    assert set(sid) == {0, 1}


def test_flatten_and_sort_data_matrices_default_weights():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    t = [np.array([0.1, 0.2]), np.array([0.15])]
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t)
    assert_allclose(ww, [1.0, 1.0, 1.0])


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
