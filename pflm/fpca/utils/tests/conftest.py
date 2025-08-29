import numpy as np
import pytest

from pflm.utils.utility import flatten_and_sort_data_matrices


@pytest.fixture
def raw_cov(request):
    dtype = getattr(request, "param", np.float64)
    return np.array(
        [
            [0.0, 0.1, 0.1, 1.0, 2.25],
            [0.0, 0.1, 0.2, 1.0, 0.75],
            [0.0, 0.1, 0.3, 1.0, 2.5],
            [0.0, 0.2, 0.1, 1.0, 0.75],
            [0.0, 0.2, 0.2, 1.0, 0.25],
            [0.0, 0.2, 0.3, 1.0, 0.83333333],
            [0.0, 0.3, 0.1, 1.0, 2.5],
            [0.0, 0.3, 0.2, 1.0, 0.83333333],
            [0.0, 0.3, 0.3, 1.0, 2.77777778],
            [1.0, 0.2, 0.2, 2.0, 0.25],
            [1.0, 0.2, 0.3, 2.0, 0.16666667],
            [1.0, 0.3, 0.2, 2.0, 0.16666667],
            [1.0, 0.3, 0.3, 2.0, 0.11111111],
            [2.0, 0.1, 0.1, 2.0, 2.25],
            [2.0, 0.1, 0.3, 2.0, 2.0],
            [2.0, 0.3, 0.1, 2.0, 2.0],
            [2.0, 0.3, 0.3, 2.0, 1.77777778],
        ],
        dtype=dtype,
    )


def _make_func_data(dtype):
    y = [np.array([1.0, 2.0, 2.0], dtype=dtype), np.array([3.0, 4.0], dtype=dtype), np.array([4.0, 5.0], dtype=dtype)]
    t = [np.array([0.1, 0.2, 0.3], dtype=dtype), np.array([0.2, 0.3], dtype=dtype), np.array([0.1, 0.3], dtype=dtype)]
    w = np.array([1.0, 2.0, 2.0], dtype=dtype)
    return y, t, w


@pytest.fixture
def flatten_data(request):
    dtype = getattr(request, "param", np.float64)
    y, t, w = _make_func_data(dtype)
    ffd = flatten_and_sort_data_matrices(y, t, dtype, w)
    mu = (np.bincount(ffd.tid, ffd.y) / np.bincount(ffd.tid)).astype(dtype, copy=False)
    return ffd, mu
