import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.interp.interp import find_le_indices_memview_f32, find_le_indices_memview_f64


@pytest.mark.parametrize("dtype, func", [(np.float32, find_le_indices_memview_f32), (np.float64, find_le_indices_memview_f64)])
def test_find_le_indices(dtype, func):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], dtype=dtype)
    expected = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 3, -1], dtype=np.int64)

    result = func(a, b)
    assert_allclose(result, expected)
