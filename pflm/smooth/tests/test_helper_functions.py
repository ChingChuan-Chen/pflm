from pflm.smooth._polyfit import (
    search_lower_bound_f64, search_lower_bound_f32, search_location_f64, search_location_f32
)
import numpy as np
import pytest
from numpy.testing import assert_allclose

@pytest.mark.parametrize("dtype, func", [
    (np.float32, search_lower_bound_f32),
    (np.float64, search_lower_bound_f64)
])
def test_search_lower_bound(dtype, func):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], dtype=dtype)
    expected_right_inclusive = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    expected_not_right_inclusive = np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int64)
    for i, v in enumerate(b):
        assert func(a, v, right_inclusive=1) == expected_right_inclusive[i], f"Failed at index {i} with value {v}"
        assert func(a, v, right_inclusive=0) == expected_not_right_inclusive[i], f"Failed at index {i} with value {v}"


@pytest.mark.parametrize("dtype, func", [
    (np.float32, search_location_f32),
    (np.float64, search_location_f64)
])
def test_search_location(dtype, func):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], dtype=dtype)
    expected = np.array([-1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.int64)
    for i, v in enumerate(b):
        assert func(a, v) == expected[i], f"Failed at index {i} with value {v}"
