import numpy as np
import pytest

from pflm.smooth.kernel import KernelType
from pflm.utils.utility import rotate_polyfit2d


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotate_polyfit2d_happy_path(dtype):
    # prepare simple data
    x_grid_tmp = np.array(
        [
            [1.0, 1.5],
            [1.0, 2.0],
            [1.0, 2.5],
            [1.0, 3.0],
            [1.5, 2.0],
            [1.5, 2.5],
            [1.5, 3.0],
            [2.0, 2.5],
            [2.0, 3.0],
            [2.5, 3.0],
        ],
        dtype=dtype,
    )
    x_grid = np.concatenate([x_grid_tmp, x_grid_tmp[:, [1, 0]]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.2, 4.5, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.2, 4.5, 3.0, 2.0], dtype=dtype)
    w = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0], dtype=dtype)
    new_grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=dtype)
    bandwidth = 1.5
    expected_results = {
        KernelType.GAUSSIAN: np.array([1.785728769758, 3.936673578508, 3.831632268517], dtype=dtype),
        KernelType.EPANECHNIKOV: np.array([-0.842750718794, 4.144551119171, 0.41162802378], dtype=dtype),
    }
    for kernel_type, expected_array in expected_results.items():
        output = rotate_polyfit2d(x_grid, y, w, new_grid, bandwidth, kernel_type)
        # check shapes
        assert output.shape == (3,)
        # check types
        assert output.dtype == dtype
        # check values
        assert np.allclose(output, expected_array)
