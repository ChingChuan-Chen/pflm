import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.smooth.kernel import KernelType
from pflm.utils import rotate_polyfit2d


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotate_polyfit2d_happy_path(dtype):
    # Prepare structured grid (duplicated with swapped columns to enlarge sample size)
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
        KernelType.LOGISTIC: np.array([2.221844386042, 3.868094087128, 4.436053556877], dtype=dtype),
        KernelType.SIGMOID: np.array([1.981201947617, 3.927923978742, 4.068217767209], dtype=dtype),
        KernelType.RECTANGULAR: np.array([-0.946409431940, 3.800545371913, 0.933812949640], dtype=dtype),
        KernelType.TRIANGULAR: np.array([-0.809524801663, 4.368192974876, 0.485585525118], dtype=dtype),
        KernelType.EPANECHNIKOV: np.array([-0.842750718794, 4.144551119171, 0.41162802378], dtype=dtype),
        KernelType.BIWEIGHT: np.array([-0.801664077584, 4.419861019520, 0.528322109587], dtype=dtype),
        KernelType.TRIWEIGHT: np.array([-0.741935753091, 4.592953369351, 0.669094930205], dtype=dtype),
        KernelType.TRICUBE: np.array([-0.813345353665, 4.381220324320, 0.515851779944], dtype=dtype),
        KernelType.COSINE: np.array([-0.834688674032, 4.200387860920, 0.426987172622], dtype=dtype),
    }
    for kernel_type, expected_array in expected_results.items():
        output = rotate_polyfit2d(x_grid, y, w, new_grid, bandwidth, kernel_type)
        assert output.shape == (3,)
        assert output.dtype == dtype
        assert_allclose(output, expected_array, rtol=1e-5, atol=0.0)


def _make_basic_inputs(dtype=np.float64):
    # Minimal synthetic data for branch / error coverage
    x = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0], [1.5, 2.5]], dtype=dtype)
    y = np.array([1.0, 2.0, 1.5, 1.2], dtype=dtype)
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    new_x = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=dtype)
    return x, y, w, new_x


@pytest.mark.parametrize("bad_type", [float("nan"), np.nan])
def test_rotate_polyfit2d_bandwidth_nan(bad_type):
    x, y, w, new_x = _make_basic_inputs()
    with pytest.raises(ValueError, match="bandwidth must not be NaN"):
        rotate_polyfit2d(x, y, w, new_x, bandwidth=bad_type, kernel_type=KernelType.GAUSSIAN)


@pytest.mark.parametrize("bad_type", ["2", [1], (2,), {"a": 1}])
def test_rotate_polyfit2d_invalid_bandwidth_type(bad_type):
    x, y, w, new_x = _make_basic_inputs()
    with pytest.raises(ValueError, match="bandwidth must be a numeric value"):
        rotate_polyfit2d(x, y, w, new_x, bandwidth=bad_type, kernel_type=KernelType.GAUSSIAN)


def test_rotate_polyfit2d_non_positive_bandwidth():
    x, y, w, new_x = _make_basic_inputs()
    with pytest.raises(ValueError, match="bandwidth must be a positive number"):
        rotate_polyfit2d(x, y, w, new_x, bandwidth=0.0, kernel_type=KernelType.GAUSSIAN)
    with pytest.raises(ValueError, match="bandwidth must be a positive number"):
        rotate_polyfit2d(x, y, w, new_x, bandwidth=-1.0, kernel_type=KernelType.GAUSSIAN)


def test_rotate_polyfit2d_invalid_kernel_type():
    x, y, w, new_x = _make_basic_inputs()
    with pytest.raises(ValueError, match="kernel must be one of"):
        rotate_polyfit2d(x, y, w, new_x, 1.0, kernel_type=999)  # invalid enum-like value


def test_rotate_polyfit2d_x_grid_wrong_shape_second_dim_not_2():
    # Force shape mismatch: want shape (n, 2), provide (2, 3)
    x = np.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]], dtype=np.float64).T  # (3,2)
    x_bad = x.T  # (2,3) triggers failure
    y = np.array([1.0, 2.0], dtype=np.float64)
    w = np.array([1.0, 1.0], dtype=np.float64)
    new_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="x_grid must be a 2D array with shape \\(n, 2\\)"):
        rotate_polyfit2d(x_bad, y, w, new_x, 1.0, KernelType.GAUSSIAN)


def test_rotate_polyfit2d_new_grid_wrong_shape():
    x, y, w, _ = _make_basic_inputs()
    new_x = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).T
    with pytest.raises(ValueError, match="new_grid must be a 2D array with shape \\(m, 2\\)"):
        rotate_polyfit2d(x, y, w, new_x, 1.0, KernelType.GAUSSIAN)

    with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead:"):
        rotate_polyfit2d(x, y, w, np.array([0.0, 0.0]), 1.0, KernelType.GAUSSIAN)


def test_rotate_polyfit2d_y_not_1d():
    x, y, w, new_x = _make_basic_inputs()
    y2 = y.reshape(-1, 1)
    with pytest.raises(ValueError, match="y must be a 1D array"):
        rotate_polyfit2d(x, y2, w, new_x, 1.0, KernelType.GAUSSIAN)


def test_rotate_polyfit2d_w_not_1d():
    x, y, w, new_x = _make_basic_inputs()
    w2 = w.reshape(-1, 1)
    with pytest.raises(ValueError, match="w must be a 1D array"):
        rotate_polyfit2d(x, y, w2, new_x, 1.0, KernelType.GAUSSIAN)


def test_rotate_polyfit2d_permutation_invariance():
    x, y, w, new_x = _make_basic_inputs(dtype=np.float64)
    rng = np.random.default_rng(42)
    perm = rng.permutation(x.shape[0])
    x_shuffled = x[perm]
    y_shuffled = y[perm]
    w_shuffled = w[perm]
    out_orig = rotate_polyfit2d(x, y, w, new_x, 1.0, KernelType.GAUSSIAN)
    out_perm = rotate_polyfit2d(x_shuffled, y_shuffled, w_shuffled, new_x, 1.0, KernelType.GAUSSIAN)
    assert_allclose(out_orig, out_perm)


def test_rotate_polyfit2d_float32_path_and_kernel_enum():
    x, y, w, new_x = _make_basic_inputs(dtype=np.float32)
    out = rotate_polyfit2d(x, y, w, new_x, 1.0, KernelType.GAUSSIAN)
    assert out.dtype == np.float32
    assert out.shape[0] == new_x.shape[0]


def test_rotate_polyfit2d_kernel_each_branch_minimal():
    x, y, w, new_x = _make_basic_inputs(dtype=np.float64)
    for k in KernelType:
        out = rotate_polyfit2d(x, y, w, new_x, 1.0, k)
        assert out.shape
