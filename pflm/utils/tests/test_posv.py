import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.utils._lapack_helper import _posv_memview_f32, _posv_memview_f64


@pytest.mark.parametrize("dtype, posv_func", [(np.float64, _posv_memview_f64), (np.float32, _posv_memview_f32)])
def test_posv_matrix_lower_triangular_matrix(dtype, posv_func):
    A = np.array(
        [
            [3.14, 0.0, 0.0, 0.0, 0.0],
            [0.17, 0.79, 0.0, 0.0, 0.0],
            [-0.90, 0.83, 4.53, 0.0, 0.0],
            [1.65, -0.65, -3.7, 5.32, 0.0],
            [-0.72, 0.28, 1.6, -1.37, 1.98],
        ],
        dtype=dtype,
    )
    b = np.array([-7.29, 9.25, 5.99, -1.94, -8.3], dtype=dtype)
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    info = posv_func(A_c, b, 108, 5, 1, 5, 5)  # 108 = 'l'
    assert info == 0
    assert_allclose(b, np.array([-6.023842441873, 15.618630181394, 3.022190200411, 3.251997882301, -8.783156682244], dtype=dtype), rtol=1e-5)


@pytest.mark.parametrize("dtype, posv_func", [(np.float64, _posv_memview_f64), (np.float32, _posv_memview_f32)])
def test_posv_matrix_upper_triangular_matrix(dtype, posv_func):
    A = np.array(
        [
            [3.14, 0.0, 0.0, 0.0, 0.0],
            [0.17, 0.79, 0.0, 0.0, 0.0],
            [-0.90, 0.83, 4.53, 0.0, 0.0],
            [1.65, -0.65, -3.7, 5.32, 0.0],
            [-0.72, 0.28, 1.6, -1.37, 1.98],
        ],
        dtype=dtype,
    )
    b = np.array([-7.29, 9.25, 5.99, -1.94, -8.3], dtype=dtype)
    A_c = np.ascontiguousarray(A.T.ravel(order="F"))
    info = posv_func(A_c, b, 117, 5, 1, 5, 5)  # 117 = 'u'
    assert info == 0
    assert_allclose(b, np.array([-6.023842441873, 15.618630181394, 3.022190200411, 3.251997882301, -8.783156682244], dtype=dtype), rtol=1e-5)


@pytest.mark.parametrize("dtype, posv_func", [(np.float64, _posv_memview_f64), (np.float32, _posv_memview_f32)])
def test_posv_matrix_nrhs_greater_than_1(dtype, posv_func):
    A = np.array(
        [
            [3.14, 0.0, 0.0, 0.0, 0.0],
            [0.17, 0.79, 0.0, 0.0, 0.0],
            [-0.90, 0.83, 4.53, 0.0, 0.0],
            [1.65, -0.65, -3.7, 5.32, 0.0],
            [-0.72, 0.28, 1.6, -1.37, 1.98],
        ],
        dtype=dtype,
    )
    b = np.array([[-7.29, 6.11, 0.59], [9.25, 2.9, 8.88], [5.99, -5.05, 7.57], [-1.94, -3.8, 5.57], [-8.3, 9.66, -1.67]], dtype=dtype)
    A_c = np.ascontiguousarray(A.ravel(order="F"))
    b_c = np.ascontiguousarray(b.ravel(order="F"))
    info = posv_func(A_c, b_c, 108, 5, 3, 5, 5)
    assert info == 0
    assert_allclose(
        b_c.reshape(3, 5).T,
        np.array(
            [
                [-6.023842441873, 3.954825759466, -3.141148550011],
                [15.618630181394, 4.317904879938, 13.053952843527],
                [3.022190200411, -8.254843834968, 4.905002828673],
                [3.251997882301, -4.827330115082, 6.108520441323],
                [-8.783156682244, 9.036752049858, -3.56871983508],
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
    )
