import numpy as np
import pytest
from numpy.testing import assert_allclose
from pflm.smooth.kernel import KernelType
from pflm.smooth.polyfit import polyfit1d

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_happy_case(dtype):
    bw = 0.25
    x = np.linspace(0.0, 1.0, 11, dtype=dtype)
    y = 2.0 * x**2 + 3 * x
    w = np.ones_like(x)
    x_new = np.linspace(0.0, 1.0, 11, dtype=dtype)

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.EPANECHNIKOV, 1, 0),
        np.array(
            [
                -0.004684015,
                0.337087794,
                0.706823529,
                1.106823529,
                1.546823529,
                2.026823529,
                2.546823529,
                3.106823529,
                3.706823529,
                4.337087794,
                4.995315985,
            ]
        ),
        rtol=1e-5,
        atol=0,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.GAUSSIAN, 1, 0),
        np.array(
            [-0.05916294, 0.32503678, 0.73101491, 1.16175986, 1.61943037, 2.10520439,
              2.61943037, 3.16175986, 3.73101491, 4.32503678, 4.94083706]
        ),
        rtol=1e-5,
        atol=0,
    )


# @pytest.mark.parametrize("dtype", [np.float32, np.float64])
# def test_polyfit1d_big(dtype):
#     bw = 5.445
#     x = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 10.0, 8.0, 9.0], dtype=dtype)
#     y = np.linspace(0.1, 1.0, 10, dtype=dtype)
#     w = np.ones_like(x, dtype=dtype) / len(x)

#     x_new = np.linspace(1.0, 10.0, 21, dtype=dtype)
#     sorted_idx = np.argsort(x)
#     xs = x[sorted_idx]
#     ys = y[sorted_idx]
#     ws = w[sorted_idx]

#     assert_allclose(
#         polyfit1d(xs, ys, ws, x_new, bw, KernelType.EPANECHNIKOV, 1, 0),
#         np.array(
#             [
#                 0.11106948,
#                 0.15270080,
#                 0.19955290,
#                 0.24470157,
#                 0.28841027,
#                 0.33296682,
#                 0.37863591,
#                 0.42584445,
#                 0.47321237,
#                 0.51640379,
#                 0.55934666,
#                 0.60205077,
#                 0.64452157,
#                 0.68613319,
#                 0.72715539,
#                 0.76718717,
#                 0.80637783,
#                 0.84394526,
#                 0.87681739,
#                 0.90155526,
#                 0.92525725,
#             ]
#         ),
#         rtol=1e-5,
#         atol=0,
#     )

#     assert_allclose(
#         polyfit1d(xs, ys, ws, x_new, bw, KernelType.EPANECHNIKOV, 3, 1),
#         np.array(
#             [
#                 0.22939675,
#                 0.16441384,
#                 0.10691559,
#                 0.10127414,
#                 0.09594557,
#                 0.09034451,
#                 0.08838902,
#                 0.08917251,
#                 0.09578496,
#                 0.11396229,
#                 0.12031462,
#                 0.12371511,
#                 0.13072795,
#                 0.12891310,
#                 0.12311516,
#                 0.10845017,
#                 0.07325234,
#                 0.02871813,
#                 -0.03237792,
#                 -0.16330836,
#                 -0.33960212,
#             ]
#         ),
#         rtol=1e-5,
#         atol=0,
#     )

#     assert_allclose(
#         polyfit1d(xs, ys, ws, x_new, bw, KernelType.GAUSSIAN, 1, 0),
#         np.array(
#             [
#                 0.11458635,
#                 0.15908777,
#                 0.20347649,
#                 0.24774155,
#                 0.29187096,
#                 0.33585169,
#                 0.37966961,
#                 0.42330946,
#                 0.46675486,
#                 0.50998821,
#                 0.55299071,
#                 0.59574225,
#                 0.63822145,
#                 0.68040555,
#                 0.72227039,
#                 0.76379037,
#                 0.80493842,
#                 0.84568594,
#                 0.88600279,
#                 0.92585727,
#                 0.96521608,
#             ]
#         ),
#         rtol=1e-5,
#         atol=0,
#     )

#     assert_allclose(
#         polyfit1d(xs, ys, ws, x_new, bw, KernelType.GAUSSIAN, 3, 1),
#         np.array(
#             [
#                 0.06600843,
#                 0.07464421,
#                 0.08334085,
#                 0.09184819,
#                 0.09990986,
#                 0.10726300,
#                 0.11363786,
#                 0.11875738,
#                 0.12233682,
#                 0.12408326,
#                 0.12369510,
#                 0.12086162,
#                 0.11526236,
#                 0.10656663,
#                 0.09443291,
#                 0.07850827,
#                 0.05842778,
#                 0.03381386,
#                 0.00427569,
#                 -0.03059144,
#                 -0.07120678,
#             ]
#         ),
#         rtol=1e-5,
#         atol=0,
#     )


# @pytest.mark.parametrize("dtype", [np.float32, np.float64])
# def test_polyfit1d_raise_exception(dtype):
#     x = np.array([1, 3, 4, 2, 5, 7, 6, 10, 8], dtype=dtype)
#     xs = np.sort(x)
#     y = np.linspace(0.1, 1.0, 10, dtype=dtype)
#     w = np.ones_like(x, dtype=dtype)
#     x_new = np.linspace(1.0, 10.0, 21, dtype=dtype)
#     with pytest.raises(ValueError):
#         polyfit1d(x, y, w, x_new, 5.445, KernelType.EPANECHNIKOV, 1, 0)

#     with pytest.raises(ValueError):
#         polyfit1d(xs, y, w, x_new, 0.0, KernelType.EPANECHNIKOV, 1, 0)

#     with pytest.raises(ValueError):
#         polyfit1d(xs, y, w, x_new, -0.1, KernelType.EPANECHNIKOV, 1, 0)

#     with pytest.raises(ValueError):
#         polyfit1d(xs, y, w, x_new, 5.445, KernelType.EPANECHNIKOV, 1, 2)

#     with pytest.raises(ValueError):
#         polyfit1d(xs, y, w, x_new, 5.445, KernelType.EPANECHNIKOV, 0, 0)

#     x_new_unsorted = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
#     with pytest.raises(ValueError):
#         polyfit1d(xs, y, w, x_new_unsorted, 5.445, KernelType.EPANECHNIKOV, 1, 0)
