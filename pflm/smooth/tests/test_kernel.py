from pflm.smooth._polyfit import calculate_kernel_value_f64, calculate_kernel_value_f32
from pflm.smooth.kernel import KernelType
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_kernel_types():
    """Test the kernel types."""
    assert KernelType.GAUSSIAN.value == 0
    assert KernelType.LOGISTIC.value == 1
    assert KernelType.SIGMOID.value == 2
    assert KernelType.GAUSSIAN_VAR.value == 3
    assert KernelType.RECTANGULAR.value == 4
    assert KernelType.TRIANGULAR.value == 5
    assert KernelType.EPANECHNIKOV.value == 6
    assert KernelType.BIWEIGHT.value == 7
    assert KernelType.TRIWEIGHT.value == 8
    assert KernelType.TRICUBE.value == 9
    assert KernelType.COSINE.value == 10

    assert str(KernelType.GAUSSIAN) == "GAUSSIAN"
    assert str(KernelType.LOGISTIC) == "LOGISTIC"
    assert str(KernelType.SIGMOID) == "SIGMOID"
    assert str(KernelType.GAUSSIAN_VAR) == "GAUSSIAN_VAR"
    assert str(KernelType.RECTANGULAR) == "RECTANGULAR"
    assert str(KernelType.TRIANGULAR) == "TRIANGULAR"
    assert str(KernelType.EPANECHNIKOV) == "EPANECHNIKOV"
    assert str(KernelType.BIWEIGHT) == "BIWEIGHT"
    assert str(KernelType.TRIWEIGHT) == "TRIWEIGHT"
    assert str(KernelType.TRICUBE) == "TRICUBE"
    assert str(KernelType.COSINE) == "COSINE"

    assert repr(KernelType.GAUSSIAN) == "KernelType.GAUSSIAN"
    assert repr(KernelType.LOGISTIC) == "KernelType.LOGISTIC"
    assert repr(KernelType.SIGMOID) == "KernelType.SIGMOID"
    assert repr(KernelType.GAUSSIAN_VAR) == "KernelType.GAUSSIAN_VAR"
    assert repr(KernelType.RECTANGULAR) == "KernelType.RECTANGULAR"
    assert repr(KernelType.TRIANGULAR) == "KernelType.TRIANGULAR"
    assert repr(KernelType.EPANECHNIKOV) == "KernelType.EPANECHNIKOV"
    assert repr(KernelType.BIWEIGHT) == "KernelType.BIWEIGHT"
    assert repr(KernelType.TRIWEIGHT) == "KernelType.TRIWEIGHT"
    assert repr(KernelType.TRICUBE) == "KernelType.TRICUBE"
    assert repr(KernelType.COSINE) == "KernelType.COSINE"


@pytest.mark.parametrize("dtype, func, weight", [
    (np.float32, calculate_kernel_value_f32, 1.0),
    (np.float64, calculate_kernel_value_f64, 1.0),
    (np.float32, calculate_kernel_value_f32, 0.5),
    (np.float64, calculate_kernel_value_f64, 0.5)
])
def test_calculate_kernel_value(dtype, func, weight):
    u = np.linspace(-1.1, 1.1, 111, dtype=dtype)
    wj = np.ones_like(u, dtype=dtype) * weight

    # Test Gaussian kernel
    result = np.array([func(ui, KernelType.GAUSSIAN.value, wj[i]) for i, ui in enumerate(u)])
    expected = wj * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Logistic kernel
    result = np.array([func(ui, KernelType.LOGISTIC.value, wj[i]) for i, ui in enumerate(u)])
    expected = wj / (np.exp(u) + 2.0 + np.exp(-u))
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Sigmoid kernel
    result = np.array([func(ui, KernelType.SIGMOID.value, wj[i]) for i, ui in enumerate(u)])
    expected = wj * 2.0 / np.pi / (np.exp(u) + np.exp(-u))
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Gaussian variance kernel
    result = np.array([func(ui, KernelType.GAUSSIAN_VAR.value, wj[i]) for i, ui in enumerate(u)])
    expected = wj * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2) * (1.25 - 0.25 * u**2)
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # The index where u is within the range (-1, 1)
    idx = np.nonzero(np.abs(u) <= 1.0)

    # Test Rectangular kernel
    result = np.array([func(ui, KernelType.RECTANGULAR.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    expected[idx] = wj[idx] * 0.5
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Triangular kernel
    result = np.array([func(ui, KernelType.TRIANGULAR.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    expected[idx] = wj[idx] * (1 - np.abs(u[idx]))
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Epanechnikov kernel
    result = np.array([func(ui, KernelType.EPANECHNIKOV.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    expected[idx] = wj[idx] * 0.75 * (1 - u[idx]**2)
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Biweight kernel
    result = np.array([func(ui, KernelType.BIWEIGHT.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    expected[idx] = wj[idx] * (15 / 16) * (1 - u[idx]**2)**2
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Triweight kernel
    result = np.array([func(ui, KernelType.TRIWEIGHT.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    expected[idx] = wj[idx] * (35 / 32) * (1 - u[idx]**2)**3
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Tricube kernel
    result = np.array([func(ui, KernelType.TRICUBE.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    expected[idx] = wj[idx] * (70 / 81) * (1 - np.abs(u[idx])**3)**3
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Cosine kernel
    result = np.array([func(ui, KernelType.COSINE.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    expected[idx] = wj[idx] * np.pi / 4.0 * np.abs(np.cos(np.pi / 2.0 * u[idx]))
    expected[np.abs(np.abs(u) - 1.0) <= 1e-7] = 0.0  # Cosine kernel is zero at the boundaries
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("dtype, func, weight", [
    (np.float32, calculate_kernel_value_f32, 1.0),
    (np.float64, calculate_kernel_value_f64, 1.0),
    (np.float32, calculate_kernel_value_f32, 0.5),
    (np.float64, calculate_kernel_value_f64, 0.5)
])
def test_calculate_kernel_value_gausvar(dtype, func, weight):
    u = np.linspace(-4.0, 4.0, 161, dtype=dtype)
    wj = np.ones_like(u, dtype=dtype) * weight

    # Test Gaussian variance kernel with large u
    result = np.array([func(ui, KernelType.GAUSSIAN_VAR.value, wj[i]) for i, ui in enumerate(u)])
    expected = np.zeros_like(u)
    u_sq = u**2
    idx = np.nonzero(u_sq < 5.0)
    expected[idx] = wj[idx] * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u_sq[idx]) * (1.25 - 0.25 * u_sq[idx])
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)
