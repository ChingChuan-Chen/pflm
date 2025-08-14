import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.smooth._polyfit import calculate_kernel_value_f32, calculate_kernel_value_f64
from pflm.smooth.kernel import KernelType


def test_kernel_types():
    assert KernelType.GAUSSIAN.value == 0
    assert KernelType.LOGISTIC.value == 1
    assert KernelType.SIGMOID.value == 2
    assert KernelType.RECTANGULAR.value == 100
    assert KernelType.TRIANGULAR.value == 101
    assert KernelType.EPANECHNIKOV.value == 102
    assert KernelType.BIWEIGHT.value == 103
    assert KernelType.TRIWEIGHT.value == 104
    assert KernelType.TRICUBE.value == 105
    assert KernelType.COSINE.value == 106

    assert str(KernelType.GAUSSIAN) == "GAUSSIAN"
    assert str(KernelType.LOGISTIC) == "LOGISTIC"
    assert str(KernelType.SIGMOID) == "SIGMOID"
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
    assert repr(KernelType.RECTANGULAR) == "KernelType.RECTANGULAR"
    assert repr(KernelType.TRIANGULAR) == "KernelType.TRIANGULAR"
    assert repr(KernelType.EPANECHNIKOV) == "KernelType.EPANECHNIKOV"
    assert repr(KernelType.BIWEIGHT) == "KernelType.BIWEIGHT"
    assert repr(KernelType.TRIWEIGHT) == "KernelType.TRIWEIGHT"
    assert repr(KernelType.TRICUBE) == "KernelType.TRICUBE"
    assert repr(KernelType.COSINE) == "KernelType.COSINE"


def test_calculate_kernel_value_at_0():
    # Test Gaussian kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.GAUSSIAN.value)
    expected = 1 / np.sqrt(2 * np.pi)
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Logistic kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.LOGISTIC.value)
    expected = 1 / (np.exp(0) + 2.0 + np.exp(-0))
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Sigmoid kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.SIGMOID.value)
    expected = 2.0 / np.pi / (np.exp(0) + np.exp(-0))
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Rectangular kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.RECTANGULAR.value)
    expected = 0.5
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Triangular kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.TRIANGULAR.value)
    expected = 1.0
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Epanechnikov kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.EPANECHNIKOV.value)
    expected = 0.75
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Biweight kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.BIWEIGHT.value)
    expected = 15 / 16
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Triweight kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.TRIWEIGHT.value)
    expected = 35 / 32
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Tricube kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.TRICUBE.value)
    expected = 70 / 81
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Cosine kernel at 0
    result = calculate_kernel_value_f32(0.0, KernelType.COSINE.value)
    expected = np.pi / 4.0
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize(
    "dtype, func",
    [
        (np.float32, calculate_kernel_value_f32),
        (np.float64, calculate_kernel_value_f64),
    ],
)
def test_calculate_kernel_value(dtype, func):
    u = np.linspace(-1.1, 1.1, 111, dtype=dtype)

    # Test Gaussian kernel
    result = np.array([func(ui, KernelType.GAUSSIAN.value) for ui in u])
    expected = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Logistic kernel
    result = np.array([func(ui, KernelType.LOGISTIC.value) for ui in u])
    expected = 1 / (np.exp(u) + 2.0 + np.exp(-u))
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Sigmoid kernel
    result = np.array([func(ui, KernelType.SIGMOID.value) for ui in u])
    expected = 2.0 / np.pi / (np.exp(u) + np.exp(-u))
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    idx = np.nonzero(np.abs(u) <= 1.0)

    # Test Rectangular kernel
    result = np.array([func(ui, KernelType.RECTANGULAR.value) for ui in u])
    expected = np.zeros_like(u)
    expected[idx] = 0.5
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Triangular kernel
    result = np.array([func(ui, KernelType.TRIANGULAR.value) for ui in u])
    expected = np.zeros_like(u)
    expected[idx] = 1 - np.abs(u[idx])
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Epanechnikov kernel
    result = np.array([func(ui, KernelType.EPANECHNIKOV.value) for ui in u])
    expected = np.zeros_like(u)
    expected[idx] = 0.75 * (1 - u[idx] ** 2)
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Biweight kernel
    result = np.array([func(ui, KernelType.BIWEIGHT.value) for ui in u])
    expected = np.zeros_like(u)
    expected[idx] = (15 / 16) * (1 - u[idx] ** 2) ** 2
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Triweight kernel
    result = np.array([func(ui, KernelType.TRIWEIGHT.value) for ui in u])
    expected = np.zeros_like(u)
    expected[idx] = (35 / 32) * (1 - u[idx] ** 2) ** 3
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Tricube kernel
    result = np.array([func(ui, KernelType.TRICUBE.value) for ui in u])
    expected = np.zeros_like(u)
    expected[idx] = (70 / 81) * (1 - np.abs(u[idx]) ** 3) ** 3
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)

    # Test Cosine kernel
    result = np.array([func(ui, KernelType.COSINE.value) for ui in u])
    expected = np.zeros_like(u)
    expected[idx] = np.pi / 4.0 * np.abs(np.cos(np.pi / 2.0 * u[idx]))
    expected[np.abs(np.abs(u) - 1.0) <= 1e-7] = 0.0
    assert_allclose(result, expected, rtol=1e-5, atol=0.0)
