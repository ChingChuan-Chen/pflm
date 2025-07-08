import numpy as np
import pytest
from pflm.utils.trapz import trapz

def test_trapz_1d_and_2d():
    x = np.linspace(0, 1, 5)
    y = x**2
    val = trapz(y, x)
    assert np.isscalar(val)
    y2 = np.vstack([y, y])
    val2 = trapz(y2, x)
    assert val2.shape == (2,)
    # shape mismatch
    with pytest.raises(ValueError):
        trapz(np.ones(4), np.ones(5))
    with pytest.raises(ValueError):
        trapz(np.ones((2, 4)), np.ones(5))
    # wrong dimension of x
    with pytest.raises(ValueError):
        trapz(np.ones((2, 2, 2)), x)
