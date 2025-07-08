import numpy as np
from pflm import FunctionalDataGenerator


def test_functional_data_generator():
    n = 20
    t = np.linspace(0, 10, 5)
    fdg = FunctionalDataGenerator(
        t,
        lambda x: np.sin(0.4 * x),
        lambda x: 2.0 * np.log(x + 5.5)
    )
    true_y = fdg.generate(n, 100)
    y = FunctionalDataGenerator.make_missing(true_y, 2, 101)

    assert y.shape == (n, 5)
    assert np.isnan(true_y).sum() == 0 # Ensure no NaNs in the original data
    assert np.isnan(y).sum() == 40  # 2 missing values per sample
    assert np.all(np.isfinite(true_y))  # Ensure no NaNs in the original data
    assert fdg.get_num_fpc() == 5
    assert fdg.get_fpca_phi().shape == (5, 5)
