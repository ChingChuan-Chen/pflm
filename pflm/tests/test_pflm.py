import pytest
import pflm


def test_pflm():
    pflm_items = set(dir(pflm))
    expected_items = set(
        [
            "interp",
            "smooth",
            "utils",
            "__version__",
            "FunctionalPCAMuCovParams",
            "FunctionalPCAUserDefinedParams",
            "FunctionalPCA",
            "FunctionalDataGenerator",
        ]
    )
    assert pflm_items.issuperset(expected_items)
    assert pflm_items.issubset(expected_items)
