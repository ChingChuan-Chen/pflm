import pflm


def test_pflm():
    pflm_items = set(dir(pflm))
    expected_items = set(
        [
            "fpca",
            "interp",
            "smooth",
            "utils",
            "__version__",
        ]
    )
    # validate that pflm has the expected items
    assert pflm_items.issuperset(expected_items)
    assert pflm_items.issubset(expected_items)
