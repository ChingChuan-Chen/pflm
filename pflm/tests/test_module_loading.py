import pytest
import pflm


@pytest.fixture
def submodule_list():
    return ["fpca", "interp", "smooth", "utils"]


def test_pflm(submodule_list):
    pflm_items = set(dir(pflm))
    expected_items = set(submodule_list + ["__version__"])
    assert pflm_items.issuperset(expected_items)
    assert pflm_items.issubset(expected_items)


def test_import_submodules(submodule_list):
    for submodule in submodule_list:
        module = getattr(pflm, submodule, None)
        assert module is not None
        assert hasattr(module, "__name__")
