# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
DEFAULT_MESON_BUILD_DIR = build/cp$(shell python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")' )

all:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  dev                  build scikit-learn with Meson"
	@echo "  clean                clean scikit-learn Meson build. Very rarely needed,"
	@echo "  format               run code formatting tools"
	@echo "                       since meson-python recompiles on import."
	@echo "  docs                 generate HTML documentation"
	@echo "  docs-clean           clean generated documentation"

.PHONY: all

dev: dev-meson

dev-meson:
	pip install --verbose --no-build-isolation --editable . --config-settings editable-verbose=true

format:
	@echo "Running code formatting tools..."
	ruff check --fix pflm
	ruff format pflm

clean: clean-meson

clean-meson:
	pip uninstall -y pflm
	# It seems in some cases removing the folder avoids weird compilation errors.
	# For some reason ninja clean -C $(DEFAULT_MESON_BUILD_DIR) is not enough.
	rm -rf $(DEFAULT_MESON_BUILD_DIR)

.PHONY: docs docs-clean

docs:
	@echo "Generating API stubs..."
	sphinx-apidoc -o docs/api pflm -f
	@echo "Building HTML documentation..."
	sphinx-build -b html docs docs/_build/html
	@echo "Docs built at docs/_build/html/index.html"

docs-clean:
	@echo "Cleaning documentation build artifacts..."
	@rm -rf docs/_build docs/api
