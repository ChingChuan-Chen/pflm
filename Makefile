# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
DEFAULT_MESON_BUILD_DIR = build/cp$(shell python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")' )

all:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  dev                  build pflm with Meson"
	@echo "  clean                clean pflm Meson build. Very rarely needed,"
	@echo "  format               run code formatting tools"
	@echo "                       since meson-python recompiles on import."
	@echo "  docker-build         build the Linux dev Docker image"
	@echo "  docker-shell         start an interactive shell in the Linux dev image"
	@echo "  docker-test          run pytest in the Linux dev image"
	@echo "  test                 run tests"

.PHONY: all

DOCKER_IMAGE ?= pflm-dev
DOCKER_PYTHON_VERSION ?= 3.13

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

test:
	pytest .

docker-build:
	docker build --build-arg PYTHON_VERSION=$(DOCKER_PYTHON_VERSION) -f Dockerfile.dev -t $(DOCKER_IMAGE)-$(DOCKER_PYTHON_VERSION) .

docker-shell: docker-build
	docker run --rm -it -v $(CURDIR):/work -w /work $(DOCKER_IMAGE)-$(DOCKER_PYTHON_VERSION) bash /work/tools/docker-dev-env.sh

docker-test: docker-build
	docker run --rm -v $(CURDIR):/work -w /work $(DOCKER_IMAGE)-$(DOCKER_PYTHON_VERSION) bash /work/tools/docker-dev-env.sh pytest . -v
