#!/usr/bin/env bash

set -euo pipefail

build_dir="build/docker-cp$(python -c 'import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))')"

python -m pip install --verbose --no-build-isolation --editable . \
    --config-settings editable-verbose=true \
    --config-settings "build-dir=${build_dir}"

python -m pip install "pytest>=7.1.2" "pytest-cov>=2.9.0"

if [ "$#" -eq 0 ]; then
    exec bash
fi

exec "$@"