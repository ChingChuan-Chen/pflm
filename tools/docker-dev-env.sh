#!/usr/bin/env bash

set -euo pipefail

python -m pip install .[tests]
python -m pip uninstall -y pflm

build_dir="build/docker-cp$(python -c 'import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))')"
python -m pip install --verbose --no-build-isolation --editable . \
    --config-settings editable-verbose=true \
    --config-settings "build-dir=${build_dir}"

if [ "$#" -eq 0 ]; then
    exec bash
fi

exec "$@"
