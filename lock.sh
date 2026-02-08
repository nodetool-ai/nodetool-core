#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(pwd)
mkdir -p conda-lock

# Workaround for local build.py shadowing 'build' module
pushd /tmp > /dev/null

python3 -m conda_lock lock \
  -f "$ROOT_DIR/environment.yml" \
  --platform linux-64 \
  --platform linux-aarch64 \
  --kind env \
  --filename-template "$ROOT_DIR/conda-lock/environment-{platform}.lock"

popd > /dev/null

cp environment.yml conda-lock/environment.yml
