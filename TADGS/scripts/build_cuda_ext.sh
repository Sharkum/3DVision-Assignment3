#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"          # .../TADGS  (holds data/, runs/, results/, TAD-GS/)
REPO_ROOT="$(dirname "$WORKSPACE")"           # repo root  (holds the shared .env/)
# Symlinks avoid spaces in the path ("Assignment 3") breaking the CUDA build
ln -sfn "$REPO_ROOT/.env/tadgs" /tmp/tadgs_env
ln -sfn "$WORKSPACE/TAD-GS" /tmp/tadgs_repo

export PATH="/tmp/tadgs_env/bin:$PATH"
export CUDA_HOME="/tmp/tadgs_env"
export TORCH_CUDA_ARCH_LIST="8.6"
export CC="/tmp/tadgs_env/bin/x86_64-conda-linux-gnu-gcc"
export CXX="/tmp/tadgs_env/bin/x86_64-conda-linux-gnu-g++"

echo "Building gaussian_rasterization_ch3..."
cd /tmp/tadgs_repo/submodules/gaussian_rasterization_ch3
/tmp/tadgs_env/bin/python setup.py install

echo "CUDA extensions built successfully!"
