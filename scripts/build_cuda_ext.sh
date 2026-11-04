#!/bin/bash
set -e
ln -sfn "/home/sharkum/Desktop/Vision/Assignment 3/.env/tadgs" /tmp/tadgs_env
ln -sfn "/home/sharkum/Desktop/Vision/Assignment 3/TAD-GS" /tmp/tadgs_repo

export PATH="/tmp/tadgs_env/bin:$PATH"
export CUDA_HOME="/tmp/tadgs_env"
export TORCH_CUDA_ARCH_LIST="8.6"
export CC="/tmp/tadgs_env/bin/x86_64-conda-linux-gnu-gcc"
export CXX="/tmp/tadgs_env/bin/x86_64-conda-linux-gnu-g++"

echo "Building gaussian_rasterization_ch3..."
cd /tmp/tadgs_repo/submodules/gaussian_rasterization_ch3
/tmp/tadgs_env/bin/python setup.py install

echo "CUDA extensions built successfully!"
