#!/bin/bash
set -e
WORKSPACE="/home/sharkum/Desktop/Vision/Assignment 3"
SRC_ENV="/home/sharkum/Desktop/Vision/Project/.env/signavatar"
DST_ENV="$WORKSPACE/.env/tadgs"

mkdir -p "$WORKSPACE/.env"
if [ ! -d "$DST_ENV" ]; then
    echo "Cloning environment from $SRC_ENV to $DST_ENV ..."
    cp -a "$SRC_ENV" "$DST_ENV"
fi

# Fix prefix in nvcc.profile and bin scripts if needed
sed -i "s|$SRC_ENV|$DST_ENV|g" "$DST_ENV/bin/nvcc.profile" 2>/dev/null || true
sed -i "s|$SRC_ENV|$DST_ENV|g" "$DST_ENV/bin/pip" "$DST_ENV/bin/pip3" "$DST_ENV/bin/pip3.10" 2>/dev/null || true

export PATH="$DST_ENV/bin:$PATH"
export CUDA_HOME="$DST_ENV"
export TORCH_CUDA_ARCH_LIST="8.6"

echo "Installing missing Python packages (plyfile, natsort, kornia, torchmetrics, pycolmap)..."
"$DST_ENV/bin/pip" install --no-build-isolation plyfile natsort kornia "torchmetrics<1.5" pycolmap

echo "Building TAD-GS CUDA extensions (gaussian_rasterization_ch3 and simple-knn)..."
"$DST_ENV/bin/pip" install --no-build-isolation "$WORKSPACE/TAD-GS/submodules/gaussian_rasterization_ch3"
"$DST_ENV/bin/pip" install --no-build-isolation "$WORKSPACE/TAD-GS/submodules/simple-knn"

echo "Environment setup complete!"

