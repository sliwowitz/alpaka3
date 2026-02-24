#!/bin/bash

echo "Building CUDA version with source tree alpaka..."
rm -rf build-cuda
mkdir -p build-cuda
cd build-cuda

# Configure with CUDA backend only (compile-time selection)
cmake -S .. \
    -Dalpaka_USE_SOURCE_TREE=ON \
    -Dalpaka_DEP_CUDA=ON \
    -Dalpaka_DEP_OMP=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
cd ..
echo "CUDA build complete!"

# Test the build
echo "Testing CUDA version:"
./build-cuda/matrixMul 100 50 200 2>&1 | grep -E "(Using device|completed successfully|Sample results|C\[.*\])"