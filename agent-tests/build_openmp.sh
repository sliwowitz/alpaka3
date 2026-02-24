#!/bin/bash

echo "Building OpenMP version with source tree alpaka..."
rm -rf build-omp
mkdir -p build-omp
cd build-omp

# Configure with OpenMP backend only (compile-time selection)
cmake -S .. \
    -Dalpaka_USE_SOURCE_TREE=ON \
    -Dalpaka_DEP_OMP=ON \
    -Dalpaka_DEP_CUDA=OFF \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
cd ..
echo "OpenMP build complete!"

# Test the build
echo "Testing OpenMP version:"
./build-omp/matrixMul 100 50 200 2>&1 | grep -E "(Using device|completed successfully|Sample results|C\[.*\])"