#!/bin/bash

echo "=== Comprehensive Performance Test ==="
echo "Testing all executables on both OpenMP and CUDA backends"
echo ""

# Test parameters
M=1000
N=500
K=2000

echo "=== OpenMP Backend (CPU) ==="
cd build-omp

echo "1. vectorAdd10M:"
./vectorAdd10M 1000000 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo "2. matrixAdd:"
./matrixAdd 1000 500 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo "3. matrixMul (unoptimized):"
./matrixMul $M $N $K 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo "4. matrixMulOptimized:"
./matrixMulOptimized $M $N $K 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo ""
echo "=== CUDA Backend (GPU) ==="
cd ../build-cuda

echo "5. vectorAdd10M:"
./vectorAdd10M 1000000 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo "6. matrixAdd:"
./matrixAdd 1000 500 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo "7. matrixMul (unoptimized):"
./matrixMul $M $N $K 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo "8. matrixMulOptimized:"
./matrixMulOptimized $M $N $K 2>&1 | grep -E "(Using device|Kernel execution|Total computation)"

echo ""
echo "✅ All tests complete!"
