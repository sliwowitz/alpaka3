#!/bin/bash

echo "=== Performance Data Collection ==="
echo ""

# Test parameters
VECTOR_SIZE=1000000
MATRIX_ADD_ROWS=1000
MATRIX_ADD_COLS=500
MATRIX_MUL_M=1000
MATRIX_MUL_N=500
MATRIX_MUL_K=2000

echo "Test Configuration:"
echo "- Vector size: $VECTOR_SIZE elements"
echo "- Matrix addition: $MATRIX_ADD_ROWS x $MATRIX_ADD_COLS"
echo "- Matrix multiplication: $MATRIX_MUL_M x $MATRIX_MUL_N * $MATRIX_MUL_N x $MATRIX_MUL_K"
echo ""

echo "=== OpenMP (CPU) Performance ==="
cd build-omp

echo "1. vectorAdd10M:"
./vectorAdd10M $VECTOR_SIZE 2>&1 | grep -A 2 "Using device: AMD EPYC" | tail -2

echo "2. matrixAdd:"
./matrixAdd $MATRIX_ADD_ROWS $MATRIX_ADD_COLS 2>&1 | grep -A 2 "Using device: AMD EPYC" | tail -2

echo "3. matrixMul (unoptimized):"
./matrixMul $MATRIX_MUL_M $MATRIX_MUL_N $MATRIX_MUL_K 2>&1 | grep -A 2 "Using device: AMD EPYC" | tail -2

echo "4. matrixMulOptimized:"
./matrixMulOptimized $MATRIX_MUL_M $MATRIX_MUL_N $MATRIX_MUL_K 2>&1 | grep -A 2 "Using device: AMD EPYC" | tail -2

echo ""
echo "=== CUDA (GPU) Performance ==="
cd ../build-cuda

echo "5. vectorAdd10M:"
./vectorAdd10M $VECTOR_SIZE 2>&1 | grep -A 2 "Using device: NVIDIA" | tail -2

echo "6. matrixAdd:"
./matrixAdd $MATRIX_ADD_ROWS $MATRIX_ADD_COLS 2>&1 | grep -A 2 "Using device: NVIDIA" | tail -2

echo "7. matrixMul (unoptimized):"
./matrixMul $MATRIX_MUL_M $MATRIX_MUL_N $MATRIX_MUL_K 2>&1 | grep -A 2 "Using device: NVIDIA" | tail -2

echo "8. matrixMulOptimized:"
./matrixMulOptimized $MATRIX_MUL_M $MATRIX_MUL_N $MATRIX_MUL_K 2>&1 | grep -A 2 "Using device: NVIDIA" | tail -2

echo ""
echo "✅ Performance data collection complete!"
