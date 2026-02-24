#!/bin/bash

echo "=== Comprehensive Matrix Multiplication Comparison ==="
echo "Testing: 1000×500 × 500×2000 = 1000×2000 (2M elements)"
echo ""

echo "1. UNOPTIMIZED (1D indexing):"
cd build-omp && ./matrixMul 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | tail -2

echo "2. OPTIMIZED (2D indexing):"
cd build-omp && ./matrixMulOptimized 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | tail -2

echo "3. TILED (cache blocking):"
cd build-omp && ./matrixMulTiled 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | tail -2

echo ""
echo "=== Performance Summary ==="
echo "Version | Kernel Time (ms) | Total Time (ms) | Relative Speed"
echo "--------|------------------|-----------------|----------------"

# Extract timing data
unopt_kernel=$(cd build-omp && ./matrixMul 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | grep "Kernel execution time" | awk '{print $4}')
opt_kernel=$(cd build-omp && ./matrixMulOptimized 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | grep "Kernel execution time" | awk '{print $4}')
tiled_kernel=$(cd build-omp && ./matrixMulTiled 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | grep "Kernel execution time" | awk '{print $4}')

unopt_total=$(cd build-omp && ./matrixMul 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | grep "Total computation time" | awk '{print $4}')
opt_total=$(cd build-omp && ./matrixMulOptimized 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | grep "Total computation time" | awk '{print $4}')
tiled_total=$(cd build-omp && ./matrixMulTiled 1000 500 2000 2>&1 | grep -A 2 "Using device: AMD EPYC" | grep "Total computation time" | awk '{print $4}')

unopt_speed="1.0×"
opt_speed=$(awk "BEGIN {printf \"%.1f×\", $unopt_kernel / $opt_kernel}")
tiled_speed=$(awk "BEGIN {printf \"%.1f×\", $unopt_kernel / $tiled_kernel}")

echo "Unoptimized | $unopt_kernel | $unopt_total | $unopt_speed"
echo "Optimized | $opt_kernel | $opt_total | $opt_speed"
echo "Tiled | $tiled_kernel | $tiled_total | $tiled_speed"

echo ""
echo "✅ Comparison complete!"
