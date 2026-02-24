#!/bin/bash

echo "=== Performance Test: Single Backend Execution ==="
echo ""

M=1000
N=500
K=2000

echo "Test: ${M}x${N} * ${N}x${K} = ${M}x${K} matrix multiplication"
echo ""

echo "1. OpenMP Backend (CPU only):"
cd /workspace/agent-tests/build-omp
./matrixMul $M $N $K 2>&1 | grep -E "(Using device|Kernel execution|Total computation|completed successfully)"

echo ""
echo "2. CUDA Backend (GPU only):"
cd /workspace/agent-tests/build-cuda
./matrixMul $M $N $K 2>&1 | grep -E "(Using device|Kernel execution|Total computation|completed successfully)"

echo ""
echo "✅ Performance test complete!"