#!/bin/bash

echo "=== Matrix Multiplication Optimization Comparison ==="
echo "Testing different matrix sizes to compare unoptimized vs optimized implementations"
echo ""

# Test configurations
sizes=(
    "100 50 200"
    "500 250 1000"
    "1000 500 2000"
    "2000 1000 4000"
)

echo "| Size (M×N×K) | Backend | Version | Kernel Time (ms) | Total Time (ms) | Speedup |"
echo "|---------------|---------|---------|------------------|-----------------|---------|"

for size in "${sizes[@]}"; do
    read M N K <<< "$size"
    total_elements=$((M * K))
    
    echo "Testing size: $M × $N × $K = $total_elements elements"
    
    # OpenMP Unoptimized
    omp_unopt_kernel=$(cd build-omp && ./matrixMul $M $N $K 2>&1 | grep "Using device: AMD EPYC" -A 2 | grep "Kernel execution time" | awk '{print $4}')
    omp_unopt_total=$(cd build-omp && ./matrixMul $M $N $K 2>&1 | grep "Using device: AMD EPYC" -A 2 | grep "Total computation time" | awk '{print $4}')
    
    # OpenMP Optimized  
    omp_opt_kernel=$(cd build-omp && ./matrixMulOptimized $M $N $K 2>&1 | grep "Using device: AMD EPYC" -A 2 | grep "Kernel execution time" | awk '{print $4}')
    omp_opt_total=$(cd build-omp && ./matrixMulOptimized $M $N $K 2>&1 | grep "Using device: AMD EPYC" -A 2 | grep "Total computation time" | awk '{print $4}')
    
    # CUDA Unoptimized
    cuda_unopt_kernel=$(cd build-cuda && ./matrixMul $M $N $K 2>&1 | grep "Using device: NVIDIA" -A 2 | grep "Kernel execution time" | awk '{print $4}')
    cuda_unopt_total=$(cd build-cuda && ./matrixMul $M $N $K 2>&1 | grep "Using device: NVIDIA" -A 2 | grep "Total computation time" | awk '{print $4}')
    
    # CUDA Optimized
    cuda_opt_kernel=$(cd build-cuda && ./matrixMulOptimized $M $N $K 2>&1 | grep "Using device: NVIDIA" -A 2 | grep "Kernel execution time" | awk '{print $4}')
    cuda_opt_total=$(cd build-cuda && ./matrixMulOptimized $M $N $K 2>&1 | grep "Using device: NVIDIA" -A 2 | grep "Total computation time" | awk '{print $4}')
    
    # Calculate speedups
    omp_speedup=$(awk "BEGIN {printf \"%.1f\", $omp_unopt_kernel / $omp_opt_kernel}")
    cuda_speedup=$(awk "BEGIN {printf \"%.1f\", $cuda_unopt_kernel / $cuda_opt_kernel}")
    
    echo "| $M×$N×$K | OpenMP | Unoptimized | $omp_unopt_kernel | $omp_unopt_total | - |"
    echo "| $M×$N×$K | OpenMP | Optimized | $omp_opt_kernel | $omp_opt_total | ${omp_speedup}× |"
    echo "| $M×$N×$K | CUDA | Unoptimized | $cuda_unopt_kernel | $cuda_unopt_total | - |"
    echo "| $M×$N×$K | CUDA | Optimized | $cuda_opt_kernel | $cuda_opt_total | ${cuda_speedup}× |"
    echo ""
    
done

echo "✅ Matrix multiplication comparison complete!"
