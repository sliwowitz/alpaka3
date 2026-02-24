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
    
    # Function to extract timing data
    extract_timing() {
        local dir=$1
        local exec=$2
        local pattern=$3
        
        local kernel_time=$(cd $dir && ./$exec $M $N $K 2>&1 | grep "$pattern" -A 5 | grep "Kernel execution time" | awk '{print $4}')
        local total_time=$(cd $dir && ./$exec $M $N $K 2>&1 | grep "$pattern" -A 5 | grep "Total computation time" | awk '{print $4}')
        
        echo "$kernel_time $total_time"
    }
    
    # OpenMP Unoptimized
    read omp_unopt_kernel omp_unopt_total <<< $(extract_timing "build-omp" "matrixMul" "AMD EPYC")
    
    # OpenMP Optimized  
    read omp_opt_kernel omp_opt_total <<< $(extract_timing "build-omp" "matrixMulOptimized" "AMD EPYC")
    
    # CUDA Unoptimized
    read cuda_unopt_kernel cuda_unopt_total <<< $(extract_timing "build-cuda" "matrixMul" "NVIDIA")
    
    # CUDA Optimized
    read cuda_opt_kernel cuda_opt_total <<< $(extract_timing "build-cuda" "matrixMulOptimized" "NVIDIA")
    
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
