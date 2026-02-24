# Alpaka Matrix Operations - Performance Benchmark

## Summary

This project demonstrates performance-portable C++ code using the alpaka library to implement matrix operations that compile and run on both OpenMP (CPU) and CUDA (GPU) backends from the same source code.

## Programs

1. **vectorAdd.cpp** - Adds two vectors of 10 million random numbers
2. **matrixAdd.cpp** - Adds two 2000×5000 matrices (10M elements)  
3. **matMul.cpp** - Multiplies two matrices using naive O(n³) algorithm
4. **matMulOpt.cpp** - Matrix multiplication with blocked tiled algorithm (16×16 blocks)

## Compilation

### Prerequisites
- CMake 3.25+
- C++20 compatible compiler (GCC recommended)
- NVIDIA HPC SDK or CUDA Toolkit (for CUDA backend)
- OpenMP library

### Build Commands

#### OpenMP Backend (CPU)
```bash
cd agent-tests
rm -rf build && mkdir build && cd build
cmake .. -DBACKEND=OPENMP
make
```

#### CUDA Backend (GPU)
```bash
cd agent-tests
rm -rf build && mkdir build && cd build
cmake .. -DBACKEND=CUDA
make
```

### Running

```bash
# Vector addition (10M elements)
./vectorAdd

# Matrix addition (2000×5000)
./matrixAdd

# Matrix multiplication (specify M K N)
./matMul <M> <K> <N>
./matMul 1000 1000 1000

# Optimized matrix multiplication
./matMulOpt <M> <K> <N>
./matMulOpt 1000 1000 1000
```

## Performance Results

### 1. Vector Addition (10M elements)

| Backend | Device | Kernel Time | Total Time |
|---------|--------|-------------|------------|
| OpenMP | AMD EPYC | ~330-350 ms | ~440-520 ms |
| CUDA | NVIDIA A40 | ~7 ms | ~22 ms |

**Speedup: ~50x on GPU**

### 2. Matrix Addition (2000×5000 = 10M elements)

| Backend | Device | Kernel Time | Total Time |
|---------|--------|-------------|------------|
| OpenMP | AMD EPYC | ~330 ms | ~440 ms |
| CUDA | NVIDIA A40 | ~7 ms | ~22 ms |

**Speedup: ~50x on GPU**

### 3. Matrix Multiplication - Naive Algorithm

| Size (M×K×N) | Total Elements | OpenMP | CUDA | Speedup |
|--------------|---------------|--------|------|---------|
| 200×200×200 | 120K | 18.7 ms | 0.94 ms | 20x |
| 500×500×500 | 750K | 235 ms | 11 ms | 21x |
| 1000×1000×1000 | 3M | ~45s (est) | 2273 ms | ~20x |

### 4. Matrix Multiplication - Optimized (16×16 Block Tiling)

| Size (M×K×N) | Total Elements | OpenMP | CUDA | Speedup |
|--------------|---------------|--------|------|---------|
| 200×200×200 | 120K | 8.3 ms | ~0.5 ms | 17x |
| 500×500×500 | 750K | 140 ms | 25 ms | 5.6x |
| 1000×1000×1000 | 3M | 819 ms | 190 ms | 4.3x |
| 1500×1000×1500 | 5.25M | 1734 ms | 427 ms | 4.1x |
| 2000×1000×2000 | 8M | - | 772 ms | - |
| 2000×2000×2000 | 12M | - | 1468 ms | - |

**Optimization Impact (CUDA, naive → optimized):**
- 500×500×500: 11 ms → 25 ms (slower due to overhead)
- 1000×1000×1000: 2273 ms → 190 ms (**12x faster**)
- 2000×1000×2000: 6671 ms → 772 ms (**8.6x faster**)

## Key Observations

1. **Performance Portability**: Same source code runs on both CPU and GPU with different performance characteristics
2. **GPU Speedup**: 4-50x speedup depending on workload
3. **Memory-bound**: Vector/matrix addition is memory-bound, showing ~50x GPU speedup
4. **Compute-bound**: Matrix multiplication benefits from blocking optimization on GPU (8-12x improvement for large matrices)
5. **OpenMP Limitations**: The naive algorithm on OpenMP shows O(n³) scaling, while the optimized version with blocking shows better scalability

## Files

- `vectorAdd.cpp` - Vector addition (10M elements)
- `matrixAdd.cpp` - Matrix addition (2000×5000)
- `matMul.cpp` - Naive matrix multiplication
- `matMulOpt.cpp` - Optimized blocked matrix multiplication
- `CMakeLists.txt` - Build configuration
- `README.md` - This file