# Agent Tests — Alpaka Benchmark Programs

Conversation date: 2026-02-24

## Overview

Four programs were written to exercise alpaka's portability layer across OpenMP (CPU)
and CUDA (GPU) backends on an NVIDIA A40 GPU system.

Build environment:
- GCC 14.2 (OpenMP builds), NVHPC 26.1 / nvcc 12.9 (CUDA builds)
- CMake 3.28, alpaka 3.0.0
- NVIDIA A40 (compute capability 8.6, driver 12.4)

Build commands:
```bash
# OpenMP
cmake -S agent-tests -B agent-tests/build-omp \
  -Dalpaka_USE_SOURCE_TREE=ON -Dalpaka_DEP_OMP=ON -Dalpaka_DEP_CUDA=OFF \
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
cmake --build agent-tests/build-omp -j$(nproc)

# CUDA
cmake -S agent-tests -B agent-tests/build-cuda \
  -Dalpaka_USE_SOURCE_TREE=ON -Dalpaka_DEP_CUDA=ON -Dalpaka_DEP_OMP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build agent-tests/build-cuda -j$(nproc)
```

---

## 1. vectorAdd10M — Vector Addition (10M elements)

**File:** `src/vectorAdd10M.cpp`

Adds two vectors of 10 million random floats using alpaka's `SimdAlgo` pattern
with 1D `threadsInGrid` work distribution.

| Backend | Kernel Time |
|---|---|
| OpenMP (CpuOmpBlocks) | 15.6 ms |
| CPU Serial | 15.7 ms |
| **CUDA (GpuCuda)** | **0.38 ms** |

---

## 2. matrixAdd — Matrix Addition (2000 x 5000)

**File:** `src/matrixAdd.cpp`

Element-wise addition of two 2000x5000 random float matrices using 2D
`makeIdxMap` with `threadsInGrid` and 16x16 thread blocks.

| Backend | Kernel Time |
|---|---|
| OpenMP (CpuOmpBlocks) | 6.8 ms |
| CPU Serial | 23.7 ms |
| **CUDA (GpuCuda)** | **0.38 ms** |

---

## 3. matMul — Naive Matrix Multiplication

**File:** `src/matMul.cpp`
**Usage:** `matMul M K N` — computes C(M,N) = A(M,K) * B(K,N)

Naive matmul where each thread computes one element of C via a full dot product
over the K dimension. Uses 2D `makeIdxMap` with `threadsInGrid`.

Selected results (CUDA GpuCuda backend):

| Shape | GFLOP/s |
|---|---|
| 500x500 * 500x500 | 827 |
| 1000x1000 * 1000x1000 | 1,651 |
| 2000x1000 * 1000x2000 | 2,155 |

---

## 4. matMulOpt — Optimized Tiled Matrix Multiplication

**File:** `src/matMulOpt.cpp`
**Usage:** `matMulOpt M K N`

Optimized matmul with three key techniques:

1. **Shared-memory tiling** — A (64x16) and B (16x64) tiles loaded into shared
   memory, reducing global memory traffic by ~16x.
2. **Register micro-tiles** — Each thread computes a 4x4 sub-block of C,
   reusing A/B values via rank-1 (outer product) updates.
3. **Shared-memory accumulator** — Persists partial sums across K-tiles
   (same pattern as the alpaka nBody example).

Tile parameters: BM=64, BN=64, BK=16, TM=4, TN=4, thread block 16x16=256.

### CUDA — Naive vs Optimized

| Shape (MxKxN) | Naive (GFLOP/s) | Optimized (GFLOP/s) | Speedup |
|---|---|---|---|
| 500x500x500 | 934 | 2,504 | 2.7x |
| 1000x100x1000 | 1,036 | 3,778 | 3.6x |
| 1000x1000x1000 | 1,700 | 5,045 | 3.0x |
| 2000x500x1000 | 1,782 | 5,800 | 3.3x |
| 1000x1000x2000 | 2,095 | 5,959 | 2.8x |
| 2000x1000x2000 | 2,177 | **6,449** | 3.0x |
| 1000x2000x2000 | 2,101 | 5,615 | 2.7x |

Peak: **6.4 TFLOP/s** (17% of A40 FP32 theoretical peak of 37.4 TFLOP/s).

### OpenMP — Naive vs Optimized

| Shape (MxKxN) | Naive (GFLOP/s) | Optimized (GFLOP/s) | Speedup |
|---|---|---|---|
| 500x500x500 | 13.3 | 95.5 | 7.2x |
| 1000x1000x1000 | 21.0 | 136.9 | 6.5x |
| 2000x1000x2000 | 15.6 | 103.9 | 6.7x |
| 1000x2000x2000 | 6.1 | **107.9** | 17.8x |

Cache-friendly tiling gives up to **18x speedup** on CPU.

---

## Files

```
agent-tests/
  CMakeLists.txt
  SUMMARY.md
  src/
    vectorAdd10M.cpp
    matrixAdd.cpp
    matMul.cpp
    matMulOpt.cpp
```
