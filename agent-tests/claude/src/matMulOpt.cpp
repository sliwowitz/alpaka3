#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#ifndef MATRIX_M
#define MATRIX_M 1000
#endif

#ifndef MATRIX_K
#define MATRIX_K 1000
#endif

#ifndef MATRIX_N
#define MATRIX_N 1000
#endif

using namespace alpaka;

constexpr int BLOCK_SIZE = 16;

struct MatMulKernelOpt
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, auto a, auto b, auto c, int k, int n, int m) const
    {
        auto ext = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
        auto idx = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);

        auto blocksPerRow = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto blockRow = static_cast<std::size_t>(idx[0]) / static_cast<std::size_t>(blocksPerRow);
        auto blockCol = static_cast<std::size_t>(idx[0]) % static_cast<std::size_t>(blocksPerRow);

        auto row = blockRow * BLOCK_SIZE;
        auto col = blockCol * BLOCK_SIZE;

        auto n_val = static_cast<std::size_t>(n);
        auto k_val = static_cast<std::size_t>(k);
        auto m_val = static_cast<std::size_t>(m);

        if(row >= m_val || col >= n_val)
            return;

        float sum[BLOCK_SIZE][BLOCK_SIZE];
        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE; ++i)
            #pragma unroll
            for(int j = 0; j < BLOCK_SIZE; ++j)
                sum[i][j] = 0.0f;

        for(int t = 0; t < k; t += BLOCK_SIZE)
        {
            float aTile[BLOCK_SIZE][BLOCK_SIZE];
            float bTile[BLOCK_SIZE][BLOCK_SIZE];

            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE; ++i)
            {
                #pragma unroll
                for(int j = 0; j < BLOCK_SIZE; ++j)
                {
                    auto aRow = row + i;
                    auto aCol = t + j;
                    auto bRow = static_cast<std::size_t>(t) + i;
                    auto bCol = col + j;

                    aTile[i][j] = (aRow < m_val && aCol < k_val) 
                        ? a[aRow * k_val + aCol] : 0.0f;
                    bTile[i][j] = (bRow < k_val && bCol < n_val) 
                        ? b[bRow * n_val + bCol] : 0.0f;
                }
            }

            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE; ++i)
            {
                #pragma unroll
                for(int j = 0; j < BLOCK_SIZE; ++j)
                {
                    #pragma unroll
                    for(int l = 0; l < BLOCK_SIZE; ++l)
                    {
                        sum[i][j] += aTile[i][l] * bTile[l][j];
                    }
                }
            }
        }

        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE; ++i)
        {
            #pragma unroll
            for(int j = 0; j < BLOCK_SIZE; ++j)
            {
                auto cRow = row + i;
                auto cCol = col + j;
                if(cRow < m_val && cCol < n_val)
                {
                    c[cRow * n_val + cCol] = sum[i][j];
                }
            }
        }
    }
};

template<typename T>
void runMatMul(int m, int k, int n)
{
#if defined(ALPAKA_DISABLE_EXEC_GpuCuda)
    auto api = api::host;
    auto deviceKind = deviceKind::cpu;
    auto exec = exec::cpuOmpBlocks;
#else
    auto api = api::cuda;
    auto deviceKind = deviceKind::nvidiaGpu;
    auto exec = exec::gpuCuda;
#endif

    auto deviceSpec = onHost::DeviceSpec{api, deviceKind};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device devAcc = devSelector.makeDevice(0);
    std::cout << "Using device: " << onHost::getName(devAcc) << std::endl;

    onHost::Queue queue = devAcc.makeQueue();

    std::size_t totalElementsA = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
    std::size_t totalElementsB = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
    std::size_t totalElementsC = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    std::size_t totalElements = totalElementsA + totalElementsB + totalElementsC;

    std::vector<T> h_a(totalElementsA), h_b(totalElementsB), h_c(totalElementsC);
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    for(std::size_t i = 0; i < totalElementsA; ++i)
        h_a[i] = dist(gen);
    for(std::size_t i = 0; i < totalElementsB; ++i)
        h_b[i] = dist(gen);

    auto buf_a = onHost::alloc<T>(devAcc, totalElementsA);
    auto buf_b = onHost::alloc<T>(devAcc, totalElementsB);
    auto buf_c = onHost::alloc<T>(devAcc, totalElementsC);

    std::cout << "Copying to device..." << std::endl;
    auto startTotal = std::chrono::high_resolution_clock::now();

    memcpy(queue, buf_a, h_a);
    memcpy(queue, buf_b, h_b);
    onHost::wait(queue);

    int blocksX = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocksY = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int totalBlocks = blocksX * blocksY;
    auto extents = static_cast<std::size_t>(totalBlocks);
    auto dataBlocking = onHost::FrameSpec{extents, static_cast<std::size_t>(1)};

    std::cout << "Running optimized kernel..." << std::endl;
    auto startKernel = std::chrono::high_resolution_clock::now();
    queue.enqueue(exec, dataBlocking, KernelBundle{MatMulKernelOpt{}, buf_a, buf_b, buf_c, k, n, m});
    onHost::wait(queue);
    auto endKernel = std::chrono::high_resolution_clock::now();

    std::cout << "Copying back..." << std::endl;
    memcpy(queue, h_c, buf_c);
    onHost::wait(queue);

    auto endTotal = std::chrono::high_resolution_clock::now();

    T sum = 0;
    for(std::size_t i = 0; i < totalElementsC; ++i)
        sum += h_c[i];

    auto kernelTime = std::chrono::duration<double>(endKernel - startKernel).count();
    auto totalTime = std::chrono::duration<double>(endTotal - startTotal).count();

    std::cout << "Matrix sizes: " << m << "x" << k << " * " << k << "x" << n << " = " << m << "x" << n << std::endl;
    std::cout << "Block size: " << BLOCK_SIZE << "x" << BLOCK_SIZE << std::endl;
    std::cout << "Grid: " << blocksX << "x" << blocksY << " = " << totalBlocks << " blocks" << std::endl;
    std::cout << "Total elements: " << totalElements << std::endl;
    std::cout << "Sum of result: " << sum << std::endl;
    std::cout << "Kernel time: " << kernelTime * 1000.0 << " ms" << std::endl;
    std::cout << "Total time (including transfers): " << totalTime * 1000.0 << " ms" << std::endl;
    std::cout << "Matrix multiplication completed successfully!" << std::endl;
}

int main(int argc, char* argv[])
{
    int m = MATRIX_M;
    int k = MATRIX_K;
    int n = MATRIX_N;

    if(argc >= 4)
    {
        m = std::atoi(argv[1]);
        k = std::atoi(argv[2]);
        n = std::atoi(argv[3]);
    }

    std::cout << "Optimized Matrix multiplication (" << m << "x" << k << " * " << k << "x" << n << ")" << std::endl;

#if defined(ALPAKA_DISABLE_EXEC_GpuCuda)
    std::cout << "\n--- Using OpenMP backend ---" << std::endl;
#else
    std::cout << "\n--- Using CUDA backend ---" << std::endl;
#endif

    runMatMul<float>(m, k, n);

    return 0;
}