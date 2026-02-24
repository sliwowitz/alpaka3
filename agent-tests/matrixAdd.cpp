#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#ifndef MATRIX_ROWS
#define MATRIX_ROWS 2000
#endif

#ifndef MATRIX_COLS
#define MATRIX_COLS 5000
#endif

using namespace alpaka;

struct MatrixAddKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, auto a, auto b, auto c) const
    {
        auto ext = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
        auto idx = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
        for(auto i = static_cast<std::size_t>(idx[0]); i < static_cast<std::size_t>(ext[0]); i += static_cast<std::size_t>(ext[0]))
        {
            c[i] = a[i] + b[i];
        }
    }
};

template<typename T>
void runMatrixAdd()
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

    constexpr std::size_t rows = MATRIX_ROWS;
    constexpr std::size_t cols = MATRIX_COLS;
    constexpr std::size_t totalElements = rows * cols;

    std::vector<T> h_a(totalElements), h_b(totalElements), h_c(totalElements);
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    for(std::size_t i = 0; i < totalElements; ++i)
    {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    auto buf_a = onHost::alloc<T>(devAcc, totalElements);
    auto buf_b = onHost::alloc<T>(devAcc, totalElements);
    auto buf_c = onHost::alloc<T>(devAcc, totalElements);

    auto startTotal = std::chrono::high_resolution_clock::now();

    memcpy(queue, buf_a, h_a);
    memcpy(queue, buf_b, h_b);
    onHost::wait(queue);

    auto extents = onHost::getExtents(buf_a);
    auto dataBlocking = onHost::FrameSpec{extents, static_cast<std::size_t>(1)};

    auto startKernel = std::chrono::high_resolution_clock::now();
    queue.enqueue(exec, dataBlocking, KernelBundle{MatrixAddKernel{}, buf_a, buf_b, buf_c});
    onHost::wait(queue);
    auto endKernel = std::chrono::high_resolution_clock::now();

    memcpy(queue, h_c, buf_c);
    onHost::wait(queue);

    auto endTotal = std::chrono::high_resolution_clock::now();

    T sum = 0;
    for(std::size_t i = 0; i < totalElements; ++i)
    {
        sum += h_c[i];
    }

    auto kernelTime = std::chrono::duration<double>(endKernel - startKernel).count();
    auto totalTime = std::chrono::duration<double>(endTotal - startTotal).count();

    std::cout << "Matrix dimensions: " << MATRIX_ROWS << " x " << MATRIX_COLS << " (" << totalElements << " elements)" << std::endl;
    std::cout << "Sum of result matrix: " << sum << std::endl;
    std::cout << "Kernel time: " << kernelTime * 1000.0 << " ms" << std::endl;
    std::cout << "Total time (including transfers): " << totalTime * 1000.0 << " ms" << std::endl;
    std::cout << "Matrix addition completed successfully!" << std::endl;
}

int main()
{
    std::cout << "Matrix addition (" << MATRIX_ROWS << " x " << MATRIX_COLS << ")" << std::endl;

#if defined(ALPAKA_DISABLE_EXEC_GpuCuda)
    std::cout << "\n--- Using OpenMP backend ---" << std::endl;
#else
    std::cout << "\n--- Using CUDA backend ---" << std::endl;
#endif

    runMatrixAdd<float>();

    return 0;
}