#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifndef NUM_ELEMENTS
#define NUM_ELEMENTS 10000000
#endif

using namespace alpaka;

struct VecAddKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, auto a, auto b, auto c) const
    {
        auto ext = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
        auto idx = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
        for(decltype(ext[0]) i = idx[0]; i < ext[0]; i += ext[0])
        {
            c[i] = a[i] + b[i];
        }
    }
};

template<typename T>
void runVectorAdd()
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

    constexpr std::size_t N = NUM_ELEMENTS;

    std::vector<T> h_a(N), h_b(N), h_c(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    for(std::size_t i = 0; i < N; ++i)
    {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    auto buf_a = onHost::alloc<T>(devAcc, N);
    auto buf_b = onHost::alloc<T>(devAcc, N);
    auto buf_c = onHost::alloc<T>(devAcc, N);

    memcpy(queue, buf_a, h_a);
    memcpy(queue, buf_b, h_b);
    onHost::wait(queue);

    auto extents = onHost::getExtents(buf_a);
    auto dataBlocking = onHost::FrameSpec{extents, static_cast<std::size_t>(1)};

    queue.enqueue(exec, dataBlocking, KernelBundle{VecAddKernel{}, buf_a, buf_b, buf_c});
    onHost::wait(queue);

    memcpy(queue, h_c, buf_c);
    onHost::wait(queue);

    T sum = 0;
    for(std::size_t i = 0; i < N; ++i)
    {
        sum += h_c[i];
    }
    std::cout << "Sum of result vector: " << sum << std::endl;
    std::cout << "Vector addition completed successfully!" << std::endl;
}

int main()
{
    std::cout << "Vector addition with " << NUM_ELEMENTS << " elements" << std::endl;

#if defined(ALPAKA_DISABLE_EXEC_GpuCuda)
    std::cout << "\n--- Using OpenMP backend ---" << std::endl;
#else
    std::cout << "\n--- Using CUDA backend ---" << std::endl;
#endif

    runVectorAdd<float>();

    return 0;
}