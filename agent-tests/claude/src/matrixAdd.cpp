/* Element-wise addition of two 2000x5000 random float matrices using alpaka. */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace alpaka;

static constexpr std::size_t M = 2000; // rows
static constexpr std::size_t N = 5000; // columns

class MatrixAddKernel
{
public:
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const A,
        alpaka::concepts::IMdSpan auto const B,
        alpaka::concepts::IMdSpan auto C,
        auto const& extent) const -> void
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{extent}))
        {
            C[idx] = A[idx] + B[idx];
        }
    }
};

auto run() -> int
{
#if defined(ALPAKA_DISABLE_EXEC_GpuCuda)
    auto deviceSpec = onHost::DeviceSpec{api::host, deviceKind::cpu};
    auto exec = exec::cpuOmpBlocks;
#else
    auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    auto exec = exec::gpuCuda;
#endif

    using Data = float;
    using IdxVec = Vec<std::size_t, 2u>;

    IdxVec const extent{M, N};

    std::string execName = "cpuOmpBlocks";
    try { execName = onHost::demangledName(exec); } catch(...) {}
    std::cout << "--- Backend: " << execName << " ("
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName()
              << ") ---\n";
    std::cout << "Matrix size: " << M << " x " << N << "\n";

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available\n";
        return EXIT_FAILURE;
    }
    
    onHost::Device devAcc = devSelector.makeDevice(0);
    onHost::Queue queue = devAcc.makeQueue();

    auto hA = onHost::allocHost<Data>(extent);
    auto hB = onHost::allocHostLike(hA);
    auto hC = onHost::allocHostLike(hA);

    std::mt19937 rng{42};
    std::uniform_real_distribution<Data> dist(0.0f, 1.0f);
    for(std::size_t r = 0; r < M; ++r)
    {
        for(std::size_t c = 0; c < N; ++c)
        {
            auto idx = Vec{r, c};
            hA[idx] = dist(rng);
            hB[idx] = dist(rng);
        }
    }

    auto dA = onHost::allocLike(devAcc, hA);
    auto dB = onHost::allocLike(devAcc, hB);
    auto dC = onHost::allocLike(devAcc, hC);

    onHost::memcpy(queue, dA, hA);
    onHost::memcpy(queue, dB, hB);
    onHost::memset(queue, dC, uint8_t{0});
    onHost::wait(queue);

    Vec<std::size_t, 2u> chunkSize{16u, 16u};
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, chunkSize), chunkSize};

    MatrixAddKernel kernel;
    auto const taskKernel = KernelBundle{kernel, dA, dB, dC, extent};

    auto const t0 = std::chrono::high_resolution_clock::now();
    queue.enqueue(exec, dataBlocking, taskKernel);
    onHost::wait(queue);
    auto const t1 = std::chrono::high_resolution_clock::now();

    double const kernelMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Kernel time: " << kernelMs << " ms\n";

    onHost::memcpy(queue, hC, dC);
    onHost::wait(queue);

    int errors = 0;
    for(std::size_t r = 0; r < M; ++r)
    {
        for(std::size_t c = 0; c < N; ++c)
        {
            auto idx = Vec{r, c};
            float expected = hA[idx] + hB[idx];
            if(std::abs(hC[idx] - expected) > 1e-6f)
            {
                if(errors < 10)
                    std::cerr << "  MISMATCH at [" << r << "," << c << "]: " << hC[idx] << " != " << expected << "\n";
                ++errors;
            }
        }
    }

    if(errors == 0)
        std::cout << "Validation PASSED (" << M << " x " << N << " = " << M * N << " elements)\n\n";
    else
        std::cout << "Validation FAILED with " << errors << " errors\n\n";

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto main() -> int
{
    std::cout << "Matrix addition (" << M << " x " << N << ")\n";

#if defined(ALPAKA_DISABLE_EXEC_GpuCuda)
    std::cout << "\n--- Using OpenMP backend ---\n" << std::endl;
#else
    std::cout << "\n--- Using CUDA backend ---\n" << std::endl;
#endif

    return run();
}