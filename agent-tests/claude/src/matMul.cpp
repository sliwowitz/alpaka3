/* Matrix multiplication C(M,N) = A(M,K) * B(K,N) using alpaka.
 * Usage: matMul M K N
 */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace alpaka;

class MatMulKernel
{
public:
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const A,
        alpaka::concepts::IMdSpan auto const B,
        alpaka::concepts::IMdSpan auto C,
        auto const& cExtent,
        std::size_t K) const -> void
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{cExtent}))
        {
            float sum = 0.0f;
            for(std::size_t k = 0; k < K; ++k)
                sum += A[Vec{idx[0], k}] * B[Vec{k, idx[1]}];
            C[idx] = sum;
        }
    }
};

auto run(auto const deviceSpec, auto const exec, std::size_t M, std::size_t K, std::size_t N) -> int
{
    using Data = float;

    Vec<std::size_t, 2u> const extA{M, K};
    Vec<std::size_t, 2u> const extB{K, N};
    Vec<std::size_t, 2u> const extC{M, N};

    std::string execName = "cpuOmpBlocks";
    try { execName = onHost::demangledName(exec); } catch(...) {}
    std::cout << "--- Backend: " << execName << " ("
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << ") ---\n";
    std::cout << "A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")\n";
    std::cout << "Total elements: " << M * K + K * N + M * N << "\n";

    // Device and queue
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    onHost::Queue queue = devAcc.makeQueue();

    // Host buffers
    auto hA = onHost::allocHost<Data>(extA);
    auto hB = onHost::allocHost<Data>(extB);
    auto hC = onHost::allocHost<Data>(extC);

    // Fill with random data in [-1,1]
    std::mt19937 rng{42};
    std::uniform_real_distribution<Data> dist(-1.0f, 1.0f);
    for(std::size_t i = 0; i < M; ++i)
        for(std::size_t j = 0; j < K; ++j)
            hA[Vec{i, j}] = dist(rng);
    for(std::size_t i = 0; i < K; ++i)
        for(std::size_t j = 0; j < N; ++j)
            hB[Vec{i, j}] = dist(rng);

    // Device buffers
    auto dA = onHost::allocLike(devAcc, hA);
    auto dB = onHost::allocLike(devAcc, hB);
    auto dC = onHost::allocLike(devAcc, hC);

    // Copy host -> device
    onHost::memcpy(queue, dA, hA);
    onHost::memcpy(queue, dB, hB);
    onHost::memset(queue, dC, uint8_t{0});
    onHost::wait(queue);

    // 2D work division over C's extent
    Vec<std::size_t, 2u> chunkSize{16u, 16u};
    auto dataBlocking = onHost::FrameSpec{divCeil(extC, chunkSize), chunkSize};

    // Launch kernel
    MatMulKernel kernel;
    auto const taskKernel = KernelBundle{kernel, dA, dB, dC, extC, K};

    auto const t0 = std::chrono::high_resolution_clock::now();
    queue.enqueue(exec, dataBlocking, taskKernel);
    onHost::wait(queue);
    auto const t1 = std::chrono::high_resolution_clock::now();

    double const kernelMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double const gflops = 2.0 * M * N * K / (kernelMs * 1e6);
    std::cout << "Kernel time: " << kernelMs << " ms  (" << gflops << " GFLOP/s)\n";

    // Copy result back
    onHost::memcpy(queue, hC, dC);
    onHost::wait(queue);

    // Validate a random sample of 256 elements
    int errors = 0;
    std::mt19937 sampleRng{123};
    constexpr int NUM_CHECKS = 256;
    for(int s = 0; s < NUM_CHECKS; ++s)
    {
        std::size_t i = sampleRng() % M;
        std::size_t j = sampleRng() % N;
        // Compute reference using double for accuracy
        double ref = 0.0;
        for(std::size_t k = 0; k < K; ++k)
            ref += static_cast<double>(hA[Vec{i, k}]) * static_cast<double>(hB[Vec{k, j}]);
        float expected = static_cast<float>(ref);
        float got = hC[Vec{i, j}];
        float relErr = std::abs(got - expected) / (std::abs(expected) + 1e-7f);
        if(relErr > 1e-3f)
        {
            if(errors < 10)
                std::cerr << "  MISMATCH at [" << i << "," << j << "]: " << got << " vs " << expected
                          << " (relErr=" << relErr << ")\n";
            ++errors;
        }
    }

    if(errors == 0)
        std::cout << "Validation PASSED (checked " << NUM_CHECKS << " random elements)\n\n";
    else
        std::cout << "Validation FAILED with " << errors << "/" << NUM_CHECKS << " errors\n\n";

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto main(int argc, char* argv[]) -> int
{
    if(argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " M K N\n"
                  << "  Computes C(M,N) = A(M,K) * B(K,N)\n";
        return EXIT_FAILURE;
    }

    std::size_t M = std::stoul(argv[1]);
    std::size_t K = std::stoul(argv[2]);
    std::size_t N = std::stoul(argv[3]);

#if defined(ALPAKA_DISABLE_EXEC_GpuCuda)
    std::cout << "\n--- Using OpenMP backend ---\n" << std::endl;
    auto deviceSpec = onHost::DeviceSpec{api::host, deviceKind::cpu};
    auto exec = exec::cpuOmpBlocks;
#else
    std::cout << "\n--- Using CUDA backend ---\n" << std::endl;
    auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    auto exec = exec::gpuCuda;
#endif

    return run(deviceSpec, exec, M, K, N);
}
