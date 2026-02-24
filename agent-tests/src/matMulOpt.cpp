/* Optimized tiled matrix multiplication C(M,N) = A(M,K) * B(K,N).
 *
 * Optimizations over the naive version:
 *   1. Shared-memory tiling: A and B tiles (BM×BK, BK×BN) are loaded into
 *      fast on-chip memory, reducing global memory traffic by ~BM/4 ×.
 *   2. Register-level micro-tiles: each thread computes a TM×TN sub-block of C,
 *      reusing loaded A/B values across the outer product.
 *   3. Shared-memory accumulator: persists partial sums across K-tiles using
 *      the same pattern as the alpaka nBody example.
 *
 * Usage: matMulOpt M K N
 */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace alpaka;

// --- Tile parameters ---
// Block output tile: BM × BN.  K-tile width: BK.
// Each thread computes a TM × TN micro-tile of C.
// Thread block: (BM/TM) × (BN/TN) = 16 × 16 = 256 threads.
static constexpr uint32_t BM = 64;
static constexpr uint32_t BN = 64;
static constexpr uint32_t BK = 16;
static constexpr uint32_t TM = 4;
static constexpr uint32_t TN = 4;

class MatMulTiledKernel
{
public:
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const A,
        alpaka::concepts::IMdSpan auto const B,
        alpaka::concepts::IMdSpan auto C,
        std::size_t M,
        std::size_t K,
        std::size_t N) const -> void
    {
        // Iterate over output tiles of C
        for(concepts::Dim<2u> auto blockStart : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{
                    Vec<std::size_t, 2u>{0, 0},
                    Vec<std::size_t, 2u>{M, N},
                    Vec<std::size_t, 2u>{BM, BN}}))
        {
            // Shared memory tiles
            auto tileA = onAcc::declareSharedMdArray<float, uniqueId()>(acc, CVec<uint32_t, BM, BK>{});
            auto tileB = onAcc::declareSharedMdArray<float, uniqueId()>(acc, CVec<uint32_t, BK, BN>{});
            // Shared accumulator — persists across K-tiles (same pattern as nBody accelerations)
            auto accumC = onAcc::declareSharedMdArray<float, uniqueId()>(acc, CVec<uint32_t, BM, BN>{});

            // Zero the accumulator (4096 elems / 256 threads = 16 per thread)
            onAcc::syncBlockThreads(acc);
            for(auto idx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{CVec<uint32_t, BM, BN>{}}))
            {
                accumC[idx] = 0.0f;
            }

            // --- K-tile loop (plain loop, like nBody's otherBlockStartIdx loop) ---
            for(std::size_t kt = 0; kt < K; kt += BK)
            {
                // Load tileA  [BM × BK] from A  (1024 elems / 256 threads = 4 each)
                onAcc::syncBlockThreads(acc);
                for(auto idx : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<uint32_t, BM, BK>{}}))
                {
                    auto r = blockStart[0] + static_cast<std::size_t>(idx[0]);
                    auto c = kt + static_cast<std::size_t>(idx[1]);
                    tileA[idx] = (r < M && c < K) ? A[Vec{r, c}] : 0.0f;
                }

                // Load tileB  [BK × BN] from B  (1024 elems / 256 threads = 4 each)
                for(auto idx : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<uint32_t, BK, BN>{}}))
                {
                    auto r = kt + static_cast<std::size_t>(idx[0]);
                    auto c = blockStart[1] + static_cast<std::size_t>(idx[1]);
                    tileB[idx] = (r < K && c < N) ? B[Vec{r, c}] : 0.0f;
                }

                onAcc::syncBlockThreads(acc);

                // Compute micro-tiles: each thread owns a TM×TN sub-block
                for(auto tIdx : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<uint32_t, BM / TM, BN / TN>{}}))
                {
                    // Register-level partial sums for this K-tile
                    float rc[TM][TN] = {};

                    for(uint32_t k = 0; k < BK; ++k)
                    {
                        // Preload A column and B row into registers
                        float ra[TM], rb[TN];
                        for(uint32_t i = 0; i < TM; ++i)
                            ra[i] = tileA[Vec{tIdx[0] * TM + i, k}];
                        for(uint32_t j = 0; j < TN; ++j)
                            rb[j] = tileB[Vec{k, tIdx[1] * TN + j}];

                        // Rank-1 update (outer product)
                        for(uint32_t i = 0; i < TM; ++i)
                            for(uint32_t j = 0; j < TN; ++j)
                                rc[i][j] += ra[i] * rb[j];
                    }

                    // Flush partial sums to shared accumulator
                    for(uint32_t i = 0; i < TM; ++i)
                        for(uint32_t j = 0; j < TN; ++j)
                            accumC[Vec{tIdx[0] * TM + i, tIdx[1] * TN + j}] += rc[i][j];
                }
            } // end K-tile loop

            // Write final results to global memory
            onAcc::syncBlockThreads(acc);
            for(auto idx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{CVec<uint32_t, BM, BN>{}}))
            {
                auto r = blockStart[0] + static_cast<std::size_t>(idx[0]);
                auto c = blockStart[1] + static_cast<std::size_t>(idx[1]);
                if(r < M && c < N)
                    C[Vec{r, c}] = accumC[idx];
            }
        }
    }
};

auto run(auto const deviceSpec, auto const exec, std::size_t M, std::size_t K, std::size_t N) -> int
{
    using Data = float;

    Vec<std::size_t, 2u> const extA{M, K};
    Vec<std::size_t, 2u> const extB{K, N};
    Vec<std::size_t, 2u> const extC{M, N};

    std::cout << "--- Backend: " << onHost::demangledName(exec) << " ("
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << ") ---\n";
    std::cout << "A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")\n";

    // Device and queue
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    onHost::Queue queue = devAcc.makeQueue();

    // Host buffers
    auto hA = onHost::allocHost<Data>(extA);
    auto hB = onHost::allocHost<Data>(extB);
    auto hC = onHost::allocHost<Data>(extC);

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

    onHost::memcpy(queue, dA, hA);
    onHost::memcpy(queue, dB, hB);
    onHost::memset(queue, dC, uint8_t{0});
    onHost::wait(queue);

    // FrameSpec: 16×16 thread block, ceil(M/BM) × ceil(N/BN) blocks
    Vec<std::size_t, 2u> threadBlock{BM / TM, BN / TN}; // 16 × 16
    Vec<std::size_t, 2u> numBlocks{divCeil(M, std::size_t{BM}), divCeil(N, std::size_t{BN})};
    auto dataBlocking = onHost::FrameSpec{numBlocks, threadBlock};

    MatMulTiledKernel kernel;
    auto const taskKernel = KernelBundle{kernel, dA, dB, dC, M, K, N};

    // Warmup
    queue.enqueue(exec, dataBlocking, taskKernel);
    onHost::wait(queue);

    // Timed run
    onHost::memset(queue, dC, uint8_t{0});
    onHost::wait(queue);
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

    // Validate sample
    int errors = 0;
    std::mt19937 sampleRng{123};
    constexpr int NUM_CHECKS = 256;
    for(int s = 0; s < NUM_CHECKS; ++s)
    {
        std::size_t i = sampleRng() % M;
        std::size_t j = sampleRng() % N;
        double ref = 0.0;
        for(std::size_t k = 0; k < K; ++k)
            ref += static_cast<double>(hA[Vec{i, k}]) * static_cast<double>(hB[Vec{k, j}]);
        float expected = static_cast<float>(ref);
        float got = hC[Vec{i, j}];
        float relErr = std::abs(got - expected) / (std::abs(expected) + 1e-7f);
        if(relErr > 1e-3f)
        {
            if(errors < 10)
                std::cerr << "  MISMATCH [" << i << "," << j << "]: " << got << " vs " << expected
                          << " (relErr=" << relErr << ")\n";
            ++errors;
        }
    }

    if(errors == 0)
        std::cout << "Validation PASSED (" << NUM_CHECKS << " samples)\n\n";
    else
        std::cout << "Validation FAILED " << errors << "/" << NUM_CHECKS << "\n\n";

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto main(int argc, char* argv[]) -> int
{
    if(argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " M K N\n"
                  << "  C(M,N) = A(M,K) * B(K,N)   [tiled: BM=" << BM << " BN=" << BN << " BK=" << BK
                  << " TM=" << TM << " TN=" << TN << "]\n";
        return EXIT_FAILURE;
    }

    std::size_t M = std::stoul(argv[1]);
    std::size_t K = std::stoul(argv[2]);
    std::size_t N = std::stoul(argv[3]);

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return run(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec], M, K, N); },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}
