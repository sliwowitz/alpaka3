/* Vector addition of 10 million random floats using alpaka. */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace alpaka;

static constexpr std::size_t NUM_ELEMENTS = 10'000'000;

class VectorAddKernel
{
public:
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const A,
        alpaka::concepts::IMdSpan auto const B,
        alpaka::concepts::IMdSpan auto C,
        auto const& numElements) const -> void
    {
        static_assert(ALPAKA_TYPEOF(numElements)::dim() == 1);

        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            numElements,
            [&](auto const&, auto&& simdA, auto&& simdB, auto&& simdC) constexpr
            { simdC = simdA.load() + simdB.load(); },
            A,
            B,
            C);
    }
};

auto run(auto const deviceSpec, auto const exec) -> int
{
    using Data = float;
    using IdxVec = Vec<std::size_t, 1u>;

    IdxVec const extent(NUM_ELEMENTS);

    std::cout << "--- Backend: " << onHost::demangledName(exec) << " ("
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName()
              << ") ---\n";

    // Device and queue
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    onHost::Queue queue = devAcc.makeQueue();

    // Host buffers
    auto hA = onHost::allocHost<Data>(extent);
    auto hB = onHost::allocHostLike(hA);
    auto hC = onHost::allocHostLike(hA);

    // Fill with random data
    std::mt19937 rng{42};
    std::uniform_real_distribution<Data> dist(0.0f, 1.0f);
    for(std::size_t i = 0; i < NUM_ELEMENTS; ++i)
    {
        hA[i] = dist(rng);
        hB[i] = dist(rng);
    }

    // Device buffers
    auto dA = onHost::allocLike(devAcc, hA);
    auto dB = onHost::allocLike(devAcc, hB);
    auto dC = onHost::allocLike(devAcc, hC);

    // Copy host -> device
    onHost::memcpy(queue, dA, hA);
    onHost::memcpy(queue, dB, hB);
    onHost::memset(queue, dC, uint8_t{0});
    onHost::wait(queue);

    // Work division
    Vec<std::size_t, 1u> chunkSize = 256u;
    uint32_t elementsPerWorker = getNumElemPerThread<Data>(queue);
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, chunkSize * elementsPerWorker), chunkSize};

    // Launch kernel
    VectorAddKernel kernel;
    auto const taskKernel = KernelBundle{kernel, dA, dB, dC, extent};

    auto const t0 = std::chrono::high_resolution_clock::now();
    queue.enqueue(exec, dataBlocking, taskKernel);
    onHost::wait(queue);
    auto const t1 = std::chrono::high_resolution_clock::now();

    double const kernelMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Kernel time: " << kernelMs << " ms\n";

    // Copy result back
    onHost::memcpy(queue, hC, dC);
    onHost::wait(queue);

    // Validate
    int errors = 0;
    for(std::size_t i = 0; i < NUM_ELEMENTS; ++i)
    {
        float expected = hA[i] + hB[i];
        if(std::abs(hC[i] - expected) > 1e-6f)
        {
            if(errors < 10)
                std::cerr << "  MISMATCH at [" << i << "]: " << hC[i] << " != " << expected << "\n";
            ++errors;
        }
    }

    if(errors == 0)
        std::cout << "Validation PASSED (" << NUM_ELEMENTS << " elements)\n\n";
    else
        std::cout << "Validation FAILED with " << errors << " errors\n\n";

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto main() -> int
{
    return onHost::executeForEachIfHasDevice(
        [](auto const& backend) { return run(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]); },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}
