#include <alpaka/alpaka.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

using namespace alpaka;

// Define kernel for vector addition
struct VectorAddKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::IMdSpan auto const A,
        alpaka::concepts::IMdSpan auto const B,
        alpaka::concepts::IMdSpan auto C,
        auto const& numElements) const -> void
    {
        using namespace alpaka;
        static_assert(ALPAKA_TYPEOF(numElements)::dim() == 1, "The VectorAddKernel expects 1-dimensional indices!");

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

auto vectorAddExample(auto const deviceSpec, auto const exec, std::vector<float> const& h_a, std::vector<float> const& h_b) -> int
{
    const std::size_t vectorSize = h_a.size(); // Use the size from the input vector
    
    std::cout << "Vector addition with " << vectorSize << " elements" << std::endl;
    
    // Create device selector and get device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    
    std::cout << "Using device: " << devSelector.getDeviceProperties(0).getName() << std::endl;
    
    // Create queue
    onHost::Queue queue = devAcc.makeQueue();
    
    // Allocate host memory
    auto bufHostA = onHost::allocHost<float>(Vec<std::size_t, 1u>(vectorSize));
    auto bufHostB = onHost::allocHostLike(bufHostA);
    auto bufHostResult = onHost::allocHostLike(bufHostA);
    
    // Copy data to host buffers
    for (std::size_t i = 0; i < vectorSize; ++i) {
        bufHostA[i] = h_a[i];
        bufHostB[i] = h_b[i];
    }
    
    // Allocate device memory
    auto bufAccA = onHost::allocLike(devAcc, bufHostA);
    auto bufAccB = onHost::allocLike(devAcc, bufHostB);
    auto bufAccResult = onHost::allocLike(devAcc, bufHostResult);
    
    // Set up work distribution
    Vec<size_t, 1u> chunkSize = 256u;
    uint32_t elementsPerWorker = getNumElemPerThread<float>(queue);
    auto dataBlocking = onHost::FrameSpec{divCeil(Vec<std::size_t, 1u>(vectorSize), chunkSize * elementsPerWorker), chunkSize};
    
    // Instantiate the kernel function object
    VectorAddKernel kernel;
    auto const taskKernel = KernelBundle{kernel, bufAccA, bufAccB, bufAccResult, Vec<std::size_t, 1u>(vectorSize)};
    
    // Copy Host -> Device
    onHost::memcpy(queue, bufAccA, bufHostA);
    onHost::memcpy(queue, bufAccB, bufHostB);
    
    // Execute kernel with timing
    onHost::wait(queue);
    auto kernelStart = std::chrono::high_resolution_clock::now();
    queue.enqueue(exec, dataBlocking, taskKernel);
    onHost::wait(queue);
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    
    // Copy result back to host
    auto copyStart = std::chrono::high_resolution_clock::now();
    onHost::memcpy(queue, bufHostResult, bufAccResult);
    onHost::wait(queue);
    auto copyEnd = std::chrono::high_resolution_clock::now();
    
    // Calculate timing
    auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelEnd - kernelStart).count() / 1000.0;
    auto copyTime = std::chrono::duration_cast<std::chrono::microseconds>(copyEnd - copyStart).count() / 1000.0;
    auto totalTime = kernelTime + copyTime;
    
    std::cout << "Kernel execution time: " << kernelTime << " ms" << std::endl;
    std::cout << "Data copy time: " << copyTime << " ms" << std::endl;
    std::cout << "Total computation time: " << totalTime << " ms" << std::endl;
    
    // Verify results
    bool success = true;
    int falseResults = 0;
    const int MAX_PRINT_FALSE_RESULTS = 5;
    
    for (std::size_t i = 0; i < vectorSize; ++i) {
        float expected = h_a[i] + h_b[i];
        float actual = bufHostResult[i];
        if (std::abs(actual - expected) > 1e-5f) {
            if (falseResults < MAX_PRINT_FALSE_RESULTS) {
                std::cerr << "Mismatch at index " << i << ": expected " << expected 
                          << " but got " << actual << std::endl;
            }
            falseResults++;
            success = false;
            if (falseResults >= MAX_PRINT_FALSE_RESULTS) break;
        }
    }
    
    if (success) {
        std::cout << "Vector addition completed successfully!" << std::endl;
        std::cout << "First 5 results: ";
        for (std::size_t i = 0; i < 5; ++i) {
            std::cout << bufHostResult[i] << " ";
        }
        std::cout << "..." << std::endl;
    } else {
        std::cerr << "Vector addition failed with " << falseResults << " mismatches!" << std::endl;
        return 1;
    }
    
    return 0;
}

int main()
{
    const std::size_t vectorSize = 10'000'000; // 10 million elements
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Generate random vectors on host
    std::vector<float> h_a(vectorSize);
    std::vector<float> h_b(vectorSize);
    
    for (std::size_t i = 0; i < vectorSize; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }
    
    // Use alpaka's automatic backend selection based on build configuration
    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend) {
            return vectorAddExample(
                backend[alpaka::object::deviceSpec],
                backend[alpaka::object::exec],
                h_a, h_b);
        },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}