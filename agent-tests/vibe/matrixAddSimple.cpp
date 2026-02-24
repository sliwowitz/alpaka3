#include <alpaka/alpaka.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

using namespace alpaka;

// Define kernel for matrix addition (treat matrix as 1D array)
struct MatrixAddKernel
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
        static_assert(ALPAKA_TYPEOF(numElements)::dim() == 1, "The MatrixAddKernel expects 1-dimensional indices!");

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

auto matrixAddExample(auto const deviceSpec, auto const exec, 
                      std::vector<float> const& h_a, std::vector<float> const& h_b,
                      size_t rows, size_t cols) -> int
{
    std::cout << "Matrix addition: " << rows << "x" << cols 
              << " matrix (" << (rows * cols) << " elements)" << std::endl;
    
    // Create device selector and get device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    
    std::cout << "Using device: " << devSelector.getDeviceProperties(0).getName() << std::endl;
    
    // Create queue
    onHost::Queue queue = devAcc.makeQueue();
    
    // Define total size
    size_t totalElements = rows * cols;
    
    // Allocate host memory for matrices (1D buffers)
    auto bufHostA = onHost::allocHost<float>(Vec<std::size_t, 1u>(totalElements));
    auto bufHostB = onHost::allocHostLike(bufHostA);
    auto bufHostResult = onHost::allocHostLike(bufHostA);
    
    // Copy data to host buffers
    for (size_t i = 0; i < totalElements; ++i) {
        bufHostA[i] = h_a[i];
        bufHostB[i] = h_b[i];
    }
    
    // Allocate device memory
    auto bufAccA = onHost::allocLike(devAcc, bufHostA);
    auto bufAccB = onHost::allocLike(devAcc, bufHostB);
    auto bufAccResult = onHost::allocLike(devAcc, bufHostResult);
    
    // Set up work distribution - one thread per row
    Vec<size_t, 1u> chunkSize(1u); // Process one row per thread
    auto dataBlocking = onHost::FrameSpec{
        Vec<size_t, 1u>(rows), // Global size: number of rows
        chunkSize
    };
    
    // Instantiate the kernel function object
    MatrixAddKernel kernel;
    auto const taskKernel = KernelBundle{kernel, bufAccA, bufAccB, bufAccResult, Vec<std::size_t, 1u>(totalElements)};
    
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
    
    // Verify results (check a few sample points)
    bool success = true;
    int falseResults = 0;
    const int MAX_PRINT_FALSE_RESULTS = 5;
    
    // Check some sample points
    std::vector<std::pair<size_t, size_t>> samplePoints = {
        {0, 0}, {rows-1, cols-1}, {rows/2, cols/2}, 
        {rows/4, cols/4}, {3*rows/4, 3*cols/4}
    };
    
    for (auto const& [i, j] : samplePoints) {
        size_t linearIndex = i * cols + j;
        float expected = h_a[linearIndex] + h_b[linearIndex];
        float actual = bufHostResult[linearIndex];
        if (std::abs(actual - expected) > 1e-5f) {
            if (falseResults < MAX_PRINT_FALSE_RESULTS) {
                std::cerr << "Mismatch at (" << i << "," << j << ") expected " 
                          << expected << " but got " << actual << std::endl;
            }
            falseResults++;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "Matrix addition completed successfully!" << std::endl;
        std::cout << "Sample results:" << std::endl;
        for (auto const& [i, j] : samplePoints) {
            size_t linearIndex = i * cols + j;
            std::cout << "  C[" << i << "][" << j << "] = " 
                      << bufHostResult[linearIndex] << std::endl;
        }
    } else {
        std::cerr << "Matrix addition failed with " << falseResults << " mismatches!" << std::endl;
        return 1;
    }
    
    return 0;
}

int main()
{
    const size_t M = 2000;  // rows
    const size_t N = 5000;  // columns
    const size_t totalElements = M * N;
    
    std::cout << "Matrix Addition: " << M << "x" << N << " = " << totalElements 
              << " elements (" << (totalElements / 1'000'000) << "M)" << std::endl;
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Generate random matrices on host
    std::vector<float> h_a(totalElements);
    std::vector<float> h_b(totalElements);
    
    for (size_t i = 0; i < totalElements; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }
    
    // Use alpaka's automatic backend selection based on build configuration
    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend) {
            return matrixAddExample(
                backend[alpaka::object::deviceSpec],
                backend[alpaka::object::exec],
                h_a, h_b, M, N);
        },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}