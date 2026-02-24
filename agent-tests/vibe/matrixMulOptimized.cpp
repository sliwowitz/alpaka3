#include <alpaka/alpaka.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace alpaka;

// Optimized matrix multiplication kernel with memory access improvements
struct MatrixMulKernelOptimized
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::IMdSpan auto const A,
        alpaka::concepts::IMdSpan auto const B,
        alpaka::concepts::IMdSpan auto C,
        size_t a_rows, size_t a_cols, size_t b_cols) const -> void
    {
        using namespace alpaka;
        // static_assert(ALPAKA_TYPEOF(A)::dim() == 2, "Expected 2D buffers!");

        // Get 2D thread indices for better work distribution
        auto const globalIdx = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        size_t row = globalIdx[0];
        size_t col = globalIdx[1];

        // Check bounds
        if (row < a_rows && col < b_cols) {
            float sum = 0.0f;

            // Use 1D indexing for 1D buffers
            size_t a_start_idx = row * a_cols;
            size_t c_start_idx = row * b_cols;
            
            for (size_t k = 0; k < a_cols; ++k) {
                sum += A[a_start_idx + k] * B[k * b_cols + col];
            }

            C[c_start_idx + col] = sum;
        }
    }
};

auto matrixMulExample(auto const deviceSpec, auto const exec,
                     std::vector<float> const& h_a, std::vector<float> const& h_b,
                     std::vector<float>& h_c,
                     size_t a_rows, size_t a_cols, size_t b_cols) -> int
{
    std::cout << "Optimized matrix multiplication: " << a_rows << "x" << a_cols
              << " * " << a_cols << "x" << b_cols
              << " = " << a_rows << "x" << b_cols
              << " (" << (a_rows * b_cols) << " elements)" << std::endl;

    // Create device selector and get device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    std::cout << "Using device: " << devSelector.getDeviceProperties(0).getName() << std::endl;

    // Create queue
    onHost::Queue queue = devAcc.makeQueue();

    // Allocate host memory for matrices (1D buffers - same as working version)
    auto bufHostA = onHost::allocHost<float>(Vec{a_rows * a_cols});
    auto bufHostB = onHost::allocHost<float>(Vec{a_cols * b_cols});
    auto bufHostC = onHost::allocHost<float>(Vec{a_rows * b_cols});

    // Copy data to host buffers
    for (size_t i = 0; i < a_rows * a_cols; ++i) {
        bufHostA[i] = h_a[i];
    }
    for (size_t i = 0; i < a_cols * b_cols; ++i) {
        bufHostB[i] = h_b[i];
    }

    // Allocate device memory
    auto bufAccA = onHost::allocLike(devAcc, bufHostA);
    auto bufAccB = onHost::allocLike(devAcc, bufHostB);
    auto bufAccC = onHost::allocLike(devAcc, bufHostC);

    // Copy Host -> Device
    onHost::memcpy(queue, bufAccA, bufHostA);
    onHost::memcpy(queue, bufAccB, bufHostB);

    // Use 1D work distribution (same as working version)
    Vec<size_t, 1u> chunkSize(1u);
    auto dataBlocking = onHost::FrameSpec{
        Vec<size_t, 1u>(a_rows * b_cols), // 1D global size
        chunkSize
    };

    // Use the optimized kernel
    MatrixMulKernelOptimized kernel;
    auto const taskKernel = KernelBundle{kernel, bufAccA, bufAccB, bufAccC, a_rows, a_cols, b_cols};

    // Execute kernel with timing
    onHost::wait(queue);
    auto kernelStart = std::chrono::high_resolution_clock::now();
    queue.enqueue(exec, dataBlocking, taskKernel);
    onHost::wait(queue);
    auto kernelEnd = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    auto copyStart = std::chrono::high_resolution_clock::now();
    onHost::memcpy(queue, bufHostC, bufAccC);
    onHost::wait(queue);
    auto copyEnd = std::chrono::high_resolution_clock::now();

    // Copy result to output vector
    for (size_t i = 0; i < a_rows * b_cols; ++i) {
        h_c[i] = bufHostC[i];
    }

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
    std::vector<std::tuple<size_t, size_t, size_t>> samplePoints = {
        {0, 0, 0}, {a_rows-1, a_cols-1, b_cols-1}, {a_rows/2, a_cols/2, b_cols/2},
        {a_rows/4, a_cols/4, b_cols/4}, {3*a_rows/4, 3*a_cols/4, 3*b_cols/4}
    };

    for (auto const& [i, k, j] : samplePoints) {
        // Compute expected value: C[i][j] = sum(A[i][k] * B[k][j])
        float expected = 0.0f;
        for (size_t m = 0; m < a_cols; ++m) {
            expected += h_a[i * a_cols + m] * h_b[m * b_cols + j];
        }
        size_t c_index = i * b_cols + j;
        float actual = h_c[c_index];

        if (std::abs(actual - expected) > 1e-5f) {
            if (falseResults < MAX_PRINT_FALSE_RESULTS) {
                std::cerr << "Mismatch at C[" << i << "][" << j << "] expected "
                          << expected << " but got " << actual << std::endl;
            }
            falseResults++;
            success = false;
        }
    }

    if (success) {
        std::cout << "Optimized matrix multiplication completed successfully!" << std::endl;
        std::cout << "Sample results:" << std::endl;
        for (auto const& [i, k, j] : samplePoints) {
            size_t c_index = i * b_cols + j;
            std::cout << "  C[" << i << "][" << j << "] = "
                      << h_c[c_index] << std::endl;
        }
    } else {
        std::cerr << "Optimized matrix multiplication failed with " << falseResults << " mismatches!" << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        std::cerr << "  M: rows in matrix A" << std::endl;
        std::cerr << "  N: columns in matrix A = rows in matrix B" << std::endl;
        std::cerr << "  K: columns in matrix B" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1000 500 2000" << std::endl;
        return 1;
    }

    // Parse command line arguments
    size_t M = std::atol(argv[1]); // rows in A
    size_t N = std::atol(argv[2]); // cols in A = rows in B
    size_t K = std::atol(argv[3]); // cols in B

    if (M == 0 || N == 0 || K == 0) {
        std::cerr << "Error: Matrix dimensions must be positive" << std::endl;
        return 1;
    }

    size_t totalElementsA = M * N;
    size_t totalElementsB = N * K;
    size_t totalElementsC = M * K;
    size_t totalElements = totalElementsA + totalElementsB + totalElementsC;

    std::cout << "Optimized Matrix Multiplication: " << M << "x" << N
              << " * " << N << "x" << K
              << " = " << M << "x" << K << std::endl;
    std::cout << "Total elements: " << totalElements
              << " (" << (totalElements / 1'000'000) << "M)" << std::endl;

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Generate random matrices on host
    std::vector<float> h_a(totalElementsA);
    std::vector<float> h_b(totalElementsB);
    std::vector<float> h_c(totalElementsC, 0.0f);

    for (size_t i = 0; i < totalElementsA; ++i) {
        h_a[i] = dist(gen);
    }
    for (size_t i = 0; i < totalElementsB; ++i) {
        h_b[i] = dist(gen);
    }

    // Use alpaka's backend selection based on compile-time configuration
    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend) {
            return matrixMulExample(
                backend[alpaka::object::deviceSpec],
                backend[alpaka::object::exec],
                h_a, h_b, h_c, M, N, K);
        },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}