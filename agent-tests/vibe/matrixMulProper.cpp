#include <alpaka/alpaka.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace alpaka;

// Matrix multiplication kernel - compute entire matrix in parallel
struct MatrixMulKernel
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
        static_assert(ALPAKA_TYPEOF(A)::dim() == 1, "Expected 1D buffers!");

        // Get the global thread index
        auto const globalIdx = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        size_t idx = globalIdx[0]; // 1D index across entire result matrix
        
        // Map 1D index to 2D matrix coordinates
        size_t row = idx / b_cols;
        size_t col = idx % b_cols;
        
        // Check if we're within bounds
        if (row < a_rows && col < b_cols) {
            // Compute one element of the result matrix: C[row][col]
            float sum = 0.0f;
            // Compute dot product: sum(A[row][k] * B[k][col] for k=0 to a_cols-1)
            for (size_t k = 0; k < a_cols; ++k) {
                size_t a_index = row * a_cols + k;
                size_t b_index = k * b_cols + col;
                sum += A[a_index] * B[b_index];
            }
            C[row * b_cols + col] = sum;
        }
    }
};

auto matrixMulExample(auto const deviceSpec, auto const exec, 
                     std::vector<float> const& h_a, std::vector<float> const& h_b,
                     std::vector<float>& h_c,
                     size_t a_rows, size_t a_cols, size_t b_cols) -> int
{
    std::cout << "Matrix multiplication: " << a_rows << "x" << a_cols 
              << " * " << a_cols << "x" << b_cols 
              << " = " << a_rows << "x" << b_cols 
              << " (" << (a_rows * b_cols) << " elements)" << std::endl;
    
    // Create device selector and get device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    
    std::cout << "Using device: " << devSelector.getDeviceProperties(0).getName() << std::endl;
    
    // Create queue
    onHost::Queue queue = devAcc.makeQueue();
    
    // Allocate host memory for matrices (1D buffers)
    auto bufHostA = onHost::allocHost<float>(Vec<std::size_t, 1u>(a_rows * a_cols));
    auto bufHostB = onHost::allocHost<float>(Vec<std::size_t, 1u>(a_cols * b_cols));
    auto bufHostC = onHost::allocHost<float>(Vec<std::size_t, 1u>(a_rows * b_cols));
    
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
    
    // Set up work distribution - one thread per element of result matrix (1D)
    Vec<size_t, 1u> chunkSize(1u);
    auto dataBlocking = onHost::FrameSpec{
        Vec<size_t, 1u>(a_rows * b_cols), // Global size: total elements in result
        chunkSize
    };
    
    // Instantiate the kernel function object
    MatrixMulKernel kernel;
    auto const taskKernel = KernelBundle{kernel, bufAccA, bufAccB, bufAccC, a_rows, a_cols, b_cols};
    
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
        std::cout << "Matrix multiplication completed successfully!" << std::endl;
        std::cout << "Sample results:" << std::endl;
        for (auto const& [i, k, j] : samplePoints) {
            size_t c_index = i * b_cols + j;
            std::cout << "  C[" << i << "][" << j << "] = " 
                      << h_c[c_index] << std::endl;
        }
    } else {
        std::cerr << "Matrix multiplication failed with " << falseResults << " mismatches!" << std::endl;
        return 1;
    }
    
    return 0;
}

void printUsage(char* progName) {
    std::cerr << "Usage: " << progName << " <M> <N> <K>" << std::endl;
    std::cerr << "  M: rows in matrix A" << std::endl;
    std::cerr << "  N: columns in matrix A = rows in matrix B" << std::endl;
    std::cerr << "  K: columns in matrix B" << std::endl;
    std::cerr << "Example: " << progName << " 1000 500 2000" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    size_t M = std::atol(argv[1]); // rows in A
    size_t N = std::atol(argv[2]); // cols in A = rows in B
    size_t K = std::atol(argv[3]); // cols in B
    
    if (M == 0 || N == 0 || K == 0) {
        std::cerr << "Error: Matrix dimensions must be positive" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    size_t totalElementsA = M * N;
    size_t totalElementsB = N * K;
    size_t totalElementsC = M * K;
    size_t totalElements = totalElementsA + totalElementsB + totalElementsC;
    
    std::cout << "Matrix Multiplication: " << M << "x" << N 
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
    
    // Use the backend configured at compile time
    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend) {
            return matrixMulExample(
                backend[alpaka::object::deviceSpec],
                backend[alpaka::object::exec],
                h_a, h_b, h_c, M, N, K);
        },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}