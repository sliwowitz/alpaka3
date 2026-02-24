#include <alpaka/alpaka.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace alpaka;

// Tiled matrix multiplication kernel with cache blocking
struct MatrixMulKernelTiled
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
        
        // Get 2D thread indices
        auto const globalIdx = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        size_t row = globalIdx[0];
        size_t col = globalIdx[1];
        
        // Check bounds
        if (row < a_rows && col < b_cols) {
            float sum = 0.0f;
            
            // Tiling parameters - adjust based on cache size
            constexpr size_t TILE_SIZE = 32;
            
            // Pre-calculate base indices
            size_t a_start_idx = row * a_cols;
            
            // Tiled matrix multiplication
            for (size_t kk = 0; kk < a_cols; kk += TILE_SIZE) {
                // Process a tile of the inner dimension
                size_t tile_end = std::min(kk + TILE_SIZE, a_cols);
                
                // Process the tile
                for (size_t k = kk; k < tile_end; ++k) {
                    sum += A[a_start_idx + k] * B[k * b_cols + col];
                }
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
    std::cout << "Tiled matrix multiplication: " << a_rows << "x" << a_cols
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
    
    // Use 2D work distribution for better mapping to matrix structure
    Vec<size_t, 2u> chunkSize(16u, 16u); // 16x16 thread blocks
    auto dataBlocking = onHost::FrameSpec{
        Vec<size_t, 2u>(a_rows, b_cols), // 2D global size
        chunkSize
    };
    
    // Use the tiled kernel
    MatrixMulKernelTiled kernel;
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
    
    std::cout << "Tiled Matrix Multiplication: " << M << "x" << N
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