cmake_minimum_required(VERSION 3.25)

set(alpaka_USE_SOURCE_TREE ON)
set(alpaka_BUILD_BENCHMARKS OFF)
set(alpaka_BUILD_EXAMPLES OFF)
set(alpaka_TESTING OFF)

if(BACKEND STREQUAL "OPENMP")
    project(vibeTests LANGUAGES CXX)
    set(alpaka_DEP_CUDA OFF)
elseif(BACKEND STREQUAL "CUDA")
    project(vibeTests LANGUAGES CXX CUDA)
    set(alpaka_DEP_CUDA ON)
else()
    message(FATAL_ERROR "Please specify BACKEND: cmake .. -DBACKEND=OPENMP or -DBACKEND=CUDA")
endif()

if(NOT TARGET alpaka::alpaka)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../" "${CMAKE_BINARY_DIR}/alpaka")
endif()

# vectorAdd10M
add_executable(vectorAdd10M vectorAdd.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(vectorAdd10M PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(vectorAdd10M PUBLIC alpaka::alpaka)

# matrixAdd
add_executable(matrixAdd matrixAddSimple.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(matrixAdd PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(matrixAdd PUBLIC alpaka::alpaka)

# matrixMul proper
add_executable(matrixMul matrixMulProper.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(matrixMul PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(matrixMul PUBLIC alpaka::alpaka)

# matrixMulOptimized
add_executable(matrixMulOptimized matrixMulOptimized.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(matrixMulOptimized PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(matrixMulOptimized PUBLIC alpaka::alpaka)

# matrixMulTiled
add_executable(matrixMulTiled matrixMulTiled.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(matrixMulTiled PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(matrixMulTiled PUBLIC alpaka::alpaka)

include(${CMAKE_BINARY_DIR}/alpaka/cmake/alpakaCommon.cmake OPTIONAL)
alpaka_finalize(vectorAdd10M)
alpaka_finalize(matrixAdd)
alpaka_finalize(matrixMul)
alpaka_finalize(matrixMulOptimized)
alpaka_finalize(matrixMulTiled)

if(BACKEND STREQUAL "CUDA")
    set_target_properties(vectorAdd10M PROPERTIES CUDA_ARCHITECTURES "86")
    set_target_properties(matrixAdd PROPERTIES CUDA_ARCHITECTURES "86")
    set_target_properties(matrixMul PROPERTIES CUDA_ARCHITECTURES "86")
    set_target_properties(matrixMulOptimized PROPERTIES CUDA_ARCHITECTURES "86")
    set_target_properties(matrixMulTiled PROPERTIES CUDA_ARCHITECTURES "86")
endif()