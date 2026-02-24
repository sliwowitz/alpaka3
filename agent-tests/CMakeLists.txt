cmake_minimum_required(VERSION 3.25)

set(alpaka_USE_SOURCE_TREE ON)
set(alpaka_BUILD_BENCHMARKS OFF)
set(alpaka_BUILD_EXAMPLES OFF)
set(alpaka_TESTING OFF)

if(BACKEND STREQUAL "OPENMP")
    project(matMulOptOpenMP LANGUAGES CXX)
    set(alpaka_DEP_CUDA OFF)
elseif(BACKEND STREQUAL "CUDA")
    project(matMulOptCUDA LANGUAGES CXX CUDA)
    set(alpaka_DEP_CUDA ON)
else()
    message(FATAL_ERROR "Please specify BACKEND: cmake .. -DBACKEND=OPENMP or -DBACKEND=CUDA")
endif()

if(NOT TARGET alpaka::alpaka)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_BINARY_DIR}/alpaka")
endif()

add_executable(matMulOpt matMulOpt.cpp)
target_compile_definitions(matMulOpt PRIVATE MATRIX_M=1000 MATRIX_K=1000 MATRIX_N=1000)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(matMulOpt PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(matMulOpt PRIVATE alpaka::alpaka)

include(${CMAKE_BINARY_DIR}/alpaka/cmake/alpakaCommon.cmake OPTIONAL)
alpaka_finalize(matMulOpt)

if(BACKEND STREQUAL "CUDA")
    set_target_properties(matMulOpt PROPERTIES CUDA_ARCHITECTURES "86")
endif()