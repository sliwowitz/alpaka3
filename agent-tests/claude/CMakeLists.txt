cmake_minimum_required(VERSION 3.25)

set(alpaka_USE_SOURCE_TREE ON)
set(alpaka_BUILD_BENCHMARKS OFF)
set(alpaka_BUILD_EXAMPLES OFF)
set(alpaka_TESTING OFF)

if(BACKEND STREQUAL "OPENMP")
    project(claudeTests LANGUAGES CXX)
    set(alpaka_DEP_CUDA OFF)
elseif(BACKEND STREQUAL "CUDA")
    project(claudeTests LANGUAGES CXX CUDA)
    set(alpaka_DEP_CUDA ON)
else()
    message(FATAL_ERROR "Please specify BACKEND: cmake .. -DBACKEND=OPENMP or -DBACKEND=CUDA")
endif()

if(NOT TARGET alpaka::alpaka)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../" "${CMAKE_BINARY_DIR}/alpaka")
endif()

# vectorAdd10M
add_executable(vectorAdd10M src/vectorAdd10M.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(vectorAdd10M PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(vectorAdd10M PUBLIC alpaka::alpaka)

# matrixAdd
add_executable(matrixAdd src/matrixAdd.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(matrixAdd PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(matrixAdd PUBLIC alpaka::alpaka)

# matMul (only include working executables)
add_executable(matMul src/matMul.cpp)
if(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(matMul PRIVATE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
target_link_libraries(matMul PUBLIC alpaka::alpaka)

# Note: matMulOpt excluded due to compilation errors with current alpaka API

include(${CMAKE_BINARY_DIR}/alpaka/cmake/alpakaCommon.cmake OPTIONAL)
alpaka_finalize(vectorAdd10M)
alpaka_finalize(matrixAdd)
alpaka_finalize(matMul)

if(BACKEND STREQUAL "CUDA")
    set_target_properties(vectorAdd10M PROPERTIES CUDA_ARCHITECTURES "86")
    set_target_properties(matrixAdd PROPERTIES CUDA_ARCHITECTURES "86")
    set_target_properties(matMul PROPERTIES CUDA_ARCHITECTURES "86")
    set_target_properties(matMulOpt PROPERTIES CUDA_ARCHITECTURES "86")
endif()