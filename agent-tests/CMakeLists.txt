cmake_minimum_required(VERSION 3.25)

project(agentTests LANGUAGES CXX)

# Use alpaka source tree for proper backend configuration
option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" ON)

if(alpaka_USE_SOURCE_TREE)
    set(alpaka_BUILD_EXAMPLES OFF)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_BINARY_DIR}/alpaka")
else()
    find_package(alpaka REQUIRED)
endif()

# Executables - single source code for all backends
add_executable(vectorAdd10M vectorAdd.cpp)
target_link_libraries(vectorAdd10M PUBLIC alpaka::alpaka)
alpaka_finalize(vectorAdd10M)

add_executable(matrixAdd matrixAddSimple.cpp)
target_link_libraries(matrixAdd PUBLIC alpaka::alpaka)
alpaka_finalize(matrixAdd)

add_executable(matrixMul matrixMulProper.cpp)
target_link_libraries(matrixMul PUBLIC alpaka::alpaka)
alpaka_finalize(matrixMul)

# Optimized versions
add_executable(matrixMulOptimized matrixMulOptimized.cpp)
target_link_libraries(matrixMulOptimized PUBLIC alpaka::alpaka)
alpaka_finalize(matrixMulOptimized)

add_executable(vectorAddOptimized vectorAdd.cpp)
target_link_libraries(vectorAddOptimized PUBLIC alpaka::alpaka)
alpaka_finalize(vectorAddOptimized)

add_executable(matrixAddOptimized matrixAddSimple.cpp)
target_link_libraries(matrixAddOptimized PUBLIC alpaka::alpaka)
alpaka_finalize(matrixAddOptimized)

# Advanced tiled version
add_executable(matrixMulTiled matrixMulTiled.cpp)
target_link_libraries(matrixMulTiled PUBLIC alpaka::alpaka)
alpaka_finalize(matrixMulTiled)