cmake_minimum_required(VERSION 3.25)

project(agentTests LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

# vectorAdd10M
add_executable(vectorAdd10M src/vectorAdd10M.cpp)
target_link_libraries(vectorAdd10M PUBLIC alpaka::alpaka)
alpaka_finalize(vectorAdd10M)

# matrixAdd
add_executable(matrixAdd src/matrixAdd.cpp)
target_link_libraries(matrixAdd PUBLIC alpaka::alpaka)
alpaka_finalize(matrixAdd)

# matMul
add_executable(matMul src/matMul.cpp)
target_link_libraries(matMul PUBLIC alpaka::alpaka)
alpaka_finalize(matMul)

# matMulOpt
add_executable(matMulOpt src/matMulOpt.cpp)
target_link_libraries(matMulOpt PUBLIC alpaka::alpaka)
alpaka_finalize(matMulOpt)
