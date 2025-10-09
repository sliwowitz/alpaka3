#
# Copyright 2025 Ivan Andriievskyi, Jiří Vyskočil
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(minibude LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Optional performance experimentation toggles
# ---------------------------------------------------------------------------
option(MINIBUDE_USE_HOST_PERF_FLAGS "Enable host CPU tuning flags (-O3 -ffast-math ...)" OFF)
option(MINIBUDE_USE_HOST_PERF_OFAST "Enable host CPU tuning flags (-Ofast)" OFF)
option(MINIBUDE_USE_MARCH_NATIVE "Build miniBUDE with -march=native" OFF)

option(MINIBUDE_USE_SINCOS "Use combined sincos evaluations in score kernel" OFF)
option(MINIBUDE_USE_FMA "Use fused multiply-add where profitable" OFF)
option(MINIBUDE_ASSUME_ALIGNED "Provide alignment hints for small kernel arrays" OFF)
option(MINIBUDE_UNROLL_LIGAND "Request additional unrolling of ligand loop" OFF)

# ---------------------------------------------------------------------------
# Resolve Alpaka (either via install or in-tree checkout)
# ---------------------------------------------------------------------------
if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(alpaka_USE_SOURCE_TREE)
        # Prevent recursion when we add Alpaka itself
        set(alpaka_BUILD_BENCHMARKS OFF)

        # Path goes two levels up:  minibude/  ->  benchmark/  ->  <alpaka root>
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
set(_TARGET_NAME "minibude")

file(GLOB_RECURSE _FILES_SOURCE "./src/*.cpp")

add_executable(${_TARGET_NAME} ${_FILES_SOURCE})
target_include_directories(${_TARGET_NAME} PRIVATE "src")
target_link_libraries(${_TARGET_NAME} PRIVATE alpaka::alpaka)
alpaka_finalize(${_TARGET_NAME})
set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER benchmarks/minibude)

target_compile_features(${_TARGET_NAME} PRIVATE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if(MINIBUDE_USE_HOST_PERF_OFAST)
        target_compile_options(${_TARGET_NAME} PRIVATE -Ofast -fno-math-errno -fno-trapping-math)
        set(_minibude_build_tag_list "ofast")
    elseif(MINIBUDE_USE_HOST_PERF_FLAGS)
        target_compile_options(
            ${_TARGET_NAME}
            PRIVATE -O3 -ffast-math -fno-math-errno -fno-trapping-math)
        set(_minibude_build_tag_list "o3-fast-math")
    endif()

    if(MINIBUDE_USE_MARCH_NATIVE)
        target_compile_options(${_TARGET_NAME} PRIVATE -march=native)
        list(APPEND _minibude_build_tag_list "native")
    endif()
endif()

set(_minibude_kernel_tag_list "")

if(MINIBUDE_USE_SINCOS)
    target_compile_definitions(${_TARGET_NAME} PRIVATE MINIBUDE_USE_ALPAKA_SINCOS)
    list(APPEND _minibude_kernel_tag_list "sincos")
endif()

if(MINIBUDE_USE_FMA)
    target_compile_definitions(${_TARGET_NAME} PRIVATE MINIBUDE_USE_FMA)
    list(APPEND _minibude_kernel_tag_list "fma")
endif()

if(MINIBUDE_ASSUME_ALIGNED)
    target_compile_definitions(${_TARGET_NAME} PRIVATE MINIBUDE_ASSUME_ALIGNED)
    list(APPEND _minibude_kernel_tag_list "align")
endif()

if(MINIBUDE_UNROLL_LIGAND)
    target_compile_definitions(${_TARGET_NAME} PRIVATE MINIBUDE_UNROLL_LIGAND)
    list(APPEND _minibude_kernel_tag_list "unroll")
endif()

if(_minibude_build_tag_list)
    list(JOIN _minibude_build_tag_list "+" _minibude_build_tag)
    target_compile_definitions(
        ${_TARGET_NAME} PRIVATE MINIBUDE_ENABLE_CSV_BUILD_TAG MINIBUDE_BUILD_FLAGS_TAG="${_minibude_build_tag}")
endif()

if(_minibude_kernel_tag_list)
    list(JOIN _minibude_kernel_tag_list "+" _minibude_kernel_tag)
    target_compile_definitions(
        ${_TARGET_NAME} PRIVATE MINIBUDE_ENABLE_CSV_KERNEL_TAG MINIBUDE_KERNEL_PATH_TAG="${_minibude_kernel_tag}")
endif()

# ---------------------------------------------------------------------------
# Optional: allow running via CTest
# ---------------------------------------------------------------------------
if(alpaka_CI)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
    endif()
else()
    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} --deck "${CMAKE_CURRENT_LIST_DIR}/data/bm1")
endif()
