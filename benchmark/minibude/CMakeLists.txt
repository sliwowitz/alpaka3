#
# Copyright 2025 Ivan Andriievskyi
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(minibude LANGUAGES CXX)

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

alpaka_add_executable(${_TARGET_NAME} ${_FILES_SOURCE})
target_include_directories(${_TARGET_NAME} PRIVATE "src")
target_link_libraries(${_TARGET_NAME} PRIVATE alpaka::alpaka)
set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER benchmarks/minibude)

target_compile_features(${_TARGET_NAME} PRIVATE cxx_std_20)

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
