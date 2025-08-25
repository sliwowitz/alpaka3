#
# Copyright 2025 Ivan Andriievskyi
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(minibude LANGUAGES CXX)

add_executable(minibude
  src/main.cpp
)

target_compile_features(minibude PRIVATE cxx_std_20)