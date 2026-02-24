cmake_minimum_required(VERSION 3.25)

set(BACKEND "CUDA")
message(STATUS "BACKEND: '${BACKEND}'")

if(BACKEND MATCHES "CUDA")
    message(STATUS "CUDA matched!")
else()
    message(STATUS "CUDA did not match")
endif()
