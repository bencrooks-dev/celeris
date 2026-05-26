cmake_minimum_required(VERSION 3.15)
project(celeris_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

option(CELERIS_LLVM "Build the C++ LLVM lowering seam" OFF)

pybind11_add_module(celeris_native
    src/celeris/_native/bindings.cpp
    src/celeris/_native/celeris_core.cpp)

target_include_directories(celeris_native PRIVATE
    src/celeris/_native
    src/celeris/_native/third_party)

target_compile_options(celeris_native PRIVATE -O3)

if(CELERIS_LLVM)
    target_compile_definitions(celeris_native PRIVATE CELERIS_LLVM)
endif()
