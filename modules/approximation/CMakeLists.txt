# =============================================================================
# modules/approximation -- builds the module's C++ kernel, `_approx_kernel`.
#
#   cmake -B modules/approximation/build -S modules/approximation
#   cmake --build modules/approximation/build --config Release
#
# The extension is written straight into this package directory (next to
# __init__.py), on every generator and configuration, so
# `from modules.approximation import _approx_kernel` works without any
# PYTHONPATH entry of its own. The engine is compiled in from ../../cdx as a
# dependency (the module depends on the engine; nothing in cdx/ knows about
# this file).
# =============================================================================
cmake_minimum_required(VERSION 3.15)
project(approximation_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Same flags as the engine, and for the same reason: no -ffast-math.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_compile_options(-O3 -Wall -Wextra)
endif()

# The engine's core library, built here as a dependency (Python bindings off:
# this module is its own extension).
set(CDX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../cdx ${CMAKE_CURRENT_BINARY_DIR}/cdx EXCLUDE_FROM_ALL)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_FOUND_RC)
if(NOT PYBIND11_FOUND_RC EQUAL 0)
  message(FATAL_ERROR "pybind11 not found. Install it with:  pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_approx_kernel kernel.cpp)
target_link_libraries(_approx_kernel PRIVATE cdx_core)
find_package(Threads REQUIRED)
target_link_libraries(_approx_kernel PRIVATE Threads::Threads)

# Output next to __init__.py for every configuration (Visual Studio is multi-config).
set_target_properties(_approx_kernel PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER ${cfg} CFG)
  set_target_properties(_approx_kernel PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
message(STATUS "approximation module kernel -> ${CMAKE_CURRENT_SOURCE_DIR}")
