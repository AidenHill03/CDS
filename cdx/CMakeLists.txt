cmake_minimum_required(VERSION 3.15)
project(cdx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # needed for the Python module
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -O3 plus warnings. Deliberately NOT -ffast-math: the escape and NaN guards
# rely on IEEE semantics.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_compile_options(-O3 -Wall -Wextra)
endif()

# ---- core library -----------------------------------------------------------
add_library(cdx_core STATIC src/renderer.cpp src/expr.cpp src/rational.cpp src/roots.cpp src/analysis.cpp src/rational_parser.cpp)
target_include_directories(cdx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
target_link_libraries(cdx_core PUBLIC Threads::Threads)

# ---- C++ tests ----------------------------------------------------------------
add_executable(cdx_test test/test_renderer.cpp)
target_link_libraries(cdx_test PRIVATE cdx_core)

add_executable(cdx_test_expr test/test_expr.cpp)
target_link_libraries(cdx_test_expr PRIVATE cdx_core)

add_executable(cdx_test_rational test/test_rational.cpp)
target_link_libraries(cdx_test_rational PRIVATE cdx_core)

add_executable(cdx_test_roots test/test_roots.cpp)
target_link_libraries(cdx_test_roots PRIVATE cdx_core)

add_executable(cdx_test_custom test/test_custom.cpp)
target_link_libraries(cdx_test_custom PRIVATE cdx_core)

add_executable(cdx_test_analysis test/test_analysis.cpp)
target_link_libraries(cdx_test_analysis PRIVATE cdx_core)

add_executable(cdx_test_rational_parser test/test_rational_parser.cpp)
target_link_libraries(cdx_test_rational_parser PRIVATE cdx_core)

add_executable(cdx_test_rational_pq test/test_rational_pq.cpp)
target_link_libraries(cdx_test_rational_pq PRIVATE cdx_core)

add_executable(cdx_test_rational_params test/test_rational_params.cpp)
target_link_libraries(cdx_test_rational_params PRIVATE cdx_core)

# Measurement tool, not a PASS/FAIL suite -- see its own header comment.
# Diagnoses the Parameter_basin unresolved/slow problem on a Nova-style
# family; prints a report rather than asserting production behavior.
add_executable(cdx_diagnose_parameter_basin test/diagnose_parameter_basin.cpp)
target_link_libraries(cdx_diagnose_parameter_basin PRIVATE cdx_core)

# ---- Python module (optional) ----------------------------------------------
# Enable with:  cmake -B build -DCDX_BUILD_PYTHON=ON
# Requires:     pip install pybind11
option(CDX_BUILD_PYTHON "Build the Python bindings" OFF)

if(CDX_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

  # locate pybind11 via the installed Python package
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FOUND_RC)

  if(NOT PYBIND11_FOUND_RC EQUAL 0)
    message(FATAL_ERROR
      "pybind11 not found. Install it with:  pip install pybind11")
  endif()

  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(cdx python/bindings.cpp)
  target_link_libraries(cdx PRIVATE cdx_core)
  message(STATUS "Python bindings enabled -> module 'cdx'")
endif()
