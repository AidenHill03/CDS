cmake_minimum_required(VERSION 3.15)
project(cdx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -O3 plus vectorization hints. Deliberately NOT -ffast-math: the escape and
# NaN guards rely on IEEE semantics.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_compile_options(-O3 -Wall -Wextra)
endif()

add_library(cdx STATIC src/renderer.cpp)
target_include_directories(cdx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
target_link_libraries(cdx PUBLIC Threads::Threads)

add_executable(cdx_test test/test_renderer.cpp)
target_link_libraries(cdx_test PRIVATE cdx)
