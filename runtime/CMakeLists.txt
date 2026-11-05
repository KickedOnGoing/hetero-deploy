cmake_minimum_required(VERSION 3.16)
project(hx_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wno-unused-parameter)

find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

add_library(hx_runtime STATIC
  src/runtime.cc
  src/backend.cc
  src/ops.cc
)
target_include_directories(hx_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                     PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hx_runtime PUBLIC Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hx_runtime PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(hx_runtime PUBLIC HX_HAVE_OPENMP)
endif()

add_executable(hx_run tools/hx_run.cc)
target_link_libraries(hx_run PRIVATE hx_runtime)

add_executable(hx_bench tools/hx_bench.cc)
target_link_libraries(hx_bench PRIVATE hx_runtime)

# 生产对接（需要 MTK NeuroPilot / LiteRT SDK 时开启）
option(HX_ENABLE_MTK "Link MTK NeuroPilot / LiteRT Dispatch backend" OFF)
option(HX_ENABLE_LITERT_GPU "Link LiteRT GPU backend" OFF)
if(HX_ENABLE_MTK OR HX_ENABLE_LITERT_GPU)
  target_compile_definitions(hx_runtime PRIVATE HX_ENABLE_MTK HX_ENABLE_LITERT_GPU)
endif()
