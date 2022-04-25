# MIT License
#
# Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ###########################
# hipCUB dependencies
# ###########################

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
  message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# CUB (only for CUDA platform)
if(HIP_COMPILER STREQUAL "nvcc")

  if(NOT DOWNLOAD_CUB)
    find_package(cub QUIET)
    find_package(thrust QUIET)
  endif()

  if(NOT DEFINED CUB_INCLUDE_DIR)
    file(
      DOWNLOAD https://github.com/NVIDIA/cub/archive/1.16.0.zip
      ${CMAKE_CURRENT_BINARY_DIR}/cub-1.16.0.zip
      STATUS cub_download_status LOG cub_download_log
    )
    list(GET cub_download_status 0 cub_download_error_code)
    if(cub_download_error_code)
      message(FATAL_ERROR "Error: downloading "
        "https://github.com/NVIDIA/cub/archive/1.16.0.zip failed "
        "error_code: ${cub_download_error_code} "
        "log: ${cub_download_log} "
      )
    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/cub-1.16.0.zip
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE cub_unpack_error_code
    )
    if(cub_unpack_error_code)
      message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/cub-1.16.0.zip failed")
    endif()
    set(CUB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub-1.16.0/ CACHE PATH "")
  endif()

  if(NOT DEFINED THRUST_INCLUDE_DIR)
    file(
      DOWNLOAD https://github.com/NVIDIA/thrust/archive/1.16.0.zip
      ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.16.0.zip
      STATUS thrust_download_status LOG thrust_download_log
    )
    list(GET thrust_download_status 0 thrust_download_error_code)
    if(thrust_download_error_code)
      message(FATAL_ERROR "Error: downloading "
        "https://github.com/NVIDIA/thrust/archive/1.16.0.zip failed "
        "error_code: ${thrust_download_error_code} "
        "log: ${thrust_download_log} "
      )
    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.16.0.zip
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE thrust_unpack_error_code
    )
    if(thrust_unpack_error_code)
      message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.16.0.zip failed")
    endif()
    set(THRUST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.16.0/ CACHE PATH "")
  endif()
else()
  # rocPRIM (only for ROCm platform)
  if(NOT DOWNLOAD_ROCPRIM)
    find_package(rocprim)
  endif()
  if(NOT rocprim_FOUND)
    message(STATUS "Downloading and building rocprim.")
    download_project(
      PROJ                rocprim
      GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
      GIT_TAG             develop
      GIT_SHALLOW         TRUE
      INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim
      CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
    find_package(rocprim REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim NO_DEFAULT_PATH)
  endif()
endif()

# Test dependencies
if(BUILD_TEST)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Test (https://github.com/google/googletest)
    find_package(GTest QUIET)
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    # Google Test (https://github.com/google/googletest)
    if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$|.*/nvcc$")
      # hip-clang cannot compile googletest for some reason
      set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
    endif()
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.11.0
      GIT_SHALLOW         TRUE
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE}
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
    find_package(GTest REQUIRED)
  endif()
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  # Google Benchmark (https://github.com/google/benchmark.git)
  message(STATUS "Downloading and building Google Benchmark.")
  if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$|.*/nvcc$")
    # hip-clang cannot compile googlebenchmark for some reason
    set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
  endif()
  # Download, build and install googlebenchmark library
  set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark CACHE PATH "")
  download_project(
    PROJ           googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.6.1
    GIT_SHALLOW    TRUE
    INSTALL_DIR    ${GOOGLEBENCHMARK_ROOT}
    CMAKE_ARGS     -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_ENABLE_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE}
    LOG_DOWNLOAD   TRUE
    LOG_CONFIGURE  TRUE
    LOG_BUILD      TRUE
    LOG_INSTALL    TRUE
    BUILD_PROJECT  TRUE
    ${UPDATE_DISCONNECTED_IF_AVAILABLE}
  )
  find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT})
endif()
