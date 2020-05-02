# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

# Find HIP package and verify that correct C++ compiler was selected for available
# platfrom. On ROCm platform host and device code is compiled by the same compiler:
# hcc. On CUDA host can be compiled by any C++ compiler while device code is compiled
# by nvcc compiler (CMake's CUDA package handles this).

# Find HIP package
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
find_package(hip REQUIRED CONFIG PATHS /opt/rocm)

if(HIP_COMPILER STREQUAL "nvcc")
  include(SetupNVCC)
  message(STATUS "CUB will be used as hipCUB's backend.")
elseif(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
  if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
    message(FATAL_ERROR "On ROCm platform 'hcc' or 'clang' must be used as C++ compiler.")
  else()
    # Determine if CXX Compiler is hcc, hip-clang or other
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
    string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
    if(CXX_VERSION_STRING MATCHES "HCC")
      set(HIP_COMPILER "hcc" CACHE STRING "HIP Compiler")
    elseif(CXX_VERSION_STRING MATCHES "clang")
      set(HIP_COMPILER "clang" CACHE STRING "HIP Compiler")
    else()
      message(FATAL_ERROR "CXX Compiler version ${CXX_VERSION_STRING} unsupported.")
    endif()
    message(STATUS "HIP Compiler: " ${HIP_COMPILER})

    if(HIP_COMPILER STREQUAL "hcc")
      list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
      find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)
    endif()
  endif()
else()
  message(FATAL_ERROR "HIP_COMPILER must be 'hcc' (AMD ROCm platform) or `nvcc` (NVIDIA CUDA platform).")
endif()
