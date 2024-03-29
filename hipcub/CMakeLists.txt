# MIT License
#
# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# hipCUB header-only library

# Configure a header file to pass the hipCUB version. Changing the out path as it conflict with header wrapper
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/hipcub/hipcub_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cub/include/hipcub/hipcub_version.hpp"
  @ONLY
)

# Create wrapper for generated version file
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_file(
      "hipcub_version.hpp"
      WRAPPER_LOCATIONS cub/${CMAKE_INSTALL_INCLUDEDIR}/hipcub
      OUTPUT_LOCATIONS cub/wrapper/include/hipcub
      ORIGINAL_FILES ${CMAKE_CURRENT_BINARY_DIR}/cub/include/hipcub/hipcub_version.hpp
  )
endif()

# Only header target, does not include dependencies
add_library(hipcub INTERFACE)
target_include_directories(hipcub
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/cub/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/cub/include/hipcub>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

if(HIP_COMPILER STREQUAL "clang")
  target_link_libraries(hipcub
    INTERFACE
      roc::rocprim_hip
  )
else()
  # On CUDA platform we don't want to export hipcub target with
  # INTERFACE_INCLUDE_DIRECTORIES property containing a path to
  # CUB library headers.
  # hipcub_cub target is only for internal use.
  add_library(hipcub_cub INTERFACE)
  target_link_libraries(hipcub_cub
    INTERFACE hipcub CUB::CUB Thrust::Thrust libcudacxx::libcudacxx
  )
endif()

# Installation

# We need to install headers manually as rocm_install_targets
# does not support header-only libraries (INTERFACE targets)
rocm_install_targets(
  TARGETS hipcub
)
rocm_install(
  DIRECTORY
  "include/"
  "${PROJECT_BINARY_DIR}/hipcub/cub/include/"
  COMPONENT devel
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

# Install the wrapper to hipcub folder. 
# Wrappers would be in /opt/rocm-xxx/hipcub/include/hipcub
if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_install(
    DIRECTORY
    "${PROJECT_BINARY_DIR}/cub/wrapper/"
    DESTINATION hipcub/ 
  )
endif()

include(ROCMExportTargetsHeaderOnly)

# Export targets
if(HIP_COMPILER STREQUAL "clang")
  rocm_export_targets_header_only(
    TARGETS hip::hipcub
    DEPENDS PACKAGE rocprim
    NAMESPACE hip::
  )
else()
  rocm_export_targets_header_only(
    TARGETS hip::hipcub
    NAMESPACE hip::
  )
endif()

