# hipCUB

hipCUB is a thin wrapper library on top of [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) or
[CUB](https://github.com/NVlabs/cub). It enables developers to port project using CUB library to the
[HIP](https://github.com/ROCm-Developer-Tools/HIP) layer and to run them on AMD hardware. In [ROCm](https://rocm.github.io/)
environment hipCUB uses rocPRIM library as the backend, however, on CUDA platforms it uses CUB instead.

## Requirements

* Git
* CMake (3.5.1 or later)
* For AMD GPUs:
  * AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.0 or later)
    * Including [HCC](https://github.com/RadeonOpenCompute/hcc) compiler, which must be
      set as C++ compiler on ROCm platform.
  * [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library
    * It will be automatically downloaded and built by CMake script.
* For NVIDIA GPUs:
  * CUDA Toolkit
  * CUB library (automatically downloaded and by CMake script)

Optional:

* [GTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is enabled by default.
  * It will be automatically downloaded and built by CMake script.

## Build And Install

```shell
git clone https://github.com/ROCmSoftwarePlatform/hipCUB.git

# Go to hipCUB directory, create and go to the build directory.
cd hipCUB; mkdir build; cd build

# Configure hipCUB, setup options for your system.
# Build options:
#   BUILD_TEST - OFF by default,
#   DOWNLOAD_ROCPRIM - OFF by default and at ON the rocPRIM will be downloaded to build folder,
#
# ! IMPORTANT !
# On ROCm platform set C++ compiler to HCC. You can do it by adding 'CXX=<path-to-hcc>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the HCC compiler.
#
[CXX=hcc] cmake ../. # or cmake-gui ../.

# To configure rocRAND for Nvidia platforms, 'CXX=<path-to-nvcc>', `CXX=nvcc` or omitting the flag
# entirely before 'cmake' is sufficient
[CXX=nvcc] cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.
# or
cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.

# Build
make -j4

# Optionally, run tests if they're enabled.
ctest --output-on-failure

# Package
make package

# Install
[sudo] make install
```

### Using hipCUB In A Project

Recommended way of including hipCUB into a CMake project is by using its package
configuration files.

```cmake
# On ROCm hipCUB requires rocPRIM
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

# "/opt/rocm" - default install prefix
find_package(hipcub REQUIRED CONFIG PATHS "/opt/rocm/hipcub")

...
# On ROCm: includes hipCUB headers and roc::rocprim_hip target
# On CUDA: includes only hipCUB headers, user has to include CUB directory
target_link_libraries(<your_target> hip::hipcub)
```

Include only the main header file:

```cpp
#include <hipcub/hipcub.hpp>
```

CUB or rocPRIM headers are included by hipCUB depending on the current HIP platform.

## Running Unit Tests

```shell
# Go to hipCUB build directory
cd hipCUB; cd build

# To run all tests
ctest

# To run unit tests for hipCUB
./test/hipcub/<unit-test-name>
```

## Using custom seeds for the tests

Go to the `hipCUB/test/hipcub/test_seed.hpp` file.
```cpp
//(1)
static constexpr int random_seeds_count = 10;

//(2)
static constexpr unsigned int seeds [] = {0, 2, 10, 1000};

//(3)
static constexpr size_t seed_size = sizeof(seeds) / sizeof(seeds[0]);
```

(1) defines a constant that sets how many passes over the tests will be done with runtime-generated seeds. Modify at will.

(2) defines the user generated seeds. Each of the elements of the array will be used as seed for all tests. Modify at will. If no static seeds are desired, the array should be left empty.

```cpp
static constexpr unsigned int seeds [] = {};
```

(3) this line should never be modified.

## Documentation

```shell
# go to hipCUB doc directory
cd hipCUB; cd doc

# run doxygen
doxygen Doxyfile

# open html/index.html

```

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/hipCUB/issues).

## Contributions and License

Contributions of any kind are most welcome! More details are found at [CONTRIBUTING](./CONTRIBUTING.md)
and [LICENSE](./LICENSE.txt).
