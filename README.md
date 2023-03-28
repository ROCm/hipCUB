# hipCUB

hipCUB is a thin wrapper library on top of [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) or
[CUB](https://github.com/thrust/cub). It enables developers to port a project using the CUB library to the
[HIP](https://github.com/ROCm-Developer-Tools/HIP) layer to run on AMD hardware. In the [ROCm](https://rocm.github.io/)
environment, hipCUB uses the rocPRIM library as the backend.  However, on CUDA platforms it uses CUB instead.

## Documentation

Information about the library API and other user topics can be found in the [hipCUB documentation](https://hipcub.readthedocs.io/en/latest).

## Requirements

* Git
* CMake (3.16 or later)
* For AMD GPUs:
  * AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.0 or later)
    * Including [HIP-clang](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang) compiler, which must be
      set as C++ compiler on ROCm platform.
  * [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library
    * Automatically downloaded and built by CMake script.
    * Requires CMake 3.16.9 or later.
* For NVIDIA GPUs:
  * CUDA Toolkit
  * CUB library
    * Automatically downloaded and built by CMake script.
    * Requires CMake 3.15.0 or later.
* Python 3.6 or higher (HIP on Windows only, only required for install scripts)
* Visual Studio 2019 with clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is enabled by default.
  * It will be automatically downloaded and built by CMake script.
* [Google Benchmark](https://github.com/google/benchmark)
  * Required only for benchmarks. Building benchmarks is off by default.
  * It will be automatically downloaded and built by cmake script.

## Build And Install

```shell
git clone https://github.com/ROCmSoftwarePlatform/hipCUB.git

# Go to hipCUB directory, create and go to the build directory.
cd hipCUB; mkdir build; cd build

# Configure hipCUB, setup options for your system.
# Build options:
#   BUILD_TEST - OFF by default,
#   BUILD_BENCHMARK - OFF by default.
#   DOWNLOAD_ROCPRIM - OFF by default and at ON the rocPRIM will be downloaded to build folder,
#
# ! IMPORTANT !
# Set C++ compiler to HCC or HIP-clang. You can do it by adding 'CXX=<path-to-compiler>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the compiler.
#
[CXX=hipcc] cmake ../. # or cmake-gui ../.

# To configure hipCUB for Nvidia platforms, 'CXX=<path-to-nvcc>', `CXX=nvcc` or omitting the flag
# entirely before 'cmake' is sufficient
[CXX=nvcc] cmake -DBUILD_TEST=ON ../. # or cmake-gui ../.
# or
cmake -DBUILD_TEST=ON ../. # or cmake-gui ../.
# or to build benchmarks
cmake -DBUILD_BENCHMARK=ON ../.

# Build
make -j4

# Optionally, run tests if they're enabled.
ctest --output-on-failure

# Package
make package

# Install
[sudo] make install
```

### HIP on Windows

Initial support for HIP on Windows has been added.  To install, use the provided rmake.py python script:
```shell
git clone https://github.com/ROCmSoftwarePlatform/hipCUB.git
cd hipCUB

# the -i option will install rocPRIM to C:\hipSDK by default
python rmake.py -i

# the -c option will build all clients including unit tests
python rmake.py -c
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

## Running Benchmarks

```shell
# Go to hipCUB build directory
cd hipCUB; cd build

# To run benchmark for warp functions:
# Further option can be found using --help
# [] Fields are optional
./benchmark/benchmark_warp_<function_name> [--size <size>] [--trials <trials>]

# To run benchmark for block functions:
# Further option can be found using --help
# [] Fields are optional
./benchmark/benchmark_block_<function_name> [--size <size>] [--trials <trials>]

# To run benchmark for device functions:
# Further option can be found using --help
# [] Fields are optional
./benchmark/benchmark_device_<function_name> [--size <size>] [--trials <trials>]
```

## Building Documentation

```shell
# Go to the hipCUB docs directory
cd hipCUB; cd docs

# Install required pip packages
python3 -m pip install -r .sphinx/requirements.txt

# Build the documentation
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

# For e.g. serve the HTML docs locally
cd _build/html
python3 -m http.server
```

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/hipCUB/issues).

## Contributions and License

Contributions of any kind are most welcome! More details are found at [CONTRIBUTING](./CONTRIBUTING.md)
and [LICENSE](./LICENSE.txt).
