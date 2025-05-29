# hipCUB

## The hipCUB repository is retired, please use the [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) repository

hipCUB is a thin wrapper library on top of
[rocPRIM](https://github.com/ROCm/rocPRIM) or
[CUB](https://github.com/thrust/cub). You can use it to port a CUB project into
[HIP](https://github.com/ROCm/HIP) so you can use AMD hardware (and
[ROCm](https://rocm.docs.amd.com/en/latest/) software).

In the [ROCm](https://rocm.docs.amd.com/en/latest/)
environment, hipCUB uses the rocPRIM library as the backend. On CUDA platforms, it uses CUB as the
backend.

## Requirements

* Git
* CMake (3.16 or later)
* For AMD GPUs:
  * AMD [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html) software (1.8.0 or later)
    * The [HIP-clang](https://github.com/ROCm/HIP/blob/master/INSTALL.md#hip-clang) compiler (you
      must, set this as the C++ compiler for ROCm)
  * The [rocPRIM](https://github.com/ROCm/rocPRIM) library
    * Automatically downloaded and built by the CMake script
    * Requires CMake 3.16.9 or later
* For NVIDIA GPUs:
  * CUDA Toolkit
  * CCCL library (>= 2.7.0)
    * Automatically downloaded and built by the CMake script
    * Requires CMake 3.15.0 or later
* Python 3.6 or higher (for HIP on Windows only; this is only required for install scripts)
* Visual Studio 2019 with Clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GoogleTest](https://github.com/google/googletest)
* [Google Benchmark](https://github.com/google/benchmark)

GoogleTest and Google Benchmark are automatically downloaded and built by the CMake script.

## Documentation

> [!NOTE]
> The published hipCUB documentation is available [here](https://rocm.docs.amd.com/projects/hipCUB/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).
