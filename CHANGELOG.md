# Change Log for hipCUB

See README.md on how to build the hipCUB documentation using Doxygen.

## (Unreleased) hipCUB-2.10.12 for ROCm 4.5.0
### Addded
- Initial HIP on Windows support. See README for instructions on how to build and install.
### Changed
- Packaging changed to a development package (called hipcub-dev for `.deb` packages, and hipcub-devel for `.rpm` packages). As hipCUB is a header-only library, there is no runtime package. To aid in the transition, the development package sets the "provides" field to provide the package hipcub, so that existing packages depending on hipcub can continue to work. This provides feature is introduced as a deprecated feature and will be removed in a future ROCm release.

## [Unreleased hipCUB-2.10.11 for ROCm 4.4.0]
### Added
- gfx1030 support added.
- Address Sanitizer build option
### Fixed
- BlockRadixRank unit test failure fixed.

## [Unreleased hipCUB-2.10.10 for ROCm 4.3.0]
### Added
- DiscardOutputIterator to backend header

## [hipCUB-2.10.9 for ROCm 4.2.0]
### Added
- Support for TexObjInputIterator and TexRefInputIterator
- Support for DevicePartition
### Changed
- Minimum cmake version required is now 3.10.2
- CUB backend has been updated to 1.11.0
### Fixed
- Benchmark build fixed
- nvcc build fixed

## [hipCUB-2.10.8 for ROCm 4.1.0]
### Added
- Support for DiscardOutputIterator

## [hipCUB-2.10.7 for ROCm 4.0.0]
### Added
- No new features

## [hipCUB-2.10.6 for ROCm 3.10]
### Added
- No new features

## [hipCUB-2.10.5 for ROCm 3.9.0]
### Added
- No new features

## [hipCUB-2.10.4 for ROCm 3.8.0]
### Added
- No new features

## [hipCUB-2.10.3 for ROCm 3.7.0]
### Added
- No new features

## [hipCUB-2.10.2 for ROCm 3.6.0]
### Added
- No new features

## [hipCUB-2.10.1 for ROCm 3.5.0]
### Added
- Improved tests with fixed and random seeds for test data
### Changed
- Switched to hip-clang as default compiler
- CMake searches for rocPRIM locally first; downloads from github if local search fails
### Deprecated
- HCC build deprecated
### Known Issues
- The following unit test failures have been observed. These are due to issues in rocclr runtime.
    - BlockDiscontinuity
    - BlockExchange
    - BlockHistogram
    - BlockRadixSort
    - BlockReduce
    - BlockScan
