# Change Log for hipCUB

See README.md on how to build the hipCUB documentation using Doxygen.

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
