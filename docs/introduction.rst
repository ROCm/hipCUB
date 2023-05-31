
*************
Introduction
*************

.. toctree::
   :maxdepth: 4
   :caption: Contents:

Overview
==================

hipCUB is a thin wrapper library on top of rocPRIM or CUB. It enables developers to port project
using CUB library to the `HIP <https://github.com/ROCm-Developer-Tools/HIP>`_ layer and to run them
on AMD hardware. In the `ROCm <https://rocmdocs.amd.com/en/latest/>`_ environment, hipCUB uses
rocPRIM library as the backend, however, on CUDA platforms it uses CUB instead.

- When using hipCUB you should only include ``<hipcub/hipcub.hpp>`` header.
- When rocPRIM is used as backend ``HIPCUB_ROCPRIM_API`` is defined.
- When CUB is used as backend ``HIPCUB_CUB_API`` is defined.
- Backends are automaticaly selected based on platform detected by HIP layer
  (``__HIP_PLATFORM_AMD__``, ``__HIP_PLATFORM_NVIDIA__``).

rocPRIM backend
====================================

hipCUB with rocPRIM backend may not support all function and features CUB has because of the
differences between ROCm (HIP) platform and CUDA platform.

Not-supported features and differences:

- Functions, classes and macros which are not in the public API or not documented are not
  supported.
- Device-wide primitives can't be called from kernels (dynamic parallelism is not supported in HIP
  on ROCm).
- Storage management and debug functions:

  - ``Debug``, ``PtxVersion``, ``SmVersion`` functions and ``CubDebug``, ``CubDebugExit``,
    ``_CubLog`` macros are not supported.
- Intrinsics:

  - ``ThreadExit``, ``ThreadTrap`` - not supported.
  - Warp thread masks (when used) are 64-bit unsigned integers.
  - ``member_mask`` input argument is ignored in ``WARP_*`` functions.
  - Arguments ``first_thread``, ``last_thread``, and ``member_mask`` are ignored in ``Shuffle*``
    functions.
