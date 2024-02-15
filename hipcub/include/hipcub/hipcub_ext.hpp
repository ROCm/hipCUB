#ifndef HIBCUB_EXT
#define HIBCUB_EXT
#include "hip/hip_runtime.h"
#include "hipcub.hpp"

#ifndef HIPCUB_QUOTIENT_CEILING
    /// Quotient of x/y rounded up to nearest integer
    #define HIPCUB_QUOTIENT_CEILING(x, y) (((x) + (y) - 1) / (y))
#endif

#if defined(__HIP_DEVICE_COMPILE__)
    #define HIPCUB_IS_DEVICE_CODE 1
    #define HIPCUB_IS_HOST_CODE 0
    #define HIPCUB_INCLUDE_DEVICE_CODE 1
    #define HIPCUB_INCLUDE_HOST_CODE 0
#else
    #define HIPCUB_IS_DEVICE_CODE 0
    #define HIPCUB_IS_HOST_CODE 1
    #define HIPCUB_INCLUDE_DEVICE_CODE 0
    #define HIPCUB_INCLUDE_HOST_CODE 1
#endif
/// Maximum number of devices supported.
#ifndef HIPCUB_MAX_DEVICES
    #define HIPCUB_MAX_DEVICES 128
#endif

#if HIPCUB_CPP_DIALECT >= 2011
    static_assert(HIPCUB_MAX_DEVICES > 0, "HIPCUB_MAX_DEVICES must be greater than 0.");
#endif

/// Whether or not the source targeted by the active compiler pass is allowed to  invoke device kernels or methods from the CUDA runtime API.
#if !defined(__HIP_DEVICE_COMPILE__)
    #define HIPCUB_RUNTIME_ENABLED
#endif

namespace hipcub_extensions {

namespace detail
{
template <bool Test, class T1, class T2>
using conditional_t = typename std::conditional<Test, T1, T2>::type;

/**
 * Call `cudaDeviceSynchronize()` using the proper API for the current CUB and
 * CUDA configuration.
 */
// HIPCUB_EXEC_CHECK_DISABLE
HIPCUB_RUNTIME_FUNCTION inline hipError_t device_synchronize()
{
  hipError_t result = hipErrorUnknown;

  if (HIPCUB_IS_HOST_CODE)
  {
#if HIPCUB_INCLUDE_HOST_CODE
    result = hipDeviceSynchronize();
#endif
  }
  else
  {
    // Device code with the CUDA runtime.
#if defined(HIPCUB_INCLUDE_DEVICE_CODE) && defined(HIPCUB_RUNTIME_ENABLED)

#if defined(__HIPCC__) &&                                                     \
  ((__HIPCC_VER_MAJOR__ > 11) ||                                              \
   ((__HIPCC_VER_MAJOR__ == 11) && (__HIPCC_VER_MINOR__ >= 6)))
    // CUDA >= 11.6
    result = __hipDeviceSynchronizeDeprecationAvoidance();
#else // CUDA < 11.6
    result = hipDeviceSynchronize();
#endif

#else // Device code without the CUDA runtime.
    // Device side CUDA API calls are not supported in this configuration.
    result = hipErrorInvalidConfiguration;
#endif
  }

  return result;
}
}

#include "util_device.hpp"
#include "single_pass_scan_operators.hpp"
}// End of namespace hipcub_extensions

#endif