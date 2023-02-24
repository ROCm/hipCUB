// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_test_header.hpp"
#include "test_utils_argminmax.hpp"

// hipcub API
#include "hipcub/device/device_reduce.hpp"
#include <bitset>

// Params for tests
template<
    class InputType,
    class OutputType = InputType
>
struct DeviceReduceParams
{
    using input_type = InputType;
    using output_type = OutputType;
};

// ---------------------------------------------------------
// Test for reduction ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceReduceTests : public ::testing::Test
{
public:
    using input_type                        = typename Params::input_type;
    using output_type                       = typename Params::output_type;
    static constexpr bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceReduceParams<int, long>,
    DeviceReduceParams<unsigned long>,
    DeviceReduceParams<short>,
    DeviceReduceParams<float>,
    DeviceReduceParams<short, float>,
    DeviceReduceParams<int, double>,
    DeviceReduceParams<test_utils::half, test_utils::half>,
    DeviceReduceParams<test_utils::bfloat16, test_utils::bfloat16>
#ifdef __HIP_PLATFORM_AMD__
    ,
    DeviceReduceParams<test_utils::half,
                       float>, // Doesn't work on NVIDIA / CUB
    DeviceReduceParams<test_utils::bfloat16,
                       float> // Doesn't work on NVIDIA / CUB
#endif
#ifdef HIPCUB_ROCPRIM_API
    ,
    DeviceReduceParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<float>>,
    DeviceReduceParams<test_utils::custom_test_type<int>, test_utils::custom_test_type<float>>
#endif
    >
    HipcubDeviceReduceTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 17) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384, rand());
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

// BEGIN - Code has been added because hipcub::Sum() and hipcub::Min() don't work with half and bfloat16 types (HOST-SIDE)
/**
 * \brief ExtendedFloatSum functor - Because hipcub::Sum() doesn't work with input types
 * <test_utils::half, test_utils::half>, <test_utils::bfloat16, test_utils::bfloat16>,
 * <test_utils::half, float> and <test_utils::bfloat16, float>.
 *
 * As explained in https://github.com/NVIDIA/cub/blob/main/test/test_device_reduce.cu#L193-L200,
 * when using e.g. a constant input iterator of value 2 the CPU accumulator fails to keep adding
 * 2 to 4096 because of precision limitations, as 2 (in half binary representation
 * 0 10000 0000000000 = 1.0 x 2e1) needs to be converted to be able to sum it with 4096
 *  (in half binary representation 0 11011 000000000 = 1.0 x 2e12), that is, the mantisa of 2
 * needs to be shifted to the left 11 times, but that yields a 0 and thus 4096 + 2 = 4096.
 *
 * Code extracted from https://github.com/NVIDIA/cub/pull/618/files (test/test_device_reduce.cu) with added
 * functions for <test_utils::half, float> and <test_utils::bfloat16, float> tests for AMD backend.
 */
struct ExtendedFloatSum
{
    template<class T>
    HIPCUB_HOST_DEVICE T operator()(T a, T b) const
    {
        T result{};
        result.__x = a.raw() + b.raw();
        return result;
    }

    HIPCUB_HOST_DEVICE float operator()(float a, float b) const
    {
        return a + b;
    }

    HIPCUB_HOST_DEVICE test_utils::half operator()(test_utils::half a, test_utils::half b) const
    {
        uint16_t    result
            = this->operator()(test_utils::native_half{a}, test_utils::native_half(b)).raw();
        return reinterpret_cast<test_utils::half&>(result);
    }

    HIPCUB_HOST_DEVICE test_utils::bfloat16 operator()(test_utils::bfloat16 a,
                                                       test_utils::bfloat16 b) const
    {
        uint16_t    result
            = this->operator()(test_utils::native_bfloat16{a}, test_utils::native_bfloat16(b))
                  .raw();
        return reinterpret_cast<test_utils::bfloat16&>(result);
    }

    HIPCUB_HOST_DEVICE float operator()(float a, test_utils::half b) const
    {
        return this->operator()(a, float(b));
    }

    HIPCUB_HOST_DEVICE float operator()(float a, test_utils::bfloat16 b) const
    {
        return this->operator()(a, float(b));
    }
};

/**
 * \brief DeviceReduce function selector - Because we need to resolve at compile time which function
 * from namespace DeviceReduce we are calling: Sum or Reduce.
 *
 * When we want to compute the reduction using the hipcub::Sum operator() and extended float types
 * we need to define our own functor due to extended floats not being arithmetically associative on CPU.
 *
 * But this new functor doesn't have an associated function in DeviceReduce, so we need to call
 * to DeviceReduce::Reduce directly passing this functor, and thus we need to determine at
 * compile time which function will be called so we don't get compile errors.
 * For more clarity, we do get compile errors if we do a simple if..else because the compiler
 * cannot determine which function will be called, and the new functor doesn't compile for all
 * the types used in the tests.
 *
 * Note: with c++17 this selector can be substituted for an if..else in the test that uses
 * "if constexpr", but currently we are using c++14.
 */
template<typename T, typename U>
struct DeviceReduceSelector
{
    void reduce_sum_impl(std::true_type,
                         void*       d_temp_storage,
                         size_t&     temp_storage_size_bytes,
                         T*          d_input,
                         U*          d_output,
                         int         num_items,
                         hipStream_t stream,
                         bool        debug_synchronous)
    {
        HIP_CHECK(hipcub::DeviceReduce::Reduce(d_temp_storage,
                                               temp_storage_size_bytes,
                                               d_input,
                                               d_output,
                                               num_items,
                                               ExtendedFloatSum(),
                                               U(0.f),
                                               stream,
                                               debug_synchronous));
    }

    void reduce_sum_impl(std::false_type,
                         void*       d_temp_storage,
                         size_t&     temp_storage_size_bytes,
                         T*          d_input,
                         U*          d_output,
                         int         num_items,
                         hipStream_t stream,
                         bool        debug_synchronous)
    {
        HIP_CHECK(hipcub::DeviceReduce::Sum(d_temp_storage,
                                            temp_storage_size_bytes,
                                            d_input,
                                            d_output,
                                            num_items,
                                            stream,
                                            debug_synchronous));
    }

    void reduce_sum(void*       d_temp_storage,
                    size_t&     temp_storage_size_bytes,
                    T*          d_input,
                    U*          d_output,
                    int         num_items,
                    hipStream_t stream,
                    bool        debug_synchronous)
    {
        reduce_sum_impl(std::integral_constant < bool,
                        std::is_same<T, test_utils::half>::value
                            || std::is_same<T, test_utils::bfloat16>::value > {},
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_input,
                        d_output,
                        num_items,
                        stream,
                        debug_synchronous);
    }
};

/**
 * \brief Common type specialization - Because hipcub::Min() doesn't work with
 * <float, test_utils::half>.
 */
template<>
struct std::common_type<float, test_utils::half>
{
    using type = float;
};

/**
 * \brief ExtendedFloatMin functor - Because hipcub::Min() doesn't work with input types
 * <test_utils::half, float> and <test_utils::bfloat16, float>.
 *
 * The operators with (__half& a, __half& b) and (__nv_bfloat16& a, __nv_bfloat16& b) input
 * parameters are needed only in NVIDIA because otherwise the program crushes. The code
 * is extracted from https://github.com/NVIDIA/cub/pull/618/files (test/test_device_reduce.cu).
 */
struct ExtendedFloatMin
{
    template<class T>
    HIPCUB_HOST_DEVICE T operator()(T a, T b) const
    {
        return a < b ? a : b;
    }

#ifdef __HIP_PLATFORM_NVIDIA__
    HIPCUB_HOST_DEVICE __half operator()(__half& a, __half& b) const
    {
        NV_IF_TARGET(NV_PROVIDES_SM_53,
                     (return CUB_MIN(a, b);),
                     (return CUB_MIN(__half2float(a), __half2float(b));));
    }

    HIPCUB_HOST_DEVICE __nv_bfloat16 operator()(__nv_bfloat16& a, __nv_bfloat16& b) const
    {
        NV_IF_TARGET(NV_PROVIDES_SM_53,
                     (return CUB_MIN(a, b);),
                     (return CUB_MIN(__bfloat162float(a), __bfloat162float(b));));
    }
#endif

    HIPCUB_HOST_DEVICE float operator()(float a, test_utils::half b) const
    {
        return this->operator()(a, float(b));
    }

    HIPCUB_HOST_DEVICE float operator()(float a, test_utils::bfloat16 b) const
    {
        return this->operator()(a, float(b));
    }
};
// END - Code has been added because hipcub::Sum() and hipcub::Min() don't work with half and bfloat16 types (HOST-SIDE)

// BEGIN - Code has been added because some thread operators don't work with custom_test_type (HOST-SIDE)
/**
 * \brief Common type specialization - Because hipcub::Min() doesn't work with
 * <custom_test_type<T>, custom_test_type<U>> for different types T and U.
 */
template<class T, class U>
struct std::common_type<test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>
{
    using type = test_utils::custom_test_type<typename std::common_type<T, U>::type>;
};

/**
 * \brief CustomTestSum functor - Because hipcub::Sum() doesn't work with
 * <custom_test_type<T>, custom_test_type<U>> for different types T and U.
 */
struct CustomTestSum
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr auto operator()(test_utils::custom_test_type<T> t,
                                                        test_utils::custom_test_type<U> u) const
        -> decltype(std::forward<test_utils::custom_test_type<T>>(t)
                    + std::forward<test_utils::custom_test_type<U>>(u))
    {
        using common_type = typename std::common_type<test_utils::custom_test_type<T>,
                                                      test_utils::custom_test_type<U>>::type;
        const common_type common_t(t);
        const common_type common_u(u);

        return common_t + common_u;
    }
};

/**
 * \brief CustomTestMin functor - Because hipcub::Min() doesn't work with
 * <custom_test_type<T>, custom_test_type<U>> for different types T and U.
 */
struct CustomTestMin
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr auto operator()(test_utils::custom_test_type<T> t,
                                                        test_utils::custom_test_type<U> u) const
    {
        using common_type = typename std::common_type<test_utils::custom_test_type<T>,
                                                      test_utils::custom_test_type<U>>::type;
        const common_type common_t(t);
        const common_type common_u(u);

        return common_t < common_u ? common_t : common_u;
    }
};

// Sum functor selector.
template<typename T, typename U>
struct SumSelector
{
    typedef hipcub::Sum type;
};

template<typename T, typename U>
struct SumSelector<test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>
{
    typedef CustomTestSum type;
};

template<typename U>
struct SumSelector<test_utils::half, U>
{
    typedef ExtendedFloatSum type;
};

template<typename U>
struct SumSelector<test_utils::bfloat16, U>
{
    typedef ExtendedFloatSum type;
};

// Min functor selector.
template<typename T, typename U>
struct MinSelector
{
    typedef hipcub::Min type;
};

template<typename T, typename U>
struct MinSelector<test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>
{
    typedef CustomTestMin type;
};

template<typename U>
struct MinSelector<test_utils::half, U>
{
    typedef ExtendedFloatMin type;
};

template<typename U>
struct MinSelector<test_utils::bfloat16, U>
{
    typedef ExtendedFloatMin type;
};
// END - Code has been added because some thread operators don't work with custom_test_type (HOST-SIDE)

// BEGIN - Code has been added because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
/**
 * \brief Arg max functor - Because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
 */
struct ArgMax {
    template<typename OffsetT, class T, std::enable_if_t<std::is_same<T, test_utils::half>::value ||
                                                         std::is_same<T, test_utils::bfloat16>::value, bool> = true>
    HIPCUB_HOST_DEVICE __forceinline__ hipcub::KeyValuePair <OffsetT, T> operator()(
        const hipcub::KeyValuePair <OffsetT, T> &a,
        const hipcub::KeyValuePair <OffsetT, T> &b) const {
        const hipcub::KeyValuePair <OffsetT, float> native_a(a.key,a.value);
        const hipcub::KeyValuePair <OffsetT, float> native_b(b.key,b.value);

        if ((native_b.value > native_a.value) || ((native_a.value == native_b.value) && (native_b.key < native_a.key)))
            return b;
        return a;
    }
};
/**
 * \brief Arg min functor - Because NVIDIA's hipcub::ArgMin doesn't work with bfloat16 (HOST-SIDE)
 */
struct ArgMin {
    template<typename OffsetT, class T, std::enable_if_t<std::is_same<T, test_utils::half>::value ||
                                                         std::is_same<T, test_utils::bfloat16>::value, bool> = true>
    HIPCUB_HOST_DEVICE __forceinline__ hipcub::KeyValuePair <OffsetT, T> operator()(
        const hipcub::KeyValuePair <OffsetT, T> &a,
        const hipcub::KeyValuePair <OffsetT, T> &b) const {
        const hipcub::KeyValuePair <OffsetT, float> native_a(a.key,a.value);
        const hipcub::KeyValuePair <OffsetT, float> native_b(b.key,b.value);

        if ((native_b.value < native_a.value) || ((native_a.value == native_b.value) && (native_b.key < native_a.key)))
            return b;
        return a;
    }
};

// Maximum to operator selector
template<typename T>
struct ArgMaxSelector {
    typedef hipcub::ArgMax type;
};

template<>
struct ArgMaxSelector<test_utils::half> {
    typedef ArgMax type;
};

template<>
struct ArgMaxSelector<test_utils::bfloat16> {
    typedef ArgMax type;
};

// Minimum to operator selector
template<typename T>
struct ArgMinSelector {
    typedef hipcub::ArgMin type;
};

#ifdef __HIP_PLATFORM_NVIDIA__
template<>
struct ArgMinSelector<test_utils::half> {
    typedef ArgMin type;
};

template<>
struct ArgMinSelector<test_utils::bfloat16> {
    typedef ArgMin type;
};
#endif
// END - Code has been added because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)

TYPED_TEST_SUITE(HipcubDeviceReduceTests, HipcubDeviceReduceTestsParams);

TYPED_TEST(HipcubDeviceReduceTests, ReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(
                size,
                1.0f,
                100.0f,
                seed_value
            );
            std::vector<U> output(1, (U) 0.0f);

            T * d_input;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host using the same accumulator type than on device
            using Sum    = typename SumSelector<T, U>::type; // For custom_type_test tests
            using AccumT = hipcub::detail::accumulator_t<Sum, U, T>;
            Sum    sum_op;
            AccumT tmp_result = U(0.0f); // hipcub::Sum uses as initial type the output type
            for(unsigned int i = 0; i < input.size(); i++)
            {
                tmp_result = sum_op(tmp_result, input[i]);
            }
            const U expected = static_cast<U>(tmp_result);

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            DeviceReduceSelector<T, U> reduce_selector;
            reduce_selector.reduce_sum(d_temp_storage,
                                       temp_storage_size_bytes,
                                       d_input,
                                       d_output,
                                       input.size(),
                                       stream,
                                       debug_synchronous);

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            reduce_selector.reduce_sum(d_temp_storage,
                                       temp_storage_size_bytes,
                                       d_input,
                                       d_output,
                                       input.size(),
                                       stream,
                                       debug_synchronous);
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, test_utils::precision_threshold<T>::percentage));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceMinimum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1.0f, 100.0f, seed_value);
            std::vector<U> output(1, U(0.0f));

            T * d_input;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host using the same accumulator type than on device
            using Min    = typename MinSelector<T, U>::type; // For custom_type_test tests
            using AccumT = hipcub::detail::accumulator_t<hipcub::Min, U, T>;
            Min    min_op;
            AccumT tmp_result = test_utils::numeric_limits<
                T>::max(); // hipcub::Min uses as initial type the input type
            for(unsigned int i = 0; i < input.size(); i++)
            {
                tmp_result = min_op(tmp_result, input[i]);
            }
            const U expected = static_cast<U>(tmp_result);

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                hipcub::DeviceReduce::Min(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                hipcub::DeviceReduce::Min(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, 0.01f));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
}

struct ArgMinDispatch
{
    template<typename InputIteratorT, typename OutputIteratorT>
    auto operator()(void*           d_temp_storage,
                    size_t&         temp_storage_bytes,
                    InputIteratorT  d_in,
                    OutputIteratorT d_out,
                    int             num_items,
                    hipStream_t     stream,
                    bool            debug_synchronous) const
    {
        return hipcub::DeviceReduce::ArgMin(d_temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            d_out,
                                            num_items,
                                            stream,
                                            debug_synchronous);
    }
};

struct ArgMaxDispatch
{
    template<typename InputIteratorT, typename OutputIteratorT>
    auto operator()(void*           d_temp_storage,
                    size_t&         temp_storage_bytes,
                    InputIteratorT  d_in,
                    OutputIteratorT d_out,
                    int             num_items,
                    hipStream_t     stream,
                    bool            debug_synchronous) const
    {
        return hipcub::DeviceReduce::ArgMax(d_temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            d_out,
                                            num_items,
                                            stream,
                                            debug_synchronous);
    }
};

template<typename TestFixture, typename DispatchFunction, typename HostOp>
void test_argminmax(typename TestFixture::input_type empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T         = typename TestFixture::input_type;
    using Iterator  = typename hipcub::ArgIndexInputIterator<T*, int>;
    using key_value = typename Iterator::value_type;

    const bool          debug_synchronous = TestFixture::debug_synchronous;
    DispatchFunction    function;
    std::vector<size_t> sizes = get_sizes();
    sizes.push_back(0);

    for(auto size : sizes)
    {
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T>         input = test_utils::get_random_data<T>(size, 0, 200, seed_value);
            std::vector<key_value> output(1);

            T*         d_input;
            key_value* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(key_value)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            key_value expected;
            if(size > 0)
            {
                // Calculate expected results on host
                Iterator        x(input.data());
                const key_value max = x[0];
                expected            = std::accumulate(x, x + size, max, HostOp());
            }
            else
            {
                // Empty inputs result in a special value
                expected = key_value(1, empty_value);
            }

            size_t temp_storage_size_bytes{};
            void*  d_temp_storage{};
            HIP_CHECK(function(d_temp_storage,
                               temp_storage_size_bytes,
                               d_input,
                               d_output,
                               input.size(),
                               stream,
                               debug_synchronous));

            // temp_storage_size_bytes must be > 0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(function(d_temp_storage,
                               temp_storage_size_bytes,
                               d_input,
                               d_output,
                               input.size(),
                               stream,
                               debug_synchronous));
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(key_value),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, expected.key));
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output[0].value, expected.value, 0.01f));
        }
    }
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArgMinimum)
{
    using T = typename TestFixture::input_type;
    // Because NVIDIA's hipcub::ArgMin doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMinSelector<T>::type;
    test_argminmax<TestFixture, ArgMinDispatch, HostOp>(test_utils::numeric_limits<T>::max());
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArgMaximum)
{
    using T = typename TestFixture::input_type;
    // Because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMaxSelector<T>::type;
    test_argminmax<TestFixture, ArgMaxDispatch, HostOp>(test_utils::numeric_limits<T>::lowest());
}

template<class T>
class HipcubDeviceReduceArgMinMaxSpecialTests : public testing::Test
{};

using HipcubDeviceReduceArgMinMaxSpecialTestsParams
    = ::testing::Types<float, test_utils::half, test_utils::bfloat16>;
TYPED_TEST_SUITE(HipcubDeviceReduceArgMinMaxSpecialTests,
                 HipcubDeviceReduceArgMinMaxSpecialTestsParams);

template<typename TypeParam, typename DispatchFunction>
void test_argminmax_allinf(TypeParam value, TypeParam empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T         = TypeParam;
    using Iterator  = typename hipcub::ArgIndexInputIterator<T*, int>;
    using key_value = typename Iterator::value_type;

    constexpr bool   debug_synchronous = false;
    hipStream_t      stream            = 0; // default
    DispatchFunction function;
    constexpr size_t size = 100'000;

    // Generate data
    std::vector<T>         input(size, value);
    std::vector<key_value> output(1);

    T*         d_input;
    key_value* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(key_value)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    size_t temp_storage_size_bytes{};
    void*  d_temp_storage{};

    HIP_CHECK(function(d_temp_storage,
                       temp_storage_size_bytes,
                       d_input,
                       d_output,
                       input.size(),
                       stream,
                       debug_synchronous));

    // temp_storage_size_bytes must be > 0
    ASSERT_GT(temp_storage_size_bytes, 0U);

    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(function(d_temp_storage,
                       temp_storage_size_bytes,
                       d_input,
                       d_output,
                       input.size(),
                       stream,
                       debug_synchronous));
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(),
                        d_output,
                        output.size() * sizeof(key_value),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));

    if(size > 0)
    {
        // all +/- infinity should produce +/- infinity
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, 0));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].value, value));
    }
    else
    {
        // empty input should produce a special value
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, 1));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].value, empty_value));
    }
}

// TODO: enable for NVIDIA platform once CUB backend incorporates fix
#ifdef __HIP_PLATFORM_AMD__
/// ArgMin with all +Inf should result in +Inf.
TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMinInf)
{
    test_argminmax_allinf<TypeParam, ArgMinDispatch>(
        test_utils::numeric_limits<TypeParam>::infinity(),
        test_utils::numeric_limits<TypeParam>::max());
}

/// ArgMax with all -Inf should result in -Inf.
TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMaxInf)
{
    test_argminmax_allinf<TypeParam, ArgMaxDispatch>(
        test_utils::numeric_limits<TypeParam>::infinity_neg(),
        test_utils::numeric_limits<TypeParam>::lowest());
}
#endif // __HIP_PLATFORM_AMD__
