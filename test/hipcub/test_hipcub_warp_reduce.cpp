// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipcub/warp/warp_reduce.hpp"

template<
    class T,
    unsigned int WarpSize
>
struct params
{
    using type = T;
    static constexpr unsigned int warp_size = WarpSize;
};

template<class Params>
class HipcubWarpReduceTests : public ::testing::Test {
public:
    using type = typename Params::type;
    static constexpr unsigned int warp_size = Params::warp_size;
};

typedef ::testing::Types<
    // shuffle based reduce
    // Integer
    params<int, 2U>,
    params<int, 4U>,
    params<int, 8U>,
    params<int, 16U>,
    params<int, 32U>,
#ifdef __HIP_PLATFORM_AMD__
    params<int, 64U>,
#endif
    // Float
    params<float, 2U>,
    params<float, 4U>,
    params<float, 8U>,
    params<float, 16U>,
    params<float, 32U>,
#ifdef __HIP_PLATFORM_AMD__
    params<float, 64U>,
#endif
    // shared memory reduce
    // Integer
    params<int, 3U>,
    params<int, 7U>,
    params<int, 15U>,
#ifdef __HIP_PLATFORM_AMD__
    params<int, 37U>,
    params<int, 61U>,
#endif
    // Float
    params<float, 3U>,
    params<float, 7U>,
    params<float, 15U>
#ifdef __HIP_PLATFORM_AMD__
    ,params<float, 37U>,
    params<float, 61U>
#endif
> HipcubWarpReduceTestParams;

TYPED_TEST_SUITE(HipcubWarpReduceTests, HipcubWarpReduceTestParams);

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_reduce_kernel(T* device_input, T* device_output)
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = hipcub::WarpReduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::TempStorage storage[warps_no];
    auto reduce_op = hipcub::Sum();
    value = wreduce_t(storage[warp_id]).Reduce(value, reduce_op);

    if (hipThreadIdx_x % LogicalWarpSize == 0)
    {
        device_output[index / LogicalWarpSize] = value;
    }
}

TYPED_TEST(HipcubWarpReduceTests, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
            : test_utils::max<size_t>((ws32/logical_warp_size) * logical_warp_size, 1);

    // Block size of warp size 64
    constexpr size_t block_size_ws64 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
            : test_utils::max<size_t>((ws64/logical_warp_size) * logical_warp_size, 1);

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100, seed_value);
        std::vector<T> output(size / logical_warp_size, 0);
        std::vector<T> expected(output.size(), 1);

        // Calculate expected results on host
        for(size_t i = 0; i < output.size(); i++)
        {
            T value = 0;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                value += input[idx];
            }
            expected[i] = value;
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output
            );
        }

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output.size(); i++)
        {
            auto diff = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            if(std::is_integral<T>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_reduce_valid_kernel(T* device_input, T* device_output, const int valid)
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = hipcub::WarpReduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::TempStorage storage[warps_no];
    auto reduce_op = hipcub::Sum();
    value = wreduce_t(storage[warp_id]).Reduce(value, reduce_op, valid);

    if (hipThreadIdx_x % LogicalWarpSize == 0)
    {
        device_output[index / LogicalWarpSize] = value;
    }
}

TYPED_TEST(HipcubWarpReduceTests, ReduceValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
            : test_utils::max<size_t>((ws32/logical_warp_size) * logical_warp_size, 1);

    // Block size of warp size 64
    constexpr size_t block_size_ws64 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
            : test_utils::max<size_t>((ws64/logical_warp_size) * logical_warp_size, 1);

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;
    const int valid = logical_warp_size - 1;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100, seed_value);
        std::vector<T> output(size / logical_warp_size, 0);
        std::vector<T> expected(output.size(), 1);

        // Calculate expected results on host
        for(size_t i = 0; i < output.size(); i++)
        {
            T value = 0;
            for(size_t j = 0; j < valid; j++)
            {
                auto idx = i * logical_warp_size + j;
                value += input[idx];
            }
            expected[i] = value;
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_valid_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output, valid
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_valid_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output, valid
            );
        }

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output.size(); i++)
        {
            auto diff = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            if(std::is_integral<T>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void head_segmented_warp_reduce_kernel(T* input, Flag* flags, T* output)
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = input[index];
    auto flag = flags[index];

    using wreduce_t = hipcub::WarpReduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::TempStorage storage[warps_no];
    value = wreduce_t(storage[warp_id]).HeadSegmentedSum(value, flag);

    output[index] = value;
}

TYPED_TEST(HipcubWarpReduceTests, HeadSegmentedReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    using flag_type = unsigned char;
    // logical warp side for warp primitive, execution warp size
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
            : test_utils::max<size_t>((ws32/logical_warp_size) * logical_warp_size, 1);

    // Block size of warp size 64
    constexpr size_t block_size_ws64 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
            : test_utils::max<size_t>((ws64/logical_warp_size) * logical_warp_size, 1);

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }


    #ifdef HIPCUB_CUB_API
    // Bug in CUB
    auto x = logical_warp_size;
    if(x%2 != 0)
    {
        return;
    }
    #endif

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(
            size,
            1,
            10,
            seed_value)
        ; // used for input
        std::vector<flag_type> flags = test_utils::get_random_data01<flag_type>(
            size,
            0.25f,
            seed_value + seed_value_addition
        );
        for(size_t i = 0; i < flags.size(); i+= logical_warp_size)
        {
            flags[i] = 1;
        }
        std::vector<T> output(input.size());

        T* device_input;
        flag_type* device_flags;
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_flags, flags.size() * sizeof(typename decltype(flags)::value_type)));
        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                device_flags, flags.data(),
                flags.size() * sizeof(flag_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<T> expected(output.size());
        size_t segment_head_index = 0;
        T reduction = input[0];
        for(size_t i = 0; i < output.size(); i++)
        {
            if(i%logical_warp_size == 0 || flags[i])
            {
                expected[segment_head_index] = reduction;
                segment_head_index = i;
                reduction = input[i];
            }
            else
            {
                reduction = reduction + input[i];
            }
        }
        expected[segment_head_index] = reduction;

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(head_segmented_warp_reduce_kernel<
                    T, flag_type, block_size_ws32, logical_warp_size
                >),
                dim3(size/block_size_ws32), dim3(block_size_ws32), 0, 0,
                device_input, device_flags, device_output
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(head_segmented_warp_reduce_kernel<
                    T, flag_type, block_size_ws64, logical_warp_size
                >),
                dim3(size/block_size_ws64), dim3(block_size_ws64), 0, 0,
                device_input, device_flags, device_output
            );
        }

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        for(size_t i = 0; i < output.size(); i++)
        {
            if(flags[i])
            {
                auto diff = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
                if(std::is_integral<T>::value) diff = 0;
                ASSERT_NEAR(output[i], expected[i], diff);
            }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_flags));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void tail_segmented_warp_reduce_kernel(T* input, Flag* flags, T* output)
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = input[index];
    auto flag = flags[index];

    using wreduce_t = hipcub::WarpReduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::TempStorage storage[warps_no];
    auto reduce_op = hipcub::Sum();
    value = wreduce_t(storage[warp_id]).TailSegmentedReduce(value, flag, reduce_op);

    output[index] = value;
}

TYPED_TEST(HipcubWarpReduceTests, TailSegmentedReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    using flag_type = unsigned char;
    // logical warp side for warp primitive, execution warp size
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
            : test_utils::max<size_t>((ws32/logical_warp_size) * logical_warp_size, 1);

    // Block size of warp size 64
    constexpr size_t block_size_ws64 =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
            : test_utils::max<size_t>((ws64/logical_warp_size) * logical_warp_size, 1);

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    #ifdef HIPCUB_CUB_API
    // Bug in CUB
    auto x = logical_warp_size;
    if(x%2 != 0)
    {
        return;
    }
    #endif

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(
            size,
            1,
            10,
            seed_value
        ); // used for input
        std::vector<flag_type> flags = test_utils::get_random_data01<flag_type>(
            size,
            0.25f,
            seed_value + seed_value_addition
        );
        for(size_t i = logical_warp_size - 1; i < flags.size(); i+= logical_warp_size)
        {
            flags[i] = 1;
        }
        std::vector<T> output(input.size());

        T* device_input;
        flag_type* device_flags;
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_flags, flags.size() * sizeof(typename decltype(flags)::value_type)));
        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                device_flags, flags.data(),
                flags.size() * sizeof(flag_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<T> expected(output.size());
        std::vector<size_t> segment_indexes;
        size_t segment_index = 0;
        T reduction;
        for(size_t i = 0; i < output.size(); i++)
        {
            // single value segments
            if(flags[i])
            {
                expected[i] = input[i];
                segment_indexes.push_back(i);
            }
            else
            {
                segment_index = i;
                reduction = input[i];
                auto next = i + 1;
                while(next < output.size() && !flags[next])
                {
                    reduction = reduction + input[next];
                    i++;
                    next++;
                }
                i++;
                expected[segment_index] = reduction + input[i];
                segment_indexes.push_back(segment_index);
            }
        }

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(tail_segmented_warp_reduce_kernel<
                    T, flag_type, block_size_ws32, logical_warp_size
                >),
                dim3(size/block_size_ws32), dim3(block_size_ws32), 0, 0,
                device_input, device_flags, device_output
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(tail_segmented_warp_reduce_kernel<
                    T, flag_type, block_size_ws64, logical_warp_size
                >),
                dim3(size/block_size_ws64), dim3(block_size_ws64), 0, 0,
                device_input, device_flags, device_output
            );
        }

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        for(size_t i = 0; i < segment_indexes.size(); i++)
        {
            auto index = segment_indexes[i];
            auto diff = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            if(std::is_integral<T>::value) diff = 0;
            ASSERT_NEAR(output[index], expected[index], diff);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_flags));
        HIP_CHECK(hipFree(device_output));
    }
}
