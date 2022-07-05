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

// hipcub API
#include "hipcub/block/block_radix_sort.hpp"
#include "hipcub/block/block_load.hpp"
#include "hipcub/block/block_store.hpp"



template<
    class Key,
    class Value,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Descending = false,
    bool ToStriped = false,
    unsigned int StartBit = 0,
    unsigned int EndBit = sizeof(Key) * 8
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool descending = Descending;
    static constexpr bool to_striped = ToStriped;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int end_bit = EndBit;
};

template<class Params>
class HipcubBlockRadixSort : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, int, 64U, 1>,
    params<int, int, 128U, 1>,
    params<unsigned int, int, 256U, 1>,
    params<unsigned short, char, 1024U, 1, true>,

    // Non-power of 2 BlockSize
    params<double, unsigned int, 65U, 1>,
    params<float, int, 37U, 1>,
    params<long long, char, 510U, 1, true>,
    params<unsigned int, long long, 162U, 1, false, true>,
    params<unsigned char, float, 255U, 1>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<float, char, 64U, 2, true>,
    params<int, short, 128U, 4>,
    params<unsigned short, char, 256U, 7>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<double, int, 33U, 5>,
    params<char, double, 464U, 2, true, true>,
    params<unsigned short, int, 100U, 3>,
    params<short, int, 234U, 9>,

    // StartBit and EndBit
    params<unsigned long long, char, 64U, 1, false, false, 8, 20>,
    params<unsigned short, int, 102U, 3, true, false, 4, 10>,
    params<unsigned int, short, 162U, 2, true, true, 3, 12>,

    // Stability (a number of key values is lower than BlockSize * ItemsPerThread: some keys appear
    // multiple times with different values or key parts outside [StartBit, EndBit))
    params<unsigned char, int, 512U, 2, false, true>,
    params<unsigned short, double, 60U, 1, true, false, 8, 11>
> Params;

TYPED_TEST_SUITE(HipcubBlockRadixSort, Params);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type
>
__global__
__launch_bounds__(BlockSize)
void sort_key_kernel(
    key_type* device_keys_output,
    bool to_striped,
    bool descending,
    unsigned int start_bit,
    unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    key_type keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);

    hipcub::BlockRadixSort<key_type, BlockSize, ItemsPerThread> bsort;
    if(to_striped)
    {
        if(descending)
            bsort.SortDescendingBlockedToStriped(keys, start_bit, end_bit);
        else
            bsort.SortBlockedToStriped(keys, start_bit, end_bit);

        hipcub::StoreDirectStriped<BlockSize>(lid, device_keys_output + block_offset, keys);
    }
    else
    {
        if(descending)
            bsort.SortDescending(keys, start_bit, end_bit);
        else
            bsort.Sort(keys, start_bit, end_bit);

        hipcub::StoreDirectBlocked(lid, device_keys_output + block_offset, keys);
    }
}

TYPED_TEST(HipcubBlockRadixSort, SortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool descending = TestFixture::params::descending;
    constexpr bool to_striped = TestFixture::params::to_striped;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 1134;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> keys_output;
        if(std::is_floating_point<key_type>::value)
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                (key_type)-1000,
                (key_type)+1000,
                seed_value
            );
        }
        else
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max(),
                seed_value
            );
        }

        // Calculate expected results on host
        std::vector<key_type> expected(keys_output);
        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(
                expected.begin() + (i * items_per_block),
                expected.begin() + ((i + 1) * items_per_block),
                test_utils::key_comparator<key_type, descending, start_bit, end_bit>()
            );
        }

        // Preparing device
        key_type* device_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys_output, keys_output.size() * sizeof(key_type)));

        HIP_CHECK(
            hipMemcpy(
                device_keys_output, keys_output.data(),
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(sort_key_kernel<block_size, items_per_thread, key_type>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_keys_output, to_striped, descending, start_bit, end_bit
        );

        // Getting results to host
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), device_keys_output,
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(keys_output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_keys_output));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    class value_type
>
__global__
__launch_bounds__(BlockSize)
void sort_key_value_kernel(
    key_type* device_keys_output,
    value_type* device_values_output,
    bool to_striped,
    bool descending,
    unsigned int start_bit,
    unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    key_type keys[ItemsPerThread];
    value_type values[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);
    hipcub::LoadDirectBlocked(lid, device_values_output + block_offset, values);

    hipcub::BlockRadixSort<key_type, BlockSize, ItemsPerThread, value_type> bsort;
    if(to_striped)
    {
        if(descending)
            bsort.SortDescendingBlockedToStriped(keys, values, start_bit, end_bit);
        else
            bsort.SortBlockedToStriped(keys, values, start_bit, end_bit);

        hipcub::StoreDirectStriped<BlockSize>(lid, device_keys_output + block_offset, keys);
        hipcub::StoreDirectStriped<BlockSize>(lid, device_values_output + block_offset, values);
    }
    else
    {
        if(descending)
            bsort.SortDescending(keys, values, start_bit, end_bit);
        else
            bsort.Sort(keys, values, start_bit, end_bit);

        hipcub::StoreDirectBlocked(lid, device_keys_output + block_offset, keys);
        hipcub::StoreDirectBlocked(lid, device_values_output + block_offset, values);
    }
}


TYPED_TEST(HipcubBlockRadixSort, SortKeysValues)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool descending = TestFixture::params::descending;
    constexpr bool to_striped = TestFixture::params::to_striped;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 1134;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> keys_output;
        if(std::is_floating_point<key_type>::value)
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                (key_type)-1000,
                (key_type)+1000,
                seed_value
            );
        }
        else
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max(),
                seed_value
            );
        }

        std::vector<value_type> values_output;
        if(std::is_floating_point<value_type>::value)
        {
            values_output = test_utils::get_random_data<value_type>(
                size,
                (value_type)-1000,
                (value_type)+1000,
                seed_value + seed_value_addition
            );
        }
        else
        {
            values_output = test_utils::get_random_data<value_type>(
                size,
                std::numeric_limits<value_type>::min(),
                std::numeric_limits<value_type>::max(),
                seed_value + seed_value_addition
            );
        }

        using key_value = std::pair<key_type, value_type>;

        // Calculate expected results on host
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_output[i], values_output[i]);
        }

        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(
                expected.begin() + (i * items_per_block),
                expected.begin() + ((i + 1) * items_per_block),
                test_utils::key_value_comparator<key_type, value_type, descending, start_bit, end_bit>()
            );
        }

        key_type* device_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys_output, keys_output.size() * sizeof(key_type)));
        value_type* device_values_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_values_output, values_output.size() * sizeof(value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_keys_output, keys_output.data(),
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_values_output, values_output.data(),
                values_output.size() * sizeof(typename decltype(values_output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(sort_key_value_kernel<block_size, items_per_thread, key_type, value_type>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_keys_output, device_values_output, to_striped, descending, start_bit, end_bit
        );

        // Getting results to host
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), device_keys_output,
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                values_output.data(), device_values_output,
                values_output.size() * sizeof(typename decltype(values_output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(keys_output[i], expected[i].first);
            ASSERT_EQ(values_output[i], expected[i].second);
        }

        HIP_CHECK(hipFree(device_keys_output));
        HIP_CHECK(hipFree(device_values_output));
    }
}
