// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_hipcub_device_radix_sort.hpp"

TYPED_TEST(HipcubDeviceRadixSort, SortPairs)
{
    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr bool check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    const std::vector<unsigned int> sizes = get_sizes();
    for(unsigned int size : sizes)
    {
        if(size > (1 << 20) && !check_large_sizes) continue;

        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<key_type> keys_input;
            keys_input = test_utils::get_random_data<key_type>(
                size,
                test_utils::numeric_limits<key_type>::min(),
                test_utils::numeric_limits<key_type>::max(),
                seed_value + seed_value_addition
            );
            test_utils::add_special_values(keys_input, seed_value);
            std::vector<value_type> values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type * d_keys_input;
            key_type * d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_keys_input, keys_input.data(),
                    size * sizeof(key_type),
                    hipMemcpyHostToDevice
                )
            );

            value_type * d_values_input;
            value_type * d_values_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_values_input, values_input.data(),
                    size * sizeof(value_type),
                    hipMemcpyHostToDevice
                )
            );

            using key_value = std::pair<key_type, value_type>;

            // Calculate expected results on host
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            std::stable_sort(
                expected.begin(), expected.end(),
                test_utils::key_value_comparator<key_type, value_type, descending, start_bit, end_bit>()
            );

            void * d_temporary_storage = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortPairs(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    start_bit, end_bit
                )
            );

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceRadixSort::SortPairsDescending(
                        d_temporary_storage, temporary_storage_bytes,
                        d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                        start_bit, end_bit,
                        stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceRadixSort::SortPairs(
                        d_temporary_storage, temporary_storage_bytes,
                        d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                        start_bit, end_bit,
                        stream, debug_synchronous
                    )
                );
            }

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_values_input));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(
                hipMemcpy(
                    keys_output.data(), d_keys_output,
                    size * sizeof(key_type),
                    hipMemcpyDeviceToHost
                )
            );

            std::vector<value_type> values_output(size);
            HIP_CHECK(
                hipMemcpy(
                    values_output.data(), d_values_output,
                    size * sizeof(value_type),
                    hipMemcpyDeviceToHost
                )
            );

            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));

            std::vector<key_type> keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i] = expected[i].first;
                values_expected[i] = expected[i].second;
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(values_output, values_expected));
        }
    }
}
