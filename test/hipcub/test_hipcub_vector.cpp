// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipcub/util_type.hpp"

template<class Value>
struct params
{
    using value_type = Value;
};

template<class Params>
class HipcubVector : public ::testing::Test
{
public:
    using params = Params;
};

typedef ::testing::Types<params<char>,
                         params<unsigned char>,
                         params<short>,
                         params<unsigned short>,
                         params<int>,
                         params<uint>,
                         params<long>,
                         params<unsigned long>,
                         params<long long>,
                         params<unsigned long long>,
                         params<float>,
                         params<double>
#ifdef __HIP_PLATFORM_AMD__
                         ,
                         params<bool> // Doesn't work on NVIDIA / CUB
#endif
                         >
    Params;

TYPED_TEST_SUITE(HipcubVector, Params);

template<class T, unsigned int VecSize, unsigned int BlockSize>
__global__
void vector_double_kernel(T* device_input, T* device_output)
{
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * BlockSize);

    device_output[index] = device_input[index] + device_input[index];
}

template<class T, unsigned int vec_size>
void run_vector_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Vector                      = hipcub::CubVector<T, vec_size>;
    constexpr unsigned int size       = 128;
    constexpr unsigned int block_size = 16;

    Vector* device_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, size * sizeof(Vector)));

    T input_num = 10;

    Vector input_vec;
    for(unsigned int i = 0; i < vec_size; i++)
    {
        *(reinterpret_cast<T*>(&input_vec) + i) = input_num;
    }

    std::vector<Vector> input(size, input_vec);
    HIP_CHECK(hipMemcpy(device_input,
                        input.data(),
                        input.size() * sizeof(Vector),
                        hipMemcpyHostToDevice));

    Vector* device_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, size * sizeof(Vector)));

    vector_double_kernel<Vector, vec_size, block_size>
        <<<size / block_size, block_size>>>(device_input, device_output);

    std::vector<Vector> output(size);
    HIP_CHECK(hipMemcpy(output.data(),
                        device_output,
                        output.size() * sizeof(Vector),
                        hipMemcpyDeviceToHost));

    const T expected_num = input_num + input_num;

    for(unsigned int i = 0; i < size; i++)
    {
        const Vector output_vec = output[i];
        for(unsigned int j = 0; j < vec_size; j++)
        {
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_eq(*(reinterpret_cast<const T*>(&output_vec) + j),
                                      expected_num));
        }
    }
}

TYPED_TEST(HipcubVector, Vector1)
{
    run_vector_test<typename TestFixture::params::value_type, 1>();
}
TYPED_TEST(HipcubVector, Vector2)
{
    run_vector_test<typename TestFixture::params::value_type, 2>();
}
TYPED_TEST(HipcubVector, Vector3)
{
    run_vector_test<typename TestFixture::params::value_type, 3>();
}
TYPED_TEST(HipcubVector, Vector4)
{
    run_vector_test<typename TestFixture::params::value_type, 4>();
}

TEST(HipcubVectorCustomType, VectorCustomType)
{
    // Custom types do not support operators
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T      = test_utils::custom_test_type<int>;
    using Vector = hipcub::CubVector<T, 2>;

    Vector vector1;
    vector1.x = 10;
    vector1.y = 8;
    Vector vector2;
    vector2.x = 8;
    vector2.y = 10;

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(vector1.x, vector2.y));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(vector1.y, vector2.x));
}
