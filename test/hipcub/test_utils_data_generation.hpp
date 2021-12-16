// MIT License
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_HIPCUB_TEST_UTILS_DATA_GENERATION_HPP_
#define HIPCUB_TEST_HIPCUB_TEST_UTILS_DATA_GENERATION_HPP_

// Std::memcpy and std::memcmp
#include <cstring>

namespace test_utils
{

// helpers to produce special values for floating-point types, also compile for integrals.
#define special_values_methods(T) \
static T pNaN(){ T r; std::memcpy(&r, &s[0], sizeof(T)); return r; } \
static T nNaN(){ T r; std::memcpy(&r, &s[1], sizeof(T)); return r; } \
static T pInf(){ T r; std::memcpy(&r, &s[2], sizeof(T)); return r; } \
static T nInf(){ T r; std::memcpy(&r, &s[3], sizeof(T)); return r; } \
static T p0(){ T r; std::memcpy(&r, &s[4], sizeof(T)); return r; } \
static T n0(){ T r; std::memcpy(&r, &s[5], sizeof(T)); return r; } \
static std::vector<T> vector(){ std::vector<T> r = { pNaN(), pInf(), nInf(), p0(), n0() }; return r; }

template<class T>
struct special_values {
    static constexpr T s[] = {0, 0, 0, 0, 0, 0};
    special_values_methods(T);
};

template<>
struct special_values<float>{         // +NaN,           -NaN,           +Inf,           -Inf,           +0.0,           -0.0
    static constexpr uint32_t s[] = {0x7fffffff, 0xffffffff, 0x7F800000, 0xFF800000, 0x00000000, 0x80000000}; // float
    special_values_methods(float);
};
template<>
struct special_values<test_utils::bfloat16>{
    static constexpr uint16_t s[] = {0x7fff, 0xffff, 0x7F80, 0xFF80, 0x0000, 0x8000}; // bfloat16
    special_values_methods(test_utils::bfloat16);
};
template<>
struct special_values<test_utils::half>{
    static constexpr uint16_t s[] = {0x7fff, 0xffff, 0x7C00, 0xFC00, 0x0000, 0x8000}; // half
    special_values_methods(test_utils::half);
};
template<>
struct special_values<double>{
    static constexpr uint64_t s[] = {0x7fffffffffffffff, 0xffffffffffffffff, 0x7FF0000000000000, 0xFFF0000000000000, 0x0000000000000000, 0x8000000000000000}; // double
    special_values_methods(double);
};
// end of special_values helpers

template<class T>
inline auto get_random_data(size_t size, T min, T max, int seed_value, bool = false)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T, class S, class U>
inline auto get_random_data(size_t size, S min, U max, int seed_value, bool use_special_values = false)
    -> typename std::enable_if<!std::is_integral<T>::value && !is_custom_test_type<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    using dis_type = typename std::conditional<std::is_same<test_utils::half, T>::value || std::is_same<test_utils::bfloat16, T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution(static_cast<dis_type>(min), static_cast<dis_type>(max));
    std::vector<T> data(size);
    std::generate(
        data.begin(),
        data.end(),
        [&]() { return static_cast<T>(distribution(gen)); }
    );
    if(use_special_values && size > 6){
        int start = gen() % (size-6);
        std::vector<T> vals = test_utils::special_values<T>::vector();
        std::copy(vals.begin(), vals.end(), data.begin()+start);
    }
    return data;
}

#if defined(_WIN32) && defined(__clang__)
template<>
inline std::vector<unsigned char> get_random_data(size_t size, unsigned char min, unsigned char max, int seed_value)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    std::vector<unsigned char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<unsigned char>(distribution(gen)); });
    return data;
}

template<>
inline std::vector<signed char> get_random_data(size_t size, signed char min, signed char max, int seed_value)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    std::vector<signed char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<signed char>(distribution(gen)); });
    return data;
}

template<>
inline std::vector<char> get_random_data(size_t size, char min, char max, int seed_value)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    std::vector<char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<char>(distribution(gen)); });
    return data;
}
#endif

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max, int seed_value, bool = false)
    -> typename std::enable_if<
        is_custom_test_type<T>::value && std::is_integral<typename T::value_type>::value,
        std::vector<T>
        >::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    return data;
}

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max, int seed_value, bool = false)
    -> typename std::enable_if<
        is_custom_test_type<T>::value && std::is_floating_point<typename T::value_type>::value,
        std::vector<T>
        >::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_real_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    return data;
}

template<class T>
inline auto get_random_value(T min, T max, int seed_value)
    -> typename std::enable_if<std::is_arithmetic<T>::value, T>::type
{
    return get_random_data<T>(1, min, max, seed_value)[0];
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p, int seed_value)
{
    const size_t max_random_size = 1024 * 1024;
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::bernoulli_distribution distribution(p);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

} // end test_utils namespace

#endif  // HIPCUB_TEST_HIPCUB_TEST_UTILS_DATA_GENERATION_HPP_
