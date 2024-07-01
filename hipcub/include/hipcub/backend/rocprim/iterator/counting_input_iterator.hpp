/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2024, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_ITERATOR_COUNTING_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_COUNTING_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"

#include "iterator_category.hpp"

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

/// \class CountingInputIterator
/// \brief A random-access input (read-only) iterator over a sequence of consecutive integer values.
///
/// \par Overview
/// * A CountingInputIterator represents a pointer into a range of sequentially increasing values.
/// * Using it for simulating a range filled with a sequence of consecutive values saves
/// memory capacity and bandwidth.
///
/// \tparam Incrementable - type of value that can be obtained by dereferencing the iterator.
/// \tparam Difference - a type used for identify distance between iterators
template<class Incrementable, class Difference = std::ptrdiff_t>
class CountingInputIterator
{
public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename std::remove_const<Incrementable>::type;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's same as `value_type` since constant_iterator is a read-only
    /// iterator and does not have underlying buffer.
    using reference = value_type; // CountingInputIterator is not writable
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since CountingInputIterator is a read-only iterator.
    using pointer = const value_type*; // CountingInputIterator is not writable
    /// A type used for identify distance between iterators.
    using difference_type = Difference;
    /// The category of the iterator.
    using iterator_category = IteratorCategory<value_type, reference>;

    static_assert(std::is_integral<value_type>::value, "Incrementable must be integral type");

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = CountingInputIterator;
#endif

    __host__ __device__ __forceinline__ CountingInputIterator() = default;

    /// \brief Creates CountingInputIterator with its initial value initialized
    /// to its default value (usually 0).
    __host__ __device__ __forceinline__ ~CountingInputIterator() = default;

    /// \brief Creates CountingInputIterator and sets its initial value to \p value_.
    ///
    /// \param value initial value
    __host__ __device__ __forceinline__ explicit CountingInputIterator(const value_type value)
        : value_(value)
    {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    __host__ __device__ __forceinline__ CountingInputIterator& operator++()
    {
        value_++;
        return *this;
    }

    __host__ __device__ __forceinline__ CountingInputIterator operator++(int)
    {
        CountingInputIterator old_ci = *this;
        value_++;
        return old_ci;
    }

    __host__ __device__ __forceinline__ CountingInputIterator& operator--()
    {
        value_--;
        return *this;
    }

    __host__ __device__ __forceinline__ CountingInputIterator operator--(int)
    {
        CountingInputIterator old_ci = *this;
        value_--;
        return old_ci;
    }

    __host__ __device__ __forceinline__ value_type operator*() const
    {
        return value_;
    }

    __host__ __device__ __forceinline__ pointer operator->() const
    {
        return &value_;
    }

    __host__ __device__ __forceinline__ CountingInputIterator
        operator+(difference_type distance) const
    {
        return CountingInputIterator(value_ + static_cast<value_type>(distance));
    }

    __host__ __device__ __forceinline__ CountingInputIterator& operator+=(difference_type distance)
    {
        value_ += static_cast<value_type>(distance);
        return *this;
    }

    __host__ __device__ __forceinline__ CountingInputIterator
        operator-(difference_type distance) const
    {
        return CountingInputIterator(value_ - static_cast<value_type>(distance));
    }

    __host__ __device__ __forceinline__ CountingInputIterator& operator-=(difference_type distance)
    {
        value_ -= static_cast<value_type>(distance);
        return *this;
    }

    __host__ __device__ __forceinline__ difference_type operator-(CountingInputIterator other) const
    {
        return static_cast<difference_type>(value_ - other.value_);
    }

    // CountingInputIterator is not writable, so we don't return reference,
    // just something convertible to reference. That matches requirement
    // of RandomAccessIterator concept
    __host__ __device__ __forceinline__ value_type operator[](difference_type distance) const
    {
        return value_ + static_cast<value_type>(distance);
    }

    __host__ __device__ __forceinline__ bool operator==(CountingInputIterator other) const
    {
        return this->equal_value(value_, other.value_);
    }

    __host__ __device__ __forceinline__ bool operator!=(CountingInputIterator other) const
    {
        return !(*this == other);
    }

    __host__ __device__ __forceinline__ bool operator<(CountingInputIterator other) const
    {
        return distance_to(other) > 0;
    }

    __host__ __device__ __forceinline__ bool operator<=(CountingInputIterator other) const
    {
        return distance_to(other) >= 0;
    }

    __host__ __device__ __forceinline__ bool operator>(CountingInputIterator other) const
    {
        return distance_to(other) < 0;
    }

    __host__ __device__ __forceinline__ bool operator>=(CountingInputIterator other) const
    {
        return distance_to(other) <= 0;
    }

    [[deprecated]] friend std::ostream& operator<<(std::ostream&                os,
                                                   const CountingInputIterator& iter)
    {
        os << "[" << iter.value_ << "]";
        return os;
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
    template<class T>
    inline bool equal_value(const T& x, const T& y) const
    {
        return (x == y);
    }

    inline difference_type distance_to(const CountingInputIterator& other) const
    {
        return difference_type(other.value_) - difference_type(value_);
    }

    value_type value_;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class Incrementable, class Difference>
__host__ __device__ __forceinline__ CountingInputIterator<Incrementable, Difference>
    operator+(typename CountingInputIterator<Incrementable, Difference>::difference_type distance,
              const CountingInputIterator<Incrementable, Difference>&                    iter)
{
    return iter + distance;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_COUNTING_INPUT_ITERATOR_HPP_
