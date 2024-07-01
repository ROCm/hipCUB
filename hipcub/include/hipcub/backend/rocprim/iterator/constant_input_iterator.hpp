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

#ifndef HIPCUB_ROCPRIM_ITERATOR_CONSTANT_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_CONSTANT_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"

#include "iterator_category.hpp"

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

/// \class ConstantInputIterator
/// \brief A random-access input (read-only) iterator which generates a sequence
/// of homogeneous values.
///
/// \par Overview
/// * A ConstantInputIterator represents a pointer into a range of same values.
/// * Using it for simulating a range filled with a sequence of same values saves
/// memory capacity and bandwidth.
///
/// \tparam ValueType - type of value that can be obtained by dereferencing the iterator.
/// \tparam Difference - a type used for identify distance between iterators
template<class ValueType, class Difference = std::ptrdiff_t>
class ConstantInputIterator
{
public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename std::remove_const<ValueType>::type;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's same as `value_type` since ConstantInputIterator is a read-only
    /// iterator and does not have underlying buffer.
    using reference = value_type; // ConstantInputIterator is not writable
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since ConstantInputIterator is a read-only iterator.
    using pointer = const value_type*; // ConstantInputIterator is not writable
    /// A type used for identify distance between iterators.
    using difference_type = Difference;
    /// The category of the iterator.
    using iterator_category = IteratorCategory<value_type, reference>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = ConstantInputIterator;
#endif

    /// \brief Creates ConstantInputIterator and sets its initial value to \p value.
    ///
    /// \param value initial value
    /// \param index optional index for ConstantInputIterator
    __host__ __device__ __forceinline__ explicit ConstantInputIterator(const value_type value,
                                                                       const size_t     index = 0)
        : value_(value), index_(index)
    {}

    __host__ __device__ __forceinline__ ~ConstantInputIterator() = default;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    __host__ __device__ __forceinline__ value_type operator*() const
    {
        return value_;
    }

    __host__ __device__ __forceinline__ pointer operator->() const
    {
        return &value_;
    }

    __host__ __device__ __forceinline__ ConstantInputIterator& operator++()
    {
        index_++;
        return *this;
    }

    __host__ __device__ __forceinline__ ConstantInputIterator operator++(int)
    {
        ConstantInputIterator old_ci = *this;
        index_++;
        return old_ci;
    }

    __host__ __device__ __forceinline__ ConstantInputIterator& operator--()
    {
        index_--;
        return *this;
    }

    __host__ __device__ __forceinline__ ConstantInputIterator operator--(int)
    {
        ConstantInputIterator old_ci = *this;
        index_--;
        return old_ci;
    }

    __host__ __device__ __forceinline__ ConstantInputIterator
        operator+(difference_type distance) const
    {
        return ConstantInputIterator(value_, index_ + distance);
    }

    __host__ __device__ __forceinline__ ConstantInputIterator& operator+=(difference_type distance)
    {
        index_ += distance;
        return *this;
    }

    __host__ __device__ __forceinline__ ConstantInputIterator
        operator-(difference_type distance) const
    {
        return ConstantInputIterator(value_, index_ - distance);
    }

    __host__ __device__ __forceinline__ ConstantInputIterator& operator-=(difference_type distance)
    {
        index_ -= distance;
        return *this;
    }

    __host__ __device__ __forceinline__ difference_type operator-(ConstantInputIterator other) const
    {
        return static_cast<difference_type>(index_ - other.index_);
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// ConstantInputIterator is not writable, so we don't return reference,
    /// just something convertible to reference. That matches requirement
    /// of RandomAccessIterator concept
    __host__ __device__ __forceinline__ value_type operator[](difference_type) const
    {
        return value_;
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    __host__ __device__ __forceinline__ bool operator==(ConstantInputIterator other) const
    {
        return value_ == other.value_ && index_ == other.index_;
    }

    __host__ __device__ __forceinline__ bool operator!=(ConstantInputIterator other) const
    {
        return !(*this == other);
    }

    __host__ __device__ __forceinline__ bool operator<(ConstantInputIterator other) const
    {
        return distance_to(other) > 0;
    }

    __host__ __device__ __forceinline__ bool operator<=(ConstantInputIterator other) const
    {
        return distance_to(other) >= 0;
    }

    __host__ __device__ __forceinline__ bool operator>(ConstantInputIterator other) const
    {
        return distance_to(other) < 0;
    }

    __host__ __device__ __forceinline__ bool operator>=(ConstantInputIterator other) const
    {
        return distance_to(other) <= 0;
    }

    [[deprecated]] friend std::ostream& operator<<(std::ostream&                os,
                                                   const ConstantInputIterator& iter)
    {
        os << "[" << iter.value_ << "]";
        return os;
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
    inline difference_type distance_to(const ConstantInputIterator& other) const
    {
        return difference_type(other.index_) - difference_type(index_);
    }

    value_type value_;
    size_t     index_;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class ValueType, class Difference>
__host__ __device__ __forceinline__ ConstantInputIterator<ValueType, Difference>
    operator+(typename ConstantInputIterator<ValueType, Difference>::difference_type distance,
              const ConstantInputIterator<ValueType, Difference>&                    iter)
{
    return iter + distance;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_CONSTANT_INPUT_ITERATOR_HPP_
