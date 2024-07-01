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

#ifndef HIPCUB_ROCPRIM_ITERATOR_TRANSFORM_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_TRANSFORM_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"

#include "iterator_category.hpp"

#include "rocprim/type_traits.hpp"

#include <cstddef>
#include <iterator>
#include <type_traits>

BEGIN_HIPCUB_NAMESPACE

/// \class TransformInputIterator
/// \brief A random-access input (read-only) iterator adaptor for transforming dereferenced values.
///
/// \par Overview
/// * A TransformInputIterator uses functor of type UnaryFunction to transform value obtained
/// by dereferencing underlying iterator.
/// * Using it for simulating a range filled with results of applying functor of type
/// \p UnaryFunction to another range saves memory capacity and/or bandwidth.
///
/// \tparam ValueType - type of value that can be obtained by dereferencing the iterator.
/// \tparam UnaryFunction - type of the transform functor.
/// By default it is the return type of \p UnaryFunction.
/// \tparam InputIterator - type of the underlying random-access input iterator. Must be
/// a random-access iterator.

template<class ValueType,
         class UnaryFunction,
         class InputIterator,
         class OffsetT = std::ptrdiff_t // ignored
         >
class TransformInputIterator
{
public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = ValueType;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's `const` since TransformInputIterator is a read-only iterator.
    using reference = const value_type&;
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since TransformInputIterator is a read-only iterator.
    using pointer = const value_type*;
    /// A type used for identify distance between iterators.
    using difference_type = typename std::iterator_traits<InputIterator>::difference_type;
    /// The category of the iterator.
    using iterator_category = IteratorCategory<value_type, reference>;
    /// The type of unary function used to transform input range.
    using unary_function = UnaryFunction;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = TransformInputIterator;
#endif

    __host__ __device__ __forceinline__ ~TransformInputIterator() = default;

    /// \brief Creates a new TransformInputIterator.
    ///
    /// \param iterator input iterator to iterate over and transform.
    /// \param transform unary function used to transform values obtained
    /// from range pointed by \p iterator.
    __host__ __device__ __forceinline__ TransformInputIterator(InputIterator iterator,
                                                               UnaryFunction transform)
        : iterator_(iterator), transform_(transform)
    {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    __host__ __device__ __forceinline__ TransformInputIterator& operator++()
    {
        iterator_++;
        return *this;
    }

    __host__ __device__ __forceinline__ TransformInputIterator operator++(int)
    {
        TransformInputIterator old = *this;
        iterator_++;
        return old;
    }

    __host__ __device__ __forceinline__ TransformInputIterator& operator--()
    {
        iterator_--;
        return *this;
    }

    __host__ __device__ __forceinline__ TransformInputIterator operator--(int)
    {
        TransformInputIterator old = *this;
        iterator_--;
        return old;
    }

    __host__ __device__ __forceinline__ value_type operator*() const
    {
        return transform_(*iterator_);
    }

    __host__ __device__ __forceinline__ pointer operator->() const
    {
        return &(*(*this));
    }

    __host__ __device__ __forceinline__ value_type operator[](difference_type distance) const
    {
        TransformInputIterator i = (*this) + distance;
        return *i;
    }

    __host__ __device__ __forceinline__ TransformInputIterator
        operator+(difference_type distance) const
    {
        return TransformInputIterator(iterator_ + distance, transform_);
    }

    __host__ __device__ __forceinline__ TransformInputIterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        return *this;
    }

    __host__ __device__ __forceinline__ TransformInputIterator
        operator-(difference_type distance) const
    {
        return TransformInputIterator(iterator_ - distance, transform_);
    }

    __host__ __device__ __forceinline__ TransformInputIterator& operator-=(difference_type distance)
    {
        iterator_ -= distance;
        return *this;
    }

    __host__ __device__ __forceinline__ difference_type
        operator-(TransformInputIterator other) const
    {
        return iterator_ - other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator==(TransformInputIterator other) const
    {
        return iterator_ == other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator!=(TransformInputIterator other) const
    {
        return iterator_ != other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator<(TransformInputIterator other) const
    {
        return iterator_ < other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator<=(TransformInputIterator other) const
    {
        return iterator_ <= other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator>(TransformInputIterator other) const
    {
        return iterator_ > other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator>=(TransformInputIterator other) const
    {
        return iterator_ >= other.iterator_;
    }

    [[deprecated]] friend std::ostream& operator<<(std::ostream& os,
                                                   const TransformInputIterator& /* iter */)
    {
        return os;
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
    InputIterator iterator_;
    UnaryFunction transform_;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class InputIterator, class UnaryFunction, class ValueType>
__host__ __device__ __forceinline__ TransformInputIterator<InputIterator, UnaryFunction, ValueType>
                                    operator+(
        typename TransformInputIterator<InputIterator, UnaryFunction, ValueType>::difference_type
                                                                               distance,
        const TransformInputIterator<InputIterator, UnaryFunction, ValueType>& iterator)
{
    return iterator + distance;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_TRANSFORM_INPUT_ITERATOR_HPP_
