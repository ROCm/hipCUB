/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"

#include <rocprim/types/key_value_pair.hpp>

#include "iterator_category.hpp"

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

/// \class ArgIndexInputIterator
/// \brief A random-access input (read-only) iterator adaptor for pairing dereferenced values
/// with their indices.
///
/// \par Overview
/// * Dereferencing ArgIndexInputIterator return a value of \p key_value_pair<Difference, InputValueType>
/// type, which includes value from the underlying range and its index in that range.
/// * \p std::iterator_traits<InputIterator>::value_type should be convertible to \p InputValueType.
///
/// \tparam InputIterator - type of the underlying random-access input iterator. Must be
/// a random-access iterator.
/// \tparam Difference - type used for identify distance between iterators and as the index type
/// in the output pair type (see \p value_type).
/// \tparam InputValueType - value type used in the output pair type (see \p value_type).
template<class InputIterator,
         class Difference     = std::ptrdiff_t,
         class InputValueType = typename std::iterator_traits<InputIterator>::value_type>
class ArgIndexInputIterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;

public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = rocprim::key_value_pair<Difference, InputValueType>;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's `const` since ArgIndexInputIterator is a read-only iterator.
    using reference = const value_type&;
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since ArgIndexInputIterator is a read-only iterator.
    using pointer = const value_type*;
    /// A type used for identify distance between iterators.
    using difference_type = Difference;
    /// The category of the iterator.
    using iterator_category = IteratorCategory<value_type, reference>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = ArgIndexInputIterator;
#endif

    __host__ __device__ __forceinline__ ~ArgIndexInputIterator() = default;

    /// \brief Creates a new ArgIndexInputIterator.
    ///
    /// \param iterator input iterator pointing to the input range.
    /// \param offset index of the \p iterator in the input range.
    __host__ __device__ __forceinline__ ArgIndexInputIterator(InputIterator   iterator,
                                                              difference_type offset = 0)
        : iterator_(iterator), offset_(offset)
    {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    __host__ __device__ __forceinline__ ArgIndexInputIterator& operator++()
    {
        iterator_++;
        offset_++;
        return *this;
    }

    __host__ __device__ __forceinline__ ArgIndexInputIterator operator++(int)
    {
        ArgIndexInputIterator old_ai = *this;
        iterator_++;
        offset_++;
        return old_ai;
    }

    __host__ __device__ __forceinline__ value_type operator*() const
    {
        value_type ret(offset_, *iterator_);
        return ret;
    }

    __host__ __device__ __forceinline__ pointer operator->() const
    {
        return &(*(*this));
    }

    __host__ __device__ __forceinline__ ArgIndexInputIterator
        operator+(difference_type distance) const
    {
        return ArgIndexInputIterator(iterator_ + distance, offset_ + distance);
    }

    __host__ __device__ __forceinline__ ArgIndexInputIterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        offset_ += distance;
        return *this;
    }

    __host__ __device__ __forceinline__ ArgIndexInputIterator
        operator-(difference_type distance) const
    {
        return ArgIndexInputIterator(iterator_ - distance, offset_ - distance);
    }

    __host__ __device__ __forceinline__ ArgIndexInputIterator& operator-=(difference_type distance)
    {
        iterator_ -= distance;
        offset_ -= distance;
        return *this;
    }

    __host__ __device__ __forceinline__ difference_type operator-(ArgIndexInputIterator other) const
    {
        return iterator_ - other.iterator_;
    }

    __host__ __device__ __forceinline__ value_type operator[](difference_type distance) const
    {
        ArgIndexInputIterator i = (*this) + distance;
        return *i;
    }

    __host__ __device__ __forceinline__ bool operator==(ArgIndexInputIterator other) const
    {
        return (iterator_ == other.iterator_) && (offset_ == other.offset_);
    }

    __host__ __device__ __forceinline__ bool operator!=(ArgIndexInputIterator other) const
    {
        return (iterator_ != other.iterator_) || (offset_ != other.offset_);
    }

    __host__ __device__ __forceinline__ bool operator<(ArgIndexInputIterator other) const
    {
        return (iterator_ - other.iterator_) > 0;
    }

    __host__ __device__ __forceinline__ bool operator<=(ArgIndexInputIterator other) const
    {
        return (iterator_ - other.iterator_) >= 0;
    }

    __host__ __device__ __forceinline__ bool operator>(ArgIndexInputIterator other) const
    {
        return (iterator_ - other.iterator_) < 0;
    }

    __host__ __device__ __forceinline__ bool operator>=(ArgIndexInputIterator other) const
    {
        return (iterator_ - other.iterator_) <= 0;
    }

    __host__ __device__ __forceinline__ void normalize()
    {
        offset_ = 0;
    }

    [[deprecated]] friend std::ostream& operator<<(std::ostream& os,
                                                   const ArgIndexInputIterator& /* iter */)
    {
        return os;
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
    InputIterator   iterator_;
    difference_type offset_;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class InputIterator, class Difference, class InputValueType>
__host__ __device__ __forceinline__ ArgIndexInputIterator<InputIterator, Difference, InputValueType>
                                    operator+(
        typename ArgIndexInputIterator<InputIterator, Difference, InputValueType>::difference_type
                                                                                distance,
        const ArgIndexInputIterator<InputIterator, Difference, InputValueType>& iterator)
{
    return iterator + distance;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_
