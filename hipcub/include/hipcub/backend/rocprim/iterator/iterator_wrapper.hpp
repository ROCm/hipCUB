// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPCUB_ROCPRIM_WRAPPER_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_WRAPPER_ITERATOR_HPP_

#include "../../../config.hpp"

#include "iterator_category.hpp"

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

/// \class IteratorWrapper
/// \brief A wrapper for iterators to be able to make iterator_traits overwritable
///
/// \tparam WrappedIterator - the iterator that is wrapped
template<class WrappedIterator>
class IteratorWrapper
{
public:
    using value_type        = typename WrappedIterator::value_type;
    using reference         = typename WrappedIterator::reference;
    using pointer           = typename WrappedIterator::pointer;
    using difference_type   = typename WrappedIterator::difference_type;
    using iterator_category = typename IteratorCategory<value_type, reference>::type;
    using self_type         = typename WrappedIterator::self_type;

    WrappedIterator iterator_;

    __host__ __device__ __forceinline__ IteratorWrapper(const IteratorWrapper& other)
        : iterator_(other.iterator_)
    {}

    __host__ __device__ __forceinline__ IteratorWrapper(IteratorWrapper& other)
        : iterator_(other.iterator_)
    {}

    __host__ __device__ __forceinline__ IteratorWrapper(IteratorWrapper&& other)
        : iterator_(std::move(other.iterator_))
    {}

    template<class... Args>
    __host__ __device__ __forceinline__ IteratorWrapper(Args&&... args)
        : iterator_(std::forward<Args>(args)...)
    {}

    __host__ __device__ __forceinline__ IteratorWrapper& operator=(const IteratorWrapper& other)
    {
        iterator_ = other.iterator_;
        return *this;
    }

    __host__ __device__ __forceinline__ IteratorWrapper& operator=(IteratorWrapper&& other)
    {
        iterator_ = std::move(other.iterator_);
        return *this;
    }

    __host__ __device__ __forceinline__ IteratorWrapper& operator++()
    {
        iterator_++;
        return *this;
    }

    __host__ __device__ __forceinline__ IteratorWrapper operator++(int)
    {
        IteratorWrapper old_ci = *this;
        iterator_++;
        return old_ci;
    }

    __host__ __device__ __forceinline__ IteratorWrapper& operator--()
    {
        iterator_--;
        return *this;
    }

    __host__ __device__ __forceinline__ IteratorWrapper operator--(int)
    {
        IteratorWrapper old_ci = *this;
        iterator_--;
        return old_ci;
    }

    __host__ __device__ __forceinline__ value_type operator*() const
    {
        return iterator_.operator*();
    }

    __host__ __device__ __forceinline__ pointer operator->() const
    {
        return iterator_.operator->();
    }

    __host__ __device__ __forceinline__ value_type operator[](difference_type distance) const
    {
        return iterator_[distance];
    }

    __host__ __device__ __forceinline__ IteratorWrapper operator+(difference_type distance) const
    {
        return iterator_ + distance;
    }

    __host__ __device__ __forceinline__ IteratorWrapper& operator+=(difference_type distance)
    {
        iterator_ += distance;
        return *this;
    }

    __host__ __device__ __forceinline__ IteratorWrapper operator-(difference_type distance) const
    {
        return iterator_ - distance;
    }

    __host__ __device__ __forceinline__ IteratorWrapper& operator-=(difference_type distance)
    {
        iterator_ -= distance;
        return *this;
    }

    __host__ __device__ __forceinline__ difference_type operator-(IteratorWrapper other) const
    {
        return iterator_.operator-(other.iterator_);
    }

    __host__ __device__ __forceinline__ bool operator==(IteratorWrapper other) const
    {
        return iterator_ == other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator!=(IteratorWrapper other) const
    {
        return iterator_ != other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator<(IteratorWrapper other) const
    {
        return iterator_ < other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator<=(IteratorWrapper other) const
    {
        return iterator_ <= other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator>(IteratorWrapper other) const
    {
        return iterator_ > other.iterator_;
    }

    __host__ __device__ __forceinline__ bool operator>=(IteratorWrapper other) const
    {
        return iterator_ >= other.iterator_;
    }

    [[deprecated]] friend std::ostream& operator<<(std::ostream& os, const IteratorWrapper& iter)
    {
        os << iter.iterator_;
        return os;
    }
};

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WRAPPER_ITERATOR_HPP_
