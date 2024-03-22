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

#ifndef HIPCUB_CUB_DEVICE_DEVICE_SPMV_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_SPMV_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <cub/device/device_spmv.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>

BEGIN_HIPCUB_NAMESPACE

class DeviceSpmv
{

public:

template <
    typename        ValueT,              ///< Matrix and vector value type
    typename        OffsetT>             ///< Signed integer type for sequence offsets
struct SpmvParams
{
    ValueT*         d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*        d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    OffsetT*        d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT*         d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*         d_vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             num_rows;            ///< Number of rows of matrix <b>A</b>.
    int             num_cols;            ///< Number of columns of matrix <b>A</b>.
    int             num_nonzeros;        ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT          alpha;               ///< Alpha multiplicand
    ValueT          beta;                ///< Beta addend-multiplicand

    ::cub::TexRefInputIterator<ValueT, 66778899, OffsetT>  t_vector_x;
};

template<typename ValueT>
HIPCUB_RUNTIME_FUNCTION static hipError_t CsrMV(void*       d_temp_storage,
                                                size_t&     temp_storage_bytes,
                                                ValueT*     d_values,
                                                int*        d_row_offsets,
                                                int*        d_column_indices,
                                                ValueT*     d_vector_x,
                                                ValueT*     d_vector_y,
                                                int         num_rows,
                                                int         num_cols,
                                                int         num_nonzeros,
                                                hipStream_t stream = 0)
{
    ::cub::SpmvParams<ValueT, int> spmv_params;
    spmv_params.d_values          = d_values;
    spmv_params.d_row_end_offsets = d_row_offsets + 1;
    spmv_params.d_column_indices  = d_column_indices;
    spmv_params.d_vector_x        = d_vector_x;
    spmv_params.d_vector_y        = d_vector_y;
    spmv_params.num_rows          = num_rows;
    spmv_params.num_cols          = num_cols;
    spmv_params.num_nonzeros      = num_nonzeros;
    spmv_params.alpha             = 1.0;
    spmv_params.beta              = 0.0;

    return static_cast<hipError_t>(::cub::DispatchSpmv<ValueT, int>::Dispatch(d_temp_storage,
                                                                              temp_storage_bytes,
                                                                              spmv_params,
                                                                              stream));
    }

    template<typename ValueT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        CsrMV(void*       d_temp_storage,
              size_t&     temp_storage_bytes,
              ValueT*     d_values,
              int*        d_row_offsets,
              int*        d_column_indices,
              ValueT*     d_vector_x,
              ValueT*     d_vector_y,
              int         num_rows,
              int         num_cols,
              int         num_nonzeros,
              hipStream_t stream,
              bool        debug_synchronous)
    {
    HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
    return CsrMV(d_temp_storage,
                 temp_storage_bytes,
                 d_values,
                 d_row_offsets,
                 d_column_indices,
                 d_vector_x,
                 d_vector_y,
                 num_rows,
                 num_cols,
                 num_nonzeros,
                 stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_SELECT_HPP_
