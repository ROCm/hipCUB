// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_benchmark_header.hpp"

// HIP API
#include "hipcub/device/device_spmv.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 16;
#endif

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class T>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   const hipStream_t stream,
                   float probability)
{
    const T rand_min = T(1);
    const T rand_max = T(10);

    std::vector<char> probs = benchmark_utils::get_random_data01<char>(size * size, probability);
    int num_nonzeroes = 0;
    for (size_t i = 0; i < size * size; i++) {
        if (probs[i]) {
            num_nonzeroes++;
        }
    }

    std::vector<T> values = benchmark_utils::get_random_data<T>(num_nonzeroes, rand_min, rand_max);
    std::vector<int> row_offsets(size + 1);
    std::vector<int> column_indices(num_nonzeroes);

    size_t idx_matrix = 0; // index in size * size matrix
    size_t idx_sparse = 0; // index in nonzero values
    for (size_t row = 0; row < size; row++) {
        row_offsets[row] = idx_sparse;
        for (size_t col = 0; col < size; col++) {
            if (probs[idx_matrix]) {
                column_indices[idx_sparse] = col;
                idx_sparse++;
            }
            idx_matrix++; 
        }
    }
    row_offsets[size] = idx_sparse;

    std::vector<T> vector_x = benchmark_utils::get_random_data<T>(size, rand_min, rand_max);

    T *d_values;
    int *d_row_offsets;
    int *d_column_indices;
    T *d_vector_x;
    T *d_vector_y;
    HIP_CHECK(hipMalloc(&d_values,  values.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_row_offsets, row_offsets.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_column_indices, column_indices.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_vector_x, vector_x.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_vector_y, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(
        d_values, values.data(), values.size() * sizeof(T), 
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_row_offsets, row_offsets.data(), row_offsets.size() * sizeof(int), 
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_column_indices, column_indices.data(), column_indices.size() * sizeof(int), 
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_vector_x, vector_x.data(), vector_x.size() * sizeof(T), 
        hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DeviceSpmv::CsrMV(
          nullptr,
          temp_storage_size_bytes,
          d_values,
          d_row_offsets,
          d_column_indices,
          d_vector_x,
          d_vector_y,
          size,
          size,
          num_nonzeroes,
          stream));
    HIP_CHECK(hipDeviceSynchronize());

    if (temp_storage_size_bytes == 0) {
        temp_storage_size_bytes = 1;
    }

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for (size_t i = 0; i < warmup_size; i++) {
        HIP_CHECK(hipcub::DeviceSpmv::CsrMV(
            d_temp_storage,
            temp_storage_size_bytes,
            d_values,
            d_row_offsets,
            d_column_indices,
            d_vector_x,
            d_vector_y,
            size,
            size,
            num_nonzeroes,
            stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; i++) {
            HIP_CHECK(hipcub::DeviceSpmv::CsrMV(
                d_temp_storage,
                temp_storage_size_bytes,
                d_values,
                d_row_offsets,
                d_column_indices,
                d_vector_x,
                d_vector_y,
                size,
                size,
                num_nonzeroes,
                stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * (num_nonzeroes + size) * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * (num_nonzeroes + size));

    hipFree(d_temp_storage);
    hipFree(d_vector_y);
    hipFree(d_vector_x);
    hipFree(d_column_indices);
    hipFree(d_row_offsets);
    hipFree(d_values);
    HIP_CHECK(hipDeviceSynchronize());
}

#define CREATE_BENCHMARK(T, p)         \
benchmark::RegisterBenchmark(          \
    ("CsrMV<" #T ">(p = " #p")"),      \
    &run_benchmark<T>, size, stream, p \
)

#define BENCHMARK_TYPE(type)         \
    CREATE_BENCHMARK(type, 1.0e-5f), \
    CREATE_BENCHMARK(type, 1.0e-4f), \
    CREATE_BENCHMARK(type, 1.0e-3f), \
    CREATE_BENCHMARK(type, 1.0e-2f), \
    CREATE_BENCHMARK(type, 1.0e-1f)

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        BENCHMARK_TYPE(int),
        BENCHMARK_TYPE(unsigned int),
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
    };

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}