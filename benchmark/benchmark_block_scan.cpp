// MIT License
//
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

// hipCUB API
#include "hipcub/block/block_scan.hpp"


#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<class Runner,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void kernel(const T* input, T* output, const T init)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(input, output, init);
}

template<hipcub::BlockScanAlgorithm Algorithm>
struct inclusive_scan
{
    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__ static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
        __shared__ typename bscan_t::TempStorage storage;

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bscan_t(storage).InclusiveScan(values, values, hipcub::Sum());
        }

        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            output[i * ItemsPerThread + k] = values[k];
        }
    }
};

template<hipcub::BlockScanAlgorithm Algorithm>
struct exclusive_scan
{
    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__ static void run(const T* input, T* output, const T init)
    {
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
        __shared__ typename bscan_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bscan_t(storage).ExclusiveScan(values, values, init, hipcub::Sum());
        }

        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            output[i * ItemsPerThread + k] = values[k];
        }
    }
};

template<class Benchmark,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials = 100>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    // Make sure size is a multiple of BlockSize
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);
    // Allocate and fill memory
    std::vector<T> input(size, T(1));
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>),
                           dim3(size / items_per_block),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_input,
                           d_output,
                           input[0]);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

// IPT - items per thread
#define CREATE_BENCHMARK(T, BS, IPT) \
    benchmark::RegisterBenchmark( \
        (std::string("block_scan<Datatype:"#T",Block Size:"#BS",Items Per Thread:"#IPT",SubAlgorithm Name:" + algorithm_name + ">.Method Name:") + method_name).c_str(), \
        &run_benchmark<Benchmark, T, BS, IPT>, \
        stream, size \
    )

// clang-format off
#define BENCHMARK_TYPE(type, block)    \
    CREATE_BENCHMARK(type, block, 1),  \
    CREATE_BENCHMARK(type, block, 3),  \
    CREATE_BENCHMARK(type, block, 4),  \
    CREATE_BENCHMARK(type, block, 8),  \
    CREATE_BENCHMARK(type, block, 11), \
    CREATE_BENCHMARK(type, block, 16)
// clang-format on

template<class Benchmark>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string&                            method_name,
                    const std::string&                            algorithm_name,
                    hipStream_t                                   stream,
                    size_t                                        size)
{
    using custom_float2 = benchmark_utils::custom_type<float, float>;
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks = {
        // When block size is less than or equal to warp size
        BENCHMARK_TYPE(int, 64),
        BENCHMARK_TYPE(float, 64),
        BENCHMARK_TYPE(double, 64),
        BENCHMARK_TYPE(uint8_t, 64),

        BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(float, 256),
        BENCHMARK_TYPE(double, 256),
        BENCHMARK_TYPE(uint8_t, 256),

        CREATE_BENCHMARK(custom_float2, 256, 1),
        CREATE_BENCHMARK(custom_float2, 256, 4),
        CREATE_BENCHMARK(custom_float2, 256, 8),

        CREATE_BENCHMARK(custom_double2, 256, 1),
        CREATE_BENCHMARK(custom_double2, 256, 4),
        CREATE_BENCHMARK(custom_double2, 256, 8),
    };
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
}

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

    std::cout << "benchmark_block_scan" << std::endl;

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    // clang-format off
    add_benchmarks<inclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>>(
        benchmarks, "inclusive_scan", "BLOCK_SCAN_RAKING", stream, size);
    add_benchmarks<inclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE>>(
        benchmarks, "inclusive_scan", "BLOCK_SCAN_RAKING_MEMOIZE", stream, size);
    add_benchmarks<inclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>>(
        benchmarks, "inclusive_scan", "BLOCK_SCAN_WARP_SCANS", stream, size);
    add_benchmarks<exclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>>(
        benchmarks, "exclusive_scan", "BLOCK_SCAN_RAKING", stream, size);
    add_benchmarks<exclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE>>(
        benchmarks, "exclusive_scan", "BLOCK_SCAN_RAKING_MEMOIZE", stream, size);
    add_benchmarks<exclusive_scan<hipcub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>>(
        benchmarks, "exclusive_scan", "BLOCK_SCAN_WARP_SCANS", stream, size);
    // clang-format on

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
