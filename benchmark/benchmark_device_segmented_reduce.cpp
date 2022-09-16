// MIT License
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipcub/device/device_segmented_reduce.hpp"


#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif


const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

using OffsetType = int;

template<class T, class OutputT, class SegmentedReduceKernel>
void run_benchmark(benchmark::State& state,
                   size_t desired_segments,
                   hipStream_t stream,
                   size_t size,
                   SegmentedReduceKernel segmented_reduce)
{
    using value_type = T;

    // Generate data
    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    const double avg_segment_length = static_cast<double>(size) / desired_segments;
    std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

    std::vector<OffsetType> offsets;
    unsigned int segments_count = 0;
    size_t offset = 0;
    while(offset < size)
    {
        const size_t segment_length = std::round(segment_length_dis(gen));
        offsets.push_back(offset);
        segments_count++;
        offset += segment_length;
    }
    offsets.push_back(size);

    std::vector<value_type> values_input(size);
    std::iota(values_input.begin(), values_input.end(), 0);

    OffsetType * d_offsets;
    HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(OffsetType)));
    HIP_CHECK(
        hipMemcpy(
            d_offsets, offsets.data(),
            (segments_count + 1) * sizeof(OffsetType),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_values_input;
    HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    OutputT * d_aggregates_output;
    HIP_CHECK(hipMalloc(&d_aggregates_output, segments_count * sizeof(OutputT)));

    hipcub::Sum reduce_op;

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;

    HIP_CHECK(
        segmented_reduce(
            d_temporary_storage, temporary_storage_bytes,
            d_values_input, d_aggregates_output,
            segments_count,
            d_offsets, d_offsets + 1,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            segmented_reduce(
                d_temporary_storage, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                segmented_reduce(
                    d_temporary_storage, temporary_storage_bytes,
                    d_values_input, d_aggregates_output,
                    segments_count,
                    d_offsets, d_offsets + 1,
                    stream, false
                )
            );
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(value_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_offsets));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_aggregates_output));
}

template<typename T, typename Op>
struct Benchmark;

template<typename T>
struct Benchmark<T, hipcub::Sum> {
    static void run(benchmark::State& state, size_t desired_segments, const hipStream_t stream, size_t size)
    {
        run_benchmark<T, T>(state, desired_segments, stream, size, hipcub::DeviceSegmentedReduce::Sum<T*, T*, OffsetType*>);
    }
};

template<typename T>
struct Benchmark<T, hipcub::Min> {
    static void run(benchmark::State& state, size_t desired_segments, const hipStream_t stream, size_t size)
    {
        run_benchmark<T, T>(state, desired_segments, stream, size, hipcub::DeviceSegmentedReduce::Min<T*, T*, OffsetType*>);
    }
};

template<typename T>
struct Benchmark<T, hipcub::ArgMin> {
    using Difference = OffsetType;
    using Iterator = typename hipcub::ArgIndexInputIterator<T*, Difference>;
    using KeyValue = typename Iterator::value_type;

    static void run(benchmark::State& state, size_t desired_segments, const hipStream_t stream, size_t size)
    {
        run_benchmark<T, KeyValue>(state, desired_segments, stream, size, hipcub::DeviceSegmentedReduce::ArgMin<T*, KeyValue*, Difference*>);
    }
};

#define CREATE_BENCHMARK(T, SEGMENTS, REDUCE_OP) \
benchmark::RegisterBenchmark( \
    (std::string("segmented_reduce") + "<Datatype:" #T ", ReduceOp:" #REDUCE_OP ">" + \
        "(Number of segments:~" + std::to_string(SEGMENTS) + " segments)" \
    ).c_str(), \
    &Benchmark<T, REDUCE_OP>::run, \
    SEGMENTS, stream, size \
)

#define BENCHMARK_TYPE(type, REDUCE_OP) \
    CREATE_BENCHMARK(type, 1, REDUCE_OP), \
    CREATE_BENCHMARK(type, 100, REDUCE_OP), \
    CREATE_BENCHMARK(type, 10000, REDUCE_OP)

#define CREATE_BENCHMARKS(REDUCE_OP) \
    BENCHMARK_TYPE(float, REDUCE_OP), \
    BENCHMARK_TYPE(double, REDUCE_OP), \
    BENCHMARK_TYPE(int8_t, REDUCE_OP), \
    BENCHMARK_TYPE(int, REDUCE_OP)

void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARKS(hipcub::Sum),
        BENCHMARK_TYPE(custom_double2, hipcub::Sum),
        CREATE_BENCHMARKS(hipcub::Min),
        #ifdef HIPCUB_ROCPRIM_API
        BENCHMARK_TYPE(custom_double2, hipcub::Min),
        #endif
        CREATE_BENCHMARKS(hipcub::ArgMin),
        #ifdef HIPCUB_ROCPRIM_API
        BENCHMARK_TYPE(custom_double2, hipcub::ArgMin),
        #endif
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
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

    std::cout << "benchmark_device_segmented_reduce" << std::endl;

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks(benchmarks, stream, size);

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
