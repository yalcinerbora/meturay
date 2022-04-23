#include "TracerCUDA/BlockSegmentedReduce.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

static constexpr uint32_t TPB = 512;
static constexpr uint32_t SEGMENT_SIZE = 64;
static constexpr uint32_t SEGMENT_COUNT = TPB / SEGMENT_SIZE;

template <class T>
__global__ __launch_bounds__(TPB)
void KCSegmentedReduceTest(T* gOut,
                           const T* gData)
{
    static_assert(T() == static_cast<T>(0));
    const uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
    auto IsSegmentLeader = [&]() -> uint32_t
    {
        return globalId % SEGMENT_SIZE == 0;
    };
    auto SegmentId = [&]() -> uint32_t
    {
        return globalId / SEGMENT_SIZE;
    };

    // Block Segmented Scan Operating Class
    using BSegReduce = BlockSegmentedReduce<T, TPB, SEGMENT_SIZE>;

    // Shared Memory that is required by the scan
    __shared__ typename BSegReduce::TempStorage sMem;

    T myData = gData[globalId];
    T reduceResult = BSegReduce(sMem).Sum(myData, T());

    if(IsSegmentLeader())
        gOut[SegmentId()] = reduceResult;
}

TEST(BlockSegmentedReduce, FloatSumBasic)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Copy all ones to GPU
    std::vector<float> data(TPB, 1.0f);

    // GPU Allocations
    float* dData;
    float* dReduceOutputs;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dReduceOutputs),
                                   mem, {TPB, SEGMENT_COUNT});
    CUDA_CHECK(cudaMemcpy(dData, data.data(), sizeof(float) * TPB,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dReduceOutputs, 0xFF, sizeof(float) * SEGMENT_COUNT));

    // Scan Call
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCSegmentedReduceTest<float>,
                      //
                      dReduceOutputs,
                      dData);

    // Copy to Host to check
    std::vector<float> hReduceResults(SEGMENT_COUNT);
    CUDA_CHECK(cudaMemcpy(hReduceResults.data(),
                          dReduceOutputs,
                          sizeof(float) * SEGMENT_COUNT,
                          cudaMemcpyDeviceToHost));

    // Checks
    for(float r : hReduceResults)
    {
        EXPECT_FLOAT_EQ(r, static_cast<float>(SEGMENT_SIZE));
    }
}

TEST(BlockSegmentedReduce, FloatSumStress)
{
    static constexpr uint32_t ITERATION_COUNT = 100;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // CPU Allocations
    std::vector<float> hData(TPB, 1.0f);
    std::vector<float> hReduceResults(SEGMENT_COUNT);
    std::vector<float> hReduceResultsExpected(SEGMENT_COUNT);
    // GPU Allocations
    float* dData;
    float* dReduceOutputs;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dReduceOutputs),
                                   mem, {TPB, SEGMENT_COUNT});


    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<float> uniformDist(0.0f, 10.0f);

    for(uint32_t i = 0; i < ITERATION_COUNT; i++)
    {
        // Generate new batch of random numbers
        for(float& d : hData)
        {
            d = uniformDist(rng);
        }

        CUDA_CHECK(cudaMemcpy(dData, hData.data(), sizeof(float) * TPB,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dReduceOutputs, 0xFF, sizeof(float) * SEGMENT_COUNT));

        // Scan Call
        const CudaGPU& bestGPU = system.BestGPU();
        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                          //
                          KCSegmentedReduceTest<float>,
                          //
                          dReduceOutputs,
                          dData);

        // Copy to Host to check
        CUDA_CHECK(cudaMemcpy(hReduceResults.data(),
                              dReduceOutputs,
                              sizeof(float) * SEGMENT_COUNT,
                              cudaMemcpyDeviceToHost));

        // Generate results by hand to check
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            uint32_t segmentStart = i * SEGMENT_SIZE;
            uint32_t nextSegmentStart = (i + 1) * SEGMENT_SIZE;

            hReduceResultsExpected[i] = std::reduce(hData.begin() + segmentStart,
                                                    hData.begin() + nextSegmentStart);
        }
        // Checks
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            float result = hReduceResults[i];
            float expected = hReduceResultsExpected[i];
            EXPECT_NEAR(result, expected, MathConstants::VeryLargeEpsilon);
        }
    }
}

TEST(BlockSegmentedReduce, IntSumStress)
{
     static constexpr uint32_t ITERATION_COUNT = 100;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // CPU Allocations
    std::vector<uint32_t> hData(TPB, 1u);
    std::vector<uint32_t> hReduceResults(SEGMENT_COUNT);
    std::vector<uint32_t> hReduceResultsExpected(SEGMENT_COUNT);
    // GPU Allocations
    uint32_t* dData;
    uint32_t* dReduceOutputs;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dReduceOutputs),
                                   mem, {TPB, SEGMENT_COUNT});


    std::mt19937 rng;
    rng.seed(0);
    std::uniform_int_distribution<uint32_t> uniformDist(0, 10);

    for(uint32_t i = 0; i < ITERATION_COUNT; i++)
    {
        // Generate new batch of random numbers
        for(uint32_t& d : hData)
        {
            d = uniformDist(rng);
        }

        CUDA_CHECK(cudaMemcpy(dData, hData.data(), sizeof(uint32_t) * TPB,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dReduceOutputs, 0xFF, sizeof(uint32_t) * SEGMENT_COUNT));

        // Scan Call
        const CudaGPU& bestGPU = system.BestGPU();
        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                          //
                          KCSegmentedReduceTest<uint32_t>,
                          //
                          dReduceOutputs,
                          dData);

        // Copy to Host to check
        CUDA_CHECK(cudaMemcpy(hReduceResults.data(),
                              dReduceOutputs,
                              sizeof(uint32_t) * SEGMENT_COUNT,
                              cudaMemcpyDeviceToHost));

        // Generate results by hand to check
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            uint32_t segmentStart = i * SEGMENT_SIZE;
            uint32_t nextSegmentStart = (i + 1) * SEGMENT_SIZE;

            hReduceResultsExpected[i] = std::reduce(hData.begin() + segmentStart,
                                                    hData.begin() + nextSegmentStart);
        }
        // Checks
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            uint32_t result = hReduceResults[i];
            uint32_t expected = hReduceResultsExpected[i];
            EXPECT_EQ(result, expected);
        }
    }
}
