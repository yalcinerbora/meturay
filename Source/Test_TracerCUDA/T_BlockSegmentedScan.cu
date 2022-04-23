#include "TracerCUDA/BlockSegmentedScan.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

static constexpr uint32_t TPB = 64;
static constexpr uint32_t SEGMENT_SIZE = 32;
static constexpr uint32_t SEGMENT_COUNT = TPB / SEGMENT_SIZE;

template <class T>
__global__ __launch_bounds__(TPB)
void KCSegmentedInclusiveScanTest(T* gOut,
                                  T* gSegmentAggregates,
                                  const T* gData)
{
    static_assert(T() == static_cast<T>(0));

    // Block Segmented Scan Operating Class
    using BSegScan = BlockSegmentedScan<T, TPB, SEGMENT_SIZE>;

    // Shared Memory that is required by the scan
    __shared__ typename BSegScan::TempStorage sMem;

    const uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
    T myData = gData[globalId];

    T scanResult, segmentAggregates;
    BSegScan(sMem).InclusiveSum(scanResult, segmentAggregates, myData, T());

    gSegmentAggregates[globalId] = segmentAggregates;
    gOut[globalId] = scanResult;
}

template <class T>
__global__ __launch_bounds__(TPB)
void KCSegmentedExclusiveScanTest(T* gOut,
                                  T* gSegmentAggregates,
                                  const T* gData)
{
    static_assert(T() == static_cast<T>(0));

    // Block Segmented Scan Operating Class
    using BSegScan = BlockSegmentedScan<T, TPB, SEGMENT_SIZE>;

    // Shared Memory that is required by the scan
    __shared__ typename BSegScan::TempStorage sMem;

    const uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
    T myData = gData[globalId];

    T scanResult, segmentAggregates;
    BSegScan(sMem).ExclusiveSum(scanResult, segmentAggregates, myData, T());

    gSegmentAggregates[globalId] = segmentAggregates;
    gOut[globalId] = scanResult;
}

TEST(BlockSegmentedScan, FloatInclusiveSumBasic)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Copy all ones to GPU
    std::vector<float> data(TPB, 1.0f);

    // GPU Allocations
    float* dData;
    float* dScanOutputs;
    float* dSegmentAggregates;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dScanOutputs, dSegmentAggregates),
                                   mem, {TPB, TPB, TPB});

    CUDA_CHECK(cudaMemcpy(dData, data.data(), sizeof(float) * TPB,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dScanOutputs, 0xFF, sizeof(float) * TPB));
    CUDA_CHECK(cudaMemset(dSegmentAggregates, 0xFF, sizeof(float) * TPB));

    // Scan Call
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCSegmentedInclusiveScanTest<float>,
                      //
                      dScanOutputs,
                      dSegmentAggregates,
                      dData);

    // Copy to Host to check
    std::vector<float> hSegmentAggregates(TPB);
    std::vector<float> hScanResults(TPB);
    CUDA_CHECK(cudaMemcpy(hSegmentAggregates.data(),
                          dSegmentAggregates,
                          sizeof(float) * TPB,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hScanResults.data(),
                          dScanOutputs,
                          sizeof(float) * TPB,
                          cudaMemcpyDeviceToHost));

    // Checks
    for(float r : hSegmentAggregates)
    {
        EXPECT_FLOAT_EQ(r, static_cast<float>(SEGMENT_SIZE));
    }

    for(uint32_t i = 0; i < static_cast<uint32_t>(hScanResults.size()); i++)
    {
        float r = hScanResults[i];
        uint32_t segmentLocalId = i % SEGMENT_SIZE;
        EXPECT_FLOAT_EQ(r, static_cast<float>(segmentLocalId + 1));
    }
}

TEST(BlockSegmentedScan, FloatInclusiveSumStress)
{
    static constexpr uint32_t ITERATION_COUNT = 100;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // CPU Allocations
    std::vector<float> hData(TPB, 1.0f);
    std::vector<float> hSegmentAggregates(TPB);
    std::vector<float> hScanResults(TPB);
    std::vector<float> hSegAggregatesExpected(TPB);
    std::vector<float> hScanResultsExpected(TPB);
    // GPU Allocations
    float* dData;
    float* dScanOutputs;
    float* dSegmentAggregates;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dScanOutputs, dSegmentAggregates),
                                   mem, {TPB, TPB, TPB});


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
        CUDA_CHECK(cudaMemset(dScanOutputs, 0xFF, sizeof(float) * TPB));
        CUDA_CHECK(cudaMemset(dSegmentAggregates, 0xFF, sizeof(float) * TPB));

        // Scan Call
        const CudaGPU& bestGPU = system.BestGPU();
        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                          //
                          KCSegmentedInclusiveScanTest<float>,
                          //
                          dScanOutputs,
                          dSegmentAggregates,
                          dData);

        // Copy to Host to check
        CUDA_CHECK(cudaMemcpy(hSegmentAggregates.data(),
                              dSegmentAggregates,
                              sizeof(float) * TPB,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hScanResults.data(),
                              dScanOutputs,
                              sizeof(float) * TPB,
                              cudaMemcpyDeviceToHost));

        // Generate results by hand to check
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            uint32_t segmentStart = i * SEGMENT_SIZE;
            uint32_t nextSegmentStart = (i + 1) * SEGMENT_SIZE;

            std::inclusive_scan(hData.begin() + segmentStart,
                                hData.begin() + nextSegmentStart,
                                hScanResultsExpected.begin() + segmentStart);
        }
        hSegAggregatesExpected.clear();
        for(uint32_t i = 1; i <= SEGMENT_COUNT; i++)
        {
            uint32_t segmentAggregateIndex = (i * SEGMENT_SIZE) - 1;
            hSegAggregatesExpected.insert(hSegAggregatesExpected.end(),
                                          SEGMENT_SIZE,
                                          hScanResultsExpected[segmentAggregateIndex]);
        }
        // Checks
        for(uint32_t i = 0; i < TPB; i++)
        {
            float result = hScanResults[i];
            float expected = hScanResultsExpected[i];
            EXPECT_NEAR(result, expected, MathConstants::VeryLargeEpsilon);
        }
        for(uint32_t i = 0; i < TPB; i++)
        {
            float result = hSegmentAggregates[i];
            float expected = hSegAggregatesExpected[i];
            EXPECT_NEAR(result, expected, MathConstants::VeryLargeEpsilon);
        }
    }
}

TEST(BlockSegmentedScan, IntInclusiveSumStress)
{
    static constexpr uint32_t ITERATION_COUNT = 100;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // CPU Allocations
    std::vector<uint32_t> hData(TPB, 1u);
    std::vector<uint32_t> hSegmentAggregates(TPB);
    std::vector<uint32_t> hScanResults(TPB);
    std::vector<uint32_t> hSegAggregatesExpected(TPB);
    std::vector<uint32_t> hScanResultsExpected(TPB);
    // GPU Allocations
    uint32_t* dData;
    uint32_t* dScanOutputs;
    uint32_t* dSegmentAggregates;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dScanOutputs, dSegmentAggregates),
                                   mem, {TPB, TPB, TPB});


    std::mt19937 rng;
    rng.seed(0);
    std::uniform_int_distribution<uint32_t> uniformDist(0u, 10u);

    for(uint32_t i = 0; i < ITERATION_COUNT; i++)
    {
        // Generate new batch of random numbers
        for(uint32_t& d : hData)
        {
            d = uniformDist(rng);
        }

        CUDA_CHECK(cudaMemcpy(dData, hData.data(), sizeof(uint32_t) * TPB,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dScanOutputs, 0xFF, sizeof(uint32_t) * TPB));
        CUDA_CHECK(cudaMemset(dSegmentAggregates, 0xFF, sizeof(uint32_t) * TPB));

        // Scan Call
        const CudaGPU& bestGPU = system.BestGPU();
        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                          //
                          KCSegmentedInclusiveScanTest<uint32_t>,
                          //
                          dScanOutputs,
                          dSegmentAggregates,
                          dData);

        // Copy to Host to check
        CUDA_CHECK(cudaMemcpy(hSegmentAggregates.data(),
                              dSegmentAggregates,
                              sizeof(uint32_t) * TPB,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hScanResults.data(),
                              dScanOutputs,
                              sizeof(uint32_t) * TPB,
                              cudaMemcpyDeviceToHost));

        // Generate results by hand to check
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            uint32_t segmentStart = i * SEGMENT_SIZE;
            uint32_t nextSegmentStart = (i + 1) * SEGMENT_SIZE;

            std::inclusive_scan(hData.begin() + segmentStart,
                                hData.begin() + nextSegmentStart,
                                hScanResultsExpected.begin() + segmentStart);
        }
        hSegAggregatesExpected.clear();
        for(uint32_t i = 1; i <= SEGMENT_COUNT; i++)
        {
            uint32_t segmentAggregateIndex = (i * SEGMENT_SIZE) - 1;
            hSegAggregatesExpected.insert(hSegAggregatesExpected.end(),
                                          SEGMENT_SIZE,
                                          hScanResultsExpected[segmentAggregateIndex]);
        }
        // Checks
        for(uint32_t i = 0; i < TPB; i++)
        {
            uint32_t result = hScanResults[i];
            uint32_t expected = hScanResultsExpected[i];
            EXPECT_EQ(result, expected);
        }
        for(uint32_t i = 0; i < TPB; i++)
        {
            uint32_t result = hSegmentAggregates[i];
            uint32_t expected = hSegAggregatesExpected[i];
            EXPECT_EQ(result, expected);
        }
    }
}

TEST(BlockSegmentedScan, FloatExclusiveSumBasic)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Copy all ones to GPU
    std::vector<float> data(TPB, 1.0f);

    // GPU Allocations
    float* dData;
    float* dScanOutputs;
    float* dSegmentAggregates;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dScanOutputs, dSegmentAggregates),
                                   mem, {TPB, TPB, TPB});

    CUDA_CHECK(cudaMemcpy(dData, data.data(), sizeof(float) * TPB,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dScanOutputs, 0xFF, sizeof(float) * TPB));
    CUDA_CHECK(cudaMemset(dSegmentAggregates, 0xFF, sizeof(float) * TPB));

    // Scan Call
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCSegmentedExclusiveScanTest<float>,
                      //
                      dScanOutputs,
                      dSegmentAggregates,
                      dData);

    // Copy to Host to check
    std::vector<float> hSegmentAggregates(TPB);
    std::vector<float> hScanResults(TPB);
    CUDA_CHECK(cudaMemcpy(hSegmentAggregates.data(),
                          dSegmentAggregates,
                          sizeof(float) * TPB,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hScanResults.data(),
                          dScanOutputs,
                          sizeof(float) * TPB,
                          cudaMemcpyDeviceToHost));

    // Checks
    for(float r : hSegmentAggregates)
    {
        EXPECT_FLOAT_EQ(r, static_cast<float>(SEGMENT_SIZE - 1));
    }

    for(uint32_t i = 0; i < static_cast<uint32_t>(hScanResults.size()); i++)
    {
        float r = hScanResults[i];
        uint32_t segmentLocalId = i % SEGMENT_SIZE;
        EXPECT_FLOAT_EQ(r, static_cast<float>(segmentLocalId));
    }
}

TEST(BlockSegmentedScan, FloatExclusiveSumStress)
{
    static constexpr uint32_t ITERATION_COUNT = 100;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // CPU Allocations
    std::vector<float> hData(TPB, 1.0f);
    std::vector<float> hSegmentAggregates(TPB);
    std::vector<float> hScanResults(TPB);
    std::vector<float> hSegAggregatesExpected(TPB);
    std::vector<float> hScanResultsExpected(TPB);
    // GPU Allocations
    float* dData;
    float* dScanOutputs;
    float* dSegmentAggregates;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dScanOutputs, dSegmentAggregates),
                                   mem, {TPB, TPB, TPB});


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
        CUDA_CHECK(cudaMemset(dScanOutputs, 0xFF, sizeof(float) * TPB));
        CUDA_CHECK(cudaMemset(dSegmentAggregates, 0xFF, sizeof(float) * TPB));

        // Scan Call
        const CudaGPU& bestGPU = system.BestGPU();
        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                          //
                          KCSegmentedExclusiveScanTest<float>,
                          //
                          dScanOutputs,
                          dSegmentAggregates,
                          dData);

        // Copy to Host to check
        CUDA_CHECK(cudaMemcpy(hSegmentAggregates.data(),
                              dSegmentAggregates,
                              sizeof(float) * TPB,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hScanResults.data(),
                              dScanOutputs,
                              sizeof(float) * TPB,
                              cudaMemcpyDeviceToHost));

        // Generate results by hand to check
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            uint32_t segmentStart = i * SEGMENT_SIZE;
            uint32_t nextSegmentStart = (i + 1) * SEGMENT_SIZE;

            std::exclusive_scan(hData.begin() + segmentStart,
                                hData.begin() + nextSegmentStart,
                                hScanResultsExpected.begin() + segmentStart,
                                0.0f);
        }
        hSegAggregatesExpected.clear();
        for(uint32_t i = 1; i <= SEGMENT_COUNT; i++)
        {
            uint32_t segmentAggregateIndex = (i * SEGMENT_SIZE) - 1;
            hSegAggregatesExpected.insert(hSegAggregatesExpected.end(),
                                          SEGMENT_SIZE,
                                          hScanResultsExpected[segmentAggregateIndex]);
        }
        // Checks
        for(uint32_t i = 0; i < TPB; i++)
        {
            float result = hScanResults[i];
            float expected = hScanResultsExpected[i];
            EXPECT_NEAR(result, expected, MathConstants::VeryLargeEpsilon);
        }
        for(uint32_t i = 0; i < TPB; i++)
        {
            float result = hSegmentAggregates[i];
            float expected = hSegAggregatesExpected[i];
            EXPECT_NEAR(result, expected, MathConstants::VeryLargeEpsilon);
        }
    }
}

TEST(BlockSegmentedScan, IntExclusiveSumStress)
{
    static constexpr uint32_t ITERATION_COUNT = 100;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // CPU Allocations
    std::vector<uint32_t> hData(TPB, 1u);
    std::vector<uint32_t> hSegmentAggregates(TPB);
    std::vector<uint32_t> hScanResults(TPB);
    std::vector<uint32_t> hSegAggregatesExpected(TPB);
    std::vector<uint32_t> hScanResultsExpected(TPB);
    // GPU Allocations
    uint32_t* dData;
    uint32_t* dScanOutputs;
    uint32_t* dSegmentAggregates;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData, dScanOutputs, dSegmentAggregates),
                                   mem, {TPB, TPB, TPB});


    std::mt19937 rng;
    rng.seed(0);
    std::uniform_int_distribution<uint32_t> uniformDist(0u, 10u);

    for(uint32_t i = 0; i < ITERATION_COUNT; i++)
    {
        // Generate new batch of random numbers
        for(uint32_t& d : hData)
        {
            d = uniformDist(rng);
        }

        CUDA_CHECK(cudaMemcpy(dData, hData.data(), sizeof(uint32_t) * TPB,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dScanOutputs, 0xFF, sizeof(uint32_t) * TPB));
        CUDA_CHECK(cudaMemset(dSegmentAggregates, 0xFF, sizeof(uint32_t) * TPB));

        // Scan Call
        const CudaGPU& bestGPU = system.BestGPU();
        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                          //
                          KCSegmentedExclusiveScanTest<uint32_t>,
                          //
                          dScanOutputs,
                          dSegmentAggregates,
                          dData);

        // Copy to Host to check
        CUDA_CHECK(cudaMemcpy(hSegmentAggregates.data(),
                              dSegmentAggregates,
                              sizeof(uint32_t) * TPB,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hScanResults.data(),
                              dScanOutputs,
                              sizeof(uint32_t) * TPB,
                              cudaMemcpyDeviceToHost));

        // Generate results by hand to check
        for(uint32_t i = 0; i < SEGMENT_COUNT; i++)
        {
            uint32_t segmentStart = i * SEGMENT_SIZE;
            uint32_t nextSegmentStart = (i + 1) * SEGMENT_SIZE;

            std::exclusive_scan(hData.begin() + segmentStart,
                                hData.begin() + nextSegmentStart,
                                hScanResultsExpected.begin() + segmentStart,
                                0u);
        }
        hSegAggregatesExpected.clear();
        for(uint32_t i = 1; i <= SEGMENT_COUNT; i++)
        {
            uint32_t segmentAggregateIndex = (i * SEGMENT_SIZE) - 1;
            hSegAggregatesExpected.insert(hSegAggregatesExpected.end(),
                                          SEGMENT_SIZE,
                                          hScanResultsExpected[segmentAggregateIndex]);
        }
        // Checks
        for(uint32_t i = 0; i < TPB; i++)
        {
            uint32_t result = hScanResults[i];
            uint32_t expected = hScanResultsExpected[i];
            EXPECT_EQ(result, expected);
        }
        for(uint32_t i = 0; i < TPB; i++)
        {
            uint32_t result = hSegmentAggregates[i];
            uint32_t expected = hSegAggregatesExpected[i];
            EXPECT_EQ(result, expected);
        }
    }
}