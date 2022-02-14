#include <gtest/gtest.h>
#include <numeric>

#include "TracerCUDA/GPUPiecewiseDistribution.cuh"
#include "RayLib/Log.h"
#include "TracerCUDA/RNGIndependent.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"

template <class RNG>
__global__
void KCSamplePWCStaticDist2D(float* gPdfs,
                             Vector2f* gLocations,
                             const PWCDistStaticGPU2D dist,
                             RNGeneratorGPUI** gRNGs,
                             uint32_t sampleCount)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);

    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < sampleCount; globalId += blockDim.x * gridDim.x)
    {
        float xi = rng.Uniform(0.0f, static_cast<float>(sampleCount));
        uint32_t distIndex = static_cast<uint32_t>(xi);

        float pdf;
        Vector2f index;
        Vector2f uv = dist.Sample(pdf, index, rng, distIndex);
        gLocations[globalId] = uv;
        gPdfs[globalId] = pdf;
    }
}

template <class RNG>
__global__
void KCSamplePWCDistributions1D(float* locations,
                                const PWCDistributionGPU1D gpuDist,
                                RNGeneratorGPUI** gRNGs,
                                uint32_t sampleCount)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < sampleCount;
        threadId += (blockDim.x * gridDim.x))
    {
        float pdf, index;
        float loc = gpuDist.Sample(pdf, index, rng);
        locations[threadId] = loc;
    }
}

TEST(PiecewiseDistibution, OneDimension)
{
    static constexpr const uint32_t SAMPLE_COUNT = 1'000'000;

    std::vector<std::vector<float>> distList(1);
    distList[0] = std::vector<float>(64, 1.0f);

    CudaSystem system;
    CudaError e = system.Initialize();
    ASSERT_EQ(e, CudaError::OK);

    PWCDistributionGroupCPU1D pwcDist1D(distList, system);
    const PWCDistributionGPU1D& gpuDist = pwcDist1D.DistributionGPU(0);

    DeviceMemory samples(SAMPLE_COUNT * sizeof(float));

    // Init RNG
    RNGIndependentCPU rngIndep(0, system);

    const CudaGPU& gpu = system.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, SAMPLE_COUNT,
                       // Kernel
                       KCSamplePWCDistributions1D<RNGIndependentGPU>,
                       // Args
                       static_cast<float*>(samples),
                       gpuDist,
                       rngIndep.GetGPUGenerators(gpu),
                       SAMPLE_COUNT);

    gpu.WaitMainStream();


    // Unified Memory, synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    // Check Numbers
    float* h_data = static_cast<float*>(samples);

    float val = std::reduce(h_data, h_data + SAMPLE_COUNT, 0.0f);
    val *= (1.0f / static_cast<float>(SAMPLE_COUNT));
    EXPECT_NEAR(val, 0.5f, MathConstants::VeryLargeEpsilon);
}

//TEST(PiecewiseDistibution, TwoDimension)
//{
//
//}

TEST(PiecewiseStaticDistribution, TwoDimension)
{
    static constexpr Vector2ui DIMENSION = Vector2ui(64, 64);
    static constexpr uint32_t DIST_COUNT = 32;
    static constexpr uint32_t SAMPLE_COUNT = 1'000'000;
    CudaSystem system;
    CudaError err = system.Initialize();
    ASSERT_EQ(err, CudaError::OK);

    RNGIndependentCPU rngIndep(0, system);
    std::vector<float> data(DIMENSION.Multiply() * DIST_COUNT, 1.0f);
    PWCDistStaticCPU2D dist(data, DIST_COUNT, DIMENSION, false, system);

    DeviceMemory mem;
    Vector2f* dSamples;
    float* dPdfs;
    GPUMemFuncs::AllocateMultiData(std::tie(dSamples, dPdfs), mem,
                                   {SAMPLE_COUNT, SAMPLE_COUNT});

    const CudaGPU& gpu = system.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, SAMPLE_COUNT,
                       //
                       KCSamplePWCStaticDist2D<RNGIndependentGPU>,
                       //
                       dPdfs,
                       dSamples,
                       dist.DistributionGPU(),
                       rngIndep.GetGPUGenerators(gpu),
                       SAMPLE_COUNT);

    //
    std::vector<float> hPdfs(SAMPLE_COUNT);
    std::vector<Vector2f> hSamples(SAMPLE_COUNT);
    CUDA_CHECK(cudaMemcpy(hSamples.data(), dSamples,
                          sizeof(Vector2f) * SAMPLE_COUNT,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPdfs.data(), dPdfs,
                          sizeof(float) * SAMPLE_COUNT,
                          cudaMemcpyDeviceToHost));

    float pdf = std::reduce(hPdfs.cbegin(), hPdfs.cend(), 0.0f);
    Vector2f value = std::reduce(hSamples.cbegin(), hSamples.cend(), Zero2f);
    value *= (1.0f / static_cast<float>(SAMPLE_COUNT));
    pdf *= (1.0f / static_cast<float>(SAMPLE_COUNT));

    EXPECT_FLOAT_EQ(pdf, 1.0f);
    EXPECT_NEAR(value[0], 0.5f, MathConstants::VeryLargeEpsilon);
    EXPECT_NEAR(value[1], 0.5f, MathConstants::VeryLargeEpsilon);
}