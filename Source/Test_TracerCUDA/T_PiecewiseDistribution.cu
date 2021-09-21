#include <gtest/gtest.h>

#include "TracerCUDA/GPUPiecewiseDistribution.cuh"
#include "RayLib/Log.h"
#include "TracerCUDA/RNGMemory.h"

//__global__
//void KCSamplePWCDistributions1D(float* locations,
//                                const GPUDistPiecewiseConst1D gpuDist,
//                                RNGGMem gMemory,
//                                uint32_t sampleCount)
//{
//    RandomGPU rand(gMemory, LINEAR_GLOBAL_ID);
//
//    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
//        threadId < sampleCount;
//        threadId += (blockDim.x * gridDim.x))
//    {
//        float pdf;
//        float loc = gpuDist.Sample(pdf, rand);
//        locations[threadId] = loc;
//    }
//}
//
//TEST(PiecewiseDistibution, OneDimension)
//{
//    static constexpr const uint32_t SAMPLE_COUNT = 1000;
//
//    static const std::vector<float> dist =
//    {
//        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f
//    };
//    std::vector<std::vector<float>> distList(1);
//    distList[0] = dist;
//
//    CudaSystem system;
//    CudaError e = system.Initialize();
//    ASSERT_EQ(e, CudaError::OK);
//
//    CPUDistGroupPiecewiseConst1D pwcDist1D(distList, system);
//    const GPUDistPiecewiseConst1D& gpuDist = pwcDist1D.DistributionGPU(0);
//
//    DeviceMemory samples(SAMPLE_COUNT * sizeof(float));
//
//    // Init RNG
//    RNGMemory rngMemory(0, system);
//
//    const CudaGPU& gpu = system.BestGPU();
//    gpu.GridStrideKC_X(0, (cudaStream_t)0, SAMPLE_COUNT,
//                       // Kernel
//                       KCSamplePWCDistributions1D,
//                       // Args
//                       static_cast<float*>(samples),
//                       gpuDist,
//                       SAMPLE_COUNT,
//                       rngMemory.RNGData(gpu));
//
//    system.SyncGPU(gpu);
//
//    // Check Numbers
//    float* h_data = static_cast<float*>(samples);
//    for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
//    {
//        METU_LOG("{:f}", h_data[i]);
//    }
//}
//
//TEST(PiecewiseDistibution, TwoDimension)
//{
//
//}