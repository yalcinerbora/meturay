#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>

#include "RayLib/Log.h"

#include "TracerCUDA/RNGIndependent.cuh"
#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"

__global__ void KInitRandStates(uint32_t seed,
                                curandStateMRG32k3a_t* states)
{
    curand_init(seed, threadIdx.x, 0, states);
}

__global__ void KCRandomNumbers(RNGeneratorGPUI** gRNGs,
                                uint32_t* randomNumbers,
                                size_t totalNumberCount)
{
    auto& rng = RNGAccessor::Acquire<RNGIndependentGPU>(gRNGs, LINEAR_GLOBAL_ID);

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalNumberCount; globalId += blockDim.x * gridDim.x)
    {
        uint32_t r = rng.Uniform();
        randomNumbers[globalId] = r;
    }
}

TEST(RandomGPU, All)
{
    static constexpr uint32_t Seed = 0;
    static constexpr size_t NumberCount = 5'000'000;
    static constexpr  size_t NumberSize = NumberCount * sizeof(uint32_t);
    DeviceMemory numbers(NumberSize);

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    const CudaGPU& gpu = system.BestGPU();

    RNGIndependentCPU rngCPU(Seed, system);

    // Kernel Call
    uint32_t* h_data = static_cast<uint32_t*>(numbers);
    gpu.GridStrideKC_X(0, 0, NumberCount,
                       //
                       KCRandomNumbers,
                       //
                       rngCPU.GetGPUGenerators(gpu),
                       h_data,
                       static_cast<uint32_t>(NumberCount));
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO: Do some checking????
}