#include "RNGMemory.h"
#include "CudaConstants.h"

#include <random>
#include <curand_kernel.h>
#include <execution>

__global__ void KCInitRNGStates(const uint32_t* gSeeds, curandStateMRG32k3a_t* gStates,
                                size_t totalCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalCount;
        threadId += (blockDim.x * gridDim.x))
    {
        curand_init(gSeeds[threadId], threadId, 0, &gStates[threadId]);
    }
}

RNGMemory::RNGMemory(uint32_t seed)
{
    assert(CudaSystem::GPUList().size() > 0);

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);

    // Determine GPU
    size_t totalCount = 0;
    for(const auto& gpu : CudaSystem::GPUList())
    {
        totalCount += BlocksPerSM * gpu.SMCount() * StaticThreadPerBlock1D;
    }

    // Do Temp Alloc for a MT19937 seeds
    // ang generate
    DeviceMemory seeds(totalCount * sizeof(uint32_t));
    std::for_each(static_cast<uint32_t*>(seeds), 
                  static_cast<uint32_t*>(seeds) + totalCount,
                  [&](uint32_t& t) { t = rng(); });

    const uint32_t* d_seeds = static_cast<const uint32_t*>(seeds);

    // Actual Allocation
    size_t totalSize = totalCount * sizeof(curandStateMRG32k3a_t);
    memRandom = std::move(DeviceMemory(totalSize));
    curandStateMRG32k3a_t* d_ptr = static_cast<curandStateMRG32k3a_t*>(memRandom);

    size_t totalOffset = 0;
    for(const auto& gpu : CudaSystem::GPUList())
    {
        randomStacks.emplace_back(RNGGMem{d_ptr + totalOffset});
        totalOffset += BlocksPerSM * gpu.SMCount() * StaticThreadPerBlock1D;
    }
    assert(totalCount == totalOffset);

    CudaSystem::GridStrideKC_X(0, 0, 0, totalCount,
                               //
                               KCInitRNGStates,
                               d_seeds,
                               d_ptr,
                               totalCount);
}

RNGGMem RNGMemory::RNGData(uint32_t gpuId)
{
    return randomStacks[gpuId];
}