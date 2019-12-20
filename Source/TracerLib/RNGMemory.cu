#include "RNGMemory.h"
#include "CudaConstants.h"
#include <random>
#include <curand_kernel.h>

__global__ void KCInitRNGStates(uint32_t seed, curandStateMRG32k3a_t* state)
{
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, globalId, 0, &state[globalId]);
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
                               seed,
                               d_ptr);
}

RNGGMem RNGMemory::RNGData(uint32_t gpuId)
{
    return randomStacks[gpuId];
}