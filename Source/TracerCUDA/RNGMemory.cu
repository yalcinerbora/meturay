#include "RNGMemory.h"
#include "CudaConstants.h"
#include "CudaConstants.hpp"

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

RNGMemory::RNGMemory(uint32_t seed,
                     const CudaSystem& system)
{
    assert(system.GPUList().size() > 0);

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);

    // Determine GPU
    size_t totalCount = 0;
    std::vector<Vector2ul> ranges;
    for(const auto& gpu : system.GPUList())
    {
        ranges.push_back(Vector2ul(0));
        ranges.back()[0] = totalCount;
        ranges.back()[1] = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
        totalCount += ranges.back()[1];
    }

    // Do Temp Alloc for a MT19937 seeds
    // ang generate
    DeviceMemory seeds(totalCount * sizeof(uint32_t));

    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    system.SyncGPUAll();
    std::for_each(static_cast<uint32_t*>(seeds),
                  static_cast<uint32_t*>(seeds) + totalCount,
                  [&](uint32_t& t) { t = rng(); });

    const uint32_t* d_seeds = static_cast<const uint32_t*>(seeds);

    // Actual Allocation
    size_t totalSize = totalCount * sizeof(curandStateMRG32k3a_t);
    memRandom = std::move(DeviceMemory(totalSize));
    curandStateMRG32k3a_t* d_ptr = static_cast<curandStateMRG32k3a_t*>(memRandom);

    size_t totalOffset = 0;
    for(const auto& gpu : system.GPUList())
    {
        uint32_t gpuRNGStateCount = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
        randomStacks.emplace(&gpu, RNGGMem{d_ptr + totalOffset, gpuRNGStateCount});
        totalOffset += gpuRNGStateCount;
    }
    assert(totalCount == totalOffset);

    // Make all GPU do its own
    int i = 0;
    for(const auto& gpu : system.GPUList())
    {
        gpu.GridStrideKC_X(0, 0, totalCount,
                           KCInitRNGStates,
                           d_seeds + ranges[i][0],
                           d_ptr + ranges[i][0],
                           ranges[i][1]);
        i++;
    }
}

RNGGMem RNGMemory::RNGData(const CudaGPU& gpu)
{
    return randomStacks.at(&gpu);
}