#include "RNGMemory.h"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

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
    assert(system.SystemGPUs().size() > 0);

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);

    // Determine GPU
    size_t totalCount = 0;
    std::vector<Vector2ul> ranges;
    for(const auto& gpu : system.SystemGPUs())
    {
        ranges.push_back(Vector2ul(0));
        ranges.back()[0] = totalCount;
        ranges.back()[1] = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
        totalCount += ranges.back()[1];
    }

    // Do Temp Alloc for a MT19937 seeds
    DeviceMemory seeds(totalCount * sizeof(uint32_t));

    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    system.SyncAllGPUs();
    std::for_each(static_cast<uint32_t*>(seeds),
                  static_cast<uint32_t*>(seeds) + totalCount,
                  [&](uint32_t& t) { t = rng(); });

    const uint32_t* d_seeds = static_cast<const uint32_t*>(seeds);

    // Actual Allocation
    size_t totalSize = totalCount * sizeof(curandStateMRG32k3a_t);
    memRandom = std::move(DeviceMemory(totalSize));
    curandStateMRG32k3a_t* d_ptr = static_cast<curandStateMRG32k3a_t*>(memRandom);

    size_t totalOffset = 0;
    for(const auto& gpu : system.SystemGPUs())
    {
        uint32_t gpuRNGStateCount = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
        randomStacks.emplace(&gpu, RNGGMem{d_ptr + totalOffset, gpuRNGStateCount});
        totalOffset += gpuRNGStateCount;
    }
    assert(totalCount == totalOffset);

    // Make all GPU do its own initialization
    int i = 0;
    for(const auto& gpu : system.SystemGPUs())
    {
        uint32_t localCount = ranges[i][1] - ranges[i][0];

        gpu.GridStrideKC_X(0, 0, localCount,
                           KCInitRNGStates,
                           d_seeds + ranges[i][0],
                           d_ptr + ranges[i][0],
                           localCount
                           /*ranges[i][1]*/);
        i++;
    }
}

RNGMemory::RNGMemory(uint32_t seed,
                     const CudaGPU& gpu)
{
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);
    // Determine GPU
    Vector2ul range(0, gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D);
    // Do Temp Alloc for a MT19937 seeds
    DeviceMemory seeds(range[1] * sizeof(uint32_t));
    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    std::for_each(static_cast<uint32_t*>(seeds),
                  static_cast<uint32_t*>(seeds) + range[1],
                  [&](uint32_t& t) { t = rng(); });

    const uint32_t* d_seeds = static_cast<const uint32_t*>(seeds);

    // Actual Allocation
    size_t totalSize = range[1] * sizeof(curandStateMRG32k3a_t);
    memRandom = std::move(DeviceMemory(totalSize));
    curandStateMRG32k3a_t* d_ptr = static_cast<curandStateMRG32k3a_t*>(memRandom);

    size_t totalOffset = 0;
    uint32_t gpuRNGStateCount = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
    randomStacks.emplace(&gpu, RNGGMem{d_ptr + totalOffset, gpuRNGStateCount});
    totalOffset += gpuRNGStateCount;
    assert(range[1] == totalOffset);

    // Initialize the States
    gpu.GridStrideKC_X(0, 0, range[1],
                        KCInitRNGStates,
                        d_seeds + range[0],
                        d_ptr + range[0],
                        range[1]);
}

RNGGMem RNGMemory::RNGData(const CudaGPU& gpu)
{
    return randomStacks.at(&gpu);
}