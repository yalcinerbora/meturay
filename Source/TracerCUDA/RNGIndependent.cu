#include "RNGIndependent.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include <random>
#include <execution>

__global__ void KCInitRNGStates(RNGIndependentGPU* dGenerators,
                                RNGeneratorGPUI** dGenPtrs,
                                const uint32_t* gSeeds,
                                uint32_t totalCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalCount;
        threadId += (blockDim.x * gridDim.x))
    {
        new (dGenerators + threadId) RNGIndependentGPU(gSeeds[threadId],
                                                       threadId);
        dGenPtrs[threadId] = dGenerators + threadId;
    }
}

RNGIndependentCPU::RNGIndependentCPU(uint32_t seed,
                                     const CudaSystem& system)
{
    assert(system.SystemGPUs().size() > 0);

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);

    // Determine GPU
    size_t totalCount = 0;
    std::vector<Vector2ul> offsetAndCounts;
    for(const auto& gpu : system.SystemGPUs())
    {
        offsetAndCounts.push_back(Vector2ul(0));
        offsetAndCounts.back()[0] = totalCount;
        offsetAndCounts.back()[1] = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
        totalCount += offsetAndCounts.back()[1];
    }

    // Do Temp Alloc for a MT19937 seeds
    DeviceMemory seeds(totalCount * sizeof(uint32_t));

    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    system.SyncAllGPUs();
    std::for_each(static_cast<uint32_t*>(seeds),
                  static_cast<uint32_t*>(seeds) + totalCount,
                  [&](uint32_t& t) { t = rng(); });
    system.SyncAllGPUs();
    const uint32_t* d_seeds = static_cast<const uint32_t*>(seeds);

    // Actual Allocation
    RNGIndependentGPU* dGenerators;
    RNGeneratorGPUI** dGenPtrs;
    GPUMemFuncs::AllocateMultiData(std::tie(dGenerators, dGenPtrs),
                                   memRandom,
                                   {totalCount, totalCount});

    size_t totalOffset = 0;
    for(const auto& gpu : system.SystemGPUs())
    {
        uint32_t gpuRNGStateCount = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
        deviceGenerators.emplace(&gpu, dGenPtrs + totalOffset);
        totalOffset += gpuRNGStateCount;
    }
    assert(totalCount == totalOffset);

    // Make all GPU do its own initialization
    int i = 0;
    for(const auto& gpu : system.SystemGPUs())
    {
        uint32_t localCount = static_cast<uint32_t>(offsetAndCounts[i][1]);

        gpu.GridStrideKC_X(0, 0, localCount,
                           //
                           KCInitRNGStates,
                           //
                           dGenerators + offsetAndCounts[i][0],
                           dGenPtrs + offsetAndCounts[i][0],
                           d_seeds + offsetAndCounts[i][0],
                           localCount);
        i++;
    }
}

RNGIndependentCPU::RNGIndependentCPU(uint32_t seed,
                                     const CudaGPU& gpu)
{
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);
    // Determine GPU
    size_t offset = 0;
    uint32_t count = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
    // Do Temp Alloc for a MT19937 seeds
    DeviceMemory seeds(count * sizeof(uint32_t));
    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    std::for_each(static_cast<uint32_t*>(seeds), static_cast<uint32_t*>(seeds) + count,
                  [&](uint32_t& t) { t = rng(); });

    const uint32_t* d_seeds = static_cast<const uint32_t*>(seeds);

    // Actual Allocation
    RNGIndependentGPU* dGenerators;
    RNGeneratorGPUI** dGenPtrs;
    GPUMemFuncs::AllocateMultiData(std::tie(dGenerators, dGenPtrs),
                                   memRandom,
                                   {count, count});

    size_t totalOffset = 0;
    uint32_t gpuRNGStateCount = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
    deviceGenerators.emplace(&gpu, dGenPtrs + totalOffset);
    totalOffset += gpuRNGStateCount;
    assert(count == static_cast<uint32_t>(totalOffset));

    // Initialize the States
    gpu.GridStrideKC_X(0, 0, count,
                       //
                       KCInitRNGStates,
                       //
                       dGenerators + offset,
                       dGenPtrs + offset,
                       d_seeds + offset,
                       count);
}

RNGIndependentCPU::RNGIndependentCPU(uint32_t seed,
                                     const CudaGPU& gpu,
                                     uint32_t rngCount)
{
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);
    // Determine GPU
    size_t offset = 0;
    // Do Temp Alloc for a MT19937 seeds
    DeviceMemory seeds(rngCount * sizeof(uint32_t));
    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    std::for_each(static_cast<uint32_t*>(seeds),
                  static_cast<uint32_t*>(seeds) + rngCount,
                  [&](uint32_t& t) { t = rng(); });

    const uint32_t* d_seeds = static_cast<const uint32_t*>(seeds);

    // Actual Allocation
    RNGIndependentGPU* dGenerators;
    RNGeneratorGPUI** dGenPtrs;
    GPUMemFuncs::AllocateMultiData(std::tie(dGenerators, dGenPtrs),
                                   memRandom,
                                   {rngCount, rngCount});

    size_t totalOffset = 0;
    uint32_t gpuRNGStateCount = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
    deviceGenerators.emplace(&gpu, dGenPtrs + totalOffset);
    totalOffset += gpuRNGStateCount;
    assert(rngCount == static_cast<uint32_t>(totalOffset));

    // Initialize the States
    gpu.GridStrideKC_X(0, 0, rngCount,
                       //
                       KCInitRNGStates,
                       //
                       dGenerators + offset,
                       dGenPtrs + offset,
                       d_seeds + offset,
                       rngCount);
}

RNGeneratorGPUI** RNGIndependentCPU::GetGPUGenerators(const CudaGPU& gpu)
{
    return deviceGenerators.at(&gpu);
}

