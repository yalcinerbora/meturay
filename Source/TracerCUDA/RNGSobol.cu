#include "RNGSobol.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include <random>
#include <execution>

__global__ void KCInitRNGStatesSobol(RNGSobolGPU* dGenerators,
                                     RNGeneratorGPUI** dGenPtrs,
                                     curandDirectionVectors32_t* gDirectionVectors,
                                     const uint32_t* gOffsets,
                                     const uint32_t* gScrambleConsts,
                                     uint32_t totalCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalCount;
        threadId += (blockDim.x * gridDim.x))
    {
        new (dGenerators + threadId) RNGSobolGPU(gDirectionVectors[threadId],
                                                 gOffsets[threadId],
                                                 gScrambleConsts[threadId]);
        dGenPtrs[threadId] = dGenerators + threadId;
    }
}

RNGSobolCPU::RNGSobolCPU(uint32_t seed,
                         const CudaSystem& system)
{
    // RNG for seeding each thread in the gpu(s)
    std::mt19937 rng;
    rng.seed(seed);

    static constexpr uint32_t VECTOR_PER_THREAD = 32;
    // Determine GPU Sizes and Offsets
    size_t totalCount = 0;
    std::vector<Vector2ul> offsetAndCounts;
    for(const auto& gpu : system.SystemGPUs())
    {
        offsetAndCounts.push_back(Vector2ul(0));
        offsetAndCounts.back()[0] = totalCount;
        offsetAndCounts.back()[1] = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
        totalCount += offsetAndCounts.back()[1];
    }

    // Get Directions Vectors & Scramble Constants from
    // CPU API
    uint32_t* hScrambleConstants;
    curandDirectionVectors32_t* hDirectionVectors;
    curandStatus_t s = curandGetDirectionVectors32(&hDirectionVectors,
                                                   CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    if(s != CURAND_STATUS_SUCCESS) assert(false);
    s = curandGetScrambleConstants32(&hScrambleConstants);
    if(s != CURAND_STATUS_SUCCESS) assert(false);

    // Copy to temp memory
    uint32_t* dOffsets;
    uint32_t* dScrambleConstants;
    curandDirectionVectors32_t* dDirectionVectors;
    DeviceMemory tempMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dOffsets, dScrambleConstants,
                                            dDirectionVectors),
                                   tempMemory,
                                   {totalCount, totalCount,
                                   VECTOR_PER_THREAD * totalCount});

    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    system.SyncAllGPUs();
    std::for_each(dOffsets, dOffsets + totalCount,
                  [&](uint32_t& t) { t = rng(); });

    // Rest is copied from host
    CUDA_CHECK(cudaMemcpy(dScrambleConstants, hScrambleConstants,
                          totalCount * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDirectionVectors, hDirectionVectors,
                          VECTOR_PER_THREAD * totalCount * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Allocate Actual State Data
    RNGSobolGPU* dGenerators;
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
                           KCInitRNGStatesSobol,
                           //
                           dGenerators + offsetAndCounts[i][0],
                           dGenPtrs + offsetAndCounts[i][0],
                           dDirectionVectors + offsetAndCounts[i][0],
                           dOffsets + offsetAndCounts[i][0],
                           dScrambleConstants + offsetAndCounts[i][0],
                           localCount);
        i++;
    }
    // All Done!
}

RNGSobolCPU::RNGSobolCPU(uint32_t seed,
                         const CudaGPU& gpu)
{
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    static constexpr uint32_t VECTOR_PER_THREAD = 32;

    // CPU Mersenne Twister
    std::mt19937 rng;
    rng.seed(seed);
    // Determine GPU
    size_t offset = 0;
    uint32_t count = gpu.MaxActiveBlockPerSM() * gpu.SMCount() * StaticThreadPerBlock1D;
    // Get Directions Vectors & Scramble Constants from
    // CPU API
    uint32_t* hScrambleConstants;
    curandDirectionVectors32_t* hDirectionVectors;
    curandStatus_t s = curandGetDirectionVectors32(&hDirectionVectors,
                                                   CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
    if(s != CURAND_STATUS_SUCCESS) assert(false);
    s = curandGetScrambleConstants32(&hScrambleConstants);
    if(s != CURAND_STATUS_SUCCESS) assert(false);
    // Copy to temp memory
    uint32_t* dOffsets;
    uint32_t* dScrambleConstants;
    curandDirectionVectors32_t* dDirectionVectors;
    DeviceMemory tempMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dOffsets, dScrambleConstants,
                                            dDirectionVectors),
                                   tempMemory,
                                   {count, count,
                                   VECTOR_PER_THREAD * count});
    // Before touching gpu mem from cpu do a sync
    // since other initialization probably launched a kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    std::for_each(dOffsets, dOffsets + count,
                  [&](uint32_t& t) { t = rng(); });
    // Rest is copied from host
    CUDA_CHECK(cudaMemcpy(dScrambleConstants, hScrambleConstants,
                          count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDirectionVectors, hDirectionVectors,
                          VECTOR_PER_THREAD * count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    // Actual Allocation
    RNGSobolGPU* dGenerators;
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
                       KCInitRNGStatesSobol,
                       //
                       dGenerators + offset,
                       dGenPtrs + offset,
                       dDirectionVectors + offset,
                       dOffsets + offset,
                       dScrambleConstants + offset,
                       count);
}

RNGeneratorGPUI** RNGSobolCPU::GetGPUGenerators(const CudaGPU& gpu)
{
    return deviceGenerators.at(&gpu);
}