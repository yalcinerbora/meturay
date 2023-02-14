#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <numeric>

#include "TracerCUDA/AnisoSVOKC.cuh"

#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"

// TODO: Non-header file include this is bad but it is a simple
// fix (at least for template functions)


TEST(AnisoSVO, NormalReductionTest)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    static constexpr uint32_t VOXEL_COUNT = 111;
    static constexpr uint32_t UNIQUE_VOXEL_COUNT = 1;

    // Index Offsets
    std::vector<uint32_t> hVoxelIndexOffsets = {0, VOXEL_COUNT};
    // Unique voxels (all zero we have one node in the tree)
    std::vector<uint64_t> hUniqueVoxels(UNIQUE_VOXEL_COUNT, 0);
    // Sorted Voxel indices just iota it
    std::vector<uint32_t> hSortedVoxelIndices(VOXEL_COUNT);
    std::iota(hSortedVoxelIndices.begin(), hSortedVoxelIndices.end(), 0);
    // No lights
    std::vector<HitKey> hVoxelLightKeys(VOXEL_COUNT, HitKey::InvalidKey);
    // Normals (all same)
    std::vector<Vector2us> hVoxelNormals(VOXEL_COUNT, Vector2us(0, 0));
    // No light keys needed

    // Generate Data for "KCReduceVoxelPayload"
    AnisoSVOctreeGPU gpuTree;
    uint32_t* dVoxelIndexOffsets = nullptr;
    uint64_t* dUniqueVoxels = nullptr;
    uint32_t* dSortedVoxelIndices = nullptr;
    HitKey* dVoxelLightKeys = nullptr;
    Vector2us* dVoxelNormals = nullptr;
    HitKey* dLightKeys = nullptr;
    const GPULightI** dLights = nullptr;
    // Kernel Constants
    uint32_t lightCount = 0;
    uint32_t uniqueVoxCount = UNIQUE_VOXEL_COUNT;
    uint32_t lightKeyCount = VOXEL_COUNT;
    AABB3f svoAABB = 0;
    uint32_t resolutionXYZ = 2;

    DeviceMemory testMem;
    GPUMemFuncs::AllocateMultiData(std::tie(dVoxelIndexOffsets,
                                            dUniqueVoxels,
                                            dSortedVoxelIndices,
                                            dVoxelLightKeys,
                                            dVoxelNormals),
                                   testMem,
                                   {UNIQUE_VOXEL_COUNT,
                                    UNIQUE_VOXEL_COUNT,
                                    VOXEL_COUNT,
                                    VOXEL_COUNT,
                                    VOXEL_COUNT});
    CUDA_CHECK(cudaMemcpy(dVoxelIndexOffsets, hVoxelIndexOffsets.data(),
                          hVoxelIndexOffsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUniqueVoxels, hUniqueVoxels.data(),
                          hUniqueVoxels.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSortedVoxelIndices, hSortedVoxelIndices.data(),
                          hSortedVoxelIndices.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVoxelLightKeys, hVoxelLightKeys.data(),
                          hVoxelLightKeys.size() * sizeof(HitKey),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVoxelNormals, hVoxelNormals.data(),
                          hVoxelNormals.size() * sizeof(Vector2us),
                          cudaMemcpyHostToDevice));

    // Call the Kernel
    const CudaGPU& gpu = system.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, uniqueVoxCount * WARP_SIZE,
                       //
                       KCReduceVoxelPayload<StaticThreadPerBlock1D>,
                       //
                       gpuTree,
                       dVoxelIndexOffsets,
                       dUniqueVoxels,
                       dSortedVoxelIndices,
                       dVoxelLightKeys,
                       dVoxelNormals,
                       dLightKeys,
                       dLights,
                       lightCount,
                       false,
                       uniqueVoxCount,
                       lightKeyCount,
                       svoAABB,
                       resolutionXYZ);
}