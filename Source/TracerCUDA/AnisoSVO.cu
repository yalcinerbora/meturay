#include "AnisoSVO.cuh"

#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "PathNode.cuh"

#include "RayLib/ColorConversion.h"
#include "RayLib/HitStructs.h"

#include "GPUAcceleratorI.h"
#include "ParallelReduction.cuh"
#include "ParallelScan.cuh"

#include <cub/cub.cuh>
#include <numeric>

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCAccumulateRadianceToLeaf(AnisoSVOctreeGPU svo,
                                // Input
                                const PathGuidingNode* gPathNodes,
                                uint32_t nodeCount,
                                uint32_t maxPathNodePerRay)
{
    bool unableToAccum = false;

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < nodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        const uint32_t nodeIndex = threadId;
        const uint32_t pathStartIndex = nodeIndex / maxPathNodePerRay * maxPathNodePerRay;

        PathGuidingNode gPathNode = gPathNodes[nodeIndex];

        // Skip if this node cannot calculate wo
        if(!gPathNode.HasPrev()) continue;

        Vector3f wo = gPathNode.Wo<PathGuidingNode>(gPathNodes, pathStartIndex);
        float luminance = Utility::RGBToLuminance(gPathNode.totalRadiance);
        unableToAccum |= !svo.DepositRadiance(gPathNode.worldPosition, wo, luminance);
    }

    // Debug
    if(unableToAccum)
    {
        printf("Unable to accumulate radiance!");
    }
}

void AnisoSVOctreeCPU::Constrcut(const AABB3f& sceneAABB, uint32_t resolutionXYZ,
                                 const AcceleratorBatchMap& accels,
                                 const GPULightI** dSceneLights,
                                 uint32_t totalLightCount,
                                 const CudaSystem& system)
{
    // Generate Light / HitKey sorted array (for binary search)
    HitKey* dLightKeys;





    // For each accelerator
    // First allocate per prim voxel count array
    std::vector<size_t> primOffsets;
    primOffsets.reserve(accels.size() + 1);
    primOffsets.push_back(0);
    for(const auto [_, accel] : accels)
    {
        primOffsets.push_back(accel->TotalPrimitiveCount());
    }

    std::inclusive_scan(primOffsets.cbegin(), primOffsets.cend(),
                        primOffsets.begin());

    // Allocate Voxel Count memory (which will be used to allocate)
    uint64_t* dVoxelCounts;
    uint64_t* dVoxelOffsets;
    uint64_t* dPrimOffsets;
    DeviceMemory voxOffsetMem;
    GPUMemFuncs::AllocateMultiData(std::tie(dVoxelCounts,
                                            dVoxelOffsets,
                                            dPrimOffsets),
                                   voxOffsetMem,
                                   {primOffsets.back(),
                                    primOffsets.back() + 1,
                                    accels.size() + 1});

    // Copy prim offsets for segmented reduction
    CUDA_CHECK(cudaMemcpy(dPrimOffsets, primOffsets.data(),
                          sizeof(uint64_t) * accels.size(),
                          cudaMemcpyHostToDevice));

    // Ask each primitive for rasterize voxel count
    uint32_t i = 0;
    for(const auto [_, accel] : accels)
    {
        accel->EachPrimVoxelCount(dVoxelCounts + primOffsets[i],
                                  resolutionXYZ,
                                  sceneAABB,
                                  system);
        i++;
    }
    // Find Global Voxel Offsets
    ExclusiveScanArrayGPU<uint64_t, ReduceAdd<uint64_t>>(dVoxelOffsets,
                                                         dVoxelCounts,
                                                         primOffsets.back() + 1,
                                                         0u);

    // Reduce Per prim voxel count to total voxel count
    uint64_t hTotalVoxCount;
    CUDA_CHECK(cudaMemcpy(&hTotalVoxCount, dVoxelOffsets + primOffsets.back(),
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Allocate enough memory for temp voxels (these may overlap)
    uint64_t* dVoxels;
    HitKey* dVoxelLightKeys;
    DeviceMemory voxelMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dVoxels, dVoxelLightKeys),
                                   voxelMemory,
                                   {hTotalVoxCount, hTotalVoxCount});
    // For each accelerator
    // Actually rasterize the primitives
    // and push to the memory (find the light key if available here)
    i = 0;
    for(const auto [_, accel] : accels)
    {
        accel->VoxelizeSurfaces(// Outputs
                                dVoxels,
                                dVoxelLightKeys,
                                // Inputs
                                dVoxelOffsets + primOffsets[i],
                                // Light Lookup Table (Binary Search)
                                dLightKeys,
                                totalLightCount,
                                // Constants
                                resolutionXYZ,
                                sceneAABB,
                                system);
        i++;
    }

    // Temporary Data Structures are not needed from now on
    // Deallocate
    voxOffsetMem = DeviceMemory();
    //
    DeviceMemory tempMem;
    size_t rleTempMemSize;
    size_t sortTempMemSize;

    uint64_t* dSortedVoxels = nullptr;
    HitKey* dSortedLightKeys = nullptr;
    // Duplicate counts
    uint32_t* dDuplicateCounts = nullptr;
    uint64_t* dReducedCount = nullptr;
    Byte* dTempMemory = nullptr;

    // Acquire Temp Memory Requirements
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortTempMemSize,
                                               dVoxels, dSortedVoxels,
                                               dVoxelLightKeys, dSortedVoxelKeys,
                                               hTotalVoxCount));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                  rleTempMemSize,
                                                  dSortedVoxels, dVoxels,
                                                  dDuplicateCounts, dReducedCount,
                                                  hTotalVoxCount));
    size_t tempMemSize = std::max(rleTempMemSize, sortTempMemSize);

    // Allocation
    DeviceMemory reduceMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dSortedVoxels, dSortedLightKeys,
                                            dDuplicateCounts, dTempMemory,
                                            dReducedCount),
                                   reduceMemory,
                                   {hTotalVoxCount, hTotalVoxCount,
                                   hTotalVoxCount, tempMemSize,
                                   1});

    //
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTempMemory, sortTempMemSize,
                                               dVoxels, dSortedVoxels,
                                               dVoxelLightKeys, dSortedVoxelKeys,
                                               hTotalVoxCount));
    CUDA_CHECKcub::DeviceRunLengthEncode::Encode(dTempMemory,
                                                 rleTempMemSize,
                                                 dSortedVoxels, dVoxels,
                                                 dDuplicateCounts, dReducedCount,
                                                 hTotalVoxCount));

    // Top down-generate the voxels
    // For each level save the node range for
    // efficient kernel calls later (level by level kernel calls)
    // Now start voxel generation level by level
    uint32_t levelCount = std::log2(resolutionXYZ);
    for(uint32_t i = 0; i < levelCount; i++)
    {

    }
}

void AnisoSVOctreeCPU::NormalizeAndFilterRadiance(const CudaSystem&)
{
    // From leaf (leaf-write) to root
    // Average the radiance

    // Down-sample the radiance for lowest n levels as well maybe? (n= 2 or 3)
}

void AnisoSVOctreeCPU::CollapseRayCounts(uint32_t minLevel, uint32_t minRayCount,
                                         const CudaSystem& system)
{
    // Assume that the ray counts are set for leaves
    const CudaGPU& bestGPU = system.BestGPU();

    //// Leaf is implicit do it separately
    //Vector2ui leafRange(0, treeGPU.leafCount);
    //bestGPU.GridStrideKC_X(0, (cudaStream_t)0,
    //                       leafRange[1],

    //                       //
    //                       );

    //for(levelRanges)

    // From leaf to root
    // Accumulate children ray counts on the node
    // If node has enough rays (has more than minRayCount rays)
    // or node is on not on a certain level collapse the rays
    // stop collapsing
}

void AnisoSVOctreeCPU::AccumulateRaidances(const PathGuidingNode* dPGNodes,
                                           uint32_t totalNodeCount,
                                           uint32_t maxPathNodePerRay,
                                           const CudaSystem& system)
{
    // Directly call the appropriate kernel
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                           //
                           KCAccumulateRadianceToLeaf,
                           //
                           treeGPU,
                           dPGNodes,
                           totalNodeCount,
                           maxPathNodePerRay);
    bestGPU.WaitMainStream();
}

void AnisoSVOctreeCPU::ClearRayCounts(const CudaSystem&)
{
    CUDA_CHECK(cudaMemset(treeGPU.dLeafRayCounts, 0x00, sizeof(uint32_t) * treeGPU.leafCount));
    CUDA_CHECK(cudaMemset(treeGPU.dRayCounts, 0x00, sizeof(uint32_t) * treeGPU.nodeCount));
}