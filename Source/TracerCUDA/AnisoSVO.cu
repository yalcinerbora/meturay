#include "AnisoSVO.cuh"

#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "PathNode.cuh"

#include "RayLib/ColorConversion.h"
#include "RayLib/HitStructs.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/CPUTimer.h"

#include "GPUAcceleratorI.h"
#include "ParallelReduction.cuh"
#include "ParallelScan.cuh"

#include <cub/cub.cuh>
#include <numeric>

#include "TracerDebug.h"

__global__
void KCGetLightKeys(HitKey* gKeys,
                    const GPULightI** gLights,
                    uint32_t totalLightCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalLightCount;
        threadId += (blockDim.x * gridDim.x))
    {
        gKeys[threadId] = gLights[threadId]->WorkKey();
    }
}


__global__
void KCMarkMortonChanges(uint32_t* gMarks,
                         const uint64_t* gVoxels,
                         uint32_t voxelCount,
                         uint32_t level,
                         uint32_t maxLevel)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < voxelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        static constexpr uint64_t DIMENSION = 3;
        // Check 3*level bits from the MSB to find the voxel counts
        uint32_t bitCount = level * DIMENSION;
        uint32_t levelStart = (maxLevel * DIMENSION) - bitCount;
        uint64_t mask = (1ull << static_cast<uint64_t>(bitCount)) - 1;

        uint64_t voxelMorton = gVoxels[threadId];
        uint64_t voxelMortonNext = gVoxels[threadId + 1];

        voxelMorton >>= levelStart;
        voxelMorton &= mask;

        voxelMortonNext >>= levelStart;
        voxelMortonNext &= mask;

        gMarks[threadId] = (voxelMorton != voxelMortonNext) ? 1 : 0;
        if(threadId == voxelCount - 1)
            gMarks[threadId + 1] = 0;
    }
}

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

TracerError AnisoSVOctreeCPU::Constrcut(const AABB3f& sceneAABB, uint32_t resolutionXYZ,
                                        const AcceleratorBatchMap& accels,
                                        const GPULightI** dSceneLights,
                                        uint32_t totalLightCount,
                                        HitKey boundaryLightKey,
                                        const CudaSystem& system)
{
    Utility::CPUTimer timer;
    timer.Start();

    // Find The SVO AABB
    Vector3f span = sceneAABB.Span();
    int maxDimIndex = span.Max();
    float worldSizeXYZ = span[maxDimIndex];
    treeGPU.svoAABB = AABB3f(sceneAABB.Min(),
                             sceneAABB.Min() + Vector3f(worldSizeXYZ));
    treeGPU.leafDepth = Utility::FindLastSet(resolutionXYZ);
    treeGPU.leafVoxelSize = worldSizeXYZ / static_cast<float>(resolutionXYZ);
    treeGPU.voxelResolution = resolutionXYZ;


    size_t lightSortMemSize;
    HitKey* dLightKeys = nullptr;
    const GPULightI** dSortedLights = nullptr;
    HitKey* dSortedLightKeys = nullptr;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, lightSortMemSize,
                                               reinterpret_cast<HitKey::Type*>(dLightKeys),
                                               reinterpret_cast<HitKey::Type*>(dSortedLightKeys),
                                               dSceneLights, dSortedLights,
                                               totalLightCount));

    DeviceMemory lightMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dLightKeys, dSortedLights,
                                            dSortedLightKeys),
                                   lightMemory,
                                   {totalLightCount, totalLightCount,
                                    totalLightCount});

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
    Byte* dLightSortTempMem;
    DeviceMemory voxOffsetMem;
    GPUMemFuncs::AllocateMultiData(std::tie(dVoxelCounts,
                                            dVoxelOffsets,
                                            dPrimOffsets,
                                            dLightSortTempMem),
                                   voxOffsetMem,
                                   {primOffsets.back(),
                                    primOffsets.back() + 1,
                                    accels.size() + 1,
                                    lightSortMemSize});

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
                                  treeGPU.svoAABB,
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

    // Generate Light / HitKey sorted array (for binary search)
    const CudaGPU& gpu = system.BestGPU();
    if(totalLightCount != 0)
    {
        gpu.GridStrideKC_X(0, (cudaStream_t)0,
                           totalLightCount,
                           //
                           KCGetLightKeys,
                           //
                           dLightKeys,
                           dSceneLights,
                           totalLightCount);
        // Sort these for binary search
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dLightSortTempMem, lightSortMemSize,
                                                   reinterpret_cast<HitKey::Type*>(dLightKeys),
                                                   reinterpret_cast<HitKey::Type*>(dSortedLightKeys),
                                                   dSceneLights, dSortedLights,
                                                   totalLightCount));

    }

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
                                dSortedLightKeys,
                                totalLightCount,
                                // Constants
                                resolutionXYZ,
                                treeGPU.svoAABB,
                                system);
        i++;
    }

    // Temporary Data Structures are not needed from now on
    // Deallocate
    voxOffsetMem = DeviceMemory();
    dVoxelCounts = nullptr;
    dVoxelOffsets = nullptr;
    dPrimOffsets = nullptr;
    dLightSortTempMem = nullptr;

    //
    size_t rleTempMemSize;
    size_t sortTempMemSize;

    uint64_t* dSortedVoxels = nullptr;
    HitKey* dSortedVoxelKeys = nullptr;
    // Duplicate counts
    uint32_t* dDuplicateCounts = nullptr;
    uint32_t* dUniqueVoxelCount = nullptr;
    Byte* dTempMemory = nullptr;

    // Acquire Temp Memory Requirements
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortTempMemSize,
                                               dVoxels, dSortedVoxels,
                                               dVoxelLightKeys, dSortedVoxelKeys,
                                               static_cast<uint32_t>(hTotalVoxCount)));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                  rleTempMemSize,
                                                  dSortedVoxels, dVoxels,
                                                  dDuplicateCounts, dUniqueVoxelCount,
                                                  static_cast<uint32_t>(hTotalVoxCount)));
    size_t tempMemSize = std::max(rleTempMemSize, sortTempMemSize);

    // Allocation
    DeviceMemory sortedVoxelMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dSortedVoxels, dSortedVoxelKeys,
                                            dDuplicateCounts, dTempMemory,
                                            dUniqueVoxelCount),
                                   sortedVoxelMemory,
                                   {hTotalVoxCount, hTotalVoxCount,
                                   hTotalVoxCount, tempMemSize,
                                   1});

    // Sort and RLE
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTempMemory, sortTempMemSize,
                                               dVoxels, dSortedVoxels,
                                               dVoxelLightKeys, dSortedVoxelKeys,
                                               static_cast<uint32_t>(hTotalVoxCount)));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(dTempMemory,
                                                  rleTempMemSize,
                                                  dSortedVoxels, dVoxels,
                                                  dDuplicateCounts, dUniqueVoxelCount,
                                                  static_cast<uint32_t>(hTotalVoxCount)));


    // Load the found unique voxel count to host memory for kernel calls
    uint32_t hUniqueVoxelCount;
    CUDA_CHECK(cudaMemcpy(&hUniqueVoxelCount, dUniqueVoxelCount, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    assert(hUniqueVoxelCount <= hTotalVoxCount);

    // Voxel are sorted and RLE is run
    // Rename the dVoxels array to sorted unique voxels
    uint64_t* dSortedUniqueVoxels = dVoxels;
    // Rename voxel light keys buffer to difference buffer (you can?)
    uint32_t* dDiffBitBuffer = reinterpret_cast<uint32_t*>(dVoxelLightKeys);
    // Top-down find the required voxel counts by looking the morton codes
    assert(Utility::BitCount(resolutionXYZ) == 1);
    uint32_t levelCount = Utility::FindLastSet(resolutionXYZ);
    std::vector<uint32_t> levelNodeCounts(levelCount + 1, 0);

    // Root node is always available
    levelNodeCounts[0] = 1;
    for(uint32_t i = 1; i <= levelCount; i++)
    {
        // Mark the differences between neighbors
        gpu.GridStrideKC_X(0, (cudaStream_t)0, hUniqueVoxelCount,
                           //
                           KCMarkMortonChanges,
                           //
                           dDiffBitBuffer,
                           dSortedUniqueVoxels,
                           hUniqueVoxelCount - 1,
                           i,
                           levelCount);

        // Reduce the marks to find
        ReduceArrayGPU<uint32_t, ReduceAdd<uint32_t>, cudaMemcpyDeviceToHost>
        (
            levelNodeCounts[i],
            dDiffBitBuffer,
            hUniqueVoxelCount,
            0u
        );
        // n difference slices means n+1 segments
        gpu.WaitMainStream();
        levelNodeCounts[i] += 1;
    }
    assert(levelNodeCounts.back() == hUniqueVoxelCount);

    uint32_t totalNodeCount = std::reduce(levelNodeCounts.cbegin(),
                                          levelNodeCounts.cend() - 1,
                                          0u);

    treeGPU.nodeCount = totalNodeCount;
    treeGPU.leafCount = hUniqueVoxelCount;

    // Allocate required memories now
    // since we found out the total node count
    GPUMemFuncs::AllocateMultiData(std::tie(// Node Related,
                                            treeGPU.dNodes,
                                            treeGPU.dRadiance,
                                            treeGPU.dRayCounts,
                                            // Leaf Related
                                            treeGPU.dLeafParents,
                                            treeGPU.dLeafRadianceRead,
                                            treeGPU.dLeafRayCounts,
                                            treeGPU.dLeafRadianceWrite,
                                            treeGPU.dLeafSampleCountWrite),
                                   octreeMem,
                                   {totalNodeCount, totalNodeCount,
                                   totalNodeCount,
                                   hUniqueVoxelCount, hUniqueVoxelCount,
                                   hUniqueVoxelCount, hUniqueVoxelCount,
                                   hUniqueVoxelCount});

    // Top down-generate voxels
    // For each level save the node range for
    // efficient kernel calls later (level by level kernel calls)
    // Now start voxel generation level by level
    for(uint32_t i = 0; i < levelCount; i++)
    {

    }

    // Scan the reduce counts to find the light index offsets




    // Log some stuff
    timer.Stop();
    double svoMemSize = static_cast<double>(octreeMem.Size()) / 1024.0 / 1024.0f;
    METU_LOG("Scene Aniso-SVO Generated in {:f} seconds. ({:f} MiB)",
             timer.Elapsed<CPUTimeSeconds>(), svoMemSize);
    return TracerError::OK;
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