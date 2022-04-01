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
#include "ParallelMemset.cuh"

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


__global__ CUDA_LAUNCH_BOUNDS_1D
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
    }
}

__global__
void KCMarkChild(// I-O
                 uint64_t* gNodes,
                 // Input
                 const uint64_t* gVoxels,
                 // Constants
                 uint32_t voxelCount,
                 uint32_t level,
                 uint32_t maxLevel)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < voxelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        static constexpr uint64_t DIMENSION = 3;
        uint64_t voxelMortonCode = gVoxels[threadId];

        // Start traversing (Root node is the very first node)
        uint64_t* currentNode = gNodes;
        for(int curLevel = 0; curLevel < level; curLevel++)
        {
            // Fetch the current bit triples of the level from the
            // morton code
            uint32_t currentBitOffset = (maxLevel - curLevel - 1) * DIMENSION;
            uint32_t childId = (voxelMortonCode >> currentBitOffset) & 0b111;

            uint8_t childOffset = AnisoSVOctreeGPU::FindChildOffset(*currentNode, childId);
            uint8_t childrenIndex = AnisoSVOctreeGPU::ChildrenIndex(*currentNode);

            // Go to next child
            currentNode = gNodes + childrenIndex + childOffset;
        }
        // Now we are at the not that does not set its children ptr and mask is set
        // Atomically mark the required child
        uint32_t currentBitOffset = (maxLevel - level - 1) * DIMENSION;
        uint32_t childId = (voxelMortonCode >> currentBitOffset) & 0b111;
        uint32_t childBit = (1 << childId);

        AnisoSVOctreeGPU::AtomicSetChildMaskBit(*currentNode, childBit);
    }
}

__global__
void KCExtractChildrenCounts(uint32_t* gChildrenCounts,
                             const uint64_t* gLevelNodes,
                             uint32_t levelNodeCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < levelNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint64_t node = gLevelNodes[threadId];
        uint32_t childrenCount = AnisoSVOctreeGPU::ChildrenCount(node);
        // Write the count
        gChildrenCounts[threadId] = childrenCount;
    }
}

__global__
void KCSetChildrenPtrs(uint64_t* gLevelNodes,
                       const uint32_t* gChildrenOffsets,
                       uint32_t nextLevelStartIndex,
                       uint32_t levelNodeCount,
                       bool markIsChildrenLeaf)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < levelNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint64_t node = gLevelNodes[threadId];
        uint32_t offset = gChildrenOffsets[threadId];

        // Children offsets are relative to the level
        // we need to put global pointer (index)
        uint32_t globalOffset = nextLevelStartIndex + offset;
        AnisoSVOctreeGPU::SetChildrenIndex(node, globalOffset);
        // If this the last non-leaf level we need to mark the children
        if(markIsChildrenLeaf)
            AnisoSVOctreeGPU::SetIsChildrenLeaf(node, true);
        // Write back the modified node
        gLevelNodes[threadId] = node;

        printf("Node (M: %x C: %u P: %u L %u\n",
               static_cast<uint32_t>(AnisoSVOctreeGPU::ChildMask(node)),
               AnisoSVOctreeGPU::ChildrenIndex(node),
               AnisoSVOctreeGPU::ParentIndex(node),
               static_cast<uint32_t>(AnisoSVOctreeGPU::IsChildrenLeaf(node)));
    }
}

__global__
void KCSetParentOfChildren(uint64_t* gNodes,
                           const uint64_t* gLevelNodes,
                           uint32_t levelNodeCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < levelNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint64_t node = gLevelNodes[threadId];
        uint32_t childrenCount = AnisoSVOctreeGPU::ChildrenCount(node);
        uint32_t childrenIndex = AnisoSVOctreeGPU::ChildrenIndex(node);

        // Find the parent id using pointer arithmetic
        uint32_t currentNodeGlobalId  = (gLevelNodes + threadId) - gNodes;
        // Set ptrs for all children
        for(uint32_t i = 0; i < childrenCount; i++)
        {
            uint64_t* gChildNode = gNodes + childrenIndex + i;
            AnisoSVOctreeGPU::SetParentIndex(*gChildNode, currentNodeGlobalId);
        }
    }
}

__global__
void KCSetParentOfLeafChildren(uint32_t* gLeafParents,
                               const uint64_t* gNodes,
                               const uint64_t* gLevelNodes,
                               uint32_t levelNodeCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < levelNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint64_t node = gLevelNodes[threadId];
        uint32_t childrenCount = AnisoSVOctreeGPU::ChildrenCount(node);
        uint32_t childrenIndex = AnisoSVOctreeGPU::ChildrenIndex(node);
        assert(AnisoSVOctreeGPU::IsChildrenLeaf(node));
        // Find the parent id using pointer arithmetic
        uint32_t currentNodeGlobalId = (gLevelNodes + threadId) - gNodes;
        // Set ptrs for all children
        for(uint32_t i = 0; i < childrenCount; i++)
        {
            uint32_t* gChildParent = gLeafParents + childrenIndex + i;
            *gChildParent = currentNodeGlobalId;
        }
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

    // Find out the sort memory requirement of Light Keys
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
                                   {hTotalVoxCount, hTotalVoxCount + 1});

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
    size_t scanTempMemSize;

    uint64_t* dSortedVoxels = nullptr;
    HitKey* dSortedVoxelKeys = nullptr;
    // Duplicate counts
    uint32_t* dDuplicateCounts = nullptr;
    uint32_t* dUniqueVoxelCount = nullptr;
    Byte* dTempMemory = nullptr;

    // Acquire Temp Memory Requirements
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanTempMemSize,
                                             dDuplicateCounts, dDuplicateCounts,
                                             static_cast<uint32_t>(hTotalVoxCount + 1)));
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
    tempMemSize = std::max(tempMemSize, scanTempMemSize);

    // Allocation
    DeviceMemory sortedVoxelMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dSortedVoxels, dSortedVoxelKeys,
                                            dDuplicateCounts, dTempMemory,
                                            dUniqueVoxelCount),
                                   sortedVoxelMemory,
                                   {hTotalVoxCount, hTotalVoxCount,
                                   hTotalVoxCount + 1, tempMemSize,
                                   1});

    // Sort and RLE
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTempMemory, sortTempMemSize,
                                               dVoxels, dSortedVoxels,
                                               dVoxelLightKeys, dSortedVoxelKeys,
                                               static_cast<uint32_t>(hTotalVoxCount)));

    //Debug::DumpMemToFile("voxels", dVoxels, hTotalVoxCount,
    //                     false, true);

    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(dTempMemory,
                                                  rleTempMemSize,
                                                  dSortedVoxels, dVoxels,
                                                  dDuplicateCounts, dUniqueVoxelCount,
                                                  static_cast<uint32_t>(hTotalVoxCount)));


    //Debug::DumpMemToFile("sortedVoxels", dSortedVoxels, hTotalVoxCount,
    //                     false, true);
    //Debug::DumpMemToFile("rleVoxels", dVoxels, hTotalVoxCount,
    //                     false, true);

    // Load the found unique voxel count to host memory for kernel calls
    uint32_t hUniqueVoxelCount;
    CUDA_CHECK(cudaMemcpy(&hUniqueVoxelCount, dUniqueVoxelCount, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    assert(hUniqueVoxelCount <= hTotalVoxCount);

    // Temp reuse the voxel keys array for scan operation
    uint32_t* dLightKeyOffsets = reinterpret_cast<uint32_t*>(dVoxelLightKeys);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dTempMemory, scanTempMemSize,
                                             dDuplicateCounts, dLightKeyOffsets,
                                             static_cast<uint32_t>(hUniqueVoxelCount + 1)));

    // Copy the scanned result back to the duplicate counts variable
    CUDA_CHECK(cudaMemcpy(dDuplicateCounts, dLightKeyOffsets,
                          sizeof(uint32_t) * (hUniqueVoxelCount + 1),
                          cudaMemcpyDeviceToDevice));
    // Rename the allocated buffer to the proper name
    uint32_t* dLightOffsets = dDuplicateCounts;
    dDuplicateCounts = nullptr;

    // Voxel are sorted and RLE is run
    // Non-unique voxel array is not required copy the unique voxels
    // (which is in dVoxels) to dSortedVoxels array and rename
    CUDA_CHECK(cudaMemcpy(dSortedVoxels, dVoxels,
                          sizeof(uint64_t) * hUniqueVoxelCount,
                          cudaMemcpyDeviceToDevice));

    // Rename the dVoxels array to sorted unique voxels
    uint64_t* dSortedUniqueVoxels = dSortedVoxels;

    //Debug::DumpMemToFile("sortedVox", dSortedUniqueVoxels, hUniqueVoxelCount,
    //                     false, true);

    // Now we can deallocate the large non-unique voxel buffers
    voxelMemory = DeviceMemory();
    dVoxels = nullptr;
    dVoxelLightKeys = nullptr;

    // Now Allocate another temp memory for SVO Construction
    uint32_t* dDiffBitBuffer = nullptr;
    uint32_t* dChildOffsetBuffer = nullptr;
    Byte* dScanMemory = nullptr;
    DeviceMemory svoTempMemory;
    // Check Scan Memory for child reduction on SVO
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanTempMemSize,
                                             dDiffBitBuffer, dChildOffsetBuffer,
                                             static_cast<uint32_t>(hUniqueVoxelCount + 1)));

    // Allocate
    GPUMemFuncs::AllocateMultiData(std::tie(dDiffBitBuffer, dChildOffsetBuffer,
                                            dScanMemory),
                                   svoTempMemory,
                                   {hUniqueVoxelCount, hUniqueVoxelCount + 1,
                                   scanTempMemSize});

    // Top-down find the required voxel counts by looking the morton codes
    assert(Utility::BitCount(resolutionXYZ) == 1);
    uint32_t levelCount = Utility::FindLastSet(resolutionXYZ);
    std::vector<uint32_t> levelNodeCounts(levelCount + 1, 0);
    std::vector<uint32_t> levelNodeOffsets(levelCount + 2, 0);

    // Root node is always available
    levelNodeCounts[0] = 1;
    for(uint32_t i = 1; i <= levelCount; i++)
    {
        // Mark the differences between neighbors
        gpu.GridStrideKC_X(0, (cudaStream_t)0, hUniqueVoxelCount - 1,
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
            hUniqueVoxelCount - 1,
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
    std::exclusive_scan(levelNodeCounts.cbegin(), levelNodeCounts.cend(),
                        levelNodeOffsets.begin(),
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

    // Set Node and leaf parents to max to early catch errors
    // Rest is set to zero
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                       //
                       KCMemset<uint64_t>,
                       //
                       treeGPU.dNodes,
                       AnisoSVOctreeGPU::INVALID_NODE,
                       totalNodeCount);
    CUDA_CHECK(cudaMemset(treeGPU.dRadiance, 0x00, totalNodeCount * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(treeGPU.dRayCounts, 0x00, totalNodeCount * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemset(treeGPU.dLeafParents, 0xFF, hUniqueVoxelCount * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(treeGPU.dLeafRadianceRead, 0x00, hUniqueVoxelCount * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(treeGPU.dLeafRayCounts, 0x00, hUniqueVoxelCount * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(treeGPU.dLeafRadianceWrite, 0x00, hUniqueVoxelCount * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(treeGPU.dLeafSampleCountWrite, 0x00, hUniqueVoxelCount * sizeof(uint64_t)));

    //DEBUG
    remove("svoChildOffsets");

    // Top down-generate voxels
    // For each level save the node range for
    // efficient kernel calls later (level by level kernel calls)
    // Now start voxel generation level by level
    for(uint32_t i = 0; i < levelCount; i++)
    {
        gpu.GridStrideKC_X(0, (cudaStream_t)0, hUniqueVoxelCount,
                           //
                           KCMarkChild,
                           // I-O
                           treeGPU.dNodes,
                           // Input
                           dSortedUniqueVoxels,
                           // Constants
                           hUniqueVoxelCount,
                           i,
                           levelCount);

        gpu.GridStrideKC_X(0, (cudaStream_t)0, levelNodeCounts[i],
                           //
                           KCExtractChildrenCounts,
                           //
                           dDiffBitBuffer,
                           treeGPU.dNodes + levelNodeOffsets[i],
                           levelNodeCounts[i]);

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dScanMemory, scanTempMemSize,
                                                 dDiffBitBuffer, dChildOffsetBuffer,
                                                 static_cast<uint32_t>(levelNodeCounts[i] + 1)));

        Debug::DumpMemToFile("svoChildOffsets", dChildOffsetBuffer,
                             levelNodeCounts[i] + 1,
                             true);


        bool lastNonLeafLevel = (i == (levelCount - 1));
        uint32_t nextLevelOffset = (lastNonLeafLevel)
                                    ? 0
                                    : static_cast<uint32_t>(levelNodeOffsets[i + 1]);

        gpu.GridStrideKC_X(0, (cudaStream_t)0, levelNodeCounts[i],
                           //
                           KCSetChildrenPtrs,
                           //
                           treeGPU.dNodes + levelNodeOffsets[i],
                           dChildOffsetBuffer,
                           nextLevelOffset,
                           levelNodeCounts[i],
                           lastNonLeafLevel);

        if(!lastNonLeafLevel)
        {
            gpu.GridStrideKC_X(0, (cudaStream_t)0, levelNodeCounts[i],
                               //
                               KCSetParentOfChildren,
                               //
                               treeGPU.dNodes,
                               treeGPU.dNodes + levelNodeOffsets[i],
                               levelNodeCounts[i]);
        }

        METU_LOG("=============");
    }

    // Only leaf parents are left

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