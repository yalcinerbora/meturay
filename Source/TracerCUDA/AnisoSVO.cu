#include "AnisoSVO.cuh"
#include "AnisoSVOKC.cuh"

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
#include "ParallelSequence.cuh"
#include "BinarySearch.cuh"

#include <cub/cub.cuh>
#include <numeric>

#include "TracerDebug.h"

__global__ CUDA_LAUNCH_BOUNDS_1D
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
    static constexpr uint32_t DIMENSION = 3;
    const uint32_t voxelMSBStart = sizeof(uint64_t) * BYTE_BITS - (maxLevel * DIMENSION);

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < voxelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint64_t voxMorton = gVoxels[threadId];
        assert((voxMorton & (~((1ull << (maxLevel * DIMENSION)) - 1))) == 0);
        uint64_t voxMortonRev = __brevll(voxMorton) >> voxelMSBStart;
        voxMortonRev &= (1ull << level * DIMENSION) - 1;

        uint64_t voxMortonNext = gVoxels[threadId + 1];
        assert((voxMortonNext & (~((1ull << (maxLevel * DIMENSION)) - 1))) == 0);
        uint64_t voxMortonNextRev = __brevll(voxMortonNext) >> voxelMSBStart;
        voxMortonNextRev &= (1ull << level * DIMENSION) - 1;

        gMarks[threadId] = (voxMortonRev != voxMortonNextRev) ? 1 : 0;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCMarkChild(// I-O
                 uint64_t* gNodes,
                 // Input
                 const uint64_t* gVoxels,
                 // Constants
                 uint32_t voxelCount,
                 uint32_t level,
                 uint32_t maxLevel)
{
    // Useful constants
    static constexpr uint32_t DIMENSION = 3;
    static constexpr uint32_t DIM_MASK = (1 << DIMENSION) - 1;

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < voxelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint64_t voxelMortonCode = gVoxels[threadId];
        uint32_t mortonLevelShift = (maxLevel - 1) * DIMENSION;
        // Start traversing (Root node is the very first node)
        uint32_t currentNodeIndex = 0;
        for(int i = 0; i < level; i++)
        {
            uint64_t currentNode = gNodes[currentNodeIndex];
            // Fetch the current bit triples of the level from the
            // morton code
            uint32_t childId = (voxelMortonCode >> mortonLevelShift) & DIM_MASK;
            uint32_t childOffset = AnisoSVOctreeGPU::FindChildOffset(currentNode, childId);
            uint32_t childrenIndex = AnisoSVOctreeGPU::ChildrenIndex(currentNode);
            // Go to next child
            currentNodeIndex = childrenIndex + childOffset;
            // Strip the processed bits
            mortonLevelShift -= DIMENSION;
        }
        // Now we are at the not that does not set its children ptr and mask is set
        // Atomically mark the required child
        uint32_t childId = (voxelMortonCode >> mortonLevelShift) & DIM_MASK;
        uint32_t childBit = (1 << childId);
        assert(childId < 8);
        assert(__popc(childBit) == 1);
        // Atomically set the child bit on the packed node
        AnisoSVOctreeGPU::AtomicSetChildMaskBit(gNodes + currentNodeIndex,
                                                childBit);
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
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

__global__ CUDA_LAUNCH_BOUNDS_1D
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
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
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

__global__ CUDA_LAUNCH_BOUNDS_1D
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
                                const WFPGPathNode* gPathNodes,
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

        WFPGPathNode gPathNode = gPathNodes[nodeIndex];

        // Skip if this node cannot calculate wo
        if(!gPathNode.HasPrev()) continue;

        Vector3f wo = gPathNode.Wo<WFPGPathNode>(gPathNodes, pathStartIndex);
        Vector3f wi = gPathNode.Wi<WFPGPathNode>(gPathNodes, pathStartIndex);

        float luminance = Utility::RGBToLuminance(gPathNode.totalRadiance);
        unableToAccum |= !svo.DepositRadiance(gPathNode.worldPosition,
                                              gPathNode.Normal(),
                                              wi, wo, luminance);
    }
    // Debug
    if(unableToAccum)
    {
        printf("Unable to accumulate some radiance values!\n");
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCCollapseRayCounts(// I-O
                         uint16_t* gBinInfo,
                         // Input
                         const uint64_t* gNodes,
                         // Constants
                         Vector2ui levelRange,
                         uint32_t level,
                         uint32_t minLevel,
                         uint32_t minRayCount)
{
    uint32_t nodeCount = levelRange[1] - levelRange[0];

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < nodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint32_t nodeId = levelRange[0] + threadId;

        // We are at the user option limit,
        // Directly mark this node then leave
        if(level == minLevel)
        {
            AnisoSVOctreeGPU::SetBinAsMarked(gBinInfo[nodeId]);
            continue;
        }
        // Fetch Ray Count
        uint32_t rayCount = AnisoSVOctreeGPU::GetRayCount(gBinInfo[nodeId]);
        // If ray count is not enough on this voxel
        // collaborate with the other children
        if(rayCount < minRayCount)
        {
            uint32_t parent = AnisoSVOctreeGPU::ParentIndex(gNodes[nodeId]);
            AnisoSVOctreeGPU::AtomicAddUInt16(gBinInfo + parent, rayCount);
        }
        // We have enough rays in this node use it as is
        else
        {
            AnisoSVOctreeGPU::SetBinAsMarked(gBinInfo[nodeId]);
        }
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCCollapseRayCountsLeaf(// I-O
                             uint16_t* gLeafBinInfo,
                             uint16_t* gBinInfo,
                             // Input
                             const uint32_t* gLeafParents,
                             // Constants
                             uint32_t leafCount,
                             uint32_t level,
                             uint32_t minLevel,
                             uint32_t minRayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < leafCount;
        threadId += (blockDim.x * gridDim.x))
    {
        // We are at the user option limit,
        // Directly mark this node then leave
        if(level == minLevel)
        {
            AnisoSVOctreeGPU::SetBinAsMarked(gLeafBinInfo[threadId]);
            continue;
        }
        // Fetch Ray Count
        uint16_t rayCount = AnisoSVOctreeGPU::GetRayCount(gLeafBinInfo[threadId]);
        if(rayCount == 0) continue;

        // If ray count is not enough on this voxel
        // collaborate with the other children
        if(rayCount < minRayCount)
        {
            uint32_t parent = gLeafParents[threadId];
            AnisoSVOctreeGPU::AtomicAddUInt16(gBinInfo + parent, rayCount);
        }
        // We have enough rays in this node use it as is
        else
        {
            AnisoSVOctreeGPU::SetBinAsMarked(gLeafBinInfo[threadId]);
        }
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCCCopyRadianceToHalfBufferLeaf(// I-O
                                     Vector2h* dLeafRadianceRead,
                                     // Input
                                     const Vector2f* dLeafRadianceWrite,
                                     const Vector2ui* dLeafSampleCountWrite,
                                     // Constants
                                     uint32_t leafCount,
                                     float totalRadianceScene)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < leafCount;
        threadId += (blockDim.x * gridDim.x))
    {
        // Read from the value;
        Vector2ui count = dLeafSampleCountWrite[threadId];
        Vector2f irradiance = dLeafRadianceWrite[threadId];
        Vector2f avgRadiance = Vector2f(irradiance[0] / static_cast<float>(count[0]),
                                        irradiance[1] / static_cast<float>(count[1]));
        // Avoid NaN if not accumulation occured
        avgRadiance[0] = (count[0] == 0) ? 0.0f : avgRadiance[0];
        avgRadiance[1] = (count[1] == 0) ? 0.0f : avgRadiance[1];
        // Normalize & Clamp the half range for now
        Vector2f irradClampled = Vector2f::Min(avgRadiance, Vector2f(MRAY_HALF_MAX));

        Vector2h irradHalf = Vector2h(irradClampled);
        dLeafRadianceRead[threadId] = irradHalf;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCConvertToAnisoFloat(Vector2f* gAnisoOut,
                           const Vector2h* gAnisoIn,
                           uint32_t anisoCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < anisoCount;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector2f& gOut = gAnisoOut[threadId];
        Vector2h in = gAnisoIn[threadId];


        gOut[0] = in[0];
        gOut[1] = in[1];
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
    float sizePadding = (worldSizeXYZ / static_cast<float>(resolutionXYZ));
    treeGPU.svoAABB = AABB3f(sceneAABB.Min() - Vector3f(sizePadding),
                             sceneAABB.Min() + Vector3f(sizePadding + worldSizeXYZ));
    treeGPU.leafDepth = Utility::FindLastSet(resolutionXYZ);
    treeGPU.leafVoxelSize = (worldSizeXYZ + 2.0f * sizePadding) / static_cast<float>(resolutionXYZ);
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

    // Allocate Voxel Count memory
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

    // Ask each primitive for rasterized voxel count
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

    // Acquire total voxel count (last element of scan operation)
    uint64_t hTotalVoxCount;
    CUDA_CHECK(cudaMemcpy(&hTotalVoxCount, dVoxelOffsets + primOffsets.back(),
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Allocate enough memory for temp voxels (these may overlap)
    Vector2us* dVoxelNormals;
    HitKey* dVoxelLightKeys;

    uint64_t* dVoxels;
    uint32_t* dVoxelIndices;
    DeviceMemory voxelMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dVoxels, dVoxelIndices),
                                   voxelMemory,
                                   {hTotalVoxCount, hTotalVoxCount + 1});
    DeviceMemory voxelPayloadMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dVoxelNormals,
                                            dVoxelLightKeys),
                                   voxelPayloadMemory,
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

    // Generate Iota for sorting
    IotaGPU(dVoxelIndices, 0u, hTotalVoxCount);

    // For each accelerator
    // Actually rasterize the primitives
    // and push to the memory (find the light key; if available, here)
    i = 0;
    for(const auto [_, accel] : accels)
    {
        accel->VoxelizeSurfaces(// Outputs
                                dVoxels,
                                dVoxelNormals,
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

    // Cub operation temporary buffers
    size_t rleTempMemSize;
    size_t sortTempMemSize;
    size_t scanTempMemSize;

    uint64_t* dSortedVoxels = nullptr;
    uint32_t* dSortedVoxelIndices = nullptr;
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
                                               dVoxelIndices, dSortedVoxelIndices,
                                               static_cast<uint32_t>(hTotalVoxCount),
                                               0, treeGPU.leafDepth * 3 + 1));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                  rleTempMemSize,
                                                  dSortedVoxels, dVoxels,
                                                  dDuplicateCounts, dUniqueVoxelCount,
                                                  static_cast<uint32_t>(hTotalVoxCount)));
    size_t tempMemSize = std::max(rleTempMemSize, sortTempMemSize);
    tempMemSize = std::max(tempMemSize, scanTempMemSize);

    // Allocation
    DeviceMemory sortedVoxelMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dSortedVoxels, dSortedVoxelIndices,
                                            dDuplicateCounts, dTempMemory,
                                            dUniqueVoxelCount),
                                   sortedVoxelMemory,
                                   {hTotalVoxCount, hTotalVoxCount,
                                    hTotalVoxCount + 1, tempMemSize,
                                    1});

    // Sort and RLE
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTempMemory, sortTempMemSize,
                                               dVoxels, dSortedVoxels,
                                               dVoxelIndices, dSortedVoxelIndices,
                                               static_cast<uint32_t>(hTotalVoxCount),
                                               0, treeGPU.leafDepth * 3 + 1));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(dTempMemory,
                                                  rleTempMemSize,
                                                  dSortedVoxels, dVoxels,
                                                  dDuplicateCounts, dUniqueVoxelCount,
                                                  static_cast<uint32_t>(hTotalVoxCount)));

    // Load the found unique voxel count (non-duplicate) to host memory for kernel calls
    uint32_t hUniqueVoxelCount;
    CUDA_CHECK(cudaMemcpy(&hUniqueVoxelCount, dUniqueVoxelCount, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    assert(hUniqueVoxelCount <= hTotalVoxCount);

    // Temp reuse the voxel indices array for scan operation
    uint32_t* dVoxelIndexOffsets = reinterpret_cast<uint32_t*>(dVoxelIndices);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dTempMemory, scanTempMemSize,
                                             dDuplicateCounts, dVoxelIndexOffsets,
                                             hUniqueVoxelCount + 1));

    // Copy the scanned result back to the duplicate counts variable
    CUDA_CHECK(cudaMemcpy(dDuplicateCounts, dVoxelIndexOffsets,
                          sizeof(uint32_t) * (hUniqueVoxelCount + 1),
                          cudaMemcpyDeviceToDevice));
    // Rename the allocated buffer to the proper name
    uint32_t* dIndexOffsets = dDuplicateCounts;
    dDuplicateCounts = nullptr;

    // Voxel are sorted and RLE is run
    // Non-unique voxel array is not required copy the unique voxels
    // (which is in dVoxels) to dSortedVoxels array and rename
    CUDA_CHECK(cudaMemcpy(dSortedVoxels, dVoxels,
                          sizeof(uint64_t) * hUniqueVoxelCount,
                          cudaMemcpyDeviceToDevice));
    // Rename the dVoxels array to sorted unique voxels
    uint64_t* dSortedUniqueVoxels = dSortedVoxels;

    // Now we can deallocate the large non-unique voxel buffers
    voxelMemory = DeviceMemory();
    dVoxels = nullptr;
    dVoxelIndices = nullptr;

    // Now Allocate another temp memory for SVO Construction
    uint32_t* dDiffBitBuffer = nullptr;
    uint32_t* dChildOffsetBuffer = nullptr;
    Byte* dScanMemory = nullptr;
    DeviceMemory svoTempMemory;
    // Check Scan Memory for child reduction on SVO
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanTempMemSize,
                                             dDiffBitBuffer, dChildOffsetBuffer,
                                             hUniqueVoxelCount + 1));

    // Allocate
    GPUMemFuncs::AllocateMultiData(std::tie(dDiffBitBuffer, dChildOffsetBuffer,
                                            dScanMemory),
                                   svoTempMemory,
                                   {hUniqueVoxelCount, hUniqueVoxelCount + 1,
                                   scanTempMemSize});

    // Top-down find the required voxel counts by looking at morton codes
    assert(Utility::BitCount(resolutionXYZ) == 1);
    uint32_t levelCount = treeGPU.leafDepth;
    std::vector<uint32_t> levelNodeCounts(levelCount + 1, 0);
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

        // Reduce the marks to find level node count
        ReduceArrayGPU<uint32_t, ReduceAdd<uint32_t>, cudaMemcpyDeviceToHost>
        (
            levelNodeCounts[i],
            dDiffBitBuffer,
            hUniqueVoxelCount - 1,
            0u
        );
        gpu.WaitMainStream();
        // n different slices means n+1 segments
        levelNodeCounts[i] += 1;
    }
    assert(levelNodeCounts.back() == hUniqueVoxelCount);

    // Populate node offset buffer
    levelNodeOffsets.resize(levelCount + 2, 0);
    std::inclusive_scan(levelNodeCounts.cbegin(), levelNodeCounts.cend(),
                        levelNodeOffsets.begin() + 1);
    levelNodeOffsets.front() = 0;

    uint32_t totalNodeCount = levelNodeOffsets[levelNodeOffsets.size() - 2];

    treeGPU.nodeCount = totalNodeCount;
    treeGPU.leafCount = hUniqueVoxelCount;
    // Allocate required memories now
    // since we found out the total node count
    GPUMemFuncs::AllocateMultiData(std::tie(// Node Related,
                                            treeGPU.dNodes,
                                            treeGPU.dBinInfo,
                                            // Leaf Related
                                            treeGPU.dLeafParents,
                                            treeGPU.dLeafBinInfo,
                                            // Payload Node
                                            treeGPU.payload.dAvgIrradianceNode,
                                            treeGPU.payload.dNormalAndSpecNode,
                                            treeGPU.payload.dGuidingFactorNode,
                                            treeGPU.payload.dMicroQuadTreeNode,
                                            // Payload Leaf
                                            treeGPU.payload.dTotalIrradianceLeaf,
                                            treeGPU.payload.dSampleCountLeaf,
                                            treeGPU.payload.dAvgIrradianceLeaf,
                                            treeGPU.payload.dNormalAndSpecLeaf,
                                            treeGPU.payload.dGuidingFactorLeaf,
                                            treeGPU.payload.dMicroQuadTreeLeaf,
                                            // Node Offsets
                                            treeGPU.dLevelNodeOffsets),
                                   octreeMem,
                                   {
                                        // Node Related
                                        totalNodeCount, totalNodeCount,
                                        // Leaf Related
                                        hUniqueVoxelCount, hUniqueVoxelCount,
                                        // Payload Node
                                        totalNodeCount, totalNodeCount,
                                        totalNodeCount, totalNodeCount,
                                        // Payload Leaf
                                        hUniqueVoxelCount, hUniqueVoxelCount, hUniqueVoxelCount,
                                        hUniqueVoxelCount, hUniqueVoxelCount, hUniqueVoxelCount,
                                        // Offsets
                                        levelNodeOffsets.size()
                                   });

    // Set Node and leaf parents to max to early catch errors
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                       //
                       KCMemset<uint64_t>,
                       //
                       treeGPU.dNodes,
                       AnisoSVOctreeGPU::INVALID_NODE,
                       totalNodeCount);
    CUDA_CHECK(cudaMemset(treeGPU.dLeafParents, 0xFF, hUniqueVoxelCount * sizeof(uint32_t)));
    // Bin info initially should be zero (every bounce we will set it to again zero as well)
    CUDA_CHECK(cudaMemset(treeGPU.dBinInfo, 0x00, totalNodeCount * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(treeGPU.dLeafBinInfo, 0x00, hUniqueVoxelCount * sizeof(uint16_t)));
    // Set accumulators to zero
    CUDA_CHECK(cudaMemset(treeGPU.payload.dTotalIrradianceLeaf, 0x00, hUniqueVoxelCount * sizeof(Vector2f)));
    CUDA_CHECK(cudaMemset(treeGPU.payload.dSampleCountLeaf, 0x00, hUniqueVoxelCount * sizeof(Vector2ui)));
    // Copy the generated offsets
    CUDA_CHECK(cudaMemcpy(treeGPU.dLevelNodeOffsets, levelNodeOffsets.data(),
                          levelNodeOffsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
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
                                                 levelNodeCounts[i] + 1));

        // Check
        uint32_t hReducedSum;
        CUDA_CHECK(cudaMemcpy(&hReducedSum, dChildOffsetBuffer + levelNodeCounts[i],
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if(hReducedSum != levelNodeCounts[i + 1])
        {
            METU_ERROR_LOG("SVO children count allocation mismatch (Level {:d}.", i);
            return TracerError::TRACER_INTERNAL_ERROR;
        }

        bool lastNonLeafLevel = (i == (levelCount - 1));
        uint32_t nextLevelOffset = (lastNonLeafLevel) ? 0 : levelNodeOffsets[i + 1];
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
        else
        {
            gpu.GridStrideKC_X(0, (cudaStream_t)0, levelNodeCounts[i],
                               //
                               KCSetParentOfLeafChildren,
                               //
                               treeGPU.dLeafParents,
                               treeGPU.dNodes,
                               treeGPU.dNodes + levelNodeOffsets[i],
                               levelNodeCounts[i]);
        }
    }
    // Only Direct light information deposition is left
    // Call the kernel for it

    // DEBUG
    //Debug::DumpMemToFile("voxelDuplicates", dIndexOffsets,
    //                     hUniqueVoxelCount + 1);

    //std::vector<uint32_t> countsss(hUniqueVoxelCount + 1);
    //std::adjacent_difference(dIndexOffsets, dIndexOffsets + hUniqueVoxelCount + 1,
    //                         countsss.begin());
    //Debug::DumpMemToFile("voxelDuplicateCounts", countsss.data(),
    //                     hUniqueVoxelCount + 1);

    gpu.GridStrideKC_X(0, (cudaStream_t)0, hUniqueVoxelCount * WARP_SIZE,
                       //
                       KCReduceVoxelPayload<StaticThreadPerBlock1D>,
                       // I-O
                       treeGPU,
                       // Input
                       dIndexOffsets,
                       dSortedUniqueVoxels,
                       dSortedVoxelIndices, // This is non-unique (we need to reduce it)
                       dVoxelLightKeys, // Voxel payload that will be reduced
                       dVoxelNormals,   // Voxel payload that will be reduced

                       // Binary Search for light
                       dSortedLightKeys,
                       dSortedLights,
                       totalLightCount,
                       // Constants
                       hUniqueVoxelCount,
                       static_cast<uint32_t>(hTotalVoxCount),
                       treeGPU.svoAABB,
                       resolutionXYZ);

    // Log some stuff
    timer.Stop();
    double svoMemSize = static_cast<double>(octreeMem.Size()) / 1024.0 / 1024.0;

    METU_LOG("Scene Aniso-SVO [N: {:L}, L: {:L}] Generated in {:f} seconds. (Total {:.2f} MiB)",
             treeGPU.nodeCount, treeGPU.leafCount, timer.Elapsed<CPUTimeSeconds>(), svoMemSize);

    // All Done!
    return TracerError::OK;
}

void AnisoSVOctreeCPU::NormalizeAndFilterRadiance(const CudaSystem& system)
{
    // From leaf (leaf-write) to root
    // Average the radiance
    // Down-sample the radiance for lowest n levels as well maybe? (n= 2 or 3)


    // TODO: Do some proper filtering
    // Just copy it to for now
    // Assume that the ray counts are set for leaves
    const CudaGPU& bestGPU = system.BestGPU();
    // Leaf has different memory layout do it separately
    bestGPU.GridStrideKC_X(0, (cudaStream_t)0, treeGPU.leafCount,
                           //
                           KCCCopyRadianceToHalfBufferLeaf,
                           // I-O
                           treeGPU.payload.dAvgIrradianceLeaf,
                           // Input
                           treeGPU.payload.dTotalIrradianceLeaf,
                           treeGPU.payload.dSampleCountLeaf,
                           // Constants
                           treeGPU.leafCount,
                           1.0f);
}

void AnisoSVOctreeCPU::CollapseRayCounts(uint32_t minLevel, uint32_t minRayCount,
                                         const CudaSystem& system)
{
    // Assume that the ray counts are set for leaves
    const CudaGPU& bestGPU = system.BestGPU();

    // Leaf has different memory layout do it separately
    bestGPU.GridStrideKC_X(0, (cudaStream_t)0, treeGPU.leafCount,
                           //
                           KCCollapseRayCountsLeaf,
                           // I-O
                           treeGPU.dLeafBinInfo,
                           treeGPU.dBinInfo,
                           // Input
                           treeGPU.dLeafParents,
                           // Constants
                           treeGPU.leafCount,
                           treeGPU.leafDepth,
                           minLevel,
                           minRayCount);

    //Debug::DumpMemToFile(std::to_string(treeGPU.leafDepth) + std::string("_binInfo"),
    //                     treeGPU.dLeafBinInfo,
    //                     treeGPU.leafCount, false, true);

    // Bottom-up process bins
    int32_t bottomNodeLevel = static_cast<int32_t>(treeGPU.leafDepth - 1);
    for(int32_t i = bottomNodeLevel; i >= static_cast<int32_t>(minLevel); i--)
    {
        Vector2ui range(levelNodeOffsets[i],
                        levelNodeOffsets[i + 1]);
        uint32_t nodeCount = range[1] - range[0];

        bestGPU.GridStrideKC_X(0, (cudaStream_t)0, nodeCount,
                               //
                               KCCollapseRayCounts,
                               // I-O
                               treeGPU.dBinInfo,
                               // Input
                               treeGPU.dNodes,
                               // Constants
                               range,
                               i,
                               minLevel,
                               minRayCount);
    }
    // Leaf->Parent chain now there is at least a single mark
    // Rays will re-check and find their marked bin and set their id accordingly
}


void AnisoSVOctreeCPU::AccumulateRaidances(const WFPGPathNode* dPGNodes,
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
    CUDA_CHECK(cudaMemset(treeGPU.dLeafBinInfo, 0x00, sizeof(uint32_t) * treeGPU.leafCount));
    CUDA_CHECK(cudaMemset(treeGPU.dBinInfo, 0x00, sizeof(uint32_t) * treeGPU.nodeCount));
}

void AnisoSVOctreeCPU::DumpSVOAsBinary(std::vector<Byte>& data,
                                       const CudaSystem& system) const
{
    // Temp Float Buffer for Conversion
    assert(treeGPU.leafCount >= treeGPU.nodeCount);
    DeviceMemory halfConvertedMemory(treeGPU.leafCount * sizeof(Vector2f));

    // Conversion Function
    auto ConvertAnisoHalfToFloat = [&](const Vector2h* dRadiance,
                                       uint32_t totalSize)
    {
        const CudaGPU& gpu = system.BestGPU();
        gpu.GridStrideKC_X(0, (cudaStream_t)0, totalSize,
                           //
                           KCConvertToAnisoFloat,
                           //
                           static_cast<Vector2f*>(halfConvertedMemory),
                           dRadiance,
                           totalSize);
    };

    // Get Sizes
    std::array<size_t, 4> byteSizes;
    byteSizes[0]  = treeGPU.nodeCount * sizeof(uint64_t);   // "dNodes" Size
    byteSizes[1]  = treeGPU.nodeCount * sizeof(Vector2f);   // "dAvgIrradianceNode" Size
    // Leaf Related
    byteSizes[2] = treeGPU.leafCount * sizeof(uint32_t);    // "dLeafParent" Size
    byteSizes[3] = treeGPU.leafCount * sizeof(Vector2f);    // "dAvgIrradianceLeaf" Size
    // Calculate the offsets and total size
    size_t bufferTotalSize = std::reduce(byteSizes.cbegin(), byteSizes.cend(), 0ull);

    size_t totalSize = (bufferTotalSize + sizeof(AABB3f) +
                        5 * sizeof(uint32_t) +
                        sizeof(float));

    data.resize(totalSize);
    // Memcpy the data from the memory
    size_t offset = 0;
    std::memcpy(data.data() + offset, &treeGPU.svoAABB, sizeof(AABB3f));
    offset += sizeof(AABB3f);
    std::memcpy(data.data() + offset, &treeGPU.voxelResolution, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(data.data() + offset, &treeGPU.leafDepth, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(data.data() + offset, &treeGPU.nodeCount, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(data.data() + offset, &treeGPU.leafCount, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(data.data() + offset, &treeGPU.leafVoxelSize, sizeof(float));
    offset += sizeof(float);
    std::memcpy(data.data() + offset, &treeGPU.levelOffsetCount, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Nodes
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.dNodes,
                          byteSizes[0], cudaMemcpyDeviceToHost));
    offset += byteSizes[0];
    // Radiance Cache Node
    ConvertAnisoHalfToFloat(treeGPU.payload.dAvgIrradianceNode, treeGPU.nodeCount);
    CUDA_CHECK(cudaMemcpy(data.data() + offset,
                          static_cast<void*>(halfConvertedMemory),
                          byteSizes[1], cudaMemcpyDeviceToHost));
    offset += byteSizes[1];
    // Leaf Parents
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.dLeafParents,
                          byteSizes[2], cudaMemcpyDeviceToHost));
    offset += byteSizes[2];
    // Radiance Cache Leaf
    ConvertAnisoHalfToFloat(treeGPU.payload.dAvgIrradianceLeaf, treeGPU.leafCount);
    CUDA_CHECK(cudaMemcpy(data.data() + offset,
                          static_cast<void*>(halfConvertedMemory),
                          byteSizes[3], cudaMemcpyDeviceToHost));
    offset += byteSizes[3];
    assert(offset == data.size());

    // All Done!
}