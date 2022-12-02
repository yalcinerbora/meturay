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
void KCFindBoundaryLight(const GPULightI*& gBoundaryLightOut,
                         // Inputs
                         const GPULightI** gSceneLights,
                         const HitKey boundaryLightKey,
                         uint32_t totalLightCount)
{
    uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadId >= totalLightCount) return;

    HitKey key = gSceneLights[threadId]->WorkKey();

    if(boundaryLightKey == key)
    {
        gBoundaryLightOut = gSceneLights[threadId];
    }

}

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
        Vector3f wi = wo;
        // Only consider wi if valid or ignore
        // This should only fail when a SVO node has both light source and surface
        // then light will leak behind, this should be a rare case
        if(gPathNode.HasNext())
           wi = gPathNode.Wi<WFPGPathNode>(gPathNodes, pathStartIndex);

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
        // Read the float
        Vector2ui count = dLeafSampleCountWrite[threadId];
        Vector2f irradiance = dLeafRadianceWrite[threadId];
        Vector2f avgRadiance = Vector2f(irradiance[0] / static_cast<float>(count[0]),
                                        irradiance[1] / static_cast<float>(count[1]));
        // Avoid NaN if not accumulation occurred
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

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCConvertToAnisoHalf(Vector2h* gAnisoOut,
                          const Vector2f* gAnisoIn,
                          uint32_t anisoCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < anisoCount;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector2h& gOut = gAnisoOut[threadId];
        Vector2f in = gAnisoIn[threadId];


        gOut[0] = in[0];
        gOut[1] = in[1];
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCAverageNormals(//I-O
                      VoxelPayload payload,
                      // Input
                      const uint64_t* dNodes,
                      const uint32_t* dLevelNodeOffsets,
                      // Constants
                      uint32_t level,
                      uint32_t levelNodeCount,
                      bool childrenAreLeaf)
{
    static constexpr uint32_t MEAN_COUNT = 2;
    static constexpr uint32_t MAX_CHILD_COUNT = 8;
    static constexpr uint32_t K_MEANS_CLUSTER_ITER_COUNT = 4;
    // Buffers
    Vector3f normals[MAX_CHILD_COUNT];
    Vector3f meanNormals[2][MEAN_COUNT];
    uint32_t counts[2][MEAN_COUNT];

    auto KMeansCluster = [&](uint32_t validChildCount)
    {
        for(uint32_t kMeanPassId = 0; kMeanPassId < K_MEANS_CLUSTER_ITER_COUNT; kMeanPassId++)
        {
            uint32_t readIndex = kMeanPassId % 2;
            uint32_t writeIndex = (readIndex == 0) ? 1 : 0;
            // For each iteration cluster the normals
            for(uint32_t nIndex = 0; nIndex < validChildCount; nIndex++)
            {
                // Skip if we are on an edge case
                if(normals[nIndex] == Vector3f(0.0f)) continue;

                // Do the mean calculation
                // Calculate angular distance and choose your mean
                float angularDist0 = 1.0f - meanNormals[readIndex][0].Dot(normals[nIndex]);
                float angularDist1 = 1.0f - meanNormals[readIndex][1].Dot(normals[nIndex]);

                uint32_t updateIndex = (angularDist0 >= angularDist1) ? 0 : 1;
                meanNormals[writeIndex][updateIndex] += normals[nIndex];
                counts[writeIndex][updateIndex] += 1;
            }
            // Pass is done, swap buffers (automatic)
            // Clean the accumulation buffers
            meanNormals[readIndex][0] = Zero3f;
            meanNormals[readIndex][1] = Zero3f;
            counts[readIndex][0] = 0;
            counts[readIndex][1] = 0;
        }
    };

    const uint32_t levelNodeOffset = dLevelNodeOffsets[level];
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < levelNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint32_t nodeIndex = levelNodeOffset + threadId;
        uint64_t nodeData = dNodes[nodeIndex];

        // Get children count & child offset
        uint32_t childrenCount = AnisoSVOctreeGPU::ChildrenCount(nodeData);
        uint32_t childrenIndex = AnisoSVOctreeGPU::ChildrenIndex(nodeData);
        // For non-leaf this is mandatory
        assert(childrenCount > 0);

        // Load normals to the buffer
        // Might as well average the specular while at it
        float avgSpecular = 0;
        for(uint32_t i = 0; i < childrenCount; i++)
        {
            // Get the normal
            float stdDev, specular;
            Vector3f normal = payload.ReadNormalAndSpecular(stdDev, specular,
                                                            childrenIndex + i, childrenAreLeaf);
            // Reverse the Toksvig 2004
            float normalLength = 1.0f / (1.0f + stdDev);
            normals[i] = normal * normalLength;
            // Specular is classic average
            avgSpecular += specular;
        }
        float childrenCountRecip = 1.0f / static_cast<float>(childrenCount);
        avgSpecular *= childrenCountRecip;

        // Do K-Means Cluster (K=2)
        // Select extremes
        meanNormals[0][0] = normals[0];
        meanNormals[0][1] = -normals[0];
        counts[1][0] = 0;
        counts[1][1] = 0;
        KMeansCluster(childrenCount);

        // Clustering is complete,
        // choose the cluster that has maximum amount of normals in it
        uint32_t normalIndex = (counts[0][0] >= counts[0][1]) ? 0 : 1;
        meanNormals[0][normalIndex] *= (1.0f / static_cast<float>(counts[0][normalIndex]));
        uint32_t normalLength = meanNormals[0][normalIndex].Length();

        // Now set
        payload.WriteNormalAndSpecular(meanNormals[0][normalIndex],
                                       avgSpecular, nodeIndex, false);
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCBottomUpFilterIrradiance(//I-O
                                VoxelPayload payload,
                                // Input
                                const uint64_t* dNodes,
                                const uint32_t* dLevelNodeOffsets,
                                // Constants
                                uint32_t level,
                                uint32_t levelNodeCount,
                                bool childrenAreLeaf)
{
    // TODO: do proper filtering
    const uint32_t levelNodeOffset = dLevelNodeOffsets[level];
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < levelNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint32_t nodeIndex = levelNodeOffset + threadId;
        uint64_t nodeData = dNodes[nodeIndex];

        // Get children count & child offset
        uint32_t childrenCount = AnisoSVOctreeGPU::ChildrenCount(nodeData);
        uint32_t childrenIndex = AnisoSVOctreeGPU::ChildrenIndex(nodeData);

        // Child Irradiance array
        const Vector2h* dChildIrradArray = (childrenAreLeaf)
                                            ? payload.dAvgIrradianceLeaf
                                            : payload.dAvgIrradianceNode;

        // Fetch the normal
        float stdDev, specular;
        Vector3f normal = payload.ReadNormalAndSpecular(stdDev, specular,
                                                        nodeIndex, false);

        // Average
        const half HALF_MAX_LOCAL = static_cast<half>(MRAY_HALF_MAX);
        Vector2h avgIrrad = Vector2h(0.0f);
        Vector2uc validChildrenCount = Vector2uc(0);
        for(uint32_t i = 0; i < childrenCount; i++)
        {
            // Get the irradiance
            Vector2h irradiance = dChildIrradArray[childrenIndex + i];
            // Normal
            Vector3f childNormal = payload.ReadNormalAndSpecular(stdDev, specular,
                                                                 childrenIndex + i,
                                                                 childrenAreLeaf);
            // During average normals are swapped
            bool swapIrrad = (childNormal.Dot(normal) < 0);
            #pragma unroll
            for(int i = 0; i < 2; i++)
            {
                // Read index may be different if normals are different
                int readIndex = (swapIrrad) ? ((i + 1) % 2) : i;

                if(irradiance[readIndex] != HALF_MAX_LOCAL)
                {
                    validChildrenCount[i]++;
                    avgIrrad[i] += irradiance[readIndex];
                }
            }
        }

        const half ONE = static_cast<half>(1.0f);
        Vector2h childrenCountRecip = Vector2h(ONE / static_cast<half>(validChildrenCount[0]),
                                               ONE / static_cast<half>(validChildrenCount[1]));
        avgIrrad[0] *= childrenCountRecip[0];
        avgIrrad[1] *= childrenCountRecip[1];
        // Avoid NaN if not accumulation occurred
        avgIrrad[0] = (validChildrenCount[0] == 0) ? HALF_MAX_LOCAL : avgIrrad[0];
        avgIrrad[1] = (validChildrenCount[1] == 0) ? HALF_MAX_LOCAL : avgIrrad[1];
        // Normalize & Clamp the half range for now
        Vector2h irradClampled = Vector2h((avgIrrad[0] <= HALF_MAX_LOCAL) ? avgIrrad[0] : MRAY_HALF_MAX,
                                          (avgIrrad[1] <= HALF_MAX_LOCAL) ? avgIrrad[1] : MRAY_HALF_MAX);
        // Now set
        payload.dAvgIrradianceNode[nodeIndex] = irradClampled;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCTopDownFilterIrradiance(//I-O
                               VoxelPayload payload,
                               // Input
                               const uint64_t* dNodes,
                               const uint32_t* dLeafParents,
                               const uint32_t* dLevelNodeOffsets,
                               // Constants
                               uint32_t level,
                               uint32_t levelNodeCount,
                               bool isLeaf)
{
    const uint32_t levelNodeOffset = isLeaf ? 0 : dLevelNodeOffsets[level];
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < levelNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint32_t nodeIndex = levelNodeOffset + threadId;
        // Get children count & child offset
        uint32_t parentIndex = isLeaf ? dLeafParents[nodeIndex]
                                      : AnisoSVOctreeGPU::ParentIndex(dNodes[nodeIndex]);
        // Child Irradiance array
        Vector2h* dIrradArray = (isLeaf) ? payload.dAvgIrradianceLeaf
                                         : payload.dAvgIrradianceNode;

        // Fetch parents value if currently no value
        const half HALF_MAX_LOCAL = static_cast<half>(MRAY_HALF_MAX);
        // Check Irrad if HALF_MAX use parents value
        Vector2h irradiance = dIrradArray[nodeIndex];
        Vector2h parentIrrad = payload.dAvgIrradianceNode[parentIndex];
        if(irradiance[0] == HALF_MAX_LOCAL) irradiance[0] = parentIrrad[0];
        if(irradiance[1] == HALF_MAX_LOCAL) irradiance[1] = parentIrrad[1];
        // Now set
        Vector2h* dIrradOut = (isLeaf) ? payload.dAvgIrradianceLeaf
                                       : payload.dAvgIrradianceNode;
        dIrradOut[nodeIndex] = irradiance;
    }
}


TracerError AnisoSVOctreeCPU::Constrcut(const AABB3f& sceneAABB, uint32_t resolutionXYZ,
                                        const AcceleratorBatchMap& accels,
                                        const GPULightI** dSceneLights,
                                        uint32_t totalLightCount,
                                        HitKey boundaryLightKey,
                                        const CudaSystem& system)
{
    treeGPU = {};

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

    // Allocate enough memory for temp voxels (these temp voxels may have duplicates)
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
    // For light irradiance injection to SVO
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
    treeGPU.levelOffsetCount = static_cast<uint32_t>(levelNodeOffsets.size());
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
                                            //treeGPU.payload.dMicroQuadTreeNode,
                                            // Payload Leaf
                                            treeGPU.payload.dTotalIrradianceLeaf,
                                            treeGPU.payload.dSampleCountLeaf,
                                            treeGPU.payload.dAvgIrradianceLeaf,
                                            treeGPU.payload.dNormalAndSpecLeaf,
                                            treeGPU.payload.dGuidingFactorLeaf,
                                            //treeGPU.payload.dMicroQuadTreeLeaf,
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
                                        totalNodeCount,
                                        //totalNodeCount,
                                        // Payload Leaf
                                        hUniqueVoxelCount, hUniqueVoxelCount, hUniqueVoxelCount,
                                        hUniqueVoxelCount, hUniqueVoxelCount,
                                        //hUniqueVoxelCount,
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
    // Only Direct light information deposition and normal generation are left
    // Call the kernel for it
    gpu.GridStrideKC_X(0, (cudaStream_t)0, hUniqueVoxelCount * WARP_SIZE,
                       //
                       KCReduceVoxelPayload<StaticThreadPerBlock1D>,
                       // I-O
                       treeGPU,
                       // Input
                       dIndexOffsets,
                       dSortedUniqueVoxels,
                       dSortedVoxelIndices, // This is non-unique (we need to reduce it)
                       dVoxelLightKeys,     // Voxel payload that will be reduced
                       dVoxelNormals,       // Voxel payload that will be reduced

                       // Binary Search for light
                       dSortedLightKeys,
                       dSortedLights,
                       totalLightCount,
                       // Constants
                       hUniqueVoxelCount,
                       static_cast<uint32_t>(hTotalVoxCount),
                       treeGPU.svoAABB,
                       resolutionXYZ);

    // Average Normals for each level
    // Bottom-up
    for(uint32_t i = 0; i < levelCount; i++)
    {
        uint32_t levelIndex = levelCount - i - 1;
        bool lastNonLeafLevel = (levelIndex == (levelCount - 1));

        gpu.GridStrideKC_X(0, (cudaStream_t)0, levelNodeCounts[levelIndex],
                           //
                           KCAverageNormals,
                           // I-O
                           treeGPU.payload,
                           // Input
                           treeGPU.dNodes,
                           treeGPU.dLevelNodeOffsets,
                           // Constants
                           levelIndex,
                           levelNodeCounts[levelIndex],
                           lastNonLeafLevel);

    }

    // Filter the initial radiance to the system
    NormalizeAndFilterRadiance(system);

    // Finally Find the boundary light and set the pointer
    DeviceMemory mem(sizeof(GPULightI*));
    const GPULightI** dLightPtr = static_cast<const GPULightI**>(mem);
    CUDA_CHECK(cudaMemset(dLightPtr, 0x00, sizeof(GPULightI**)));
    // Call a kernel to determine the boundary light using the key
    gpu.KC_X(0, (cudaStream_t)0, totalLightCount,
             //
             KCFindBoundaryLight,
             //
             *dLightPtr,
             //
             dSceneLights,
             boundaryLightKey,
             totalLightCount);
    // Copy to host
    CUDA_CHECK(cudaMemcpy(&treeGPU.dBoundaryLight, dLightPtr,
                          sizeof(GPULightI*), cudaMemcpyDeviceToHost));

    // Log some stuff
    timer.Stop();
    double svoMemSize = static_cast<double>(octreeMem.Size()) / 1024.0 / 1024.0;
    std::locale::global(std::locale("en_US.UTF-8"));
    METU_LOG("Scene Aniso-SVO [N: {:L}, L: {:L}] Generated in {:f} seconds. (Total {:.2f} MiB)",
             treeGPU.nodeCount, treeGPU.leafCount, timer.Elapsed<CPUTimeSeconds>(), svoMemSize);
    std::locale::global(std::locale::classic());

    // All Done!
    return TracerError::OK;
}

TracerError AnisoSVOctreeCPU::Constrcut(const std::vector<Byte>& data,
                                        const GPULightI** dSceneLights,
                                        uint32_t totalLightCount,
                                        HitKey boundaryLightKey,
                                        const CudaSystem& system)
{
    Utility::CPUTimer timer;
    timer.Start();

    // Init Zeroes
    treeGPU = {};
    // Start by fetching the "header"
    size_t offset = 0;
    std::memcpy(&treeGPU.svoAABB, data.data() + offset, sizeof(AABB3f));
    offset += sizeof(AABB3f);
    std::memcpy(&treeGPU.voxelResolution, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&treeGPU.leafDepth, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&treeGPU.nodeCount, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&treeGPU.leafCount, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&treeGPU.leafVoxelSize, data.data() + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&treeGPU.levelOffsetCount, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Temp Float Buffer for Conversion
    assert(treeGPU.leafCount >= treeGPU.nodeCount);
    DeviceMemory halfConvertedMemory(treeGPU.leafCount * sizeof(Vector2f));
    // Conversion Function
    auto ConvertAnisoFloatToHalf = [&](Vector2h* dRadiance,
                                       uint32_t totalSize)
    {
        const CudaGPU& gpu = system.BestGPU();
        gpu.GridStrideKC_X(0, (cudaStream_t)0, totalSize,
                           //
                           KCConvertToAnisoHalf,
                           //
                           dRadiance,
                           static_cast<const Vector2f*>(halfConvertedMemory),
                           totalSize);
    };

    // After that generate the sizes
    std::array<size_t, 5> treeSizes = {};
    // Actual Tree Related Sizes
    // Node Amount
    treeSizes[0]  = treeGPU.nodeCount * sizeof(uint64_t);       // "dNodes" Size
    treeSizes[1] = treeGPU.nodeCount * sizeof(uint16_t);        // "dBinInfo" Size
    // Leaf Amount
    treeSizes[2] = treeGPU.leafCount * sizeof(uint32_t);        // "dLeafParents" Size
    treeSizes[3] = treeGPU.leafCount * sizeof(uint16_t);        // "dLeafBinInfo" Size
    // Misc
    treeSizes[4] = treeGPU.levelOffsetCount * sizeof(uint32_t); // "dLevelNodeOffsets" Size

    // Payload Related Sizes
    std::array<size_t, 10> payloadSizes = {};
    // Node Amount
    payloadSizes[0] = treeGPU.nodeCount * sizeof(Vector2f);     // "dAvgIrradianceNode" Size
    payloadSizes[1] = treeGPU.nodeCount * sizeof(uint32_t);     // "dNormalAndSpecNode" Size
    payloadSizes[2] = treeGPU.nodeCount * sizeof(uint8_t);      // "dGuidingFactorNode" Size
    //payloadSizes[3] = treeGPU.nodeCount * sizeof(uint64_t);   // "dMicroQuadTreeNode" Size
    //Leaf Amount
    payloadSizes[4]  = treeGPU.leafCount * sizeof(Vector2f);    // "dTotalIrradianceLeaf" Size
    payloadSizes[5] = treeGPU.leafCount * sizeof(Vector2ui);    // "dSampleCountLeaf" Size
    payloadSizes[6] = treeGPU.leafCount * sizeof(Vector2f);     // "dAvgIrradianceLeaf" Size
    payloadSizes[7] = treeGPU.leafCount * sizeof(uint32_t);     // "dNormalAndSpecLeaf" Size
    payloadSizes[8] = treeGPU.leafCount * sizeof(uint8_t);      // "dGuidingFactorLeaf" Size
    //payloadSizes[9] = treeGPU.leafCount * sizeof(uint64_t);   // "dMicroQuadTreeNode" Size

    // Calculate the offsets and total size
    size_t treeTotalSize = std::reduce(treeSizes.cbegin(), treeSizes.cend(), 0ull);
    size_t payloadTotalSize = std::reduce(payloadSizes.cbegin(), payloadSizes.cend(), 0ull);

    // Sanity check the calculated size
    size_t totalSize = (treeTotalSize + payloadTotalSize +
                        sizeof(AABB3f) + 5 * sizeof(uint32_t) +
                        sizeof(float));
    if(totalSize != data.size())
        return TracerError(TracerError::TRACER_INTERNAL_ERROR, "Read SVO size mismatch");

    // Now allocate the actual buffers
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
                                            //treeGPU.payload.dMicroQuadTreeNode,
                                            // Payload Leaf
                                            treeGPU.payload.dTotalIrradianceLeaf,
                                            treeGPU.payload.dSampleCountLeaf,
                                            treeGPU.payload.dAvgIrradianceLeaf,
                                            treeGPU.payload.dNormalAndSpecLeaf,
                                            treeGPU.payload.dGuidingFactorLeaf,
                                            //treeGPU.payload.dMicroQuadTreeLeaf,
                                            // Node Offsets
                                            treeGPU.dLevelNodeOffsets),
                                   octreeMem,
                                   {
                                        // Node Related
                                        treeGPU.nodeCount, treeGPU.nodeCount,
                                        // Leaf Related
                                        treeGPU.leafCount, treeGPU.leafCount,
                                        // Payload Node
                                        treeGPU.nodeCount, treeGPU.nodeCount,
                                        treeGPU.nodeCount,
                                        //treeGPU.nodeCount,
                                        // Payload Leaf
                                        treeGPU.leafCount, treeGPU.leafCount, treeGPU.leafCount,
                                        treeGPU.leafCount, treeGPU.leafCount,
                                        //treeGPU.leafCount,
                                        // Offsets
                                        treeGPU.levelOffsetCount
                                   });

    // Now copy stuff
    // Tree Related
    // "dNodes"
    CUDA_CHECK(cudaMemcpy(treeGPU.dNodes, data.data() + offset,
                          treeSizes[0], cudaMemcpyHostToDevice));
    offset += treeSizes[0];
    // "dBinInfo"
    CUDA_CHECK(cudaMemcpy(treeGPU.dBinInfo, data.data() + offset,
                          treeSizes[1], cudaMemcpyHostToDevice));
    offset += treeSizes[1];
    // "dLeafParents"
    CUDA_CHECK(cudaMemcpy(treeGPU.dLeafParents, data.data() + offset,
                          treeSizes[2], cudaMemcpyHostToDevice));
    offset += treeSizes[2];
    // "dLeafBinInfo"
    CUDA_CHECK(cudaMemcpy(treeGPU.dLeafBinInfo, data.data() + offset,
                          treeSizes[3], cudaMemcpyHostToDevice));
    offset += treeSizes[3];
    // "dLevelNodeOffsets"
    levelNodeOffsets.resize(treeGPU.levelOffsetCount);
    std::memcpy(levelNodeOffsets.data(), data.data() + offset, treeSizes[4]);
    CUDA_CHECK(cudaMemcpy(treeGPU.dLevelNodeOffsets, data.data() + offset,
                          treeSizes[4], cudaMemcpyHostToDevice));
    offset += treeSizes[4];
    // Payload Related
    // "dAvgIrradianceNode"
    CUDA_CHECK(cudaMemcpy(static_cast<void*>(halfConvertedMemory), data.data() + offset,
                          payloadSizes[0], cudaMemcpyHostToDevice));
    ConvertAnisoFloatToHalf(treeGPU.payload.dAvgIrradianceNode, treeGPU.nodeCount);
    offset += payloadSizes[0];
    // "dNormalAndSpecNode"
    CUDA_CHECK(cudaMemcpy(treeGPU.payload.dNormalAndSpecNode, data.data() + offset,
                          payloadSizes[1], cudaMemcpyHostToDevice));
    offset += payloadSizes[1];
    // "dGuidingFactorNode"
    CUDA_CHECK(cudaMemcpy(treeGPU.payload.dGuidingFactorNode, data.data() + offset,
                          payloadSizes[2], cudaMemcpyHostToDevice));
    offset += payloadSizes[2];
    //// "dMicroQuadTreeNode"
    //CUDA_CHECK(cudaMemcpy(treeGPU.payload.dMicroQuadTreeNode, data.data() + offset,
    //                      payloadSizes[3], cudaMemcpyHostToDevice));
    //offset += payloadSizes[3];
    // "dTotalIrradianceLeaf"
    CUDA_CHECK(cudaMemcpy(treeGPU.payload.dTotalIrradianceLeaf, data.data() + offset,
                          payloadSizes[4], cudaMemcpyHostToDevice));
    offset += payloadSizes[4];
    // "dTotalIrradianceLeaf"
    CUDA_CHECK(cudaMemcpy(treeGPU.payload.dSampleCountLeaf, data.data() + offset,
                          payloadSizes[5], cudaMemcpyHostToDevice));
    offset += payloadSizes[5];
    // "dSampleCountLeaf"
    CUDA_CHECK(cudaMemcpy(static_cast<void*>(halfConvertedMemory), data.data() + offset,
                          payloadSizes[6], cudaMemcpyHostToDevice));
    ConvertAnisoFloatToHalf(treeGPU.payload.dAvgIrradianceLeaf, treeGPU.leafCount);
    offset += payloadSizes[6];
    // "dNormalAndSpecLeaf"
    CUDA_CHECK(cudaMemcpy(treeGPU.payload.dNormalAndSpecLeaf, data.data() + offset,
                          payloadSizes[7], cudaMemcpyHostToDevice));
    offset += payloadSizes[7];
    // "dGuidingFactorLeaf"
    CUDA_CHECK(cudaMemcpy(treeGPU.payload.dGuidingFactorLeaf, data.data() + offset,
                          payloadSizes[8], cudaMemcpyHostToDevice));
    offset += payloadSizes[8];
    //// "dMicroQuadTreeLeaf"
    //CUDA_CHECK(dSampleCountLeaf(treeGPU.payload.dMicroQuadTreeLeaf, data.data() + offset,
    //                            payloadSizes[9], cudaMemcpyHostToDevice));
    //offset += payloadSizes[9];
    // All Done!
    assert(offset == data.size());

    // Finally Find the boundary light
    DeviceMemory mem(sizeof(GPULightI*));
    const GPULightI** dLightPtr = static_cast<const GPULightI**>(mem);
    CUDA_CHECK(cudaMemset(dLightPtr, 0x00, sizeof(GPULightI**)));
    // Call a kernel to determine the boundary light using the key
    const CudaGPU& gpu = system.BestGPU();
    gpu.KC_X(0, (cudaStream_t)0, totalLightCount,
             //
             KCFindBoundaryLight,
             //
             *dLightPtr,
             //
             dSceneLights,
             boundaryLightKey,
             totalLightCount);
    // Copy to host
    CUDA_CHECK(cudaMemcpy(&treeGPU.dBoundaryLight, dLightPtr,
                          sizeof(GPULightI*), cudaMemcpyDeviceToHost));

    // Log some stuff
    timer.Stop();
    double svoMemSize = static_cast<double>(octreeMem.Size()) / 1024.0 / 1024.0;
    std::locale::global(std::locale("en_US.UTF-8"));
    METU_LOG("Scene Aniso-SVO [N: {:L}, L: {:L}] Loaded from disk in {:f} seconds. (Total {:.2f} MiB)",
             treeGPU.nodeCount, treeGPU.leafCount,
             timer.Elapsed<CPUTimeSeconds>(), svoMemSize);
    std::locale::global(std::locale::classic());

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

    const CudaGPU& gpu = system.BestGPU();
    const uint32_t levelCount = treeGPU.leafDepth;
    // Bottom-up filter and populate the intermediate nodes
    for(uint32_t i = 0; i < levelCount; i++)
    {
        uint32_t levelIndex = levelCount - i - 1;
        bool lastNonLeafLevel = (levelIndex == (levelCount - 1));
        uint32_t levelNodeCount = (levelNodeOffsets[levelIndex + 1] -
                                   levelNodeOffsets[levelIndex]);
        gpu.GridStrideKC_X(0, (cudaStream_t)0, levelNodeCount,
                           //
                           KCBottomUpFilterIrradiance,
                           // I-O
                           treeGPU.payload,
                           // Input
                           treeGPU.dNodes,
                           treeGPU.dLevelNodeOffsets,
                           // Constants
                           levelIndex,
                           levelNodeCount,
                           lastNonLeafLevel);
    }
    // Top-Down saturate the values
    // Skip parent
    for(uint32_t i = 1; i <= levelCount; i++)
    {
        uint32_t levelIndex = i;
        bool isLeafLevel = (levelIndex == levelCount);
        uint32_t levelNodeCount = (levelNodeOffsets[levelIndex + 1] -
                                   levelNodeOffsets[levelIndex]);
        gpu.GridStrideKC_X(0, (cudaStream_t)0, levelNodeCount,
                           //
                           KCTopDownFilterIrradiance,
                           // I-O
                           treeGPU.payload,
                           // Input
                           treeGPU.dNodes,
                           treeGPU.dLeafParents,
                           treeGPU.dLevelNodeOffsets,
                           // Constants
                           levelIndex,
                           levelNodeCount,
                           isLeafLevel);
    }
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
    CUDA_CHECK(cudaMemset(treeGPU.dLeafBinInfo, 0x00, sizeof(uint16_t) * treeGPU.leafCount));
    CUDA_CHECK(cudaMemset(treeGPU.dBinInfo, 0x00, sizeof(uint16_t) * treeGPU.nodeCount));
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
    std::array<size_t, 5> treeSizes = {};
    // Actual Tree Related Sizes
    // Node Amount
    treeSizes[0]  = treeGPU.nodeCount * sizeof(uint64_t);       // "dNodes" Size
    treeSizes[1] = treeGPU.nodeCount * sizeof(uint16_t);        // "dBinInfo" Size
    // Leaf Amount
    treeSizes[2] = treeGPU.leafCount * sizeof(uint32_t);        // "dLeafParents" Size
    treeSizes[3] = treeGPU.leafCount * sizeof(uint16_t);        // "dLeafBinInfo" Size
    // Misc
    treeSizes[4] = treeGPU.levelOffsetCount * sizeof(uint32_t); // "dLevelNodeOffsets" Size

    // Payload Related Sizes
    std::array<size_t, 10> payloadSizes = {};
    // Node Amount
    payloadSizes[0] = treeGPU.nodeCount * sizeof(Vector2f);     // "dAvgIrradianceNode" Size
    payloadSizes[1] = treeGPU.nodeCount * sizeof(uint32_t);     // "dNormalAndSpecNode" Size
    payloadSizes[2] = treeGPU.nodeCount * sizeof(uint8_t);      // "dGuidingFactorNode" Size
    //payloadSizes[3] = treeGPU.nodeCount * sizeof(uint64_t);   // "dMicroQuadTreeNode" Size
    //Leaf Amount
    payloadSizes[4]  = treeGPU.leafCount * sizeof(Vector2f);    // "dTotalIrradianceLeaf" Size
    payloadSizes[5] = treeGPU.leafCount * sizeof(Vector2ui);    // "dSampleCountLeaf" Size
    payloadSizes[6] = treeGPU.leafCount * sizeof(Vector2f);     // "dAvgIrradianceLeaf" Size
    payloadSizes[7] = treeGPU.leafCount * sizeof(uint32_t);     // "dNormalAndSpecLeaf" Size
    payloadSizes[8] = treeGPU.leafCount * sizeof(uint8_t);      // "dGuidingFactorLeaf" Size
    //payloadSizes[9] = treeGPU.leafCount * sizeof(uint64_t);   // "dMicroQuadTreeNode" Size

    // Calculate the offsets and total size
    size_t treeTotalSize = std::reduce(treeSizes.cbegin(), treeSizes.cend(), 0ull);
    size_t payloadTotalSize = std::reduce(payloadSizes.cbegin(), payloadSizes.cend(), 0ull);

    size_t totalSize = (treeTotalSize +
                        payloadTotalSize +
                        sizeof(AABB3f) +
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

    // Tree Related
    // "dNodes"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.dNodes,
                          treeSizes[0], cudaMemcpyDeviceToHost));
    offset += treeSizes[0];
    // "dBinInfo"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.dBinInfo,
                          treeSizes[1], cudaMemcpyDeviceToHost));
    offset += treeSizes[1];
    // "dLeafParents"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.dLeafParents,
                          treeSizes[2], cudaMemcpyDeviceToHost));
    offset += treeSizes[2];
    // "dLeafBinInfo"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.dLeafBinInfo,
                          treeSizes[3], cudaMemcpyDeviceToHost));
    offset += treeSizes[3];
    // "dLevelNodeOffsets"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.dLevelNodeOffsets,
                          treeSizes[4], cudaMemcpyDeviceToHost));
    offset += treeSizes[4];
    // Payload Related
    // "dAvgIrradianceNode"
    ConvertAnisoHalfToFloat(treeGPU.payload.dAvgIrradianceNode, treeGPU.nodeCount);
    CUDA_CHECK(cudaMemcpy(data.data() + offset, static_cast<void*>(halfConvertedMemory),
                          payloadSizes[0], cudaMemcpyDeviceToHost));
    offset += payloadSizes[0];
    // "dNormalAndSpecNode"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.payload.dNormalAndSpecNode,
                          payloadSizes[1], cudaMemcpyDeviceToHost));
    offset += payloadSizes[1];
    // "dGuidingFactorNode"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.payload.dGuidingFactorNode,
                          payloadSizes[2], cudaMemcpyDeviceToHost));
    offset += payloadSizes[2];
    //// "dMicroQuadTreeNode"
    //CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.payload.dMicroQuadTreeNode,
    //                      payloadSizes[3], cudaMemcpyDeviceToHost));
    //offset += payloadSizes[3];
    // "dTotalIrradianceLeaf"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.payload.dTotalIrradianceLeaf,
                          payloadSizes[4], cudaMemcpyDeviceToHost));
    offset += payloadSizes[4];
    // "dSampleCountLeaf"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.payload.dSampleCountLeaf,
                          payloadSizes[5], cudaMemcpyDeviceToHost));
    offset += payloadSizes[5];
    // "dAvgIrradianceLeaf"
    ConvertAnisoHalfToFloat(treeGPU.payload.dAvgIrradianceLeaf, treeGPU.leafCount);
    CUDA_CHECK(cudaMemcpy(data.data() + offset, static_cast<void*>(halfConvertedMemory),
                          payloadSizes[6], cudaMemcpyDeviceToHost));
    offset += payloadSizes[6];
    // "dNormalAndSpecLeaf"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.payload.dNormalAndSpecLeaf,
                          payloadSizes[7], cudaMemcpyDeviceToHost));
    offset += payloadSizes[7];
    // "dGuidingFactorLeaf"
    CUDA_CHECK(cudaMemcpy(data.data() + offset, treeGPU.payload.dGuidingFactorLeaf,
                          payloadSizes[8], cudaMemcpyDeviceToHost));
    offset += payloadSizes[8];
    //// "dMicroQuadTreeLeaf"
    //CUDA_CHECK(dSampleCountLeaf(data.data() + offset, treeGPU.payload.dMicroQuadTreeLeaf,
    //                            payloadSizes[9], cudaMemcpyDeviceToHost));
    //offset += payloadSizes[9];
    // All Done!
    assert(offset == data.size());
}