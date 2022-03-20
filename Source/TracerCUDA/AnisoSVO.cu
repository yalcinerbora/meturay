#include "AnisoSVO.cuh"

#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "PathNode.cuh"
#include "RayLib/ColorConversion.h"

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
                                 const AcceleratorBatchMap&,
                                 const GPULightI** dSceneLights,
                                 const CudaSystem&)
{
    // Generate Light / HitKey sorted array (for binary search)

    // For each accelerator
    // Ask each primitive for rasterize voxel count

    // Allocate enough memory
    // (uint32_t for voxel id, float8 for anisotropic  radiance if a light defined for that prim)

    // For each accelerator
    // Actually rasterize the primitives
    // and push to the memory (calculate the emitted radiance if available here)

    // Sort voxels
    // Run-length encode the voxels (use cub here)
    // so that it will remove the duplicates

    // Top down-generate the voxels
    // For each level save the node range for efficient kernel calls later (level by level kernel calls)
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