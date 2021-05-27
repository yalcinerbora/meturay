#include "STree.cuh"
#include "ParallelSequence.cuh"
#include "PathNode.cuh"

#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include "ParallelPartition.cuh"

struct FetchTreeIdFunctor
{
    __device__ __host__ __forceinline__
    uint32_t operator()(const PathGuidingNode& node) const
    {
        return node.nearestDTreeIndex;
    }
};

STree::STree(const AABB3f& sceneExtents)
{

}

void STree::SplitLeaves(const CudaSystem& system)
{
    // Check the split cretaria on the leaf and respond...


}

void STree::AccumulateRaidances(const PathGuidingNode* dPGNodes,
                                uint32_t totalNodeCount,
                                uint32_t maxPathNodePerRay,                                
                                const CudaSystem& system)
{   
    const CudaGPU& bestGPU = system.BestGPU();

    std::set<ArrayPortion<uint32_t>> partitions;
    DeviceMemory sortedIndices;

    CUDA_CHECK(cudaSetDevice(bestGPU.DeviceId()));    
    PartitionGPU(partitions, sortedIndices,
                 dPGNodes, totalNodeCount,
                 FetchTreeIdFunctor(),
                 static_cast<uint32_t>(dTrees.size()));
}