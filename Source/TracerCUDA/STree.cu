#include "STree.cuh"
#include "PathNode.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "ParallelPartition.cuh"

#include "RayLib/MemoryAlignment.h"

static constexpr size_t AlignedOffsetSTreeGPU = Memory::AlignSize(sizeof(STreeGPU));

void STree::ExpandTree(size_t newNodeCount)
{
    // If its already large do not do stuff
    size_t currentCapacity = 0;
    if(memory.Size() > AlignedOffsetSTreeGPU)
        currentCapacity = (memory.Size() - AlignedOffsetSTreeGPU) / sizeof(STreeGPU);    
    if(currentCapacity >= newNodeCount) return;
    
    DeviceMemory newMem(AlignedOffsetSTreeGPU + newNodeCount * sizeof(STreeNode));
    // Copy the old stuff
    if(memory.Size() > 0)
        CUDA_CHECK(cudaMemcpy(static_cast<Byte*>(newMem),
                              static_cast<const Byte*>(memory),
                              AlignedOffsetSTreeGPU + nodeCount * sizeof(STreeNode),
                              cudaMemcpyHostToDevice));

    // Set new STree
    dSTree = static_cast<STreeGPU*>(newMem);
    // Copy the new node pointer
    Byte* nodeStart = static_cast<Byte*>(memory) + AlignedOffsetSTreeGPU;
    Byte* nodePtrLoc = static_cast<Byte*>(memory) + offsetof(STreeGPU, gRoot);
    CUDA_CHECK(cudaMemcpy(nodePtrLoc, &nodeStart, sizeof(STreeNode*),
                          cudaMemcpyHostToDevice));
    
    memory = std::move(newMem);    
}

struct FetchTreeIdFunctor
{
    __device__ __host__ __forceinline__
    uint32_t operator()(const PathGuidingNode& node) const
    {
        return node.nearestDTreeIndex;
    }
};

STree::STree(const AABB3f& sceneExtents)
    : nodeCount(0)
    , dSTree(nullptr)
{
    ExpandTree(INITIAL_NODE_CAPACITY);

    // There should be at least one node
    STreeNode node;
    node.isLeaf = true;
    node.splitAxis = STreeNode::AxisType::X;
    node.index = 0; // This shows tree index since isLeaf is true
    Byte* nodeStart = static_cast<Byte*>(memory) + AlignedOffsetSTreeGPU;
    CUDA_CHECK(cudaMemcpy(nodeStart, &node, sizeof(STreeNode),
                          cudaMemcpyHostToDevice));

    // Update total node count aswell
    nodeCount = 1;
    Byte* nodeCountLocPtr = static_cast<Byte*>(memory) + offsetof(STreeGPU, nodeCount);
    CUDA_CHECK(cudaMemcpy(nodeCountLocPtr, &nodeCount, sizeof(uint32_t),
               cudaMemcpyHostToDevice));
    // Create a deafault single tree
    dTrees.emplace_back();
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

    const GPUList& gpuList = system.GPUList();
    auto currentGPU = gpuList.cbegin();
    // Call kernels
    for(const auto& partition : partitions)
    {        
        uint32_t treeIndex = partition.portionId;
        dTrees[treeIndex].AddRadiancesFromPaths(static_cast<uint32_t*>(sortedIndices),
                                                dPGNodes, partition,
                                                maxPathNodePerRay,
                                                *currentGPU);
        // Get a next GPU if exausted all gpus
        // rool back to start
        currentGPU++;
        if(currentGPU == gpuList.cend()) currentGPU = gpuList.cbegin();
    }
    
    // Wait all gpus to finish
    system.SyncAllGPUs();
}