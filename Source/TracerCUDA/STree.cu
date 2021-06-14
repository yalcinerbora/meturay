#include "STree.cuh"
#include "PathNode.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "ParallelPartition.cuh"
#include "STreeKC.cuh"

#include "RayLib/MemoryAlignment.h"

#include <cub/cub.cuh>

static constexpr size_t AlignedOffsetSTreeGPU = Memory::AlignSize(sizeof(STreeGPU));

struct FetchTreeIdFunctor
{
    __device__ __host__ __forceinline__
    uint32_t operator()(const PathGuidingNode& node) const
    {
        return node.nearestDTreeIndex;
    }
};

struct IsSplittedLeafFunctor
{
    __device__ __host__ __forceinline__
    bool operator()(const uint32_t& index) const
    {
        return (index != INVALID_NODE);
    }
};

void STree::LinearizeDTreeGPUPtrs(DeviceMemory& mem, bool readTree, size_t offset)
{
    size_t currentSize = dTrees.size() - offset;
    std::vector<DTreeGPU*> hTreePtrs(currentSize);

    for(uint32_t i = 0; i < currentSize; i++)
    {
        hTreePtrs[i] = dTrees[offset + i].TreeGPU(readTree);
    }

    DeviceMemory::EnlargeBuffer(mem, currentSize * sizeof(DTreeGPU*));
    CUDA_CHECK(cudaMemcpy(static_cast<DTreeGPU**>(mem),
                          hTreePtrs.data(),
                          currentSize * sizeof(DTreeGPU*),
                          cudaMemcpyHostToDevice));
}

void STree::LinearizeDTreeGPUPtrs(DeviceMemory& mem)
{
    size_t currentTreeCount = dTrees.size();
    std::vector<DTreeGPU*> hReadTreePtrs(currentTreeCount);
    std::vector<DTreeGPU*> hWriteTreePtrs(currentTreeCount);

    size_t totalMemSize = currentTreeCount * sizeof(DTreeGPU*) * 2;

    for(uint32_t i = 0; i < currentTreeCount; i++)
    {
        hReadTreePtrs[i] = dTrees[i].TreeGPU(true);
        hWriteTreePtrs[i] = dTrees[i].TreeGPU(false);
    }

    DeviceMemory::EnlargeBuffer(mem, totalMemSize);
    // Set new pointers
    dReadDTrees = static_cast<const DTreeGPU**>(mem);
    dWriteDTrees = static_cast<DTreeGPU**>(mem) + currentTreeCount;
    // Copy data
    CUDA_CHECK(cudaMemcpy(dReadDTrees,
                          hReadTreePtrs.data(),
                          currentTreeCount * sizeof(DTreeGPU*),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWriteDTrees,
                          hWriteTreePtrs.data(),
                          currentTreeCount * sizeof(DTreeGPU*),
                          cudaMemcpyHostToDevice));
}

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
    Byte* nodeStart = static_cast<Byte*>(newMem) + AlignedOffsetSTreeGPU;
    Byte* nodePtrLoc = static_cast<Byte*>(newMem) + offsetof(STreeGPU, gRoot);
    CUDA_CHECK(cudaMemcpy(nodePtrLoc, &nodeStart, sizeof(STreeNode*),
                          cudaMemcpyHostToDevice));    
    memory = std::move(newMem);    
}

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
    // Copy AABB aswell
    // Copy slightly larger AABB to prevent numerical unstabilities
    AABB3f sceneAABB = AABB3f(sceneExtents.Min() - MathConstants::Epsilon,
                              sceneExtents.Max() + MathConstants::Epsilon);
    Byte* nodeAABBLoc = static_cast<Byte*>(memory) + offsetof(STreeGPU, extents);
    CUDA_CHECK(cudaMemcpy(nodeAABBLoc, &sceneAABB, sizeof(AABB3f),
                          cudaMemcpyHostToDevice));

    // Create a default single tree
    dTrees.reserve(INITIAL_TREE_RESERVE_COUNT);
    dTrees.emplace_back();

    // Adjust DTree pointers for Tracer Kernels
    LinearizeDTreeGPUPtrs(rwDTreeGPUBuffer);
}

void STree::SplitLeaves(uint32_t maxSamplesPerNode,
                        const CudaSystem& system)
{
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));    
    // Temporary memories
    size_t deviceIfTempMemSize;
    DeviceMemory writeDTreeGPUBuffer;
    DeviceMemory oldTreeIds;
    DeviceMemory tempMemory;
    DeviceMemory selectedIndices;
    DeviceMemory splitMarks;

    // Loop untill no subdivision is left
    uint32_t offset = 0;
    uint32_t processedNodeCount = static_cast<uint32_t>(nodeCount);
    while(processedNodeCount > 0)
    {
        // Resize if buffer if required
        cub::DeviceSelect::If(nullptr, deviceIfTempMemSize,
                              static_cast<uint32_t*>(splitMarks),
                              static_cast<uint32_t*>(splitMarks),
                              static_cast<uint32_t*>(splitMarks),
                              static_cast<int>(processedNodeCount),
                              IsSplittedLeafFunctor());

        DeviceMemory::EnlargeBuffer(tempMemory, deviceIfTempMemSize);
        DeviceMemory::EnlargeBuffer(splitMarks, processedNodeCount * sizeof(uint32_t));
        DeviceMemory::EnlargeBuffer(selectedIndices, (processedNodeCount + 1) *sizeof(uint32_t));
        // Get a new write tree list
        LinearizeDTreeGPUPtrs(writeDTreeGPUBuffer, false);
        DTreeGPU** dWriteDTrees = static_cast<DTreeGPU**>(writeDTreeGPUBuffer);

        // Mark Leafs
        gpu.GridStrideKC_X(0, 0, nodeCount,
                           //
                           KCMarkSTreeSplitLeaf,
                           //
                           static_cast<uint32_t*>(splitMarks),
                           *dSTree,
                           dWriteDTrees,
                           maxSamplesPerNode,
                           offset,
                           static_cast<uint32_t>(processedNodeCount));

        // Make dense leaf indices from sparse mark indices
        uint32_t* dDenseIndexCount = static_cast<uint32_t*>(selectedIndices);
        uint32_t* dDenseIndices = static_cast<uint32_t*>(selectedIndices) + 1;
        cub::DeviceSelect::If(static_cast<void*>(tempMemory), deviceIfTempMemSize,
                              static_cast<uint32_t*>(splitMarks),
                              dDenseIndices, dDenseIndexCount,
                              static_cast<int>(processedNodeCount),
                              IsSplittedLeafFunctor());

        // Check how many new trees we need to create
        // then allocate these trees
        uint32_t hSubdivisionCount;
        CUDA_CHECK(cudaMemcpy(&hSubdivisionCount, dDenseIndexCount,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        // No need to continue since there are no leaves to split
        if(hSubdivisionCount == 0) break;
        // Each individual node will create two childs
        uint32_t extraChildCount = hSubdivisionCount * 2;
        // And we need one extra tree
        uint32_t extraTreeCount = hSubdivisionCount;

        // Old Tree count will be the next "allocation"
        uint32_t oldTreeCount = static_cast<uint32_t>(dTrees.size());    
        // Expand nodes
        uint32_t oldNodeCount = static_cast<uint32_t>(nodeCount);
        ExpandTree(nodeCount + extraChildCount);
        nodeCount += extraChildCount;

        DeviceMemory::EnlargeBuffer(oldTreeIds, hSubdivisionCount * sizeof(uint32_t));
        gpu.GridStrideKC_X(0, 0, hSubdivisionCount,
                           //
                           KCSplitSTree,
                           //
                           static_cast<uint32_t*>(oldTreeIds),
                           *dSTree,
                           //
                           dDenseIndices,
                           //
                           offset,
                           oldNodeCount,
                           oldTreeCount,
                           hSubdivisionCount);
        // Copy old indices to the CPU
        std::vector<uint32_t> hOldTreeIds(hSubdivisionCount);
        CUDA_CHECK(cudaMemcpy(hOldTreeIds.data(),
                              static_cast<uint32_t*>(oldTreeIds),
                              hSubdivisionCount * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        // Create the tree copies
        for(uint32_t i = 0; i < extraTreeCount; i++)
        {
            // Copy the old tree to the new
            DTree& oldTree = dTrees[hOldTreeIds[i]];
            dTrees.push_back(oldTree);
        }

        // Now get redy for next iteration
        offset = oldNodeCount;
        processedNodeCount = extraChildCount;

    }
    // Finally copy the new node count to the GPU
    Byte* nodeCountLocPtr = static_cast<Byte*>(memory) + offsetof(STreeGPU, nodeCount);
    CUDA_CHECK(cudaMemcpy(nodeCountLocPtr, &nodeCount, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Subdivided recursively untill all leaf nodes 
    // have sample count less than "maxSamplesPerNode"
    // All done!
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
        // Skip if these nodes are invalid
        if(treeIndex == InvalidDTreeIndex) continue;

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

void STree::SwapTrees(float fluxRatio, uint32_t depthLimit, const CudaSystem& system)
{
    const GPUList& gpuList = system.GPUList();
    auto currentGPU = gpuList.cbegin();
    for(DTree& dTree : dTrees)
    {
        // Call swap function for a kernel
        dTree.SwapTrees(fluxRatio, depthLimit, *currentGPU);
        // Get a next GPU if exausted all gpus
        // rool back to start
        currentGPU++;
        if(currentGPU == gpuList.cend()) currentGPU = gpuList.cbegin();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void STree::SplitAndSwapTrees(uint32_t sTreeMaxSamplePerLeaf,
                              float dTreeFluxRatio, 
                              uint32_t dTreeDepthLimit,
                              const CudaSystem& system)
{   
    SplitLeaves(sTreeMaxSamplePerLeaf, system);
    SwapTrees(dTreeFluxRatio, dTreeDepthLimit, system);

    // Adjust DTree pointers for Tracer Kernels
    LinearizeDTreeGPUPtrs(rwDTreeGPUBuffer);
}

void STree::GetTreeToCPU(STreeGPU& treeCPU, std::vector<STreeNode>& nodesCPU) const
{
    CUDA_CHECK(cudaMemcpy(&treeCPU, dSTree, sizeof(STreeGPU),
                          cudaMemcpyDeviceToHost));
    nodesCPU.resize(nodeCount);
    const STreeNode* dSTreeNodes = treeCPU.gRoot;
    CUDA_CHECK(cudaMemcpy(nodesCPU.data(), dSTreeNodes, 
                          nodeCount * sizeof(STreeNode),
                          cudaMemcpyDeviceToHost));
}

void STree::GetAllDTreesToCPU(std::vector<DTreeGPU>& dTreeStructs,
                              std::vector<std::vector<DTreeNode>>& dTreeNodes,
                              bool fetchReadTree) const
{
    dTreeStructs.reserve(dTrees.size());
    dTreeNodes.reserve(dTrees.size());
    for(const DTree& dT : dTrees)
    {
        std::vector<DTreeNode> currentNodes;
        DTreeGPU currentStruct;
        if(fetchReadTree)
        {
            dT.GetReadTreeToCPU(currentStruct, currentNodes);
        }
        else
        {
            dT.GetWriteTreeToCPU(currentStruct, currentNodes);
        }
        
        dTreeStructs.push_back(std::move(currentStruct));
        dTreeNodes.push_back(std::move(currentNodes));
    }
}