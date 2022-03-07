#include "STree.cuh"
#include "PathNode.cuh"
#include "STreeKC.cuh"
#include "TracerDebug.h"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include "RayLib/MemoryAlignment.h"

#include <cub/cub.cuh>

static constexpr size_t AlignedOffsetSTreeGPU = Memory::AlignSize(sizeof(STreeGPU));

struct IsSplittedLeafFunctor
{
    __device__ __host__ inline
    bool operator()(const uint32_t& index) const
    {
        return (index != INVALID_NODE);
    }
};

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

void STree::LoadSDTree(const std::string& path, const CudaSystem& system)
{
    std::ifstream file(path, std::ios::binary);
    std::istreambuf_iterator<char>fileIt(file);
    static_assert(sizeof(char) == sizeof(Byte), "\"Byte\" is not have sizeof(char)");
    // Read STree Start Offset
    uint64_t sTreeOffset;
    file.read(reinterpret_cast<char*>(&sTreeOffset), sizeof(uint64_t));
    // Read STree Node Count
    uint64_t sTreeNodeCount;
    file.read(reinterpret_cast<char*>(&sTreeNodeCount), sizeof(uint64_t));
    // Read DTree Count
    uint64_t dTreeCount;
    file.read(reinterpret_cast<char*>(&dTreeCount), sizeof(uint64_t));
    // Read DTree Offset/Count Pairs
    std::vector<Vector2ul> offsetCountPairs(dTreeCount);
    file.read(reinterpret_cast<char*>(offsetCountPairs.data()), sizeof(Vector2ul) * dTreeCount);
    // Read STree
    // Extents
    AABB3f sceneExtents;
    file.read(reinterpret_cast<char*>(&sceneExtents), sizeof(AABB3f));
    // Nodes
    std::vector<STreeNode> hSTreeNodes(sTreeNodeCount);
    file.read(reinterpret_cast<char*>(hSTreeNodes.data()),
              sizeof(STreeNode) * sTreeNodeCount);
    // Expand the Tree
    ExpandTree(sTreeNodeCount);
    Byte* extentsLoc = reinterpret_cast<Byte*>(dSTree) + offsetof(STreeGPU, extents);
    CUDA_CHECK(cudaMemcpy(extentsLoc, &sceneExtents,
                          sizeof(AABB3f), cudaMemcpyHostToDevice));
    // Copy the nodes
    STreeNode* dSTreeNodes;
    const Byte* nodePtrLoc = reinterpret_cast<const Byte*>(dSTree) + offsetof(STreeGPU, gRoot);
    CUDA_CHECK(cudaMemcpy(&dSTreeNodes, nodePtrLoc,
                          sizeof(STreeNode*), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dSTreeNodes, hSTreeNodes.data(),
                          hSTreeNodes.size() * sizeof(STreeNode),
                          cudaMemcpyHostToDevice));


    // Read DTrees in order
    std::vector<std::pair<uint32_t, float>> hDTreeBases;
    std::vector<std::vector<DTreeNode>> hDTreeNodes;
    for(uint64_t i = 0; i < dTreeCount; i++)
    {
        size_t fileOffset = offsetCountPairs[i][0];
        size_t nCount = offsetCountPairs[i][1];

        file.seekg(fileOffset);
        // Read Base
        std::pair<uint32_t, float> dTreeBase;
        file.read(reinterpret_cast<char*>(&dTreeBase.first), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&dTreeBase.second), sizeof(float));
        // Read Nodes
        std::vector<DTreeNode> dTreeNodes(nCount);
        file.read(reinterpret_cast<char*>(dTreeNodes.data()), nCount * sizeof(DTreeNode));
        // Move to the struct
        hDTreeBases.emplace_back(std::move(dTreeBase));
        hDTreeNodes.emplace_back(std::move(dTreeNodes));
    }

    // Reset the d-trees using the data
    dTrees.InitializeTrees(hDTreeBases, hDTreeNodes, system);
}

STree::STree(const AABB3f& sceneExtents,
             const CudaSystem& system)
    : dSTree(nullptr)
    , nodeCount(0)
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

    // Update total node count as well
    nodeCount = 1;
    Byte* nodeCountLocPtr = static_cast<Byte*>(memory) + offsetof(STreeGPU, nodeCount);
    CUDA_CHECK(cudaMemcpy(nodeCountLocPtr, &nodeCount, sizeof(uint32_t),
               cudaMemcpyHostToDevice));
    // Copy AABB as well
    // Copy slightly larger AABB to prevent numerical instabilities
    AABB3f sceneAABB = AABB3f(sceneExtents.Min() - MathConstants::Epsilon,
                              sceneExtents.Max() + MathConstants::Epsilon);
    Byte* nodeAABBLoc = static_cast<Byte*>(memory) + offsetof(STreeGPU, extents);
    CUDA_CHECK(cudaMemcpy(nodeAABBLoc, &sceneAABB, sizeof(AABB3f),
                          cudaMemcpyHostToDevice));

    dTrees.AllocateDefaultTrees(1, system);
}

STree::STree(const std::string& sdTreePath,
             const CudaSystem& system)
{
    LoadSDTree(sdTreePath, system);
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

    // Loop until no subdivision is left
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

        GPUMemFuncs::EnlargeBuffer(tempMemory, deviceIfTempMemSize);
        GPUMemFuncs::EnlargeBuffer(splitMarks, processedNodeCount * sizeof(uint32_t));
        GPUMemFuncs::EnlargeBuffer(selectedIndices, (processedNodeCount + 1) *sizeof(uint32_t));

        // Mark Leafs
        gpu.GridStrideKC_X(0, 0, nodeCount,
                           //
                           KCMarkSTreeSplitLeaf,
                           //
                           static_cast<uint32_t*>(splitMarks),
                           *dSTree,
                           dTrees.WriteTrees(),
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
        // Each individual node will create two children
        uint32_t extraChildCount = hSubdivisionCount * 2;

        // Old Tree count will be the next "allocation"
        uint32_t oldTreeCount = dTrees.TreeCount();
        // Expand nodes
        uint32_t oldNodeCount = static_cast<uint32_t>(nodeCount);
        ExpandTree(nodeCount + extraChildCount);
        nodeCount += extraChildCount;

        GPUMemFuncs::EnlargeBuffer(oldTreeIds, hSubdivisionCount * sizeof(uint32_t));
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
        dTrees.AllocateExtra(hOldTreeIds, system);

        // Now get ready for next iteration
        offset = oldNodeCount;
        processedNodeCount = extraChildCount;

    }
    // Finally copy the new node count to the GPU
    Byte* nodeCountLocPtr = static_cast<Byte*>(memory) + offsetof(STreeGPU, nodeCount);
    CUDA_CHECK(cudaMemcpy(nodeCountLocPtr, &nodeCount, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Subdivided recursively until all leaf nodes
    // have sample count less than "maxSamplesPerNode"
    // All done!
}

void STree::AccumulateRaidances(const PPGPathNode* dPGNodes,
                                uint32_t totalNodeCount,
                                uint32_t maxPathNodePerRay,
                                const CudaSystem& system)
{
    if(totalNodeCount == 0) return;

    dTrees.AddRadiancesFromPaths(dPGNodes,
                                 totalNodeCount,
                                 maxPathNodePerRay,
                                 system);
}

void STree::SwapTrees(float fluxRatio, uint32_t depthLimit, const CudaSystem& system)
{
    dTrees.SwapTrees(fluxRatio, depthLimit, system);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void STree::SplitAndSwapTrees(uint32_t sTreeMaxSamplePerLeaf,
                              float dTreeFluxRatio,
                              uint32_t dTreeDepthLimit,
                              const CudaSystem& system)
{
    SplitLeaves(sTreeMaxSamplePerLeaf, system);
    SwapTrees(dTreeFluxRatio, dTreeDepthLimit, system);

    system.SyncAllGPUs();
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

const DTreeGroup& STree::DTrees() const
{
    return dTrees;
}

void STree::GetAllDTreesToCPU(std::vector<DTreeGPU>& dTreeStructs,
                              std::vector<std::vector<DTreeNode>>& dTreeNodes,
                              bool fetchReadTree) const
{
    dTreeStructs.reserve(dTrees.TreeCount());
    dTreeNodes.reserve(dTrees.TreeCount());
    for(uint32_t i = 0 ; i < dTrees.TreeCount(); i++)
    {
        DTreeGPU currentStruct;
        std::vector<DTreeNode> currentNodes;
        if(fetchReadTree)
        {
            dTrees.GetReadTreeToCPU(currentStruct, currentNodes, i);
        }
        else
        {
            dTrees.GetWriteTreeToCPU(currentStruct, currentNodes, i);
        }
        dTreeStructs.push_back(std::move(currentStruct));
        dTreeNodes.push_back(std::move(currentNodes));
    }
}

void STree::DumpSDTreeAsBinary(std::vector<Byte>& data,
                               bool fetchReadTree) const
{
    std::vector<Byte> sTree;
    std::vector<std::vector<Byte>> dTreeBinary(dTrees.TreeCount());
    std::vector<Vector2ul> countOffsetPairs(dTrees.TreeCount());

    uint64_t sTreeStartOffset = (sizeof(uint64_t) +
                                 sizeof(uint64_t) +
                                 sizeof(uint64_t) +
                                 sizeof(Vector2ul) * dTrees.TreeCount());


    size_t offset = (sTreeStartOffset +
                     sizeof(AABB3f) +
                     nodeCount * sizeof(STreeNode));
    for(uint32_t i = 0; i < dTrees.TreeCount(); i++)
    {
        uint32_t dTreeNodeCount;
        dTrees.DumpTreeAsBinary(dTreeBinary[i], dTreeNodeCount, i, fetchReadTree);
        countOffsetPairs[i] = Vector2ul(static_cast<uint64_t>(offset),
                                        static_cast<uint64_t>(dTreeNodeCount));
        offset += dTreeBinary[i].size();
    }
    data.reserve(offset);

    // Write STree Start Offset
    data.insert(data.end(),
                reinterpret_cast<Byte*>(&sTreeStartOffset),
                reinterpret_cast<Byte*>(&sTreeStartOffset) + sizeof(uint64_t));
    // Write STree Node Count
    uint64_t sTreeNodeCount = static_cast<uint64_t>(nodeCount);
    data.insert(data.end(),
                reinterpret_cast<Byte*>(&sTreeNodeCount),
                reinterpret_cast<Byte*>(&sTreeNodeCount) + sizeof(uint64_t));
    // Write DTree Count
    uint64_t dTreeCount = static_cast<uint64_t>(dTrees.TreeCount());
    assert(countOffsetPairs.size() == dTreeCount);
    data.insert(data.end(),
                reinterpret_cast<Byte*>(&dTreeCount),
                reinterpret_cast<Byte*>(&dTreeCount) + sizeof(uint64_t));
    // Write DTree Offset/Count Pairs
    data.insert(data.end(),
                reinterpret_cast<Byte*>(countOffsetPairs.data()),
                (reinterpret_cast<Byte*>(countOffsetPairs.data()) +
                 sizeof(Vector2ul) * countOffsetPairs.size()));
    // Write STree
    STreeGPU sTreeBase;
    std::vector<STreeNode> sTreeNodes;
    GetTreeToCPU(sTreeBase, sTreeNodes);
    data.insert(data.end(),
                reinterpret_cast<Byte*>(&sTreeBase.extents),
                reinterpret_cast<Byte*>(&sTreeBase.extents) + sizeof(AABB3f));
    data.insert(data.end(),
                reinterpret_cast<Byte*>(sTreeNodes.data()),
                (reinterpret_cast<Byte*>(sTreeNodes.data()) +
                 sizeof(STreeNode) * sTreeNodes.size()));

    // Write DTrees in order
    for(const std::vector<Byte>& dTree : dTreeBinary)
    {
        data.insert(data.end(), dTree.cbegin(), dTree.cend());
    }
}