#include "DTree.cuh"
#include "DTreeKC.cuh"
#include "PathNode.h"
#include "ParallelReduction.cuh"
#include "CudaSystem.hpp"

#include "RayLib/Types.h"
#include "RayLib/MemoryAlignment.h"

static constexpr size_t AlignedOffsetDTreeGPU = Memory::AlignSize(sizeof(DTreeGPU));

void DTree::DTreeBuffer::FixPointers()
{
    Byte* nodeStart = static_cast<Byte*>(memory) + AlignedOffsetDTreeGPU;
    Byte* nodePtrLoc = static_cast<Byte*>(memory) + offsetof(DTreeGPU, gRoot);
    CUDA_CHECK(cudaMemcpy(nodePtrLoc, &nodeStart, sizeof(DTreeNode*), 
                          cudaMemcpyHostToDevice));
}

DTree::DTreeBuffer::DTreeBuffer()
    : dDTree(nullptr)
    , nodeCount(0)
{
    nodeCount = 1;
    DeviceMemory::EnlargeBuffer(memory, AlignedOffsetDTreeGPU + sizeof(DTreeNode));
    dDTree = static_cast<DTreeGPU*>(memory);
    DTreeNode* dDTreeNodes = reinterpret_cast<DTreeNode*>(static_cast<Byte*>(memory) + AlignedOffsetDTreeGPU);

    // Init Tree
    DTreeGPU hDTree;
    hDTree.gRoot = dDTreeNodes;
    hDTree.nodeCount = 1;
    hDTree.irradiance = 0;
    hDTree.totalSamples = 0;
    CUDA_CHECK(cudaMemcpy(dDTree, &hDTree, sizeof(DTreeGPU),
                          cudaMemcpyHostToDevice));

    // Init very first node
    DTreeNode hNode;
    hNode.irradianceEstimates = Zero4f;
    hNode.childIndices = Vector4ui(std::numeric_limits<uint32_t>::max());
    hNode.parentIndex = std::numeric_limits<uint16_t>::max();
    CUDA_CHECK(cudaMemcpy(dDTreeNodes, &hNode, sizeof(DTreeNode),
                          cudaMemcpyHostToDevice));    
}

DTree::DTreeBuffer::DTreeBuffer(const DTreeBuffer& other)
    : memory(other.memory.Size())
    , nodeCount(other.nodeCount)
    , dDTree(static_cast<DTreeGPU*>(memory))
{
    CUDA_CHECK(cudaMemcpy(memory, other.memory,
                          AlignedOffsetDTreeGPU + nodeCount * sizeof(DTreeNode),
                          cudaMemcpyDeviceToDevice));
    FixPointers();
}

DTree::DTreeBuffer& DTree::DTreeBuffer::operator=(const DTreeBuffer& other)
{
    DeviceMemory::EnlargeBuffer(memory, other.memory.Size());
    nodeCount = other.nodeCount;
    dDTree = static_cast<DTreeGPU*>(memory);

    CUDA_CHECK(cudaMemcpy(memory, other.memory,
                          AlignedOffsetDTreeGPU + nodeCount * sizeof(DTreeNode),
                          cudaMemcpyDeviceToDevice));
    FixPointers();
    return *this;
}

void DTree::DTreeBuffer::ResetAndReserve(size_t newNodeCount,
                                         const CudaGPU& gpu,
                                         cudaStream_t stream)
{   
    // Check capacity and if its not large enough
    // allocate larger memory
    size_t capacity = (memory.Size() - AlignedOffsetDTreeGPU) / sizeof(DTreeNode);
    if(capacity < newNodeCount)
    {
        size_t size = AlignedOffsetDTreeGPU + (capacity * sizeof(DTreeNode));
        DeviceMemory::EnlargeBuffer(memory, size);
        dDTree = static_cast<DTreeGPU*>(memory);
        FixPointers();        
    }
    // Reset all node values
    gpu.GridStrideKC_X(0, stream, nodeCount,
                        //
                       KCInitDTreeNodes,
                       //
                       dDTree,
                       static_cast<uint32_t>(capacity));
    
    nodeCount = 0;
}

void DTree::DTreeBuffer::CopyGPUNodeCountToCPU()
{
    uint32_t constructedNodeCount;
    CUDA_CHECK(cudaMemcpy(&constructedNodeCount, reinterpret_cast<Byte*>(dDTree) + offsetof(DTreeGPU, nodeCount),
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    nodeCount = constructedNodeCount;
}

void DTree::DTreeBuffer::DumpTree(DTreeGPU& treeCPU, std::vector<DTreeNode>& nodesCPU) const
{
    CUDA_CHECK(cudaMemcpy(&treeCPU, dDTree, sizeof(DTreeGPU),
                          cudaMemcpyDeviceToHost));
    nodesCPU.resize(nodeCount);
    const DTreeNode* dDTreeNodes = treeCPU.gRoot;
    CUDA_CHECK(cudaMemcpy(nodesCPU.data(), dDTreeNodes, nodeCount * sizeof(DTreeNode),
                          cudaMemcpyDeviceToHost));
}

void DTree::SwapTrees(const CudaGPU& gpu, float fluxRatio, uint32_t depthLimit)
{
    // Get an arbitrary stream
    cudaStream_t stream = gpu.DetermineStream();

    // Currently build tree that has its only leafs
    // are valid. Write values to the all nodes
    uint32_t nodeCount = static_cast<uint32_t>(writeTree.NodeCount());
    gpu.GridStrideKC_X(0, stream, nodeCount,
                       //
                       KCCalculateParentIrradiance,
                       //
                       writeTree.TreeGPU(),
                       nodeCount);
    // We have a valid tree now
    // New tree will be reconsturcted from this tree
    // Ask each node that how many child they will need
    DeviceMemory childCountBuffer(nodeCount * sizeof(uint32_t));
    uint32_t* dNodeChildCounts = static_cast<uint32_t*>(childCountBuffer);
    gpu.GridStrideKC_X(0, stream, nodeCount,
                       //
                       KCMarkChildRequest,
                       //
                       dNodeChildCounts,
                       writeTree.TreeGPU(),
                       fluxRatio,
                       nodeCount);
    // Sum all values on the GPU
    uint32_t newNodeCount;
    ReduceArrayGPU<uint32_t, ReduceAdd<uint32_t>,
                   cudaMemcpyDeviceToHost>
    (
        newNodeCount,
        dNodeChildCounts, 
        nodeCount, 
        0u, 
        stream
    );

    // Add root node (DTree will atleast have a root node)
    // And above kernel only checks if childs should be generated
    // Root does not have any parent so we need to manually include here
    newNodeCount++;

    // Reserve enough nodes on the other tree for construction
    readTree.ResetAndReserve(newNodeCount, gpu, stream);
    // Reconstruct a new read tree from the findings
    gpu.GridStrideKC_X(0, stream, nodeCount,
                       //
                       KCReconstructEmptyTree,
                       //
                       readTree.TreeGPU(),
                       //
                       writeTree.TreeGPU(),
                       fluxRatio,
                       depthLimit,
                       nodeCount);

    readTree.CopyGPUNodeCountToCPU();

    // Finally swap the trees
    std::swap(readTree, writeTree);
}

void DTree::GetReadTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes) const
{
    readTree.DumpTree(tree, nodes);
}
void DTree::GetWriteTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes) const
{
    writeTree.DumpTree(tree, nodes);
}