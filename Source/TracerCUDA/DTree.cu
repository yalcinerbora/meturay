#include "DTree.cuh"
#include "DTreeKC.cuh"

#include "RayLib/Types.h"

void DTree::DTreeBuffer::FixPointers()
{
    Byte* nodeStart = static_cast<Byte*>(memory) + sizeof(DTreeGPU);
    Byte* nodePtrLoc = static_cast<Byte*>(memory) + offsetof(DTreeGPU, gRoot);
    CUDA_CHECK(cudaMemcpy(nodePtrLoc, &nodeStart, sizeof(DTreeNode*), 
                          cudaMemcpyHostToDevice));
}

DTree::DTreeBuffer::DTreeBuffer()
    : dDTree(nullptr)
    , nodeCount(0)
{}

DTree::DTreeBuffer::DTreeBuffer(const DTreeBuffer& other)
    : memory(other.memory.Size())
    , nodeCount(other.nodeCount)
    , dDTree(static_cast<DTreeGPU*>(memory))
{
    CUDA_CHECK(cudaMemcpy(memory, other.memory,
                          sizeof(DTreeGPU) + nodeCount * sizeof(DTreeNode),
                          cudaMemcpyDeviceToDevice));
    FixPointers();
}

DTree::DTreeBuffer& DTree::DTreeBuffer::operator=(const DTreeBuffer& other)
{
    DeviceMemory::EnlargeBuffer(memory, other.memory.Size());
    nodeCount = other.nodeCount;
    dDTree = static_cast<DTreeGPU*>(memory);

    CUDA_CHECK(cudaMemcpy(memory, other.memory,
                          sizeof(DTreeGPU) + nodeCount * sizeof(DTreeNode),
                          cudaMemcpyDeviceToDevice));
    FixPointers();
}

void DTree::SwapTrees(const CudaSystem&)
{
    // At the end of the tracing iterations
    //  we need to create a new tree and reset a new write tree

    // DTree Write has only valid points on its leaf
    // backpropogate

    // Reconstruct a new write tree from the findings


    std::swap(dReadTree, dWriteTree);
}