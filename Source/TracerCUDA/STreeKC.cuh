#pragma once


#include "RayLib/Vector.h"
#include "RayLib/AABB.h"

#include "CudaSystem.h"

struct STreeNode
{
    enum class AxisType : int8_t
    {
        X = 0,
        Y = 1,
        Z = 2,

        END
    };
    
    AxisType                splitAxis; // In which dimension this node is split
    bool                    isLeaf;    // Determines which data the index is holding

    // It is either DTree index or next child index
    // Childs are always grouped (childs + 1 is the other child)    
    uint32_t                index;

    // True: left child (the small one), False: right child (the large one)
    __device__ bool         DetermineChild(const Vector3f& normalizedCoords) const;
    // Normalize coordinates for the next iteration
    __device__ Vector3f     NormalizeCoordsForChild(bool leftRight,
                                                    const Vector3f& parentNormalizedCoords) const;
    __device__ 
    static AxisType         NextAxis(AxisType t);
};

class STreeGPU
{
    private:
        STreeNode*          gRoot;
        AABB3f              extents;

    public:
        __device__ void     AcquireDTree(uint32_t& dTreeIndex, const Vector3f& worldPos);
};

__device__ __forceinline__
bool STreeNode::DetermineChild(const Vector3f& normalizedCoords) const
{
    // Binary tree is always mid split so check half
    return normalizedCoords[static_cast<int>(splitAxis)] >= 0.5f;
}
// Normalize coordinates for the next iteration
__device__ __forceinline__
Vector3f STreeNode::NormalizeCoordsForChild(bool leftRight,
                                            const Vector3f& parentNormalizedCoords) const
{
    Vector3f result = parentNormalizedCoords;
    int axis = static_cast<int>(splitAxis);
    if(!leftRight) result[axis] -= 0.5;
    result[axis] *= 2.0f;    
    return result;
}

__device__ __forceinline__
STreeNode::AxisType STreeNode::NextAxis(STreeNode::AxisType t)
{
    int nextAxisAsInt = (static_cast<int>(t) + 1) % static_cast<int>(STreeNode::AxisType::END);
    return static_cast<STreeNode::AxisType>(nextAxisAsInt);
}

__device__ __forceinline__
void STreeGPU::AcquireDTree(uint32_t& dTreeIndex,
                            const Vector3f& worldPos)
{
    dTreeIndex == UINT32_MAX;
    if(gRoot == nullptr) return;

    // Convert to Normalized Tree Space
    Vector3f normalizedCoords = worldPos - extents.Min();
    normalizedCoords /= extents.Max() - extents.Min();

    const STreeNode* node = gRoot;    
    do
    {
        if(node->isLeaf)
        {
            dTreeIndex = node->index;
            break;
        }
        else
        {
            bool leftRight = node->DetermineChild(normalizedCoords);
            node = gRoot + node->index + ((leftRight) ? 0 : 1);
            normalizedCoords = node->NormalizeCoordsForChild(leftRight, normalizedCoords);
        }
    }
    while(true);
}


class SplitCountFunctor
{

};


__global__ //CUDA_LAUNCH_BOUNDS_1D
void SplitSTree(STreeNode* gRoot,
                const uint32_t* gLeafIndices,
                const uint32_t* gLeafAllocLocations,
                // Split Threshold
                uint32_t sTreeSplitThreshold,
                // Last empty spot on the root array
                uint32_t lastEmptySpot,
                // Total Leafs to be 
                uint32_t leafCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < leafCount; 
        threadId += (blockDim.x * gridDim.x))
    {
        // For each leaf
        STreeNode* leafNode = gRoot + gLeafIndices[threadId];

        // Check if this leaf should split
        if(m_samples...)
        {
            // We are splitting 
            uint32_t childrenLoc = gLeafAllocLocations[threadId];
            STreeNode* gLeft = gRoot + lastEmptySpot + childrenLoc;
            STreeNode* gRight = gLeft + 1;
            STreeNode::AxisType nextAxis = STreeNode::NextAxis(leafNode->splitAxis);

            uint32_t dTreeIndex = leafNode->index;
            leafNode->isLeaf = false;
            leafNode->index = childrenLoc;

            gLeft->isLeaf = true;
            gLeft->splitAxis = nextAxis;
            gLeft->index = dTreeIndex;

            gRight->isLeaf = true;
            gRight->splitAxis = nextAxis;
            gRight->index = dTreeIndex;

            // Allocate 2x
        }
    }
}

__global__ //CUDA_LAUNCH_BOUNDS_1D
void SplitSTree(STreeNode* gRoot,
                const uint32_t* gLeafIndices,
                const uint32_t* gLeafAllocLocations,
                // Options
                uint32_t sTreeSplitThreshold,
                uint32_t leafCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < leafCount; 
        threadId += (blockDim.x * gridDim.x))
    {
        // For each leaf
        STreeNode* leafNode = gRoot + gLeafIndices[threadId];

        // Check if this leaf should split
        if(m_samples...)
        {
            // We are splitting 
            uint32_t childrenLoc = gLeafAllocLocations[threadId];
            STreeNode* gLeft = gRoot + childrenLoc;
            STreeNode* gRight = gLeft + 1;
            STreeNode::AxisType nextAxis = STreeNode::NextAxis(leafNode->splitAxis);

            uint32_t dTreeIndex = leafNode->index;
            leafNode->isLeaf = false;
            leafNode->index = childrenLoc;

            gLeft->isLeaf = true;
            gLeft->splitAxis = nextAxis;
            gLeft->index = dTreeIndex;

            gRight->isLeaf = true;
            gRight->splitAxis = nextAxis;
            gRight->index = dTreeIndex;

            // Subdivide the
        }
    }
}