#pragma once


#include "RayLib/Vector.h"
#include "RayLib/AABB.h"
#include "DTreeKC.cuh"

#include "CudaSystem.h"

static constexpr uint32_t INVALID_NODE = std::numeric_limits<uint32_t>::max();

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

struct STreeGPU
{    
    STreeNode*          gRoot;
    uint32_t            nodeCount;
    AABB3f              extents;
    
    __device__ void     AcquireNearestDTree(uint32_t& dTreeIndex, const Vector3f& worldPos) const;
};

__device__ __forceinline__
bool STreeNode::DetermineChild(const Vector3f& normalizedCoords) const
{
    // Binary tree is always mid split so check half
    return normalizedCoords[static_cast<int>(splitAxis)] >= 0.5f;
}

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
void STreeGPU::AcquireNearestDTree(uint32_t& dTreeIndex,
                                   const Vector3f& worldPos) const
{
    dTreeIndex = UINT32_MAX;
    if(gRoot == nullptr) return;

    // Convert to Normalized Tree Space
    Vector3f normalizedCoords = worldPos - extents.Min();
    normalizedCoords /= (extents.Max() - extents.Min());

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

__global__ //CUDA_LAUNCH_BOUNDS_1D
static void KCMarkSTreeSplitLeaf(uint32_t* leafIndices,
                                 //
                                 const STreeGPU& gTree,
                                 DTreeGPU** gDTrees,
                                 //
                                 uint32_t treeSplitThreshold,
                                 // Total Node count
                                 uint32_t nodeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < nodeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        const STreeNode* node = gTree.gRoot + globalId;

        uint32_t output = INVALID_NODE;
        if(node->isLeaf)
        {
            // Check if this leaf should split
            uint32_t totalSamples = gDTrees[node->index]->totalSamples;
            if(totalSamples > treeSplitThreshold)
            {
                // Pre divide the sample count since this leaf will be split
                gDTrees[node->index]->totalSamples /= 2;
                output = globalId;  
            }
        }
        leafIndices[globalId] = output;
    }
}

__global__ //CUDA_LAUNCH_BOUNDS_1D
static void KCSplitSTree(uint32_t* gOldTrees,
                         STreeGPU& gTree,
                         //
                         const uint32_t* gLeafIndices,
                         // Options
                         uint32_t leafAllocStartIndex,
                         uint32_t treeAllocStartIndex,
                         uint32_t splitLeafCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < splitLeafCount;
        threadId += (blockDim.x * gridDim.x))
    {
        // For each split node
        STreeNode* leafNode = gTree.gRoot + gLeafIndices[threadId];

        // Determine child loc from allocation offsets
        uint32_t childrenLoc = leafAllocStartIndex + threadId * 2;
        uint32_t newTreeIndex = treeAllocStartIndex + threadId;
        
        // We are splitting             
        STreeNode* gLeft = gTree.gRoot + childrenLoc;
        STreeNode* gRight = gLeft + 1;
        STreeNode::AxisType nextAxis = STreeNode::NextAxis(leafNode->splitAxis);

        // Do the wiring
        uint32_t oldTreeIndex = leafNode->index;
        leafNode->isLeaf = false;
        leafNode->index = childrenLoc;

        gLeft->isLeaf = true;
        gLeft->splitAxis = nextAxis;
        // Left child acquires the old tree
        gLeft->index = oldTreeIndex;

        gRight->isLeaf = true;
        gRight->splitAxis = nextAxis;
        // Right child on the other hand gets a new tree
        gRight->index = newTreeIndex;

        // Copy the old tree so that cpu will copy it to the new tree
        // New tree location is implicit so no need to store
        gOldTrees[threadId] = oldTreeIndex;
    }
}