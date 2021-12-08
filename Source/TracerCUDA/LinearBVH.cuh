#pragma once

#include <cstdint>

#include "RayLib/TracerStructs.h"
#include "RayLib/AABB.h"

#include "DeviceMemory.h"
#include "CudaSystem.h"

struct SurfaceKdONode
{
    // Pointers
    uint32_t    parent      = std::numeric_limits<uint32_t>::max();
    uint32_t    child[8]    = {};
    // Split Locations
    Vector3f    splits      = Vector3f(std::numeric_limits<float>::infinity());

    // Actual Leaf Data Index (Data does not stored in the Node)
    // However even if node is not a leaf,
    // it holds this for ease of mem management
    uint32_t    dataIndex   = std::numeric_limits<uint32_t>::max();
};

struct SurfaceKdOOctreeGPU
{
    uint32_t                rootIndex;
    const SurfaceKdONode*   nodes;
    uint32_t                nodeCount;
    AABB3f                  extents;

    __device__ uint32_t     FindLeaf(const Vector3f& worldPos);
};

class SurfaceKdOOctreeCPU
{
    private:
        SurfaceKdOOctreeGPU treeGPU;
        DeviceMemory        memory;

        uint32_t            nodeCount;
        uint32_t            leafCount;

        uint32_t            Subdivide(Vector3f* dPositions, Vector3f* dNormals, float* dAreas,
                                      DeviceMemory& tempMemory,
                                      uint32_t elementCount,
                                      const CudaSystem&);
        float               FindMortonDelta(const AABB3f& sceneExtents);

    protected:
    public:
        // Constructors & Destructor
                                SurfaceKdOOctreeCPU();
                                SurfaceKdOOctreeCPU(const SurfaceKdOOctreeCPU&) = delete;
        SurfaceKdOOctreeCPU&    operator=(const SurfaceKdOOctreeCPU&) = delete;
                                ~SurfaceKdOOctreeCPU() = default;

        // Construct the svo using all accelerators
        // Refine the svo using thresholds
        void                        Construct(const AcceleratorBatchMap& sceneAccelerators,
                                              const AABB3f& sceneExtents,
                                              float normalAngleThreshold, float areaThreshold,
                                              const CudaSystem&);

        // Getters
        uint32_t                    NodeCount() const;
        uint32_t                    LeafCount() const;
        const SurfaceKdOOctreeGPU&  TreeGPU() const;

        size_t                      UsedGPUMemory() const;
        size_t                      UsedCPUMemory() const;
};

__device__ __forceinline__
uint32_t SurfaceKdOOctreeGPU::FindLeaf(const Vector3f& worldPos)
{
    assert(extents.IsOutside(worldPos));

    const SurfaceKdONode* currentNode = nodes + rootIndex;
    while(currentNode->dataIndex != UINT32_MAX)
    {
        // Determine child
        uint8_t childId = 0;
        childId |= (worldPos[0] > currentNode->splits[0]) ? (1 << 0) : 0;
        childId |= (worldPos[1] > currentNode->splits[1]) ? (1 << 1) : 0;
        childId |= (worldPos[2] > currentNode->splits[2]) ? (1 << 2) : 0;

        // Go to next node
        uint32_t childNodeIndex = currentNode->child[childId];
        assert(childNodeIndex < nodeCount);
        currentNode = nodes + childNodeIndex;
    }
    return currentNode->dataIndex;
}

inline uint32_t SurfaceKdOOctreeCPU::NodeCount() const
{
    return nodeCount;
}

inline uint32_t SurfaceKdOOctreeCPU::LeafCount() const
{
    return leafCount;
}

inline const SurfaceKdOOctreeGPU& SurfaceKdOOctreeCPU::TreeGPU() const
{
    return treeGPU;
}

inline size_t SurfaceKdOOctreeCPU::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t SurfaceKdOOctreeCPU::UsedCPUMemory() const
{
    return sizeof(SurfaceKdOOctreeCPU);
}