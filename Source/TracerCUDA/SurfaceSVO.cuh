#pragma once

#include <cstdint>

#include "RayLib/TracerStructs.h"
#include "RayLib/AABB.h"

#include "DeviceMemory.h"
#include "CudaSystem.h"

struct SurfaceSVONode
{
    uint32_t parent;
    uint32_t child[8];
    uint32_t dataIndex;
};

struct SurfaceSVOGPU
{
    const SurfaceSVONode*   root;
    uint32_t                nodeCount;
    AABB3f                  extents;

    __device__ uint32_t     FindLeaf(const Vector3f& worldPos);
};

class SurfaceSVOCPU
{
    private:
        SurfaceSVOGPU   svoGPU;
        DeviceMemory    memory;

        uint32_t        nodeCount;
        uint32_t        leafCount;


    protected:
    public:
        // Costructors & Destructor
                            SurfaceSVOCPU();
                            SurfaceSVOCPU(const SurfaceSVOCPU&) = delete;
        SurfaceSVOCPU&      operator=(const SurfaceSVOCPU&) = delete;
                            ~SurfaceSVOCPU() = default;

        // Construct the svo using all accelerators
        // Refine the svo using thresholds
        void                Construct(const AcceleratorBatchMap& sceneAccelerators,
                                      const AABB3f& sceneExtents,
                                      float normalAngleThreshold, float areaThreshold,
                                      const CudaSystem&);

        // Getters
        uint32_t                NodeCount() const;
        uint32_t                LeafCount() const;
        const SurfaceSVOGPU&    SVO() const;

        size_t                  UsedGPUMemory() const;
        size_t                  UsedCPUMemory() const;
};

__device__ __forceinline__
uint32_t SurfaceSVOGPU::FindLeaf(const Vector3f& worldPos)
{
    Vector3ui discreteCoords = Vector3ui((worldPos - extents.Min()) / extents.Span());

    const SurfaceSVONode* currentNode = root;
    while(currentNode->dataIndex != UINT32_MAX)
    {
        // Determine child
        uint8_t childId = 0; // TODO:

        uint32_t childNodeIndex = currentNode->child[childId];
        assert(childNodeIndex < nodeCount);
        currentNode = root + childNodeIndex;
    }
    return currentNode->dataIndex;

}

inline uint32_t SurfaceSVOCPU::NodeCount() const
{
    return nodeCount;
}

inline uint32_t SurfaceSVOCPU::LeafCount() const
{
    return leafCount;
}

inline const SurfaceSVOGPU& SurfaceSVOCPU::SVO() const
{
    return svoGPU;
}

inline size_t SurfaceSVOCPU::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t SurfaceSVOCPU::UsedCPUMemory() const
{
    return sizeof(SurfaceSVOCPU);
}