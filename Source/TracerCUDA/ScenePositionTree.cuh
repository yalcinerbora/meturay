#pragma once

#include "LinearBVH.cuh"
#include "RayLib/TracerStructs.h"

class ScenePositionTree
{
    private:
        LBVHPointCPU            lBVHPoint;

    public:
        // Constructors & Destructor
                                ScenePositionTree();
                                ScenePositionTree(const ScenePositionTree&) = delete;
        ScenePositionTree&      operator=(const ScenePositionTree&) = delete;
                                ~ScenePositionTree() = default;
        // Construct the linear bvh using all accelerators
        // Refine the svo using thresholds
        TracerError             Construct(const AcceleratorBatchMap& sceneAccelerators,
                                          float normalAngleThreshold,
                                          uint32_t samplePointCount,
                                          const CudaSystem&);

        const LBVHPointGPU&     TreeGPU() const;

        size_t                  UsedGPUMemory() const;
        size_t                  UsedCPUMemory() const;
};

inline const LBVHPointGPU& ScenePositionTree::TreeGPU() const
{
    return lBVHPoint.TreeGPU();
}

inline size_t ScenePositionTree::UsedGPUMemory() const
{
    return lBVHPoint.UsedGPUMemory();
}

inline size_t ScenePositionTree::UsedCPUMemory() const
{
    return sizeof(ScenePositionTree);
}
