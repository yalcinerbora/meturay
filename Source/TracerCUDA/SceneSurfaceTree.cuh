#pragma once

#include "RayLib/TracerStructs.h"
#include "SceneSurfaceTreeKC.cuh"

class SceneSurfaceTree
{
    private:
        LBVHSurfaceCPU          lBVHSurface;

    public:
        // Constructors & Destructor
                                SceneSurfaceTree() = default;
                                SceneSurfaceTree(const SceneSurfaceTree&) = delete;
        SceneSurfaceTree&       operator=(const SceneSurfaceTree&) = delete;
                                ~SceneSurfaceTree() = default;
        // Construct the linear bvh using all accelerators
        // Refine the svo using thresholds
        TracerError             Construct(const AcceleratorBatchMap& sceneAccelerators,
                                          float normalAngleThreshold,
                                          uint32_t samplePointCount,
                                          uint32_t seed,
                                          const CudaSystem&);

        const LBVHSurfaceGPU&   TreeGPU() const;

        size_t                  UsedGPUMemory() const;
        size_t                  UsedCPUMemory() const;
};

inline const LBVHSurfaceGPU& SceneSurfaceTree::TreeGPU() const
{
    return lBVHSurface.TreeGPU();
}

inline size_t SceneSurfaceTree::UsedGPUMemory() const
{
    return lBVHSurface.UsedGPUMemory();
}

inline size_t SceneSurfaceTree::UsedCPUMemory() const
{
    return sizeof(SceneSurfaceTree);
}
