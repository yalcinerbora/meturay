#pragma once

#include "RayLib/TracerStructs.h"
#include "SceneSurfaceTreeKC.cuh"

#include "KdTree.cuh"

class SceneSurfaceTree
{
    public:
        using TreeGPUType = KDTreeGPU;
        using TreeCPUType = KDTreeCPU;
        //LBVHSurfaceCPU          lBVHSurface;
    private:
        TreeCPUType             treeCPU;

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

        void                    DumpTreeAsBinary(std::vector<Byte>&) const;

        const TreeGPUType&      TreeGPU() const;

        size_t                  UsedGPUMemory() const;
        size_t                  UsedCPUMemory() const;
};

inline void SceneSurfaceTree::DumpTreeAsBinary(std::vector<Byte>& vec) const
{
    //lBVHSurface.DumpTreeAsBinary(vec);
}

inline const SceneSurfaceTree::TreeGPUType& SceneSurfaceTree::TreeGPU() const
{
    return treeCPU.TreeGPU();
}

inline size_t SceneSurfaceTree::UsedGPUMemory() const
{
    return treeCPU.UsedGPUMemory();
}

inline size_t SceneSurfaceTree::UsedCPUMemory() const
{
    return sizeof(SceneSurfaceTree);
}
