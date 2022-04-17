#pragma once

#include "RayCaster.h"
#include "RayMemory.h"

class GPUAcceleratorI;
class GPUSceneI;

// Default Ray Caster
class RayCasterCUDA : public RayCaster
{
    private:
    protected:
    public:
        // Constructors & Destructor
                                    RayCasterCUDA(const GPUSceneI& gpuScene,
                                                  const CudaSystem& system);
                                    RayCasterCUDA(const RayCasterCUDA&) = delete;
        RayCasterCUDA&              operator=(const RayCasterCUDA&) = delete;
                                    ~RayCasterCUDA() = default;

        // Interface
        TracerError                 ConstructAccelerators(const GPUTransformI** dTransforms,
                                                          uint32_t identityTransformIndex) override;
        void                        HitRays() override;
};

inline RayCasterCUDA::RayCasterCUDA(const GPUSceneI& gpuScene,
                                    const CudaSystem& system)
    : RayCaster(gpuScene, system)
{}