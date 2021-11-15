#pragma once


#include "RayCasterI.h"
#include "OptixSystem.h"

class GPUSceneI;
class GPUAcceleratorI;

class RayCasterOptiX : public RayCasterI
{
    private:
        Vector2i                    maxAccelBits;
        Vector2i                    maxWorkBits;
        // Combined hit struct size
        const uint32_t              maxHitSize;
        const uint32_t              boundaryTransformIndex;
        // Cuda System for GPU Kernel Launces
        const CudaSystem&           cudaSystem;
        // Accelerators
        GPUBaseAcceleratorI&        baseAccelerator;
        const AcceleratorBatchMap&  accelBatches;
        // Misc
        uint32_t                    currentRayCount;

        // OptiX Related
        OptiXSystem                 optixSystem;

        // GPU Memory

    protected:
    public:
        // Constructors & Destructor
                                    RayCasterOptiX(const GPUSceneI& gpuScene,
                                                   const CudaSystem& system);
                                    RayCasterOptiX(const RayCasterOptiX&) = delete;
        RayCasterOptiX&             operator=(const RayCasterOptiX&) = delete;
                                    ~RayCasterOptiX() = default;

        // Interface
        TracerError                 ConstructAccelerators(const GPUTransformI** dTransforms,
                                                          uint32_t identityTransformIndex) override;
        RayPartitions<uint32_t>     HitAndPartitionRays() override;
        void                        WorkRays(const WorkBatchMap& workMap,
                                             const RayPartitionsMulti<uint32_t>& outPortions,
                                             const RayPartitions<uint32_t>& inPartitions,
                                             RNGMemory& rngMemory,
                                             uint32_t totalRayOut,
                                             HitKey baseBoundMatKey) override;

        // Ray Related
        uint32_t                CurrentRayCount() const override;
        void                    ResizeRayOut(uint32_t rayCount,
                                             HitKey baseBoundMatKey) override;
        RayGMem*                RaysOut() override;
        void                    SwapRays() override;
        // Work Related
        void                    OverrideWorkBits(const Vector2i newWorkBits) override;
};


inline uint32_t RayCasterOptiX::CurrentRayCount() const
{
    return currentRayCount;
}

inline void RayCasterOptiX::OverrideWorkBits(const Vector2i newWorkBits)
{
    maxWorkBits = newWorkBits;
}