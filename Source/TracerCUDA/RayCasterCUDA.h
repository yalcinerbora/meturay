#pragma once

#include "RayCasterI.h"
#include "RayMemory.h"

class GPUAcceleratorI;
class GPUSceneI;

// Default Ray Caster
class RayCasterCUDA : public RayCasterI
{
    private:
        RayMemory                   rayMemory;

        Vector2i                    maxAccelBits;
        Vector2i                    maxWorkBits;
        // Combined hit struct size
        const uint32_t              maxHitSize;
        const uint32_t              boundaryTransformIndex;
        // CUDA System for GPU Kernel Launches
        const CudaSystem&           cudaSystem;
        // Accelerators
        GPUBaseAcceleratorI&        baseAccelerator;
        const AcceleratorBatchMap&  accelBatches;
        // Misc
        uint32_t                    currentRayCount;

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
        RayPartitions<uint32_t>     HitAndPartitionRays() override;
        void                        WorkRays(const WorkBatchMap& workMap,
                                             const RayPartitionsMulti<uint32_t>& outPortions,
                                             const RayPartitions<uint32_t>& inPartitions,
                                             RNGeneratorCPUI& rngCPU,
                                             uint32_t totalRayOut,
                                             HitKey baseBoundMatKey) override;

        // Ray Related
        uint32_t                CurrentRayCount() const override;
        void                    ResizeRayOut(uint32_t rayCount,
                                             HitKey baseBoundMatKey) override;
        RayGMem*                RaysOut() override;
        const RayGMem*          RaysIn() override;
        void                    SwapRays() override;
        // Work Related
        void                    OverrideWorkBits(const Vector2i newWorkBits) override;

        // Memory Usage
        size_t                  UsedGPUMemory() const override;
};

inline uint32_t RayCasterCUDA::CurrentRayCount() const
{
    return currentRayCount;
}

inline void RayCasterCUDA::ResizeRayOut(uint32_t rayCount,
                                        HitKey baseBoundMatKey)
{
    currentRayCount = rayCount;
    return rayMemory.ResizeRayOut(rayCount, baseBoundMatKey);
}

inline RayGMem* RayCasterCUDA::RaysOut()
{
    return rayMemory.RaysOut();
}

inline const RayGMem* RayCasterCUDA::RaysIn()
{
    return rayMemory.Rays();
}

inline void RayCasterCUDA::SwapRays()
{
    rayMemory.SwapRays();
}

inline void RayCasterCUDA::OverrideWorkBits(const Vector2i newWorkBits)
{
    maxWorkBits = newWorkBits;
}