#pragma once

#pragma once

#include "RayCasterI.h"
#include "RayMemory.h"

class GPUAcceleratorI;
class GPUSceneI;

// Default Ray Caster
class RayCaster : public RayCasterI
{
    private:
    protected:
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

    public:
        // Constructors & Destructor
                                    RayCaster(const GPUSceneI& gpuScene,
                                              const CudaSystem& system);
                                    RayCaster(const RayCaster&) = delete;
        RayCaster&                  operator=(const RayCaster&) = delete;
                                    ~RayCaster() = default;

        // Interface
        RayPartitions<uint32_t>     PartitionRaysWRTWork() override;
        void                        WorkRays(const WorkBatchMap& workMap,
                                             const RayPartitionsMulti<uint32_t>& outPortions,
                                             const RayPartitions<uint32_t>& inPartitions,
                                             RNGeneratorCPUI& rngCPU,
                                             uint32_t totalRayOut,
                                             HitKey baseBoundMatKey) override;
        //
        void                        PartitionRaysWRTNothing(HitKey baseBoundMatKey,
                                                            uint32_t totalRayOut) override;
        void                        AssumeRaysAreWorked(uint32_t newRayCount) override;

        template <class KeyType, class FetchType,
                  class FetchFunction,
                  typename = CustomKeySizeEnable<KeyType>>
        bool                        PartitionRaysWRTCustomData(// Outputs
                                                               uint32_t& hPartitionCount,
                                                               // GPU Outputs
                                                               DeviceMemory& partitionMemory,
                                                               uint32_t*& dPartitionOffsets,
                                                               KeyType*& dPartitionKeys,
                                                               // Inputs
                                                               const FetchType* dFetchData,
                                                               FetchFunction FetchFunc,
                                                               uint32_t rayCount,
                                                               const CudaSystem& system);

        // Ray Related
        uint32_t                    CurrentRayCount() const override;
        void                        ResizeRayOut(uint32_t rayCount,
                                                 HitKey baseBoundMatKey) override;
        // RayMemory Fetch
        // Input
        const RayGMem*              RaysIn() const override;
        const RayId*                RayIds() const override;
        const HitKey*               KeysIn() const override;
        const PrimitiveId*          PrimitiveIds() const override;
        const TransformId*          TransformIds() const override;
        const HitStructPtr          HitSturctPtr() const override;
        // Output
        RayGMem*                    RaysOut() override;
        HitKey*                     KeysOut() override;
        // Misc
        void                        SwapRays() override;
        // Work Related
        void                        OverrideWorkBits(const Vector2i newWorkBits) override;

        // Memory Usage
        size_t                      UsedGPUMemory() const override;
};

inline uint32_t RayCaster::CurrentRayCount() const
{
    return currentRayCount;
}

inline void RayCaster::ResizeRayOut(uint32_t rayCount,
                                    HitKey baseBoundMatKey)
{
    currentRayCount = rayCount;
    return rayMemory.ResizeRayOut(rayCount, baseBoundMatKey);
}

inline const RayGMem* RayCaster::RaysIn() const
{
    return rayMemory.Rays();
}

inline const RayId* RayCaster::RayIds() const
{
    return rayMemory.CurrentIds();
}

inline const HitKey* RayCaster::KeysIn() const
{
    return rayMemory.CurrentKeys();
}

inline const PrimitiveId* RayCaster::PrimitiveIds() const
{
    return rayMemory.PrimitiveIds();
}

inline const TransformId* RayCaster::TransformIds() const
{
    return rayMemory.TransformIds();
}

inline const HitStructPtr RayCaster::HitSturctPtr() const
{
    return rayMemory.HitStructs();
}

inline RayGMem* RayCaster::RaysOut()
{
    return rayMemory.RaysOut();
}

inline HitKey* RayCaster::KeysOut()
{
    return rayMemory.WorkKeys();
}

inline void RayCaster::SwapRays()
{
    rayMemory.SwapRays();
}

inline void RayCaster::OverrideWorkBits(const Vector2i newWorkBits)
{
    maxWorkBits = newWorkBits;
}

template <class KeyType, class FetchType,
          class FetchFunction, typename>
bool RayCaster::PartitionRaysWRTCustomData(// Outputs
                                           uint32_t& hPartitionCount,
                                           // GPU Outputs
                                           DeviceMemory& partitionMemory,
                                           uint32_t*& dPartitionOffsets,
                                           KeyType*& dPartitionKeys,
                                           // Inputs
                                           const FetchType* dFetchData,
                                           FetchFunction FetchFunc,
                                           uint32_t rayCount,
                                           const CudaSystem& system)
{
    return rayMemory.PartitionRaysCustom(hPartitionCount,
                                         partitionMemory,
                                         dPartitionOffsets,
                                         dPartitionKeys,
                                         dFetchData,
                                         FetchFunc,
                                         rayCount,
                                         system);
}