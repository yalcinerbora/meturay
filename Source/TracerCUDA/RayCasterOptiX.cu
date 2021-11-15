#include "RayCasterOptiX.h"
#include "RayLib/GPUSceneI.h"
#include "CudaSystem.h"
#include "GPUAcceleratorI.h"

RayCasterOptiX::RayCasterOptiX(const GPUSceneI& gpuScene,
                               const CudaSystem& system)
    : baseAccelerator(*gpuScene.BaseAccelerator())
    , accelBatches(gpuScene.AcceleratorBatchMappings())
    , boundaryTransformIndex(gpuScene.BoundaryTransformIndex())
    , maxAccelBits(DetermineMaxBitFromId(gpuScene.MaxAccelIds()))
    , maxWorkBits(DetermineMaxBitFromId(gpuScene.MaxMatIds()))
    , maxHitSize(gpuScene.HitStructUnionSize())
    , cudaSystem(system)
    , currentRayCount(0)
    , optixSystem(system)
{

    // Load S

}

TracerError RayCasterOptiX::ConstructAccelerators(const GPUTransformI** dTransforms,
                                                  uint32_t identityTransformIndex)
{
    TracerError e = TracerError::OK;

    // Attach Transform gpu pointer to the Accelerator Batches
    for(const auto& acc : accelBatches)
        acc.second->AttachGlobalTransformArray(dTransforms, identityTransformIndex);

    // Construct Accelerators
    SurfaceAABBList allSurfaceAABBs;
    for(const auto& accBatch : accelBatches)
    {
        GPUAcceleratorGroupI* acc = accBatch.second;
        if((e = acc->ConstructAccelerators(cudaSystem)) != TracerError::OK)
            return e;
        // Acquire surface aabb listings for base accelerator consrtuction
        allSurfaceAABBs.insert(acc->AcceleratorAABBs().cbegin(),
                               acc->AcceleratorAABBs().cend());
    }
    // Construct Base accelerator using aabb list
    if((e = baseAccelerator.Constrcut(cudaSystem, allSurfaceAABBs)) != TracerError::OK)
        return e;

    return e;
}

RayPartitions<uint32_t> RayCasterOptiX::HitAndPartitionRays()
{
    return RayPartitions<uint32_t>();
}

void RayCasterOptiX::WorkRays(const WorkBatchMap& workMap,
                         const RayPartitionsMulti<uint32_t>& outPortions,
                         const RayPartitions<uint32_t>& inPartitions,
                         RNGMemory& rngMemory,
                         uint32_t totalRayOut,
                         HitKey baseBoundMatKey)
{

}

void RayCasterOptiX::ResizeRayOut(uint32_t rayCount,
                                  HitKey baseBoundMatKey)
{

}

void RayCasterOptiX::SwapRays()
{

}

RayGMem* RayCasterOptiX::RaysOut()
{
    return nullptr;
}
