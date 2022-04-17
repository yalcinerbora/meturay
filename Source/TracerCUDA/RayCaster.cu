#include "RayCaster.h"
#include "CudaSystem.h"
#include "RayLib/GPUSceneI.h"

#include "GPUAcceleratorI.h"
#include "RayLib/TracerStructs.h"

RayCaster::RayCaster(const GPUSceneI& gpuScene,
                     const CudaSystem& system)
    : rayMemory(system.BestGPU())
    , maxAccelBits(DetermineMaxBitFromId(gpuScene.MaxAccelIds()))
    , maxWorkBits(DetermineMaxBitFromId(gpuScene.MaxMatIds()))
    , maxHitSize(gpuScene.HitStructUnionSize())
    , boundaryTransformIndex(gpuScene.BoundaryTransformIndex())
    , cudaSystem(system)
    , baseAccelerator(*gpuScene.BaseAccelerator())
    , accelBatches(gpuScene.AcceleratorBatchMappings())
    , currentRayCount(0)
{}

RayPartitions<uint32_t> RayCaster::PartitionRaysWRTWork()
{
    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
    RayId* dCurrentRayIds = rayMemory.CurrentIds();
    // Partition rays for work kernel calls
    // Copy materialKeys to currentKeys
    // to make it ready for sorting
    rayMemory.FillMatIdsForSort(currentRayCount);
    // Sort with respect to the materials keys
    rayMemory.SortKeys(dCurrentRayIds, dCurrentKeys, currentRayCount, maxWorkBits);
    // Partition w.r.t. material batch
    RayPartitions<uint32_t> workPartition;
    workPartition.clear();
    workPartition = rayMemory.Partition(currentRayCount);

    return workPartition;
}

void RayCaster::WorkRays(const WorkBatchMap& workMap,
                         const RayPartitionsMulti<uint32_t>& outPortions,
                         const RayPartitions<uint32_t>& inPartitions,
                         RNGeneratorCPUI& rngCPU,
                         uint32_t totalRayOut,
                         HitKey baseBoundMatKey)
{
    // Sort and Partition happens on leader device
    CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

    // Ray Memory Pointers
    const RayGMem* dRays = rayMemory.Rays();
    const HitStructPtr dHitStructs = rayMemory.HitStructs();
    const PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
    const TransformId* dTransformIds = rayMemory.TransformIds();
    // These are sorted etc.
    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
    RayId* dCurrentRayIds = rayMemory.CurrentIds();

    // Allocate output ray memory
    rayMemory.ResizeRayOut(totalRayOut, baseBoundMatKey);
    RayGMem* dRaysOut = rayMemory.RaysOut();
    HitKey* dBoundKeyOut = rayMemory.WorkKeys();
    // Wait that "ResizeRayOut" is completed on the leader device
    rayMemory.LeaderDevice().WaitMainStream();

    // For each partition
    for(const auto& p : inPartitions)
    {
        // Skip if null batch or not found material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // TODO: change this loop to combine iterator instead of find
        //const auto& pIn = *(workPartition.find<uint32_t>(p.portionId));
        const auto& pOut = *(outPortions.find(MultiArrayPortion<uint32_t>{p.portionId}));

        // Relativize input & output pointers
        const RayId* dRayIdStart = dCurrentRayIds + p.offset;
        const HitKey* dKeyStart = dCurrentKeys + p.offset;

        assert(pOut.counts.size() == pOut.offsets.size());
        assert(pOut.counts.size() == loc->second.size());

        // Actual Shade Calls
        int i = 0;
        for(auto& workBatch : loc->second)
        {
            // Output
            RayGMem* dRayOutStart = dRaysOut + pOut.offsets[i];
            HitKey* dBoundKeyStart = dBoundKeyOut + pOut.offsets[i];

            workBatch->Work(dBoundKeyStart,
                            dRayOutStart,
                            //  Input
                            dRays,
                            dPrimitiveIds,
                            dTransformIds,
                            dHitStructs,
                            // Ids
                            dKeyStart,
                            dRayIdStart,
                            //
                            static_cast<uint32_t>(p.count),
                            rngCPU);

            i++;
        }
    }
    currentRayCount = totalRayOut;
    // Again wait all of the GPU's since
    // CUDA functions will be on multiple-gpus
    cudaSystem.SyncAllGPUs();
    // Shading complete
    // Now make "RayOut" -> "RayIn"
    // and continue
    rayMemory.SwapRays();
}

size_t RayCaster::UsedGPUMemory() const
{
    size_t mem = 0;
    for(const auto& accel : accelBatches)
        mem += accel.second->UsedGPUMemory();
    mem += baseAccelerator.UsedGPUMemory();
    mem += rayMemory.UsedGPUMemory();
    return mem;
}