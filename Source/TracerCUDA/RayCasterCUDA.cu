#include "RayCasterCUDA.h"
#include "CudaSystem.h"

#include "RayLib/GPUSceneI.h"
#include "GPUAcceleratorI.h"
#include "RayLib/TracerStructs.h"

// Constructors & Destructor
RayCasterCUDA::RayCasterCUDA(const GPUSceneI& gpuScene,
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

TracerError RayCasterCUDA::ConstructAccelerators(const GPUTransformI** dTransforms,
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
        // Acquire surface aabb listings for base accelerator construction
        allSurfaceAABBs.insert(acc->AcceleratorAABBs().cbegin(),
                               acc->AcceleratorAABBs().cend());
    }
    // Construct Base accelerator using aabb list
    if((e = baseAccelerator.Construct(cudaSystem, allSurfaceAABBs)) != TracerError::OK)
        return e;

    return e;
}

RayPartitions<uint32_t> RayCasterCUDA::HitAndPartitionRays()
{
    // Sort and Partition happens on the leader device
    CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

    // Tracer Logic interface
    const Vector2i& accBitCounts = maxAccelBits;
    const AcceleratorBatchMap& subAccelerators = accelBatches;
    // Reset Hit Memory for hit loop
    rayMemory.ResetHitMemory(boundaryTransformIndex, currentRayCount, maxHitSize);
    // Make Base Accelerator to get ready for hitting
    baseAccelerator.GetReady(cudaSystem, currentRayCount);
    // Ray Memory Pointers
    RayGMem* dRays = rayMemory.Rays();
    HitKey* dWorkKeys = rayMemory.WorkKeys();
    TransformId* dTransfomIds = rayMemory.TransformIds();
    PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
    HitStructPtr dHitStructs = rayMemory.HitStructs();
    // These are sorted etc.
    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
    RayId*  dCurrentRayIds = rayMemory.CurrentIds();

    //CUDA_CHECK(cudaMemset(dTransfomIds, 0xFF, currentRayCount * sizeof(TransformId)));
    //Debug::DumpMemToFile("dTransforms", dTransfomIds, currentRayCount);

    // Try to hit rays until no ray is left
    // (these rays will be assigned with a material)
    // outside rays are also assigned with a material (which is special)
    uint32_t rayCount = currentRayCount;
    // At start all rays are valid
    uint32_t validRayOffset = 0;
    while(rayCount > 0)
    {
        // Traverse accelerator
        // Base accelerator provides potential hits
        // Cannot provide an absolute hit (its not its job)
        baseAccelerator.Hit(cudaSystem,
                            dCurrentKeys + validRayOffset,
                            dRays,
                            dCurrentRayIds + validRayOffset,
                            rayCount);
        // Wait all GPUs to finish...
        cudaSystem.SyncAllGPUsMainStreamOnly();

        // Base accelerator traverses the data partially
        // Updates current key (which represents inner accelerator batch and id)

        // After that, system sorts rays according to the keys
        // and partitions the array according to batches
        // Sort and Partition happens on the leader device
        CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

        // Sort initial results (in order to partition and launch kernels accordingly)
        // Sort is radix sort.
        // We sort inner indices in addition to batches results for better data locality
        // We only sort up-to a certain bit (radix sort) which is tied to
        // accelerator count
        // Move offset to skip null batches
        rayMemory.SortKeys(dCurrentRayIds, dCurrentKeys, rayCount, accBitCounts);
        // Partition to sub accelerators
        //
        // There may be invalid rays sprinkled along the array.
        // Holes occur in the structure since in previous iteration,
        // a material may required to write N rays for its output (which is defined
        // by the material) but it wrote < N rays.
        //
        // One of the main examples for such behavior can be transparent objects
        // where ray may be only reflected (instead of refracting and reflecting) because
        // of the total internal reflection phenomena.
        auto portions = rayMemory.Partition(rayCount);

        // TODO: Reorder partitions for efficient calls
        // (group partitions into gpus and order for better async access)
        const int totalGPU = static_cast<int>(cudaSystem.SystemGPUs().size());
        const auto& gpus = cudaSystem.SystemGPUs();
        auto currentGPU = gpus.begin();
        // For each partition
        for(const auto& p : portions)
        {
            // Find Accelerator
            // Since there is no batch for invalid keys
            // that partition will be automatically be skipped
            auto loc = subAccelerators.find(p.portionId);
            if(loc == subAccelerators.end()) continue;

            RayId* dRayIdStart = dCurrentRayIds + validRayOffset + p.offset;
            HitKey* dCurrentKeyStart = dCurrentKeys + validRayOffset + p.offset;

            // Run local hit kernels
            // Local hit kernels returns a material key
            // and primitive inner id.
            // Since materials are batched for both material and
            loc->second->Hit(*currentGPU,
                             // O
                             dWorkKeys,
                             dTransfomIds,
                             dPrimitiveIds,
                             dHitStructs,
                             // I-O
                             dRays,
                             // Input
                             dRayIdStart,
                             dCurrentKeyStart,
                             static_cast<uint32_t>(p.count));

            // Split to GPUs
            currentGPU++;
            if(currentGPU == gpus.end()) currentGPU = gpus.begin();

            // Hit function updates material key,
            // primitive id and struct if this hit is accepted
        }
        // Update new ray count
        // On partition array check first partition
        // it may contain invalid key meaning
        // those rays are totally processed
        // change the offset so that
        // we skip those rays
        auto nullPortion = portions.begin();
        if(nullPortion->portionId == HitKey::NullBatch)
            rayCount = static_cast<uint32_t>(nullPortion->offset);

        // Iteration is done
        // We cant continue loop until these kernels are finished
        // on gpu(s)
        //
        // Tracer logic mostly utilizes multiple GPUs so we need to
        // wait all GPUs to finish
        cudaSystem.SyncAllGPUs();
    }
    // At the end of iteration, all rays found a material, a primitive
    // and an interpolation weights (which should be on hitStruct)

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

void RayCasterCUDA::WorkRays(const WorkBatchMap& workMap,
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

// Memory Usage
size_t RayCasterCUDA::UsedGPUMemory() const
{
    size_t mem = 0;
    for(const auto& accel : accelBatches)
        mem += accel.second->UsedGPUMemory();
    mem += baseAccelerator.UsedGPUMemory();
    mem += rayMemory.UsedGPUMemory();
    return mem;
}