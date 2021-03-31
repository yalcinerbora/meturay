#include "AOTracer.h"
#include "RayAuxStruct.cuh"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/BitManipulation.h"

#include "GPUWork.cuh"
#include "TracerKC.cuh"
#include "TracerWorks.cuh"

AOTracer::AOTracer(const CudaSystem& sys,
                   const GPUSceneI& scene,
                   const TracerParameters& params)
    : RayTracer(sys, scene, params)
    , emptyMat(sys.BestGPU())
{
    workPool.AppendGenerators(AmbientOcclusionWorkerList{});
}

TracerError AOTracer::Initialize()
{
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // We will add special work key for AO misses
    // Check if max work bits support that if not increment
    const Vector2i& workMaxIds = scene.MaxMatIds();
    int32_t maxWorkBatchId = workMaxIds[0];
    int32_t aoMissWorkBatchId = maxWorkBatchId + 1;
    // Increment batch work id by one
    int32_t newBatchWorkBits = Utility::FindFirstSet32(aoMissWorkBatchId) + 1;
    maxWorkBits[0] = newBatchWorkBits;

    // Generate Combined Key
    aoMissKey = HitKey::CombinedKey(aoMissWorkBatchId, 0);

    // Add AO Miss Work
    GPUWorkBatchI* aoMissBatch = nullptr;
    if((err = workPool.GenerateWorkBatch(aoMissBatch,
                                         AmbientOcclusionMissWork::TypeName(),
                                         emptyMat, emptyPrim,
                                         dTransforms)) != TracerError::OK)
        return err;
    workMap.emplace(aoMissWorkBatchId, aoMissBatch);

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        // Dont fetch mat group since we are not going to use it
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);

        const std::string workTypeName = MangledNames::WorkBatch(pg.Type(), "AO");

        if(std::string(pg.Type()) == std::string(BaseConstants::EMPTY_PRIMITIVE_NAME))
        {
            workMap.emplace(batchId, aoMissBatch);
            continue;
        }

        // Generate work batch from appropirate work pool
        WorkPool<>& wp = workPool;
        GPUWorkBatchI* batch = nullptr;
        if((err = wp.GenerateWorkBatch(batch, workTypeName.c_str(),
                                       emptyMat, pg, dTransforms)) != TracerError::OK)
            return err;

        workMap.emplace(batchId, batch);
    }
    return TracerError::OK;
}

TracerError AOTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.maxDistance, MAX_DISTANCE_NAME)) != TracerError::OK)
        return err;
    return TracerError::OK;
}

void AOTracer::GenerateWork(int cameraId)
{
    depth = 0;
    hitPhase = false;
    GenerateRays<RayAuxAO, RayInitAO>(dCameras[cameraId],
                                      options.sampleCount,
                                      InitialAOAux);
}

void AOTracer::GenerateWork(const VisorCamera& cam)
{
    depth = 0;
    hitPhase = false;
    GenerateRays<RayAuxAO, RayInitAO>(cam, options.sampleCount,
                                      InitialAOAux);
}

bool AOTracer::Render()
{
    HitAndPartitionRays();

    cudaSystem.SyncGPUAll();

    // Generate Global Data for Work Kernels
    AmbientOcclusionGlobal globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.maxDistance = options.maxDistance;
    globalData.hitPhase = hitPhase;
    globalData.aoMissKey = aoMissKey;

    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = PartitionOutputRays(totalOutRayCount, workMap);

    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxAO);
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    for(auto p : outPartitions)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers
        RayAuxAO* dAuxOutLocal = static_cast<RayAuxAO*>(*dAuxOut) + p.offset;
        const RayAuxAO* dAuxInLocal = static_cast<const RayAuxAO*>(*dAuxIn);

        using WorkData = typename GPUWorkBatchD<AmbientOcclusionGlobal, RayAuxAO>;
        auto& wData = static_cast<WorkData&>(*loc->second);
        wData.SetGlobalData(globalData);
        wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);
    }

    // Launch Kernels
    WorkRays(workMap, outPartitions,
             totalOutRayCount,
             scene.BaseBoundaryMaterial());

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();

    hitPhase = false;
    depth++;
    // If we exausted the rays quit
    if(totalOutRayCount == 0 || depth == 2) return false;
    // Or continue
    return true;
}