#include "AOTracer.h"
#include "RayAuxStruct.cuh"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/VisorTransform.h"

#include "AOTracerWork.cuh"

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
    Vector2i newMaxIds = scene.MaxMatIds();
    // Increment batch work id by one
    newMaxIds[0] += 1;
    // Check if we flipped a new bit
    Vector2i newWorkBits = RayCasterI::DetermineMaxBitFromId(newMaxIds);
    rayCaster->OverrideWorkBits(newWorkBits);

    // Generate Combined Key
    uint32_t aoMissWorkBatchId = newMaxIds[0];
    aoMissKey = HitKey::CombinedKey(aoMissWorkBatchId, 0);

    // Add AO Miss Work
    GPUWorkBatchI* aoMissBatch = nullptr;
    if((err = workPool.GenerateWorkBatch(aoMissBatch,
                                         AmbientOcclusionMissWork::TypeName(),
                                         emptyMat, emptyPrim,
                                         dTransforms)) != TracerError::OK)
        return err;
    workMap.emplace(aoMissWorkBatchId, WorkBatchArray{aoMissBatch});

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& wInfo : infoList)
    {
        // Dont fetch mat group since we are not going to use it
        const GPUPrimitiveGroupI& pg = *std::get<1>(wInfo);
        uint32_t batchId = std::get<0>(wInfo);

        const std::string workTypeName = MangledNames::WorkBatch(pg.Type(), "AO");

        if(std::string(pg.Type()) == std::string(BaseConstants::EMPTY_PRIMITIVE_NAME))
        {
            workMap.emplace(batchId, WorkBatchArray{aoMissBatch});
            continue;
        }

        // Generate work batch from appropirate work pool
        WorkPool<>& wp = workPool;
        GPUWorkBatchI* batch = nullptr;
        if((err = wp.GenerateWorkBatch(batch, workTypeName.c_str(),
                                       emptyMat, pg, dTransforms)) != TracerError::OK)
            return err;

        workMap.emplace(batchId, WorkBatchArray{batch});
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

void AOTracer::GenerateWork(uint32_t cameraIndex)
{
    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    depth = 0;
    hitPhase = false;
    GenerateRays<RayAuxAO, RayAuxInitAO>(cameraIndex,
                                         options.sampleCount,
                                         RayAuxInitAO(InitialAOAux),
                                         true);
}

void AOTracer::GenerateWork(const VisorTransform& t, uint32_t cameraIndex)
{
    depth = 0;
    hitPhase = false;
    GenerateRays<RayAuxAO, RayAuxInitAO>(t, cameraIndex,
                                         options.sampleCount,
                                         RayAuxInitAO(InitialAOAux),
                                         true);
}

void AOTracer::GenerateWork(const GPUCameraI& dCam)
{
    depth = 0;
    hitPhase = false;
    GenerateRays<RayAuxAO, RayAuxInitAO>(dCam,
                                         options.sampleCount,
                                         RayAuxInitAO(InitialAOAux),
                                         true);
}

bool AOTracer::Render()
{
    const auto partitions = rayCaster->HitAndPartitionRays();

    cudaSystem.SyncAllGPUs();

    // Generate Global Data for Work Kernels
    AmbientOcclusionGlobalState globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.maxDistance = options.maxDistance;
    globalData.hitPhase = hitPhase;
    globalData.aoMissKey = aoMissKey;

    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = RayCasterI::PartitionOutputRays(totalOutRayCount,
                                                         partitions,
                                                         workMap);

    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxAO);
    GPUMemFuncs::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    for(auto p : outPartitions)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers
        const RayAuxAO* dAuxInLocal = static_cast<const RayAuxAO*>(*dAuxIn);
        using WorkData = GPUWorkBatchD<AmbientOcclusionGlobalState, RayAuxAO>;
        int i = 0;
        for(auto& work : loc->second)
        {
            RayAuxAO* dAuxOutLocal = static_cast<RayAuxAO*>(*dAuxOut) + p.offsets[i];

            auto& wData = static_cast<WorkData&>(*work);
            wData.SetGlobalData(globalData);
            wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);
            i++;
        }
    }

    // Launch Kernels
    rayCaster->WorkRays(workMap, outPartitions,
                        partitions,
                        rngMemory,
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