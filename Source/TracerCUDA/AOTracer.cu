#include "AOTracer.h"
#include "RayAuxStruct.cuh"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"

#include "GPUWork.cuh"
#include "TracerKC.cuh"

AOTracer::AOTracer(const CudaSystem& sys,
                   const GPUSceneI& scene,
                   const TracerParameters& params)
    : RayTracer(sys, scene, params)
{

    workPool.AppendGenerators(AOTracerWorkerList{});
    lightWorkPool.AppendGenerators(AOTracerLightWorkerList{});
}

TracerError AOTracer::Initialize()
{
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);

        // Generate work batch from appropirate work pool
        GPUWorkBatchI* batch = nullptr;
        if(mg.IsLightGroup())
        {
            bool emptyPrim = (std::string(pg.Type()) ==
                              std::string(BaseConstants::EMPTY_PRIMITIVE_NAME));

            WorkPool<>& wp = lightWorkPool;
            if((err = wp.GenerateWorkBatch(batch, mg, pg, dTransforms)) != TracerError::OK)
                return err;
        }
        else
        {
            WorkPool<>& wp = workPool;
            if((err = wp.GenerateWorkBatch(batch, mg, pg, dTransforms)) != TracerError::OK)
                return err;
        }
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
    GenerateRays<RayAuxAO, RayInitAO>(dCameras[cameraId],
                                      options.sampleCount,
                                      InitialAOAux);
}

void AOTracer::GenerateWork(const VisorCamera& cam)
{
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

    // If we exausted the rays quit
    if(totalOutRayCount == 0) return false;
    // Or continue
    return true;
}
