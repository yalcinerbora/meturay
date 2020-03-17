#include "DirectTracer.h"
#include "TracerWorks.cuh"
#include "MetaTracerWork.cuh"

#include "RayLib/GPUSceneI.h"

DirectTracer::DirectTracer(const CudaSystem& s,
                           const GPUSceneI& scene,
                           const TracerParameters& p)
    : RayTracer(s, scene, p)
{
    workPool.AppendGenerators(DirectTracerWorkerList{});
}

TracerError DirectTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
       return err;
   return TracerError::OK;
}

TracerError DirectTracer::Initialize()
{
    TracerError err = TracerError::OK;
    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);
        
        GPUWorkBatchI* batch = nullptr;
        if((err = workPool.GenerateWorkBatch(batch, mg, pg)) != TracerError::OK)
            return err;        
        // No need for custom initialization so push
        workMap.emplace(batchId, batch);
    }
    return RayTracer::Initialize();
}

bool DirectTracer::Render()
{
    // Do Hit Loop
    HitAndPartitionRays();

    // After Hit Determine Ray Aux Output size    
    uint32_t totalOutRayCount = 0;
    for(const auto& p : workPartition)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        totalOutRayCount += (static_cast<uint32_t>(p.count)*
                             loc->second->OutRayCount());
    }
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxBasic);
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Generate Global Data Struct
    DirectTracerGlobal globalData;
    globalData.gImage = imgMemory.GMem<Vector3>();

    for(auto& work : workMap)
    {
        using WorkData = typename MetaWorkBatchData<DirectTracerGlobal, RayAuxBasic>;

        auto& wData = static_cast<WorkData&>(*work.second);
        wData.SetGlobalData(globalData);
        wData.SetRayDataPtrs(static_cast<RayAuxBasic*>(*dAuxOut),
                             static_cast<const RayAuxBasic*>(*dAuxIn));
    }

    // Hit System Generated bunch of hit pairs
    WorkRays(workMap, scene.BaseBoundaryMaterial());
    SwapAuxBuffers();

    if(totalOutRayCount == 0) return false;
    return true;
}

void DirectTracer::GenerateWork(int cameraId)
{
    // Generate Rays
    GenerateRays(dSceneCameras[cameraId], options.sampleCount);
}

void DirectTracer::GenerateWork(const CPUCamera& c)
{
    LoadCameraToGPU(c);
    GenerateRays(*dCustomCamera, options.sampleCount);
}