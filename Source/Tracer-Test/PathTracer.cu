#include "PathTracer.h"
#include "TracerWorks.cuh"
#include "MetaTracerWork.cuh"

#include "RayLib/GPUSceneI.h"

PathTracer::PathTracer(CudaSystem& s, GPUSceneI& scene,
                       const TracerParameters& params)
    : RayTracer(s, scene, params)
    , currentDepth(0)
{}

TracerError PathTracer::Initialize()
{
    // Load Work Batches accoring to the scene info
    const WorkBatchCreationInfo& info = scene.WorkBatchInfo();

    // Generate Work Map

    return TracerError::OK;
}

TracerError PathTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.maximumDepth, MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.nextEventEstimation, NEE_NAME)) != TracerError::OK)
        return err;
    return TracerError::OK;
}

bool PathTracer::Render()
{
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
    PathTracerGlobal globalData;
    globalData.gImage = imgMemory.GMem<Vector3>();
    globalData.lightList = *scene.LightsGPU();
    globalData.totalLightCount = static_cast<uint32_t>(scene.LightCount());

    for(auto& work : workMap)
    {
        using WorkData = typename MetaWorkBatchData<PathTracerGlobal, RayAuxBasic>;

        auto& wData = static_cast<WorkData&>(*work.second);
        wData.SetGlobalData(globalData);
        wData.SetRayDataPtrs(static_cast<RayAuxBasic*>(*dAuxOut),
                             static_cast<const RayAuxBasic*>(*dAuxIn));
    }

    WorkRays(workMap, scene.BaseBoundaryMaterial());
    SwapAuxBuffers();
    currentDepth++;
    if(currentDepth > options.maximumDepth)
        return false;
    return true;
}

void PathTracer::GenerateWork(int cameraId)
{
    GenerateRays(*scene.CamerasGPU()[cameraId], options.sampleCount);
    currentDepth = 0;
}

void PathTracer::GenerateWork(const CPUCamera& cam)
{
    LoadCameraToGPU(cam);
    GenerateRays(**dCameraPtr, options.sampleCount);
    currentDepth = 0;
}