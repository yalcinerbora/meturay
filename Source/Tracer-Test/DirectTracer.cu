#include "DirectTracer.h"
#include "TracerWorks.cuh"

#include "RayLib/GPUSceneI.h"

DirectTracer::DirectTracer(CudaSystem& s, GPUSceneI& scene,
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
    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        //workMap.emplace()
    }
    return RayTracer::Initialize();
}

bool DirectTracer::Render()
{
    HitRays();
    WorkRays(workMap, scene.BaseBoundaryMaterial());
    SwapAuxBuffers();
    return true;
}

void DirectTracer::GenerateWork(int cameraId)
{
    // Generate Rays
    GenerateRays(*scene.CamerasGPU()[cameraId], options.sampleCount);
}

void DirectTracer::GenerateWork(const CPUCamera& c)
{
    LoadCameraToGPU(c);
    GenerateRays(**dCameraPtr, options.sampleCount);
}