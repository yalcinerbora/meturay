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
    HitRays();

    // Hit System Generated bunch of hit pairs
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