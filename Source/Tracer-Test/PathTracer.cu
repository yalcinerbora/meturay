#include "PathTracer.h"

#include "RayLib/TracerError.h"
#include "RayLib/GPUSceneI.h"

PathTracer::PathTracer(CudaSystem& s, GPUSceneI& scene,
                       const TracerParameters& params)
    : BasicTracer(s, scene, params)
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
    HitRays();
    WorkRays(workMap, scene.BaseBoundaryMaterial());
    SwapAuxBuffers();
    currentDepth++;
    if(currentDepth > options.maximumDepth)
        return false;
    return true;
}

void PathTracer::GenerateWork(int cameraId)
{
    BasicTracer::GenerateWork(cameraId);
    currentDepth = 0;
}

void PathTracer::GenerateWork(const CPUCamera& cam)
{
    BasicTracer::GenerateWork(cam);
    currentDepth = 0;
}