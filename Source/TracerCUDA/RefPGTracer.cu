#include "RefPGTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/FileUtility.h"

#include "PPGTracerWork.cuh"
#include "GPULightSamplerUniform.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"
#include "GPUAcceleratorI.h"

#include "TracerDebug.h"

#include "GPUCameraSpherical.cuh"

__global__
void KCConstructSingleGPUCameraSpherical(GPUCameraSpherical* gCameraLocations,
                                         bool deletePrevious,
                                         //
                                         float pixelRatio,
                                         Vector3 position,
                                         Vector3 direction,
                                         Vector3 up,
                                         Vector2 nearFar,
                                         //
                                         const TransformId gTransformId,
                                         const uint16_t gMediumIndex,
                                         const HitKey gCameraMaterialId,
                                         //
                                         const GPUTransformI& gTransform)
{
    if(deletePrevious) delete gCameraLocations;
    new (gCameraLocations) GPUCameraSpherical(pixelRatio,
                                              position,
                                              direction,
                                              up,
                                              nearFar,
                                              gTransform,
                                              //
                                              gMediumIndex,
                                              gCameraMaterialId);
}

RefPGTracer::RefPGTracer(const CudaSystem& s,
                         const GPUSceneI& scene,
                         const TracerParameters& p)    
    : currentPixel(0)
    , currentDepth(0)    
    , currentSample(0)
    , pathTracer(s, scene, p)
    , directTracer(s, scene, p)    
    , cudaSystem(s)
{}

TracerError RefPGTracer::Initialize()
{
    // Generate Tracers
    TracerError err = TracerError::OK;
    if((err = pathTracer.Initialize()) != TracerError::OK)
        return err;

    if((err = directTracer.Initialize()) != TracerError::OK)
        return err;

    // Allocate a SphericalCamera Memory (construct when needed)
    memory = DeviceMemory(sizeof(GPUCameraSpherical));

    return TracerError::OK;
}

TracerError RefPGTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetUInt(options.maximumDepth, MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.rrStart, RR_START_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.lightSamplerType, LIGHT_SAMPLER_TYPE_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.nextEventEstimation, NEE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.directLightMIS, DIRECT_LIGHT_MIS_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetUInt(options.maxSampleCount, MAX_SAMPLE_NAME)) != TracerError::OK)
        return err;    

    ...
    return TracerError::OK;
}

bool RefPGTracer::Render()
{
    ...
    return true;
}

void RefPGTracer::Finalize()
{
    ...;

    //
    cudaSystem.BestGPU().AsyncGridStrideKC_X(0, 1,
                                             // Function
                                             KCConstructSingleGPUCameraSpherical,
                                             // Args
                                             memory,
                                             ...);

}

void RefPGTracer::GenerateWork(int cameraId)
{
    ...
}

void RefPGTracer::GenerateWork(const VisorCamera& cam)
{
    ...
}

void RefPGTracer::SetParameters(const TracerParameters& p)
{
    directTracer.SetParameters(p);
    pathTracer.SetParameters(p);
}

void RefPGTracer::AskParameters()
{
    if(callbacks) callbacks->SendCurrentParameters(params);
}

void RefPGTracer::SetImagePixelFormat(PixelFormat f)
{
    directTracer.SetImagePixelFormat(f);
}

void RefPGTracer::ReportionImage(Vector2i start, Vector2i end)
{
    directTracer.ReportionImage(start, end);
    portionStart = start;
    portionEnd = end;
}

void RefPGTracer::ResizeImage(Vector2i resolution)
{
    directTracer.ResizeImage(resolution);
}

void RefPGTracer::ResetImage()
{
    directTracer.ResetImage();
    if(callbacks)
    {
        Vector2i start = portionStart;
        Vector2i end = portionEnd;
        callbacks->SendImageSectionReset(start, end);
    }
}