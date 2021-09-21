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
#include "GPUTransformIdentity.cuh"

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
                                         const uint16_t gMediumIndex)
{
    // Our transform is always identity
    GPUTransformIdentity identityTransform;
    // Get Camera Material Index
    // TODO: Assign a proper material (for bi-directional stuff)
    HitKey gCameraMaterialId = HitKey::InvalidKey;

    if(deletePrevious) delete gCameraLocations;
    new (gCameraLocations) GPUCameraSpherical(pixelRatio,
                                              position,
                                              direction,
                                              up,
                                              nearFar,
                                              identityTransform,
                                              //
                                              gMediumIndex,
                                              gCameraMaterialId);
}

void PathTracerMiddleCallback::SendLog(const std::string s)
{
    METU_LOG(std::move(s));
}

void PathTracerMiddleCallback::SendError(TracerError e)
{
    METU_ERROR_LOG(static_cast<std::string>(e));
}

void PathTracerMiddleCallback::SendImageSectionReset(Vector2i start, Vector2i end)
{
    // Empty callback since we dont reset visor samples
}

void PathTracerMiddleCallback::SendImage(const std::vector<Byte> data,
                                         PixelFormat pf, size_t sampleCount,
                                         Vector2i start, Vector2i end)
{
    // Accumulate image???????
    //...

    // Accumulate the data yourself ??


    // Delegate the to the visor for visual feedback
    if(callbacks) callbacks->SendImage(std::move(data), pf, sampleCount,
                                       start, end);
}

void DirectTracerMiddleCallback::SendLog(const std::string s)
{
    METU_LOG(std::move(s));
}

void DirectTracerMiddleCallback::SendError(TracerError e)
{
    METU_ERROR_LOG(static_cast<std::string>(e));
}

void DirectTracerMiddleCallback::SendImageSectionReset(Vector2i start, Vector2i end)
{
    end = Vector2i::Min(end, resolution);
    Vector2i size = portionEnd - portionStart;

    assert(portionStart == start);
    assert(portionEnd == end);
    
    // Zero (in this case infinity) out the pixels
    std::for_each(pixelLocations.begin(), pixelLocations.end(),
                  [] (Vector3f& pixel)
                  {
                      pixel = Vector3f(std::numeric_limits<float>::infinity());
                  });    
}

void DirectTracerMiddleCallback::SendImage(const std::vector<Byte> data,
                                           PixelFormat pf, size_t sampleCount,
                                           Vector2i start, Vector2i end)
{
    end = Vector2i::Min(end, resolution);
    Vector2i size = portionEnd - portionStart;

    // Check that segments match
    assert(portionStart == start);
    assert(portionEnd == end);
    // We did set this
    assert(pf == PixelFormat::RGB_FLOAT);

    // Directly copy data to pixelLocation buffer;
    const Vector3f* pixels = reinterpret_cast<const Vector3f*>(data.data());
    int pixelCount = size[0] * size[1];

    // Copy that data
    std::copy(pixels, pixels + pixelCount,
              pixelLocations.begin());

}

RefPGTracer::RefPGTracer(const CudaSystem& s,
                         const GPUSceneI& scene,
                         const TracerParameters& p)    
    : currentPixel(0)   
    , currentSampleCount(0)
    , currentCamera(std::numeric_limits<int>::max())
    , pathTracer(s, scene, p)
    , directTracer(s, scene, p)    
    , cudaSystem(s)
    , doInitCameraCreation(true)
    , scene(scene)
{}

TracerError RefPGTracer::Initialize()
{
    // Generate Tracers
    TracerError err = TracerError::OK;
    if((err = pathTracer.Initialize()) != TracerError::OK)
        return err;
    pathTracer.AttachTracerCallbacks(ptCallbacks);

    if((err = directTracer.Initialize()) != TracerError::OK)
        return err;
    directTracer.AttachTracerCallbacks(dtCallbacks);

    // Allocate a SphericalCamera Memory (construct when needed)
    memory = DeviceMemory(sizeof(GPUCameraSpherical));

    // Set Custom Options for path tracer
    // Set path tracer image format
    pathTracer.SetImagePixelFormat(PixelFormat::RGB_FLOAT);
    pathTracer.ResizeImage(options.resolution);
    pathTracer.ReportionImage();


    return TracerError::OK;
}

TracerError RefPGTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetUInt(options.maximumDepth, MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetInt(options.samplePerIteration, SAMPLE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.rrStart, RR_START_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.lightSamplerType, LIGHT_SAMPLER_TYPE_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.nextEventEstimation, NEE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.directLightMIS, DIRECT_LIGHT_MIS_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetUInt(options.totalSamplePerPixel, MAX_SAMPLE_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetVector2i(options.resolution, RESOLUTION_NAME)) != TracerError::OK)
        return err;

    // Delegate these to path tracer
    pathTracer.SetOptions(opts);

    return TracerError::OK;
}

bool RefPGTracer::Render()
{
    // Continue rendering
    return pathTracer.Render();
}

void RefPGTracer::Finalize()
{
    // Increment Sample count
    currentSampleCount += options.samplePerIteration * options.samplePerIteration;
    // Finalize the path tracer
    pathTracer.Finalize();
}

void RefPGTracer::GenerateWork(int cameraId)
{
    // Check if the camera is changed
    if(doInitCameraCreation || currentCamera != cameraId)
    {
        currentCamera = cameraId;

        directTracer.GenerateWork(cameraId);
        while(directTracer.Render());
        directTracer.Finalize();

        // Re-init
        currentPixel = 0;
        currentSampleCount = 0;
    }

    // Change camera if we had enough samples
    if(currentSampleCount >= options.totalSamplePerPixel)
    {
        currentPixel++;
        const Vector3f& position = dtCallbacks.Pixel(currentPixel);
        // Get Camera Medium index
        // Always use current camera's medium
        // TODO: change this to proper implementation
        uint16_t gMediumIndex = scene.BaseMediumIndex();
        // Construct a New camera
        cudaSystem.BestGPU().KC_X(0, (cudaStream_t)0, 1,
                                  KCConstructSingleGPUCameraSpherical,
                                  static_cast<GPUCameraSpherical*>(memory),
                                  !doInitCameraCreation,
                                  //
                                  1.0f,
                                  position,
                                  ZAxis,
                                  YAxis,
                                  Vector2f{0.01f, 1000.0f}, // Near Far
                                  //
                                  gMediumIndex);

        // Reset sample count for this img
        currentSampleCount = 0;

        // Reset the shown image
        if(callbacks) callbacks->SendImageSectionReset();
    }

    // Generate Work for current Camera
    pathTracer.GenerateWork(*dSphericalCamera);
}

void RefPGTracer::GenerateWork(const VisorCamera& cam)
{
    METU_ERROR_LOG("Cannot use custom camera for this Tracer.");
    GenerateWork(0);
}

void RefPGTracer::GenerateWork(const GPUCameraI& dCam)
{
    METU_ERROR_LOG("Cannot use custom camera for this Tracer.");
    GenerateWork(0);
}

void RefPGTracer::SetParameters(const TracerParameters& p)
{
    directTracer.SetParameters(p);
    pathTracer.SetParameters(p);
    params = p;
}

void RefPGTracer::AskParameters()
{
    if(callbacks) callbacks->SendCurrentParameters(params);
}

void RefPGTracer::SetImagePixelFormat(PixelFormat f)
{
    pathTracer.SetImagePixelFormat(f);
}

void RefPGTracer::ReportionImage(Vector2i start, Vector2i end)
{
    directTracer.ReportionImage(start, end);
    dtCallbacks.SetSection(start, end);
}

void RefPGTracer::ResizeImage(Vector2i resolution)
{
    dtCallbacks.SetResolution(resolution);
    directTracer.ResizeImage(resolution);
}

void RefPGTracer::ResetImage()
{
    directTracer.ResetImage();    
}