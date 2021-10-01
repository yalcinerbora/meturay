#include "RefPGTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/ColorConversion.h"
#include "RayLib/FileUtility.h"
#include "RayLib/FileSystemUtility.h"

#include "TracerDebug.h"

#include "GPUCameraSpherical.cuh"
#include "GPUTransformIdentity.cuh"

#include "DeviceMemory.h"

#include "ImageIO/EntryPoint.h"

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
    if(threadIdx.x != 0) return;

    // Our transform is always identity
    GPUTransformIdentity identityTransform;
    // Get Camera Material Index
    // TODO: Assign a proper material (for bi-directional stuff)
    HitKey gCameraMaterialId = HitKey::InvalidKey;

    if(deletePrevious) gCameraLocations->~GPUCameraSpherical();
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

PathTracerMiddleCallback::PathTracerMiddleCallback(const Vector2i& resolution)
    : callbacks(nullptr)
    , resolution(resolution)
    , lumPixels(resolution[0] * resolution[1], 0.0f)
    , totalSampleCounts(resolution[0] * resolution[1], 0)
{}

void PathTracerMiddleCallback::SetCallbacks(TracerCallbacksI* cb)
{
    callbacks = cb;
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
                                         PixelFormat pf, size_t offset,
                                         Vector2i start, Vector2i end)
{
    // Entire image should be here
    assert(end == resolution);
    assert(start == Zero2i);
    assert(pf == PixelFormat::RGBA_FLOAT);
    
    // Convert to Luminance & Calculate Average
    const uint32_t* newSampleCounts = reinterpret_cast<const uint32_t*>(data.data() + offset);
    const Vector4f* pixels = reinterpret_cast<const Vector4f*>(data.data());
    for(size_t i = 0; i < (resolution[0] * resolution[1]); i++)
    {
        // Incoming Samples
        float incLum = Utility::RGBToLuminance(pixels[i]);
        uint32_t incSampleCount = newSampleCounts[i];

        // Old
        float oldLum = lumPixels[i];
        uint32_t oldSampleCount = totalSampleCounts[i];

        uint32_t newSampleCount = incSampleCount + oldSampleCount;
        float newAvg = (oldLum * oldSampleCount) + (incLum * incSampleCount);
        newAvg /= static_cast<float>(newSampleCount);

        lumPixels[i] = newAvg;
        totalSampleCounts[i] = newSampleCount;
    }

    // Delegate the to the visor for visual feedback
    if(callbacks) callbacks->SendImage(std::move(data), pf, offset,
                                       start, end);
}

void PathTracerMiddleCallback::SaveImage(const std::string& baseName, Vector2i
                                         pixelId)
{

    // Generate File Name
    std::stringstream pixelIdStr;
    pixelIdStr << '['  << std::setw(4) << std::setfill('0') << pixelId[1]
               << ", " << std::setw(4) << std::setfill('0') << pixelId[0]
               << ']';
    std::string path = Utility::PrependToFileInPath(baseName, pixelIdStr.str()) + ".exr";

    // Create Directories if not available
    Utility::ForceMakeDirectoriesInPath(path);

    // Write Image
    ImageIOError e = ImageIOInstance().WriteImage(lumPixels,
                                                  Vector2ui(resolution[0], resolution[1]),
                                                  PixelFormat::R_FLOAT, ImageType::EXR,
                                                  path);
    if(callbacks && e == ImageIOError::OK)
    {
        std::string s = fmt::format("Pixel ({:d},{:d}) reference is written as \"{:s}\"",
                                    pixelId[0], pixelId[1], path);
        callbacks->SendLog(s);
    }
    else if(callbacks)
    {
        std::string s = fmt::format("Unable to Save\"{:s}\", ({})", path, e);
        callbacks->SendLog(s);
    }
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
                  [] (Vector4f& pixel)
                  {
                      pixel = Vector4f(std::numeric_limits<float>::infinity());
                  });    
}

void DirectTracerMiddleCallback::SendImage(const std::vector<Byte> data,
                                           PixelFormat pf, size_t offset,
                                           Vector2i start, Vector2i end)
{
    end = Vector2i::Min(end, resolution);
    Vector2i size = portionEnd - portionStart;

    // Check that segments match
    assert(portionStart == start);
    assert(portionEnd == end);
    // We did set this
    assert(pf == PixelFormat::RGBA_FLOAT);

    // Directly copy data to pixelLocation buffer;
    const Vector4f* pixels = reinterpret_cast<const Vector4f*>(data.data());
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
    , ptCallbacks(Zero2i)
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
    dSphericalCamera = static_cast<GPUCameraSpherical*>(memory);

    // Set Custom Options for path tracer
    // Set path tracer image format
    pathTracer.SetImagePixelFormat(PixelFormat::RGBA_FLOAT);
    pathTracer.ResizeImage(options.resolution);
    pathTracer.ReportionImage();
    // Generate a Proper Callback for Path Tracer
    ptCallbacks = std::move(PathTracerMiddleCallback(options.resolution));
    ptCallbacks.SetCallbacks(callbacks);

    // Initialize the required members

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
    if((err = opts.GetString(options.refPGOutputName, IMAGE_NAME)) != TracerError::OK)
        return err;

    // Delegate these to path tracer
    pathTracer.SetOptions(opts);

    // Set Direct Tracer Options
    directTracer.SetOptions(*GenerateDirectTracerOptions());

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

    // Save the image 
    if(currentSampleCount >= options.totalSamplePerPixel)
    {
        ptCallbacks.SaveImage(options.refPGOutputName, 
                              dtCallbacks.PixelGlobalId(currentPixel));
        currentPixel++;
    }

    uint32_t totalPixels = static_cast<uint32_t>(dtCallbacks.Resolution()[0] * dtCallbacks.Resolution()[1]);
    if(currentPixel >= totalPixels && callbacks)
    {
        callbacks->SendLog("Finished All Pixels");
        callbacks->SendCrashSignal();
    }
}

void RefPGTracer::GenerateWork(int cameraId)
{
    // Check if the camera is changed
    if(currentCamera != cameraId)
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
    if(currentSampleCount >= options.totalSamplePerPixel ||
       doInitCameraCreation)
    {
        const Vector3f& position = dtCallbacks.Pixel(currentPixel);
        // Get Camera Medium index
        // Always use current camera's medium
        // TODO: change this to proper implementation
        uint16_t gMediumIndex = scene.BaseMediumIndex();
        // Construct a New camera
        cudaSystem.BestGPU().KC_X(0, (cudaStream_t)0, 1,
                                  // Function
                                  KCConstructSingleGPUCameraSpherical,
                                  // Args
                                  dSphericalCamera,
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
        doInitCameraCreation = false;
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
    directTracer.SetImagePixelFormat(f);
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