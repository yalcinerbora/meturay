#include "RefPGTracer.h"
#include "RayTracer.hpp"
#include "Tracers.h"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/ColorConversion.h"
#include "RayLib/FileUtility.h"
#include "RayLib/FileSystemUtility.h"
#include "RayLib/Log.h"
#include "RayLib/ImageIOError.h"

#include "RefPGTracerWorks.cuh"

#include "GPUCameraPixel.cuh"
#include "GPUTransformIdentity.cuh"
#include "DeviceMemory.h"

#include "ImageIO/EntryPoint.h"

#include <iomanip>


//#include "TracerDebug.h"
//PathTracerMiddleCallback::PathTracerMiddleCallback(const Vector2i& resolution)
//    : callbacks(nullptr)
//    , resolution(resolution)
//    , lumPixels(resolution[0] * resolution[1], 0.0f)
//    , totalSampleCounts(resolution[0] * resolution[1], 0)
//{}
//
//void PathTracerMiddleCallback::SetCallbacks(TracerCallbacksI* cb)
//{
//    callbacks = cb;
//}
//
//void PathTracerMiddleCallback::SendLog(const std::string s)
//{
//    METU_LOG(std::move(s));
//}
//
//void PathTracerMiddleCallback::SendError(TracerError e)
//{
//    METU_ERROR_LOG(static_cast<std::string>(e));
//}
//
//void PathTracerMiddleCallback::SendImageSectionReset(Vector2i start, Vector2i end)
//{
//    // Empty callback since we dont reset visor samples
//}
//
//void PathTracerMiddleCallback::SendImage(const std::vector<Byte> data,
//                                         PixelFormat pf, size_t offset,
//                                         Vector2i start, Vector2i end)
//{
//    // Entire image should be here
//    assert(end == resolution);
//    assert(start == Zero2i);
//    assert(pf == PixelFormat::RGBA_FLOAT);
//    
//    // Convert to Luminance & Calculate Average
//    const uint32_t* newSampleCounts = reinterpret_cast<const uint32_t*>(data.data() + offset);
//    const Vector4f* pixels = reinterpret_cast<const Vector4f*>(data.data());
//    for(size_t i = 0; i < (resolution[0] * resolution[1]); i++)
//    {
//        // Incoming Samples
//        float incLum = Utility::RGBToLuminance(pixels[i]);
//        uint32_t incSampleCount = newSampleCounts[i];
//
//        // Old
//        float oldLum = lumPixels[i];
//        uint32_t oldSampleCount = totalSampleCounts[i];
//
//        uint32_t newSampleCount = incSampleCount + oldSampleCount;
//        float newAvg = (oldLum * oldSampleCount) + (incLum * incSampleCount);
//        newAvg /= static_cast<float>(newSampleCount);
//
//        lumPixels[i] = newAvg;
//        totalSampleCounts[i] = newSampleCount;
//    }
//
//    // Delegate the to the visor for visual feedback
//    if(callbacks) callbacks->SendImage(std::move(data), pf, offset,
//                                       start, end);
//}
//
//void PathTracerMiddleCallback::SaveImage(const std::string& baseName, Vector2i
//                                         pixelId)
//{
//
//    // Generate File Name
//    std::stringstream pixelIdStr;
//    pixelIdStr << '['  << std::setw(4) << std::setfill('0') << pixelId[1]
//               << ", " << std::setw(4) << std::setfill('0') << pixelId[0]
//               << ']';
//    std::string path = Utility::PrependToFileInPath(baseName, pixelIdStr.str()) + ".exr";
//
//    // Create Directories if not available
//    Utility::ForceMakeDirectoriesInPath(path);
//
//    // Write Image
//    ImageIOError e = ImageIOInstance().WriteImage(lumPixels,
//                                                  Vector2ui(resolution[0], resolution[1]),
//                                                  PixelFormat::R_FLOAT, ImageType::EXR,
//                                                  path);
//    if(callbacks && e == ImageIOError::OK)
//    {
//        std::string s = fmt::format("Pixel ({:d},{:d}) reference is written as \"{:s}\"",
//                                    pixelId[0], pixelId[1], path);
//        callbacks->SendLog(s);
//    }
//    else if(callbacks)
//    {
//        std::string s = fmt::format("Unable to Save\"{:s}\", ({})", path, e);
//        callbacks->SendLog(s);
//    }
//}
//
//void DirectTracerMiddleCallback::SendLog(const std::string s)
//{
//    METU_LOG(std::move(s));
//}
//
//void DirectTracerMiddleCallback::SendError(TracerError e)
//{
//    METU_ERROR_LOG(static_cast<std::string>(e));
//}
//
//void DirectTracerMiddleCallback::SendImageSectionReset(Vector2i start, Vector2i end)
//{
//    end = Vector2i::Min(end, resolution);
//    Vector2i size = portionEnd - portionStart;
//
//    assert(portionStart == start);
//    assert(portionEnd == end);
//    
//    // Zero (in this case infinity) out the pixels
//    std::for_each(pixelLocations.begin(), pixelLocations.end(),
//                  [] (Vector4f& pixel)
//                  {
//                      pixel = Vector4f(std::numeric_limits<float>::infinity());
//                  });    
//}
//
//void DirectTracerMiddleCallback::SendImage(const std::vector<Byte> data,
//                                           PixelFormat pf, size_t offset,
//                                           Vector2i start, Vector2i end)
//{
//    end = Vector2i::Min(end, resolution);
//    Vector2i size = portionEnd - portionStart;
//
//    // Check that segments match
//    assert(portionStart == start);
//    assert(portionEnd == end);
//    // We did set this
//    assert(pf == PixelFormat::RGBA_FLOAT);
//
//    // Directly copy data to pixelLocation buffer;
//    const Vector4f* pixels = reinterpret_cast<const Vector4f*>(data.data());
//    int pixelCount = size[0] * size[1];
//
//    // Copy that data
//    std::copy(pixels, pixels + pixelCount,
//              pixelLocations.begin());
//
//}

ImageIOError RefPGTracer::SaveAndResetAccumImage(const Vector2i& pixelId)
{
    // Generate File Name
    std::stringstream pixelIdStr;
    pixelIdStr << '['  << std::setw(4) << std::setfill('0') << pixelId[1]
               << ", " << std::setw(4) << std::setfill('0') << pixelId[0]
               << ']';
    std::string path = Utility::PrependToFileInPath(options.refPGOutputName, 
                                                    pixelIdStr.str()) + ".exr";

    // Create Directories if not available
    Utility::ForceMakeDirectoriesInPath(path);

    std::vector<Byte> pixels = accumulationBuffer.GetImageToCPU(cudaSystem);

    ImageIOError e = ImageIOError::OK;
    if((e = ImageIOInstance().WriteImage(pixels.data(),
                                         Vector2ui(accumulationBuffer.SegmentSize()[0],
                                                   accumulationBuffer.SegmentSize()[1]),
                                         accumulationBuffer.Format(), ImageType::EXR,

                                         path)) != ImageIOError::OK)
        return e;

    return ImageIOError::OK;
}

RefPGTracer::RefPGTracer(const CudaSystem& s,
                         const GPUSceneI& scene,
                         const TracerParameters& p)    
    : RayTracer(s, scene, p)
    , currentPixel(0)   
    , currentSampleCount(0)
    , currentCamera(std::numeric_limits<int>::max())
    , doInitCameraCreation(true)
{
    boundaryWorkPool.AppendGenerators(RPGBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(RPGPathWorkerList{});
}

TracerError RefPGTracer::Initialize()
{
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // Generate Light Sampler (for nee)
    if((err = LightSamplerCommon::ConstructLightSampler(lsMemory,
                                                        dLightSampler,
                                                        options.lightSamplerType,
                                                        dLights,
                                                        lightCount,
                                                        cudaSystem)) != TracerError::OK)
        return err;

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& wInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(wInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(wInfo);
        uint32_t batchId = std::get<0>(wInfo);

        // Generate work batch from appropirate work pool
        WorkBatchArray workBatchList;
        if(mg.IsBoundary())
        {
            // Boundary Materials have special kernels
            bool emptyPrim = (std::string(pg.Type()) ==
                              std::string(BaseConstants::EMPTY_PRIMITIVE_NAME));

            WorkPool<bool, bool, bool>& wp = boundaryWorkPool;
            GPUWorkBatchI* batch = nullptr;
            if((err = wp.GenerateWorkBatch(batch, mg, pg, dTransforms,
                                           options.nextEventEstimation,
                                           options.directLightMIS,
                                           emptyPrim)) != TracerError::OK)
                return err;
            workBatchList.push_back(batch);
        }
        else
        {
            // Generic Path work
            WorkPool<bool, bool>& wpCombo = pathWorkPool;
            GPUWorkBatchI* batch = nullptr;
            if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                                dTransforms,
                                                options.nextEventEstimation,
                                                options.directLightMIS)) != TracerError::OK)
                return err;
            workBatchList.push_back(batch);    
        }
        workMap.emplace(batchId, workBatchList);
    }
    return TracerError::OK;

    // Allocate a pixel camera
    // (it will be initialized (will be constructed later)   
    camMemory = DeviceMemory(sizeof(GPUCameraPixel));
    dPixelCamera = static_cast<GPUCameraPixel*>(camMemory);

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

    std::string lightSamplerTypeString;
    if((err = opts.GetString(lightSamplerTypeString, LIGHT_SAMPLER_TYPE_NAME)) != TracerError::OK)
        return err;
    if((err = LightSamplerCommon::StringToLightSamplerType(options.lightSamplerType, lightSamplerTypeString)) != TracerError::OK)
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

    return TracerError::OK;
}

bool RefPGTracer::Render()
{
    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(currentRayCount == 0 || currentDepth >= options.maximumDepth)
        return false;

    HitAndPartitionRays();

    // Generate Global Data Struct
    RPGTracerGlobalState globalData;
    globalData.gImage = imgMemory.GMem<float>();
    globalData.gLightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;
    globalData.directLightMIS = options.directLightMIS;
    globalData.gLightSampler = dLightSampler;

    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = PartitionOutputRays(totalOutRayCount, workMap);

    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPath);
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    //for(auto pIt = workPartition.crbegin();
    //    pIt != workPartition.crend(); pIt++)
    for(auto p : outPartitions)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers
        const RayAuxPath* dAuxInLocal = static_cast<const RayAuxPath*>(*dAuxIn);
        using WorkData = GPUWorkBatchD<RPGTracerGlobalState, RayAuxPath>;
        int i = 0;
        for(auto& work : loc->second)
        {
            RayAuxPath* dAuxOutLocal = static_cast<RayAuxPath*>(*dAuxOut) + p.offsets[i];

            auto& wData = static_cast<WorkData&>(*work);
            wData.SetGlobalData(globalData);
            wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);
            i++;
        }
    }

    // Launch Kernels
    WorkRays(workMap, outPartitions,
             totalOutRayCount,
             scene.BaseBoundaryMaterial());

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    // Increase Depth
    currentDepth++;

    return true;
}

void RefPGTracer::Finalize()
{
    // Increment Sample count
    currentSampleCount += options.samplePerIteration * options.samplePerIteration;

    // Finalize the path tracer
    // Accumulate the image to the buffer
    //....

    // If we calculated enough samples pixel out
    // Save the image 
    if(currentSampleCount >= options.totalSamplePerPixel)
    {
        Vector2i pixelId = Vector2i(currentPixel % resolution[0],
                                    currentPixel / resolution[0]);

        ImageIOError e = ImageIOError::OK;
        if((e = SaveAndResetAccumImage(pixelId)) != ImageIOError::OK)           
        {
            METU_ERROR_LOG("Tracer, {}", std::string(e));
            if(callbacks) callbacks->SendCrashSignal();
        }
        currentPixel++;
    }

    uint32_t totalPixels = static_cast<uint32_t>(resolution[0] * resolution[1]);
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
        // Reset the accum buffer
        accumulationBuffer.Reset(cudaSystem);        
        // Reset currents
        currentCamera = cameraId;
        currentPixel = 0;
        currentSampleCount = 0;
        currentDepth = 0;
    }

    // Change pixel if we had enough samples
    if(currentSampleCount >= options.totalSamplePerPixel ||
       doInitCameraCreation)
    {
        // Get Camera Medium index
        // Always use current camera's medium
        // TODO: change this to proper implementation
        uint16_t gMediumIndex = scene.BaseMediumIndex();
        // Construct a New camera
        cudaSystem.BestGPU().KC_X(0, (cudaStream_t)0, 1,
                                  // Function
                                  KCConstructSingleGPUCameraPixel,
                                  // Args
                                  dPixelCamera,
                                  !doInitCameraCreation,
                                  //
                                  *(dCameras[cameraId]),
                                  currentPixel,
                                  resolution);

        // Reset sample count for this img
        currentSampleCount = 0;

        // Reset the shown image
        if(callbacks) callbacks->SendImageSectionReset();
        doInitCameraCreation = false;
    }

    // Generate Work for current Camera
    GenerateRays<RayAuxPath, RayAuxInitPath>(*dPixelCamera, options.samplePerIteration,
                                             RayAuxInitPath(InitialPathAux));
    currentDepth = 0;
}

void RefPGTracer::GenerateWork(const VisorCamera& cam)
{
    METU_ERROR_LOG("Cannot use custom camera for this Tracer.");
    if(callbacks) callbacks->SendCrashSignal();    
}

void RefPGTracer::GenerateWork(const GPUCameraI& dCam)
{
    METU_ERROR_LOG("Cannot use custom camera for this Tracer.");
    if(callbacks) callbacks->SendCrashSignal();
}

void RefPGTracer::AskParameters()
{
    if(callbacks) callbacks->SendCurrentParameters(params);
}

void RefPGTracer::ReportionImage(Vector2i start, Vector2i end)
{
    end = Vector2i::Min(resolution, end);

    iPortionStart = start;
    iPortionEnd = end;
}

void RefPGTracer::ResizeImage(Vector2i res)
{
    // Save the resolution
    resolution = res;

    // Re-init camera etc..
    // TODO:
}