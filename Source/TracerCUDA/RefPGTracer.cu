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
#include "ParallelReduction.cuh"

#include "ImageIO/EntryPoint.h"

#include <iomanip>

#include "TracerDebug.h"

__global__
void KCAccumulateToBuffer(ImageGMem<float> accumBuffer,
                          ImageGMemConst<float> newSamples,
                          uint32_t pixelCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < pixelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        float data0 = accumBuffer.gPixels[threadId];
        uint32_t sample0 = accumBuffer.gSampleCounts[threadId];

        float data1 = newSamples.gPixels[threadId];
        uint32_t sample1 = newSamples.gSampleCounts[threadId];

        float avgData = (data0 * static_cast<float>(sample0) + data1);
        uint32_t newSampleCount = sample0 + sample1;
        avgData = (newSampleCount == 0)
                    ? 0.0f
                    : avgData / static_cast<float>(newSampleCount);

        // Make NaN bright to make them easier to find
        if(isnan(avgData)) avgData = 1e30;

        accumBuffer.gPixels[threadId] = avgData;
        accumBuffer.gSampleCounts[threadId] = newSampleCount;
    }
}

void RefPGTracer::SendPixel() const
{
    const Vector2i currentPixel2D = GlobalPixel2D();
    const size_t workCount = (imgMemory.SegmentSize()[0] *
                              imgMemory.SegmentSize()[1]);
    // Do Parallel Reduction over the image
    size_t pixelSize = ImageIOI::FormatToPixelSize(iPixelFormat);
    float accumPixel;
    uint32_t totalSamples;
    // Reduction Kernels
    CUDA_CHECK(cudaSetDevice(cudaSystem.BestGPU().DeviceId()));
    ReduceArrayGPU<float, ReduceAdd<float>, cudaMemcpyDeviceToHost>
    (
        accumPixel,
        imgMemory.GMem<float>().gPixels,
        workCount, 0.0f
    );
    ReduceArrayGPU<uint32_t, ReduceAdd<uint32_t>, cudaMemcpyDeviceToHost>
    (
        totalSamples,
        imgMemory.GMem<float>().gSampleCounts,
        workCount, 0u
    );
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0));

    // Convert Accum Pixel to Requested format
    std::array<Byte, 16> convertedPixel;
    switch(iPixelFormat)
    {
        case PixelFormat::R_FLOAT:
            *reinterpret_cast<float*>(convertedPixel.data()) = accumPixel; break;
        case PixelFormat::RG_FLOAT:
            *reinterpret_cast<Vector2f*>(convertedPixel.data()) = Vector2f(accumPixel); break;
        case PixelFormat::RGB_FLOAT:
            *reinterpret_cast<Vector3f*>(convertedPixel.data()) = Vector3f(accumPixel); break;
        case PixelFormat::RGBA_FLOAT:
            *reinterpret_cast<Vector4f*>(convertedPixel.data()) = Vector4f(Vector3f(accumPixel), 1.0f); break;
            break;
        default:
            if(callbacks) callbacks->SendCrashSignal();
            throw TracerException(TracerError::UNABLE_TO_CONVERT_TO_VISOR_PIXEL_FORMAT);
    }

    // Copy data to the vector
    std::vector<Byte> convertedData(pixelSize + sizeof(uint32_t));
    std::memcpy(convertedData.data(), convertedPixel.data(), pixelSize);
    std::memcpy(convertedData.data() + pixelSize,
                &totalSamples, sizeof(uint32_t));

    if(callbacks) callbacks->SendImage(std::move(convertedData),
                                       iPixelFormat,
                                       pixelSize,
                                       // Only send one pixel
                                       currentPixel2D,
                                       currentPixel2D + Vector2i(1));
}

Vector2i RefPGTracer::GlobalPixel2D() const
{
    Vector2i segmentSize = iPortionEnd - iPortionStart;
    Vector2i localPixel2D = Vector2i(currentPixel % segmentSize[0],
                                     currentPixel / segmentSize[0]);
    return iPortionStart + localPixel2D;
}

void RefPGTracer::ResetIterationVariables()
{
    doInitCameraCreation = true;
    currentPixel = 0;
    currentSampleCount = 0;
    currentDepth = 0;
}

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
    e = ImageIOInstance()->WriteImage(pixels.data(),
                                      Vector2ui(accumulationBuffer.SegmentSize()[0],
                                                accumulationBuffer.SegmentSize()[1]),
                                      accumulationBuffer.Format(), ImageType::EXR,
                                      path);

    if(callbacks && e == ImageIOError::OK)
    {
        std::string s = fmt::format("Pixel ({:d},{:d}) reference is written as \"{:s}\"",
                                    pixelId[0], pixelId[1], path);
        callbacks->SendLog(s);
    }
    else if(callbacks)
    {
        std::string s = fmt::format("Unable to Save\"{:s}\"", path);
        callbacks->SendLog(s);
    }
    return e;
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

    // Generate your work list
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& wInfo : infoList)
    {
        WorkBatchArray workBatchList;
        uint32_t batchId = std::get<0>(wInfo);
        const GPUPrimitiveGroupI& pg = *std::get<1>(wInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(wInfo);

        // Generic Path work
        WorkPool<bool, bool>& wpCombo = pathWorkPool;
        GPUWorkBatchI* batch = nullptr;
        if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                            dTransforms,
                                            options.nextEventEstimation,
                                            options.directLightMIS)) != TracerError::OK)
            return err;
        workBatchList.push_back(batch);
        workMap.emplace(batchId, workBatchList);
    }
    const auto& boundaryInfoList = scene.BoundarWorkBatchInfo();
    for(const auto& wInfo : boundaryInfoList)
    {
        uint32_t batchId = std::get<0>(wInfo);
        EndpointType et = std::get<1>(wInfo);
        const CPUEndpointGroupI& eg = *std::get<2>(wInfo);

        // Skip the camera types
        if(et == EndpointType::CAMERA) continue;

        WorkBatchArray workBatchList;
        BoundaryWorkPool<bool, bool>& wp = boundaryWorkPool;
        GPUWorkBatchI* batch = nullptr;
        if((err = wp.GenerateWorkBatch(batch, eg, dTransforms,
                                       options.nextEventEstimation,
                                       options.directLightMIS)) != TracerError::OK)
            return err;
        workBatchList.push_back(batch);
        workMap.emplace(batchId, workBatchList);

    }

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
    if(crashed) return false;

    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(rayCaster->CurrentRayCount() == 0 ||
       currentDepth >= options.maximumDepth)
        return false;

    const auto partitions = rayCaster->HitAndPartitionRays();

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
    globalData.resolution = options.resolution;

    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = RayCasterI::PartitionOutputRays(totalOutRayCount,
                                                         partitions,
                                                         workMap);

    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPath);
    GPUMemFuncs::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    //for(auto pIt = workPartition.crbegin();
    //    pIt != workPartition.crend(); pIt++)
    for(auto p : outPartitions)
    {
        // Skip if null batch or not found material
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
    rayCaster->WorkRays(workMap, outPartitions,
                        partitions,
                        rngMemory,
                        totalOutRayCount,
                        scene.BaseBoundaryMaterial());

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    // Increase Depth
    currentDepth++;

    return true;
}

void RefPGTracer::GenerateWork(uint32_t cameraIndex)
{
    if(crashed) return;

    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    // Check if the camera is changed
    if(currentCamera != static_cast<int>(cameraIndex))
    {
        // Reset currents
        currentCamera = cameraIndex;
        ResetIterationVariables();

        // Reset the shown image
        if(callbacks) callbacks->SendImageSectionReset(iPortionStart, iPortionEnd);
    }

    // Change pixel if we had enough samples
    if(currentSampleCount >= options.totalSamplePerPixel ||
       doInitCameraCreation)
    {
        doInitCameraCreation = false;

        // Reset the accum buffer
        accumulationBuffer.Reset(cudaSystem);

        // Find world pixel 2D id
        Vector2i pixelId = GlobalPixel2D();
        // Construct a New camera
        cudaSystem.BestGPU().KC_X(0, (cudaStream_t)0, 1,
                                  // Function
                                  KCConstructSingleGPUCameraPixel,
                                  // Args
                                  dPixelCamera,
                                  //
                                  *(dCameras[cameraIndex]),
                                  pixelId,
                                  resolution);

        // Reset sample count for this img
        currentSampleCount = 0;
    }

    // Generate Work for current Camera
    GenerateRays<RayAuxPath, RayAuxInitRefPG>(*dPixelCamera, options.samplePerIteration,
                                              RayAuxInitRefPG(InitialPathAux),
                                              false);
    currentDepth = 0;
}

void RefPGTracer::Finalize()
{
    if(crashed) return;

    cudaSystem.SyncAllGPUs();
    // Increment Sample count
    uint32_t finalizedSampleCount = options.samplePerIteration * options.samplePerIteration;
    currentSampleCount += finalizedSampleCount;

    // Finalize the tracing
    // Accumulate the image to the buffer
    const auto& gpu = cudaSystem.BestGPU();
    // Average the finalized data
    size_t workCount = imgMemory.SegmentSize()[0] * imgMemory.SegmentSize()[1];
    gpu.KC_X(0, cudaStream_t(0), workCount,
             //
             KCAccumulateToBuffer,
             //
             accumulationBuffer.GMem<float>(),
             std::as_const(imgMemory).GMem<float>(),
             static_cast<uint32_t>(workCount));

    // Send the img memory as pixel
    SendPixel();

    // Now we can reset the image
    imgMemory.Reset(cudaSystem);

    // If we calculated enough samples of this
    // Save the image & go to the next pixel
    if(currentSampleCount >= options.totalSamplePerPixel)
    {
        Vector2i pixelId2D = GlobalPixel2D();
        ImageIOError e = ImageIOError::OK;
        if((e = SaveAndResetAccumImage(pixelId2D)) != ImageIOError::OK)
            throw ImageIOException(e);

        currentPixel++;
    }

    Vector2i segmentSize = iPortionStart - iPortionEnd;
    uint32_t totalPixels = static_cast<uint32_t>(segmentSize[0] * segmentSize[1]);
    if(currentPixel >= totalPixels && callbacks)
    {
        callbacks->SendLog("Finished All Pixels");

        callbacks->SendCrashSignal();
    }
}

void RefPGTracer::GenerateWork(const VisorTransform&, uint32_t)
{
    throw TracerException(TracerError(TracerError::TRACER_INTERNAL_ERROR),
                          "Cannot use visor transformed camera for this Tracer.");
}

void RefPGTracer::GenerateWork(const GPUCameraI&)
{
    throw TracerException(TracerError(TracerError::TRACER_INTERNAL_ERROR),
                          "Cannot use custom camera for this Tracer.");
}

void RefPGTracer::AskParameters()
{
    if(callbacks) callbacks->SendCurrentParameters(params);
}

void RefPGTracer::SetImagePixelFormat(PixelFormat pf)
{
    // Ignore pixel format change calls since we generate
    // single channel images
    // but call pixel format change function anyway (it may change the state of image mem)
    imgMemory.SetPixelFormat(PixelFormat::R_FLOAT, cudaSystem);
    accumulationBuffer.SetPixelFormat(PixelFormat::R_FLOAT, cudaSystem);

    // Save the format tho
    // we will use it to convert single channel image to multi channel image
    // when we are sending it to Visor
    iPixelFormat = pf;
}

void RefPGTracer::ReportionImage(Vector2i start, Vector2i end)
{
    end = Vector2i::Min(resolution, end);

    iPortionStart = start;
    iPortionEnd = end;

    ResetIterationVariables();

    // Re-portion the image buffers as well
    imgMemory.Reportion(Zero2i, options.resolution, cudaSystem);
    accumulationBuffer.Reportion(Zero2i, options.resolution, cudaSystem);
}

void RefPGTracer::ResizeImage(Vector2i res)
{
    // Save the resolution
    resolution = res;
    ResetIterationVariables();

    imgMemory.Resize(options.resolution);
    accumulationBuffer.Resize(options.resolution);
}

void RefPGTracer::ResetImage()
{
    imgMemory.Reset(cudaSystem);
    accumulationBuffer.Reset(cudaSystem);

    // Reset currents
    doInitCameraCreation = true;
    currentCamera = std::numeric_limits<int>::max();
    ResetIterationVariables();
    // Reset the shown image
    if(callbacks) callbacks->SendImageSectionReset(iPortionStart, iPortionEnd);
}

size_t RefPGTracer::TotalGPUMemoryUsed() const
{
    return (RayTracer::TotalGPUMemoryUsed() +
            accumulationBuffer.UsedGPUMemory() +
            lsMemory.Size() + camMemory.Size());
}