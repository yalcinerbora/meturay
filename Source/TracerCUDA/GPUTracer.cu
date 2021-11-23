#include "GPUTracer.h"

#include "RayLib/Log.h"
#include "RayLib/TracerError.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/GPUSceneI.h"
#include "RayLib/MemoryAlignment.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/VisorTransform.h"

#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include "GPUAcceleratorI.h"
#include "GPUWorkI.h"
#include "GPUTransformI.h"
#include "GPUMediumI.h"
#include "GPUMaterialI.h"
#include "GPUTransformI.h"
#include "GPULightI.h"
#include "GPUCameraI.h"

#include "RayCaster.h"
#ifdef MRAY_OPTIX
    #include "RayCasterOptiX.h"
#endif

#include "TracerDebug.h"

__global__
void KCTransformCam(GPUCameraI* gCam, const VisorTransform t)
{
    if(threadIdx.x != 0) return;
    gCam->SwapTransform(t);
}

__global__
void KCFetchTransform(VisorTransform* gVTransforms,
                      const GPUCameraI** gCameras,
                      uint32_t cameraCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < cameraCount;
        threadId += (blockDim.x * gridDim.x))
    {
        gVTransforms[threadId] = gCameras[threadId]->GenVisorTransform();
    }
}

__global__
void KCSetEndpointIds(GPUEndpointI** gEndpoints,
                      uint32_t endpointCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < endpointCount;
        threadId += (blockDim.x * gridDim.x))
    {
        gEndpoints[threadId]->SetEndpointId(threadId);
    }
}

__global__
void KCSetLightIds(GPULightI** gLights,
                   uint32_t lightCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < lightCount;
        threadId += (blockDim.x * gridDim.x))
    {
        gLights[threadId]->SetGlobalLightIndex(threadId);
    }
}

TracerError GPUTracer::LoadCameras(std::vector<const GPUCameraI*>& dGPUCameras,
                                   std::vector<const GPUEndpointI*>& dGPUEndpoints)
{
    TracerError e = TracerError::OK;
    for(auto& camera : cameras)
    {
        CPUCameraGroupI& c = *(camera.second);
        if((e = c.ConstructEndpoints(dTransforms, cudaSystem)) != TracerError::OK)
            return e;
        const auto& dCList = c.GPUCameras();
        const auto& dEList = c.GPUEndpoints();
        dGPUCameras.insert(dGPUCameras.end(), dCList.begin(), dCList.end());
        dGPUEndpoints.insert(dGPUEndpoints.end(), dEList.begin(), dEList.end());

        cameraGroupNames.insert(cameraGroupNames.end(), c.EndpointCount(), camera.first);
    }
    cameraCount = static_cast<uint32_t>(dGPUCameras.size());

    // Copy the pointers to the device
    CUDA_CHECK(cudaMemcpy(const_cast<GPUCameraI**>(dCameras),
                          dGPUCameras.data(),
                          dGPUCameras.size() * sizeof(GPUCameraI*),
                          cudaMemcpyHostToDevice));

    // Calculate Visor Transforms & SceneIds
    DeviceMemory tempMem(cameraCount * sizeof(VisorTransform));
    VisorTransform* dVisorTransforms = static_cast<VisorTransform*>(tempMem);

    const auto& gpu = cudaSystem.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, cameraCount,
                       //
                       KCFetchTransform,
                       //
                       dVisorTransforms,
                       dCameras,
                       cameraCount);

    cameraVisorTransforms.resize(cameraCount);
    CUDA_CHECK(cudaMemcpy(cameraVisorTransforms.data(), dVisorTransforms,
                          sizeof(VisorTransform) * cameraCount,
                          cudaMemcpyDeviceToHost));

    return TracerError::OK;
}

TracerError GPUTracer::LoadLights(std::vector<const GPULightI*>& dGPULights,
                                  std::vector<const GPUEndpointI*>& dGPUEndpoints)
{

    TracerError e = TracerError::OK;
    for(auto& light : lights)
    {
        CPULightGroupI& l = *(light.second);
        if((e = l.ConstructEndpoints(dTransforms, cudaSystem)) != TracerError::OK)
            return e;
        const auto& dLList = l.GPULights();
        const auto& dEList = l.GPUEndpoints();
        dGPULights.insert(dGPULights.end(), dLList.begin(), dLList.end());
        dGPUEndpoints.insert(dGPUEndpoints.end(), dEList.begin(), dEList.end());
    }
    lightCount = static_cast<uint32_t>(dGPULights.size());

    // Copy the pointers to the device
    CUDA_CHECK(cudaMemcpy(const_cast<GPULightI**>(dLights),
                          dGPULights.data(),
                          dGPULights.size() * sizeof(GPULightI*),
                          cudaMemcpyHostToDevice));

    return TracerError::OK;
}

TracerError GPUTracer::LoadTransforms(std::vector<const GPUTransformI*>& dGPUTransforms)
{
    TracerError e = TracerError::OK;
    for(auto& transform : transforms)
    {
        CPUTransformGroupI& t = *(transform.second);
        if((e = t.ConstructTransforms(cudaSystem)) != TracerError::OK)
            return e;
        const auto& dTList = t.GPUTransforms();
        dGPUTransforms.insert(dGPUTransforms.end(), dTList.begin(), dTList.end());
    }
    transformCount = static_cast<uint32_t>(dGPUTransforms.size());
    return TracerError::OK;
}

TracerError GPUTracer::LoadMediums(std::vector<const GPUMediumI*>& dGPUMediums)
{
    TracerError e = TracerError::OK;
    uint32_t indexOffset = 0;
    for(auto& medium : mediums)
    {
        CPUMediumGroupI& m = *(medium.second);
        if((e = m.ConstructMediums(cudaSystem, indexOffset)) != TracerError::OK)
            return e;
        const auto& dMList = m.GPUMediums();
        dGPUMediums.insert(dGPUMediums.end(), dMList.begin(), dMList.end());
        indexOffset += m.MediumCount();
    }
    mediumCount = static_cast<uint32_t>(dGPUMediums.size());
    return TracerError::OK;
}

GPUTracer::GPUTracer(const CudaSystem& system,
                     const GPUSceneI& scene,
                     const TracerParameters& p)
    : cudaSystem(system)
    , materialGroups(scene.MaterialGroups())
    , transforms(scene.Transforms())
    , mediums(scene.Mediums())
    , cameras(scene.Cameras())
    , lights(scene.Lights())
    , workInfo(scene.WorkBatchInfo())
    , baseMediumIndex(scene.BaseMediumIndex())
    , identityTransformIndex(scene.IdentityTransformIndex())
    , boundaryTransformIndex(scene.BoundaryTransformIndex())
    , params(p)
    , maxHitSize(scene.HitStructUnionSize())
    , callbacks(nullptr)
    , crashed(false)
    , currentCameraIndex(std::numeric_limits<uint32_t>::max())
{
    #ifdef MRAY_OPTIX
        bool allOptiXScene = true;
        bool foundAnOptiXAccel = false;
        bool foundAnOtherAccel = false;
        // Check if all of the accelerators (base included)
        // are optix accelerators
        std::string s = scene.BaseAccelerator()->Type();

        // Check Base Accelerator
        bool baseIsOptix = (s.find("OptiX") != std::string::npos);
        if(!foundAnOtherAccel) foundAnOtherAccel = (!baseIsOptix);
        if(!foundAnOptiXAccel) foundAnOptiXAccel = (baseIsOptix);
        allOptiXScene &= baseIsOptix;
        for(const auto& acc : scene.AcceleratorGroups())
        {
            // TODO: primitive type maybe have a name OptiX
            // probably not thow i will leave that case
            std::string accType = acc.second->Type();
            bool accelIsOptiX = (accType.find("OptiX") != std::string::npos);

            if(!foundAnOtherAccel) foundAnOtherAccel = (!accelIsOptiX);
            if(!foundAnOptiXAccel) foundAnOptiXAccel = (accelIsOptiX);
            allOptiXScene &= accelIsOptiX;
        }

        // Only all OptiX or all non-OptiX is supported
        if(foundAnOtherAccel && foundAnOptiXAccel)
            throw TracerException(TracerError::OPTIX_ACCELERATOR_MISMATCH);

        // Construct the RayCaster
        if(allOptiXScene)
            rayCaster = std::make_unique<RayCasterOptiX>(scene, system);
        else
            rayCaster = std::make_unique<RayCaster>(scene, system);
    #else
        // Just create original RayCaster
        rayCaster = std::make_unique<RayCaster>(scene, system);
    #endif
}

TracerError GPUTracer::Initialize()
{
    // Init RNGs for each block
    TracerError e = TracerError::OK;
    rngMemory = RNGMemory(params.seed, cudaSystem);

    std::vector<const GPUTransformI*> dGPUTransforms;
    std::vector<const GPUMediumI*> dGPUMediums;
    std::vector<const GPULightI*> dGPULights;
    std::vector<const GPUCameraI*> dGPUCameras;
    std::vector<const GPUEndpointI*> dGPUEndpoints;

    // Calculate Total Sizes
    size_t tCount = 0;
    size_t mCount = 0;
    size_t lCount = 0;
    size_t cCount = 0;
    std::for_each(transforms.cbegin(), transforms.cend(),
                  [&tCount](const auto& transform)
                  {
                      tCount += transform.second->TransformCount();
                  });
    std::for_each(mediums.cbegin(), mediums.cend(),
                  [&mCount](const auto& medium)
                  {
                      mCount += medium.second->MediumCount();
                  });
    std::for_each(lights.cbegin(), lights.cend(),
                  [&lCount](const auto& light)
                  {
                      lCount += light.second->EndpointCount();
                  });
    std::for_each(cameras.cbegin(), cameras.cend(),
                  [&cCount](const auto& camera)
                  {
                      cCount += camera.second->EndpointCount();
                  });

    transformCount = static_cast<uint32_t>(tCount);
    mediumCount = static_cast<uint32_t>(mCount);
    lightCount = static_cast<uint32_t>(lCount);
    cameraCount = static_cast<uint32_t>(cCount);
    endpointCount = static_cast<uint32_t>(cCount + lCount);

    GPUMemFuncs::AllocateMultiData(std::tie(dTransforms,
                                            dMediums,
                                            dLights,
                                            dCameras,
                                            dEndpoints),
                                   commonTypeMemory,
                                   {transformCount,
                                   mediumCount,
                                   lightCount,
                                   cameraCount,
                                   endpointCount});

    // Do transforms and Mediums fist
    // since materials and accelerators requires these objects
    // Transforms
    if((e = LoadTransforms(dGPUTransforms)) != TracerError::OK)
        return e;
    CUDA_CHECK(cudaMemcpy(const_cast<GPUTransformI**>(dTransforms),
                          dGPUTransforms.data(),
                          dGPUTransforms.size() * sizeof(GPUTransformI*),
                          cudaMemcpyHostToDevice));
    // Mediums
    if((e = LoadMediums(dGPUMediums)) != TracerError::OK)
        return e;
    CUDA_CHECK(cudaMemcpy(const_cast<GPUMediumI**>(dMediums),
                          dGPUMediums.data(),
                          dGPUMediums.size() * sizeof(GPUMediumI*),
                          cudaMemcpyHostToDevice));

    // Attach Medium gpu pointer to Material Groups
    for(const auto& mg : materialGroups)
    {
        mg.second->AttachGlobalMediumArray(dMediums, baseMediumIndex);
        if((e = mg.second->ConstructTextureReferences()) != TracerError::OK)
            return e;
    }

    // Construct Accelerators
    if((e = rayCaster->ConstructAccelerators(dTransforms,
                                             identityTransformIndex)) != TracerError::OK)
       return e;

    // Finally Construct GPU Light and Camera Lists
    // Lights
    if((e = LoadLights(dGPULights, dGPUEndpoints)) != TracerError::OK)
        return e;

    // Cameras
    if((e = LoadCameras(dGPUCameras, dGPUEndpoints)) != TracerError::OK)
        return e;

    // Endpoints
    CUDA_CHECK(cudaMemcpy(const_cast<GPUEndpointI**>(dEndpoints),
                          dGPUEndpoints.data(),
                          dGPUEndpoints.size() * sizeof(GPUEndpointI*),
                          cudaMemcpyHostToDevice));

    // Now Call all endpoints to generate their unique id (iota basically)
    const auto& gpu = cudaSystem.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, endpointCount,
                       //
                       KCSetEndpointIds,
                       //
                       const_cast<GPUEndpointI**>(dEndpoints),
                       endpointCount);

    // Also Call all lights to generate their globalIds
    // (which will be used by direct light sampler etc..)
    if(lightCount != 0)
        gpu.GridStrideKC_X(0, (cudaStream_t)0, lightCount,
                           //
                           KCSetLightIds,
                           //
                           const_cast<GPULightI**>(dLights),
                           lightCount);

    cudaSystem.SyncAllGPUs();
    return TracerError::OK;
}

void GPUTracer::ResetHitMemory(uint32_t rayCount, HitKey baseBoundMatKey)
{
    rayCaster->ResizeRayOut(rayCount, baseBoundMatKey);
}

VisorTransform GPUTracer::SceneCamTransform(uint32_t cameraIndex)
{
    return cameraVisorTransforms[cameraIndex];
}

const GPUCameraI* GPUTracer::GenerateCameraWithTransform(const VisorTransform& t,
                                                         uint32_t cameraIndex)
{
    // Copy the newly selected camera
    if(cameraIndex != currentCameraIndex)
    {
        const std::string& camGroupName = cameraGroupNames[cameraIndex];
        const auto& camGroup = cameras.at(camGroupName);
        camGroup->CopyCamera(tempTransformedCam,
                             dCameras[cameraIndex],
                             cudaSystem);
        currentCameraIndex = cameraIndex;
    }

    // Apply Transform
    const auto& gpu = cudaSystem.BestGPU();
    gpu.KC_X(0, (cudaStream_t)0, 1,
                //
                KCTransformCam,
                //
                static_cast<GPUCameraI*>(tempTransformedCam),
                t);
    gpu.WaitMainStream();

    return static_cast<const GPUCameraI*>(tempTransformedCam);
}

void GPUTracer::SetParameters(const TracerParameters& p)
{
    if(params.seed != p.seed)
        rngMemory = std::move(RNGMemory(p.seed, cudaSystem));
    params = p;
}

void GPUTracer::SetImagePixelFormat(PixelFormat f)
{
    imgMemory.SetPixelFormat(f, cudaSystem);
}

void GPUTracer::ReportionImage(Vector2i start,
                                Vector2i end)
{
    imgMemory.Reportion(start, end, cudaSystem);
}

void GPUTracer::ResizeImage(Vector2i resolution)
{
    imgMemory.Resize(resolution);
}

void GPUTracer::ResetImage()
{
    imgMemory.Reset(cudaSystem);
    if(callbacks)
    {
        Vector2i start = imgMemory.SegmentOffset();
        Vector2i end = start + imgMemory.SegmentSize();
        callbacks->SendImageSectionReset(start, end);
    }
}

template <class... Args>
inline void GPUTracer::SendLog(const char* format, Args... args)
{
    if(!params.verbose) return;

    std::string s = fmt::format(format, args...);
    if(callbacks) callbacks->SendLog(s);
}

void GPUTracer::SendError(TracerError e, bool isFatal)
{
    if(callbacks) callbacks->SendError(e);
    crashed = isFatal;
}

void GPUTracer::Finalize()
{
    if(crashed) return;
    SendLog("Finalizing...");

    // Determine Size
    Vector2i pixelCount = imgMemory.SegmentSize();
    Vector2i start = imgMemory.SegmentOffset();
    Vector2i end = start + imgMemory.SegmentSize();
    size_t offset = (static_cast<size_t>(pixelCount[0]) * pixelCount[1] *
                     imgMemory.PixelSize());

    // Flush Devices and Get the Image
    cudaSystem.SyncAllGPUs();
    std::vector<Byte> imageData = imgMemory.GetImageToCPU(cudaSystem);
    //size_t pixelCount1D = static_cast<size_t>(pixelCount[0]) * pixelCount[1];

    //Debug::DumpMemToFile("OutPixels",
    //                     reinterpret_cast<Vector4*>(imageData.data()),
    //                     pixelCount1D);
    //Debug::DumpMemToFile("OutSamples",
    //                     reinterpret_cast<uint32_t*>(imageData.data() + offset),
    //                     pixelCount1D);
    // Debug::DumpImage("SentImage.png",
    //                 reinterpret_cast<Vector4*>(imageData.data()),
    //                 Vector2ui(pixelCount[0], pixelCount[1]));

    // Launch finished image
    if(callbacks) callbacks->SendImage(std::move(imageData),
                                       imgMemory.Format(),
                                       offset,
                                       start, end);
    SendLog("Image sent!");
}

void GPUTracer::AskParameters()
{
    if(callbacks) callbacks->SendCurrentParameters(params);
}