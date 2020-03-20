#include "PathTracer.h"
#include "TracerWorks.cuh"
#include "MetaTracerWork.cuh"

#include "RayLib/GPUSceneI.h"
#include "TracerLib/LightCameraKernels.cuh"

PathTracer::PathTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, params)
    , currentDepth(0)
    , dLights(nullptr)
    , dLightAlloc(nullptr)
{}

TracerError PathTracer::Initialize()
{
    const auto& lights = scene.LightsCPU();
    lightCount = static_cast<uint32_t>(lights.size());
    // Determine Size
    size_t lightPtrSize = sizeof(GPULightI*) * lightCount;
    size_t lightSize = LightCameraKernels::LightClassesUnionSize() * lightCount;
    size_t totalSize = lightPtrSize + lightSize;

    if(lights.size() != 0)
    {
        DeviceMemory::EnlargeBuffer(lightMemory, totalSize);
        // Load light and create light interfaces
        // Determine Ptrs
        size_t offset = 0;
        Byte* dMem = static_cast<Byte*>(lightMemory);
        dLights = reinterpret_cast<const GPULightI**>(dMem + offset);
        offset += lightPtrSize;
        dLightAlloc = dMem + offset;
        offset += lightSize;
        assert(offset == totalSize);

        // Construct lights on GPU
        LightCameraKernels::ConstructLights(const_cast<GPULightI**>(dLights),
                                            dLightAlloc,
                                            lights,
                                            cudaSystem);
    }

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
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.lightList = dLights;
    globalData.totalLightCount = lightCount;

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
    if(totalOutRayCount == 0 || currentDepth > options.maximumDepth)
        return false;
    return true;
}

void PathTracer::GenerateWork(int cameraId)
{
    GenerateRays(dSceneCameras[cameraId], options.sampleCount);
    currentDepth = 0;
}

void PathTracer::GenerateWork(const CPUCamera& cam)
{
    LoadCameraToGPU(cam);
    GenerateRays(*dCustomCamera, options.sampleCount);
    currentDepth = 0;
}