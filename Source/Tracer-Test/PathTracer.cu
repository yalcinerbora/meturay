#include "PathTracer.h"
#include "TracerWorks.cuh"
#include "MetaTracerWork.cuh"

#include "RayLib/GPUSceneI.h"
#include "TracerLib/LightCameraKernels.cuh"
#include "TracerLib/GenerationKernels.cuh"

__device__ __host__
inline void RayInitPath(RayAuxPath& gOutPath,
                         // Input
                         const RayAuxPath& defaults,
                         const RayReg& ray,
                         // Index
                         const uint32_t localPixelId,
                         const uint32_t pixelSampleId)
{
    RayAuxPath init = defaults;
    init.pixelIndex = localPixelId;
    init.type = RayType::CAMERA_RAY;
    init.mediumIndex = 0;
    init.depth = 1;
    gOutPath = init;
}

PathTracer::PathTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, params)
    , currentDepth(0)
    , dLights(nullptr)
    , dLightAlloc(nullptr)
{
    workPool.AppendGenerators(PathTracerWorkList{});
}

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
        if((err = workPool.GenerateWorkBatch(batch, mg, pg,
                                             options.nextEventEstimation)) != TracerError::OK)
            return err;
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
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPath);
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Generate Global Data Struct
    PathTracerGlobal globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.lightList = dLights;
    globalData.totalLightCount = lightCount;

    for(auto& work : workMap)
    {
        using WorkData = typename MetaWorkBatchData<PathTracerGlobal, RayAuxPath>;

        auto& wData = static_cast<WorkData&>(*work.second);
        wData.SetGlobalData(globalData);
        wData.SetRayDataPtrs(static_cast<RayAuxPath*>(*dAuxOut),
                             static_cast<const RayAuxPath*>(*dAuxIn));
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

void PathTracer::GenerateRays(const GPUCameraI* dCamera, int32_t sampleCount)
{
    int32_t sampleCountSqr = sampleCount * sampleCount;

    const Vector2i resolution = imgMemory.Resolution();
    const Vector2i pixelStart = imgMemory.SegmentOffset();
    const Vector2i pixelEnd = pixelStart + imgMemory.SegmentSize();

    Vector2i pixelCount = (pixelEnd - pixelStart);
    uint32_t totalRayCount = pixelCount[0] * pixelCount[1] * sampleCountSqr;
    size_t auxBufferSize = totalRayCount * sizeof(RayAuxPath);

    // Allocate enough space for ray
    rayMemory.ResizeRayOut(totalRayCount, scene.BaseBoundaryMaterial());
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxBufferSize);

    // Basic Tracer does classic camera to light tracing
    // Thus its initial rays are from camera
    // Call multi-device
    const uint32_t TPB = StaticThreadPerBlock1D;
    // GPUSplits
    const auto splits = cudaSystem.GridStrideMultiGPUSplit(totalRayCount, TPB, 0,
                                                           KCGenerateCameraRaysGPU<RayAuxPath, RayInitPath>);

    // Only use splits as guidance
    // and Split work into columns (much easier to maintain..
    // however not perfectly balanced... (as all things should be))
    int i = 0;
    Vector2i localPixelStart = Zero2i;
    for(const CudaGPU& gpu : cudaSystem.GPUList())
    {
        // If no work is splitted to this GPU skip
        if(splits[i] == 0) break;

        // Generic Args
        // Offsets
        const size_t localPixelCount1D = splits[i] / sampleCountSqr;
        const int columnCount = static_cast<int>((localPixelCount1D + pixelCount[1] - 1) / pixelCount[1]);
        const int rowCount = pixelCount[1];
        Vector2i localPixelCount = Vector2i(columnCount, rowCount);
        Vector2i localPixelEnd = Vector2i::Min(localPixelStart + localPixelCount, pixelCount);
        Vector2i localWorkCount2D = (localPixelEnd - localPixelStart) * sampleCountSqr;
        size_t localWorkCount = localWorkCount2D[0] * localWorkCount2D[1];

        // Kernel Specific Args
        // Output
        RayGMem* gRays = rayMemory.RaysOut();
        RayAuxPath* gAuxiliary = static_cast<RayAuxPath*>(*dAuxOut);
        // Input
        RNGGMem rngData = rngMemory.RNGData(gpu);
        ImageGMem<Vector4f> gImgData = imgMemory.GMem<Vector4f>();

        // Kernel Call
        gpu.AsyncGridStrideKC_X
        (
            0, localWorkCount,
            KCGenerateCameraRaysGPU<RayAuxPath, RayInitPath>,
            // Args
            // Inputs
            gRays,
            gAuxiliary,
            gImgData,
            // Input
            rngData,
            dCamera,
            sampleCount,
            resolution,
            localPixelStart,
            localPixelEnd,
            // Data to initialize auxiliary base data
            InitialPathAux
        );

        // Adjust for next call
        localPixelStart = localPixelEnd;
        i++;
    }

    SwapAuxBuffers();
    rayMemory.SwapRays();
    currentRayCount = totalRayCount;
}