#include "PathTracer.h"
#include "TracerWorks.cuh"

#include "RayLib/GPUSceneI.h"

#include "TracerLib/LightCameraKernels.cuh"
#include "TracerLib/GenerationKernels.cuh"
#include "TracerLib/GPUWork.cuh"

//#include "TracerLib/TracerDebug.h"
//std::ostream& operator<<(std::ostream& stream, const RayAuxPath& v)
//{
//    stream << std::setw(0)
//        << v.pixelIndex << ", "
//        << "{" << v.radianceFactor[0]
//        << "," << v.radianceFactor[1]
//        << "," << v.radianceFactor[2] << "} "
//        << v.endPointIndex << " ";
//    switch(v.type)
//    {
//        case RayType::CAMERA_RAY:
//            stream << "CAMERA_RAY";
//            break;
//        case RayType::NEE_RAY:
//            stream << "NEE_RAY";
//            break;
//        case RayType::TRANS_RAY:
//            stream << "TRANS_RAY";
//            break;
//        case RayType::PATH_RAY:
//            stream << "PATH_RAY";
//    }
//    return stream;
//}

__device__ __host__
inline void RayInitPath(RayAuxPath& gOutPath,
                         // Input
                        const RayAuxPath& defaults,
                        const RayReg& ray,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId)
{
    RayAuxPath init = defaults;
    init.pixelIndex = localPixelId;
    init.type = RayType::CAMERA_RAY;
    init.mediumIndex = medumIndex;
    init.depth = 1;
    gOutPath = init;
}

PathTracer::PathTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
    , dLights(nullptr)
    , dLightAlloc(nullptr)
{
    workPool.AppendGenerators(PathTracerWorkerList{});
    lightWorkPool.AppendGenerators(PathTracerLightWorkerList{});
}

TracerError PathTracer::Initialize()
{
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    //const auto& lights = scene.LightsCPU();
    //lightCount = static_cast<uint32_t>(lights.size());
    //// Determine Size
    //size_t lightPtrSize = sizeof(GPULightI*) * lightCount;
    //size_t lightSize = LightCameraKernels::LightClassesUnionSize() * lightCount;
    //size_t totalSize = lightPtrSize + lightSize;

    //if(lights.size() != 0)
    //{
    //    DeviceMemory::EnlargeBuffer(lightMemory, totalSize);
    //    // Load light and create light interfaces
    //    // Determine Ptrs
    //    size_t offset = 0;
    //    Byte* dMem = static_cast<Byte*>(lightMemory);
    //    dLights = reinterpret_cast<const GPULightI**>(dMem + offset);
    //    offset += lightPtrSize;
    //    dLightAlloc = dMem + offset;
    //    offset += lightSize;
    //    assert(offset == totalSize);

    //    // Construct lights on GPU
    //    LightCameraKernels::ConstructLights(const_cast<GPULightI**>(dLights),
    //                                        dLightAlloc,
    //                                        lights,
    //                                        cudaSystem);
    //}

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);
        
        // Generate work batch from appropirate work pool
        GPUWorkBatchI* batch = nullptr;
        if(mg.IsLightGroup())
        {
            bool emptyPrim = (std::string(pg.Type()) == 
                              std::string(BaseConstants::EMPTY_PRIMITIVE_NAME));

            WorkPool<bool, bool>& wp = lightWorkPool;
            if((err = wp.GenerateWorkBatch(batch, mg, pg, dTransforms,
                                           options.nextEventEstimation,
                                           emptyPrim)) != TracerError::OK)
                return err;
        }
        else
        {
            WorkPool<bool>& wp = workPool;
            if((err = wp.GenerateWorkBatch(batch, mg, pg, dTransforms,
                                           options.nextEventEstimation)) != TracerError::OK)
                return err;
        }
        workMap.emplace(batchId, batch);
    }
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
    if((err = opts.GetUInt(options.rrStart, RR_START_NAME)) != TracerError::OK)
        return err;
    return TracerError::OK;
}

bool PathTracer::Render()
{
    HitAndPartitionRays();

    //Debug::DumpMemToFile("auxIn",
    //                     static_cast<const RayAuxPath*>(*dAuxIn),
    //                     currentRayCount);
    //Debug::DumpMemToFile("rayIn",
    //                     rayMemory.Rays(),
    //                     currentRayCount);
    //Debug::DumpMemToFile("rayIdIn", rayMemory.CurrentIds(),
    //                     currentRayCount);

    // Generate Global Data Struct
    PathTracerGlobal globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.lightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;  
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;

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
        RayAuxPath* dAuxOutLocal = static_cast<RayAuxPath*>(*dAuxOut) + p.offset;
        const RayAuxPath* dAuxInLocal = static_cast<const RayAuxPath*>(*dAuxIn);
                                                    
        using WorkData = typename GPUWorkBatchD<PathTracerGlobal, RayAuxPath>;
        auto& wData = static_cast<WorkData&>(*loc->second);
        wData.SetGlobalData(globalData);
        wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);
    }

    // Launch Kernels
    WorkRays(workMap, outPartitions,
             totalOutRayCount, 
             scene.BaseBoundaryMaterial());

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    //Debug::DumpMemToFile("auxOut",
    //                     static_cast<const RayAuxPath*>(*dAuxOut),
    //                     totalOutRayCount);
    //// Work rays swapped the ray buffer so read input rays
    //Debug::DumpMemToFile("rayOut", rayMemory.Rays(),
    //                     totalOutRayCount);
    //Debug::DumpMemToFile("rayIdOut", rayMemory.CurrentIds(),
    //                     totalOutRayCount);

    SwapAuxBuffers();
    // Check tracer termination conditions
    currentDepth++;
    if(totalOutRayCount == 0 || currentDepth >= options.maximumDepth)
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