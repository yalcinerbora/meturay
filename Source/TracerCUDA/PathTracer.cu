#include "PathTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"

#include "PathTracerWorks.cuh"
#include "GPULightSamplerUniform.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"

//#include "TracerDebug.h"
//std::ostream& operator<<(std::ostream& stream, const RayAuxPath& v)
//{
//    stream << std::setw(0)
//        << v.pixelIndex << ", "
//        << "{" << v.radianceFactor[0]
//        << "," << v.radianceFactor[1]
//        << "," << v.radianceFactor[2] << "} "
//        << v.endPointIndex << ", "
//        << v.mediumIndex << " ";
//    switch(v.type)
//    {
//        case RayType::CAMERA_RAY:
//            stream << "CAMERA_RAY";
//            break;
//        case RayType::NEE_RAY:
//            stream << "NEE_RAY";
//            break;
//        case RayType::SPECULAR_PATH_RAY:
//            stream << "SPEC_PATH_RAY";
//            break;
//        case RayType::PATH_RAY:
//            stream << "PATH_RAY";
//    }
//    return stream;
//}

template <class T>
__global__ void KCConstructLightSampler(T* loc,
                                        const GPULightI** gLights,
                                        const uint32_t lightCount)
{
    uint32_t globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId == 0)
    {
        T* lightSampler = new (loc) T(gLights, lightCount);
    }
}

TracerError PathTracer::LightSamplerNameToEnum(PathTracer::LightSamplerType& ls,
                                               const std::string& lsName)
{
    const std::array<std::string, LightSamplerType::END> samplerNames =
    {
        "Uniform"
    };

    uint32_t i = 0;
    for(const std::string s : samplerNames)
    {
        if(lsName == s)
        {
            ls = static_cast<LightSamplerType>(i);
            return TracerError::OK;
        }
        i++;
    }
    return TracerError::UNABLE_TO_INITIALIZE;
}

TracerError PathTracer::ConstructLightSampler()
{
    LightSamplerType lst;
    TracerError e = LightSamplerNameToEnum(lst, options.lightSamplerType);

    if(e != TracerError::OK) 
        return e;

    switch(lst)
    {
        case LightSamplerType::UNIFORM:
        {
            DeviceMemory::EnlargeBuffer(memory, sizeof(GPULightSamplerUniform));
            lightSampler = static_cast<const GPUDirectLightSamplerI*>(memory);

            const auto& gpu = cudaSystem.BestGPU();
            gpu.KC_X(0, (cudaStream_t)0, 1,
                     // Kernel
                     KCConstructLightSampler<GPULightSamplerUniform>,
                     // Args
                     static_cast<GPULightSamplerUniform*>(memory),
                     dLights, 
                     lightCount);

            return TracerError::OK;
        }
    }
    return TracerError::UNABLE_TO_INITIALIZE;
}

PathTracer::PathTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
{
    // Append Work Types for generation
    boundaryWorkPool.AppendGenerators(PTBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(PTPathWorkerList{});
    neeWorkPool.AppendGenerators(PTNEEWorkerList{});
    misWorkPool.AppendGenerators(PTMISWorkerList{});
    comboWorkPool.AppendGenerators(PTComboWorkerList{});
}

TracerError PathTracer::Initialize()
{
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // Generate Light Sampler (for nee)
    ConstructLightSampler();

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);

        // Generate work batch from appropirate work pool
        WorkBatchArray workBatchList;
        if(mg.IsBoundary())
        {
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
            if constexpr(USE_SINGLE_PATH_KERNEL)            
            {
                WorkPool<bool, bool>& wpCombo = comboWorkPool;
                GPUWorkBatchI* batch = nullptr;
                if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                                    dTransforms,
                                                    options.nextEventEstimation,
                                                    options.directLightMIS)) != TracerError::OK)
                    return err;
                workBatchList.push_back(batch);
            }
            else
            {
                WorkPool<>& wpPath = pathWorkPool;
                GPUWorkBatchI* pathBatch = nullptr;
                if((err = wpPath.GenerateWorkBatch(pathBatch, mg, pg,
                                                   dTransforms)) != TracerError::OK)
                    return err;
                workBatchList.push_back(pathBatch);

                if(options.nextEventEstimation)
                {
                    WorkPool<bool>& wpNEE = neeWorkPool;
                    GPUWorkBatchI* neeBatch = nullptr;
                    if((err = wpNEE.GenerateWorkBatch(neeBatch, mg, pg,
                                                      dTransforms,
                                                      options.directLightMIS)) != TracerError::OK)
                        return err;
                    workBatchList.push_back(neeBatch);
                }

                if(options.nextEventEstimation &&
                   options.directLightMIS)
                {
                    WorkPool<>& wpMIS = misWorkPool;
                    GPUWorkBatchI* misBatch = nullptr;
                    if((err = wpMIS.GenerateWorkBatch(misBatch, mg, pg,
                                                      dTransforms)) != TracerError::OK)
                        return err;
                    workBatchList.push_back(misBatch);
                }
            }
        }
        workMap.emplace(batchId, workBatchList);
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
    if((err = opts.GetBool(options.directLightMIS, DIRECT_LIGHT_MIS_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.lightSamplerType, LIGHT_SAMPLER_TYPE_NAME)) != TracerError::OK)
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
    //Debug::DumpMemToFile("primIds", rayMemory.PrimitiveIds(),
    //                     currentRayCount);
    //Debug::DumpMemToFile("hitKeys", rayMemory.CurrentKeys(),
    //                     currentRayCount);

    // Generate Global Data Struct
    PathTracerGlobalState globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.lightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;
    globalData.directLightMIS = options.directLightMIS;
    globalData.lightSampler = lightSampler;

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
        using WorkData = typename GPUWorkBatchD<PathTracerGlobalState, RayAuxPath>;
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

    //Debug::DumpMemToFile("auxOut",
    //                     static_cast<const RayAuxPath*>(*dAuxOut),
    //                     totalOutRayCount);
    //// Work rays swapped the ray buffer so read input rays
    //Debug::DumpMemToFile("rayOut", rayMemory.Rays(),
    //                     totalOutRayCount);
    //Debug::DumpMemToFile("rayIdOut", rayMemory.CurrentIds(),
    //                     totalOutRayCount);

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    // Check tracer termination conditions
    currentDepth++;
    if(totalOutRayCount == 0 || currentDepth >= options.maximumDepth)
        return false;
    return true;
}

void PathTracer::GenerateWork(int cameraId)
{
    if(callbacks)
        callbacks->SendCurrentCamera(SceneCamToVisorCam(cameraId));

    GenerateRays<RayAuxPath, RayInitPath>(dCameras[cameraId],
                                          options.sampleCount,
                                          InitialPathAux);
    currentDepth = 0;
}

void PathTracer::GenerateWork(const VisorCamera& cam)
{
    GenerateRays<RayAuxPath, RayInitPath>(cam, options.sampleCount,
                                          InitialPathAux);
    currentDepth = 0;
}