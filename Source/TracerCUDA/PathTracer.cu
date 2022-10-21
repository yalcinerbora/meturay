#include "PathTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/VisorTransform.h"

#include "PathTracerWorks.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"

#include "RayLib/Options.h"
#include "RayLib/TracerCallbacksI.h"

//#include "TracerDebug.h"
//std::ostream& operator<<(std::ostream& stream, const RayAuxPath& v)
//{
//    stream << std::setw(0)
//        << v.sampleIndex << ", "
//        << "{" << v.radianceFactor[0]
//        << "," << v.radianceFactor[1]
//        << "," << v.radianceFactor[2] << "} "
//        << v.endpointIndex << ", "
//        << v.mediumIndex << " "
//        << v.prevPDF << " ";
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

PathTracer::PathTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
    , conservativeOutRayPerMaterial(0)
{
    // Append Work Types for generation
    boundaryWorkPool.AppendGenerators(PTBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(PTPathWorkerList{});
}

TracerError PathTracer::Initialize()
{
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // Generate Light Sampler (for nee)
    if((err = LightSamplerCommon::ConstructLightSampler(lightSamplerMemory,
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

        WorkPool<bool, bool>& wpCombo = pathWorkPool;
        GPUWorkBatchI* batch = nullptr;
        if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                            dTransforms,
                                            options.nextEventEstimation,
                                            options.directLightMIS)) != TracerError::OK)
            return err;

        if(options.runAsMegaKernel)
            conservativeOutRayPerMaterial = std::max(conservativeOutRayPerMaterial,
                                                     static_cast<uint32_t>(batch->OutRayCount()));

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
        if((err = wp.GenerateWorkBatch(batch, eg,
                                       dTransforms,
                                       options.nextEventEstimation,
                                       options.directLightMIS)) != TracerError::OK)
            return err;

        if(options.runAsMegaKernel)
            conservativeOutRayPerMaterial = std::max(conservativeOutRayPerMaterial,
                                                     static_cast<uint32_t>(batch->OutRayCount()));

        workBatchList.push_back(batch);
        workMap.emplace(batchId, workBatchList);
    }

    if(options.runAsMegaKernel)
    {
        if((err = metaSurfHandler.Initialize(scene, workMap)) != TracerError::OK)
            return err;
    }
    return TracerError::OK;
}

TracerError PathTracer::SetOptions(const OptionsI& opts)
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
    if((err = opts.GetBool(options.runAsMegaKernel, MEGA_KERNEL_NAME)) != TracerError::OK)
        return err;

    std::string lightSamplerTypeString;
    if((err = opts.GetString(lightSamplerTypeString, LIGHT_SAMPLER_TYPE_NAME)) != TracerError::OK)
        return err;
    if((err = LightSamplerCommon::StringToLightSamplerType(options.lightSamplerType, lightSamplerTypeString)) != TracerError::OK)
        return err;
    return TracerError::OK;
}

bool PathTracer::Render()
{
    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(rayCaster->CurrentRayCount() == 0 ||
       currentDepth >= options.maximumDepth)
        return false;

    // Hit Rays
    rayCaster->HitRays();

    // Generate Global Data Struct
    PathTracerGlobalState globalData;
    globalData.gSamples = sampleMemory.GMem<Vector4f>();
    globalData.gLightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;
    globalData.directLightMIS = options.directLightMIS;
    globalData.gLightSampler = dLightSampler;

    if(options.runAsMegaKernel)
    {
        uint32_t totalRayOut = (rayCaster->CurrentRayCount() * conservativeOutRayPerMaterial);
        size_t auxOutSize = totalRayOut * sizeof(RayAuxPath);
        GPUMemFuncs::EnlargeBuffer(*dAuxOut, auxOutSize);
        // Do not partition the rays
        // However call this so that ray memory is properly initialized
        rayCaster->PartitionRaysWRTNothing(scene.BaseBoundaryMaterial(),
                                           totalRayOut);

        // Get Pointers
        // Input
        const RayAuxPath* dAuxInPtr         = static_cast<const RayAuxPath*>(*dAuxIn);
        const RayGMem* dRaysIn              = rayCaster->RaysIn();
        const HitKey* dKeysIn               = rayCaster->KeysIn();
        const HitStructPtr dHitStructs      = rayCaster->HitSturctPtr();
        const TransformId* dTransformIds    = rayCaster->TransformIds();
        const PrimitiveId* dPrimIdsIn       = rayCaster->PrimitiveIds();
        // Output
        RayAuxPath* dAuxOutPtr  = static_cast<RayAuxPath*>(*dAuxOut);
        RayGMem* dRaysOut       = rayCaster->RaysOut();
        HitKey* dKeysOut        = rayCaster->KeysOut();

        // TODO: partition the work between multiple GPUs
        const auto& gpu = cudaSystem.BestGPU();
        gpu.GridStrideKC_X(0, (cudaStream_t)0, rayCaster->CurrentRayCount(),
                           //
                           KCPathTracerMegaKernel,
                           // Output
                           dKeysOut,
                           dRaysOut,
                           dAuxOutPtr,
                           conservativeOutRayPerMaterial,
                           // Input
                           dRaysIn,
                           dAuxInPtr,
                           dPrimIdsIn,
                           dTransformIds,
                           dHitStructs,
                           dKeysIn,
                           // I-O
                           globalData,
                           rngCPU->GetGPUGenerators(gpu),
                           // MetaSurface Generator
                           metaSurfHandler.GetMetaSurfaceGroup(dHitStructs, dRaysIn),
                           //
                           rayCaster->CurrentRayCount(),
                           dTransforms);

        rayCaster->AssumeRaysAreWorked(totalRayOut);
    }
    else
    {
        const auto partitions = rayCaster->PartitionRaysWRTWork();
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
        for(auto p : outPartitions)
        {
            // Skip if null batch or not found material
            if(p.portionId == HitKey::NullBatch) continue;
            auto loc = workMap.find(p.portionId);
            if(loc == workMap.end()) continue;

            // Set pointers
            const RayAuxPath* dAuxInLocal = static_cast<const RayAuxPath*>(*dAuxIn);
            using WorkData = GPUWorkBatchD<PathTracerGlobalState, RayAuxPath>;
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
                            *rngCPU.get(),
                            totalOutRayCount,
                            scene.BaseBoundaryMaterial());
    }
    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    // Increase Depth
    currentDepth++;
    return true;
}

void PathTracer::Finalize()
{
    //METU_LOG("==");

    cudaSystem.SyncAllGPUs();
    frameTimer.Stop();
    UpdateFrameAnalytics("paths / sec", options.sampleCount * options.sampleCount);

    RayTracer::Finalize();
}

void PathTracer::GenerateWork(uint32_t cameraIndex)
{
    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    GenerateRays<RayAuxPath, RayAuxInitPath, RNGIndependentGPU, Vector4f>
    (
        cameraIndex,
        options.sampleCount,
        RayAuxInitPath(InitialPathAux),
        true
    );
    currentDepth = 0;
}

void PathTracer::GenerateWork(const VisorTransform& t, uint32_t cameraIndex)
{
    GenerateRays<RayAuxPath, RayAuxInitPath, RNGIndependentGPU, Vector4f>
    (
        t, cameraIndex, options.sampleCount,
        RayAuxInitPath(InitialPathAux),
        true
    );
    currentDepth = 0;
}

void PathTracer::GenerateWork(const GPUCameraI& dCam)
{
    GenerateRays<RayAuxPath, RayAuxInitPath, RNGIndependentGPU, Vector4f>
    (
        dCam, options.sampleCount,
        RayAuxInitPath(InitialPathAux),
        true
    );
    currentDepth = 0;
}

size_t PathTracer::TotalGPUMemoryUsed() const
{
    return (RayTracer::TotalGPUMemoryUsed() +
            lightSamplerMemory.Size());
}

void PathTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(static_cast<int64_t>(options.sampleCount)));
    list.emplace(MAX_DEPTH_NAME, OptionVariable(static_cast<int64_t>(options.maximumDepth)));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));
    list.emplace(DIRECT_LIGHT_MIS_NAME, OptionVariable(options.directLightMIS));
    list.emplace(RR_START_NAME, OptionVariable(static_cast<int64_t>(options.rrStart)));
    list.emplace(MEGA_KERNEL_NAME, OptionVariable(options.runAsMegaKernel));

    std::string lightSamplerTypeString = LightSamplerCommon::LightSamplerTypeToString(options.lightSamplerType);
    list.emplace(LIGHT_SAMPLER_TYPE_NAME, OptionVariable(lightSamplerTypeString));

    if(callbacks) callbacks->SendCurrentOptions(::Options(std::move(list)));
}