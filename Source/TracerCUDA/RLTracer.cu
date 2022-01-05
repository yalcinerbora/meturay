#include "RLTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/FileUtility.h"
#include "RayLib/VisorTransform.h"

#include "RLTracerWorks.cuh"
#include "GPULightSamplerUniform.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"
#include "GPUAcceleratorI.h"

#include "RayLib/TracerOptions.h"
#include "RayLib/TracerCallbacksI.h"

void RLTracer::ResizeAndInitPathMemory()
{
    size_t totalPathNodeCount = TotalPathNodeCount();
    GPUMemFuncs::EnlargeBuffer(pathMemory, totalPathNodeCount * sizeof(PathGuidingNode));
    dPathNodes = static_cast<PathGuidingNode*>(pathMemory);

    // Initialize Paths
    const CudaGPU& bestGPU = cudaSystem.BestGPU();
    if(totalPathNodeCount > 0)
        bestGPU.KC_X(0, 0, totalPathNodeCount,
                     //
                     KCInitializePaths,
                     //
                     dPathNodes,
                     static_cast<uint32_t>(totalPathNodeCount));
}

uint32_t RLTracer::TotalPathNodeCount() const
{
    return (imgMemory.SegmentSize()[0] * imgMemory.SegmentSize()[1] *
            options.sampleCount * options.sampleCount) * MaximumPathNodePerPath();
}

uint32_t RLTracer::MaximumPathNodePerPath() const
{
    return (options.maximumDepth == 0) ? 0 : (options.maximumDepth + 1);
}

RLTracer::RLTracer(const CudaSystem& s,
                   const GPUSceneI& scene,
                   const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
    , currentTreeIteration(0)
    , nextTreeSwap(1)
{
    // Append Work Types for generation
    boundaryWorkPool.AppendGenerators(RLBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(RLPathWorkerList{});
}

TracerError RLTracer::Initialize()
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

    if((err = surfaceTree.Construct(scene.AcceleratorBatchMappings(),
                                    options.normalThreshold,
                                    options.spatialSamples,
                                    params.seed,
                                    cudaSystem)) != TracerError::OK)
        return err;

    return TracerError::OK;
}

TracerError RLTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetUInt(options.maximumDepth, MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
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

    if((err = opts.GetBool(options.rawPathGuiding, RAW_PG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetVector2i(options.directionalRes, DIRECTONAL_RES_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.spatialSamples, SPATIAL_SAMPLE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.alpha, ALPHA_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.normalThreshold, NORM_THRESHOLD_NAME)) != TracerError::OK)
        return err;

    return TracerError::OK;
}

bool RLTracer::Render()
{
    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(rayCaster->CurrentRayCount() == 0 ||
       currentDepth >= options.maximumDepth)
        return false;

    const auto partitions = rayCaster->HitAndPartitionRays();

    // Generate Global Data Struct
    RLTracerGlobalState globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.gLightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.gLightSampler = dLightSampler;
    //
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    //
    // Set Positional Tree
    globalData.gPosTree = surfaceTree.TreeGPU();
    globalData.gSpatialData = denseArray.ArrayGPU();
    //
    globalData.gPathNodes = dPathNodes;
    globalData.maximumPathNodePerRay = MaximumPathNodePerPath();

    globalData.rawPathGuiding = options.rawPathGuiding;
    globalData.nee = options.nextEventEstimation;
    globalData.directLightMIS = options.directLightMIS;
    globalData.rrStart = options.rrStart;

    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = RayCasterI::PartitionOutputRays(totalOutRayCount,
                                                         partitions,
                                                         workMap);

    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxRL);
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
        const RayAuxRL* dAuxInLocal = static_cast<const RayAuxRL*>(*dAuxIn);
        using WorkData = GPUWorkBatchD<RLTracerGlobalState, RayAuxRL>;
        int i = 0;
        for(auto& work : loc->second)
        {
            RayAuxRL* dAuxOutLocal = static_cast<RayAuxRL*>(*dAuxOut) + p.offsets[i];

            auto& wData = static_cast<WorkData&>(*work);
            wData.SetGlobalData(globalData);
            wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);
            i++;
        }
    }

    // Launch Kernels
    rayCaster->WorkRays(workMap, outPartitions,
                        partitions, *rngCPU.get(),
                        totalOutRayCount,
                        scene.BaseBoundaryMaterial());

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    currentDepth++;
    return true;
}

void RLTracer::Finalize()
{
    cudaSystem.SyncAllGPUs();

    uint32_t totalPathNodeCount = TotalPathNodeCount();
    // Accumulate the finished radiances to the STree
    //posTree->AccumulateRaidances(dPathNodes, totalPathNodeCount,
    //                             MaximumPathNodePerPath(), cudaSystem);
    // We iterated once
    currentTreeIteration += 1;// options.sampleCount* options.sampleCount;


    cudaSystem.SyncAllGPUs();
    frameTimer.Stop();
    UpdateFrameAnalytics("paths / sec", options.sampleCount * options.sampleCount);
    // Base class finalize directly sends the image
    GPUTracer::Finalize();
}

void RLTracer::GenerateWork(uint32_t cameraIndex)
{
    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    GenerateRays<RayAuxRL, RayAuxInitRL, RNGIndependentGPU>
    (
        cameraIndex,
        options.sampleCount,
        RayAuxInitRL(InitialRLAux,
                     options.sampleCount *
                     options.sampleCount),
        true
    );

    ResizeAndInitPathMemory();
    currentDepth = 0;
}

void RLTracer::GenerateWork(const VisorTransform& t, uint32_t cameraIndex)
{
    GenerateRays<RayAuxRL, RayAuxInitRL, RNGIndependentGPU>
    (
        t, cameraIndex, options.sampleCount,
        RayAuxInitRL(InitialRLAux,
                     options.sampleCount *
                     options.sampleCount),
        true
    );
    ResizeAndInitPathMemory();
    currentDepth = 0;
}

void RLTracer::GenerateWork(const GPUCameraI& dCam)
{
    GenerateRays<RayAuxRL, RayAuxInitRL, RNGIndependentGPU>
    (
        dCam, options.sampleCount,
        RayAuxInitRL(InitialRLAux,
                     options.sampleCount *
                     options.sampleCount),
        true
    );
    ResizeAndInitPathMemory();
    currentDepth = 0;
}

size_t RLTracer::TotalGPUMemoryUsed() const
{
    return (RayTracer::TotalGPUMemoryUsed() +
            surfaceTree.UsedGPUMemory() +
            lightSamplerMemory.Size() + pathMemory.Size());
}

void RLTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));
    list.emplace(MAX_DEPTH_NAME, OptionVariable(options.maximumDepth));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}
