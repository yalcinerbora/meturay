#include "WFPGTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerOptions.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/FileUtility.h"
#include "RayLib/VisorTransform.h"

#include "WFPGTracerWorks.cuh"
#include "GPULightSamplerUniform.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"
#include "GPUAcceleratorI.h"

// Constructors & Destructor
WFPGTracer::WFPGTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
{
    // Append Work Types for generation
    boundaryWorkPool.AppendGenerators(WFPGBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(WFPGPathWorkerList{});

    debugBoundaryWorkPool.AppendGenerators(WFPGDebugBoundaryWorkerList{});
    debugPathWorkPool.AppendGenerators(WFPGDebugPathWorkerList{});
}

TracerError WFPGTracer::Initialize()
{
    iterationCount = 0;

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
        GPUWorkBatchI* batch = nullptr;
        if(options.debugRender)
        {
            if(options.debugRender)
            {
                WorkPool<>& wp = debugPathWorkPool;
                if((err = wp.GenerateWorkBatch(batch, mg, pg,
                                               dTransforms)) != TracerError::OK)
                    return err;
            }
        }
        else
        {
            WorkPool<bool, bool>& wpCombo = pathWorkPool;
            if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                                dTransforms,
                                                options.nextEventEstimation,
                                                options.directLightMIS)) != TracerError::OK)
                return err;
        }

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
        GPUWorkBatchI* batch = nullptr;
        if(options.debugRender)
        {
            BoundaryWorkPool<>& wp = debugBoundaryWorkPool;
            if((err = wp.GenerateWorkBatch(batch, eg,
                                           dTransforms)) != TracerError::OK)
                return err;
        }
        else
        {
            BoundaryWorkPool<bool, bool>& wp = boundaryWorkPool;
            if((err = wp.GenerateWorkBatch(batch, eg, dTransforms,
                                           options.nextEventEstimation,
                                           options.directLightMIS)) != TracerError::OK)
                return err;
        }
        workBatchList.push_back(batch);
        workMap.emplace(batchId, workBatchList);
    }

    // Init SVO
    if((err == svo.Constrcut(scene.BaseAccelerator()->SceneExtents(),
                             (1 << options.octreeLevel),
                             scene.AcceleratorBatchMappings(),
                             dLights, lightCount,
                             scene.BaseBoundaryMaterial(),
                             cudaSystem)) != TracerError::OK)
        return err;

    return TracerError::OK;
}

TracerError WFPGTracer::SetOptions(const TracerOptionsI& opts)
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
    if((err = opts.GetUInt(options.octreeLevel, OCTREE_LEVEL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.minRayBinLevel, RAY_BIN_MIN_LEVEL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.binRayCount, BIN_RAY_COUNT_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.debugRender, DEBUG_RENDER_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.dumpDebugData, DUMP_DEBUG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.svoDumpInterval, DUMP_INTERVAL_NAME)) != TracerError::OK)
        return err;
    return TracerError::OK;
}

void WFPGTracer::AskOptions()
{
    VariableList list;

    list.emplace(MAX_DEPTH_NAME, OptionVariable(options.maximumDepth));
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));
    list.emplace(RR_START_NAME, OptionVariable(options.rrStart));
    list.emplace(LIGHT_SAMPLER_TYPE_NAME, OptionVariable(LightSamplerCommon::LightSamplerTypeToString(options.lightSamplerType)));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));
    list.emplace(DIRECT_LIGHT_MIS_NAME, OptionVariable(options.directLightMIS));
    list.emplace(OCTREE_LEVEL_NAME, OptionVariable(options.octreeLevel));
    list.emplace(RAY_BIN_MIN_LEVEL_NAME, OptionVariable(options.minRayBinLevel));
    list.emplace(BIN_RAY_COUNT_NAME, OptionVariable(options.binRayCount));
    list.emplace(DEBUG_RENDER_NAME, OptionVariable(options.debugRender));
    list.emplace(DUMP_DEBUG_NAME, OptionVariable(options.dumpDebugData));
    list.emplace(DUMP_INTERVAL_NAME, OptionVariable(options.svoDumpInterval));
    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}

void WFPGTracer::GenerateWork(uint32_t cameraIndex)
{

}

void WFPGTracer::GenerateWork(const VisorTransform&, uint32_t cameraIndex)
{

}

void WFPGTracer::GenerateWork(const GPUCameraI&)
{

}

bool WFPGTracer::Render()
{
    return true;
}

void WFPGTracer::Finalize()
{

}

size_t WFPGTracer::TotalGPUMemoryUsed() const
{
    return 0;
}