#include "RayCasterOptiX.h"
#include "RayLib/GPUSceneI.h"
#include "CudaSystem.h"
#include "GPUAcceleratorOptiX.cuh"
#include "OptixCheck.h"

#include <optix_stack_size.h>

RayCasterOptiX::RayCasterOptiX(const GPUSceneI& gpuScene,
                               const CudaSystem& system)
    : baseAccelerator(*gpuScene.BaseAccelerator())
    , accelBatches(gpuScene.AcceleratorBatchMappings())
    , boundaryTransformIndex(gpuScene.BoundaryTransformIndex())
    , maxAccelBits(DetermineMaxBitFromId(gpuScene.MaxAccelIds()))
    , maxWorkBits(DetermineMaxBitFromId(gpuScene.MaxMatIds()))
    , maxHitSize(gpuScene.HitStructUnionSize())
    , cudaSystem(system)
    , currentRayCount(0)
    , optixSystem(system)
{}

TracerError RayCasterOptiX::CreateProgramGroups(const std::string& rgFuncName,
                                                const std::vector<HitFunctionNames>& hitFuncNames)
{
    OptixProgramGroupOptions pgOpts = {};

    TracerError err = TracerError::OK;
    uint32_t i = 0;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        OptixModule gpuModule = hOptixModules[i];

        hOptixPrograms.emplace_back();
        hOptixPrograms.back().emplace_back();

        OptixProgramGroupDesc rhProgramDesc = {};
        rhProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rhProgramDesc.raygen.module = gpuModule;
        rhProgramDesc.raygen.entryFunctionName = rgFuncName.c_str();

        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &rhProgramDesc, 1,
                                            &pgOpts,
                                            nullptr, 0,
                                            &hOptixPrograms.back().back()));

        for(const auto& [chFuncName, ahFuncName, iFuncName] : hitFuncNames)
        {
            hOptixPrograms.back().emplace_back();
            auto& hitProgram = hOptixPrograms.back().back();

            OptixProgramGroupDesc hProgramDesc = {};
            hProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hProgramDesc.hitgroup.moduleCH = gpuModule;
            hProgramDesc.hitgroup.entryFunctionNameCH = (chFuncName.empty()) ? nullptr : chFuncName.c_str();
            hProgramDesc.hitgroup.moduleAH = gpuModule;
            hProgramDesc.hitgroup.entryFunctionNameAH = (ahFuncName.empty()) ? nullptr : ahFuncName.c_str();
            hProgramDesc.hitgroup.moduleIS = gpuModule;
            hProgramDesc.hitgroup.entryFunctionNameIS = (iFuncName.empty()) ? nullptr : iFuncName.c_str();

            OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                                &hProgramDesc, 1,
                                                &pgOpts,
                                                nullptr, 0,
                                                &hitProgram));
        }
        i++;
    }
    return TracerError::OK;
}

TracerError RayCasterOptiX::CreateModules(const OptixModuleCompileOptions& mOpts,
                                          const OptixPipelineCompileOptions& pOpts,
                                          const std::string& baseFileName)
{
    TracerError err = TracerError::OK;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        std::string ptxSource;
        if((err = OptiXSystem::LoadPTXFile(ptxSource, gpu, baseFileName)) != TracerError::OK)
            return err;

        hOptixModules.emplace_back();
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                             &mOpts, &pOpts,
                                             ptxSource.c_str(),
                                             ptxSource.size(),
                                             nullptr,
                                             nullptr,
                                             &hOptixModules.back()));
    }
    return TracerError::OK;
}

TracerError RayCasterOptiX::CreatePipelines(const OptixPipelineCompileOptions& pOpts,
                                            const OptixPipelineLinkOptions& lOpts)
{
    TracerError err = TracerError::OK;

    uint32_t i = 0;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        const ProgramGroups& pgs = hOptixPrograms[i];

        hOptixPipelines.emplace_back();

        OPTIX_CHECK(optixPipelineCreate(optixContext,
                                        &pOpts, &lOpts,
                                        pgs.data(),
                                        static_cast<uint32_t>(pgs.size()),
                                        nullptr, nullptr,
                                        &hOptixPipelines.back()));

        // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
        // parameters to optixPipelineSetStackSize.
        OptixStackSizes stack_sizes = {};
        for(const auto& pg : pgs)
            OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));

        uint32_t dcStackSizeTraverse;
        uint32_t dcStackSizeState;
        uint32_t contStackSize;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                               2,   // max trace depth
                                               0, 0,
                                               &dcStackSizeTraverse,
                                               &dcStackSizeState,
                                               &contStackSize));

        const uint32_t maxTraversalDepth = 2;
        OPTIX_CHECK(optixPipelineSetStackSize(hOptixPipelines.back(),
                                              dcStackSizeTraverse,
                                              dcStackSizeState,
                                              contStackSize,
                                              maxTraversalDepth));
        i++;
    }
    return TracerError::OK;
}

TracerError RayCasterOptiX::ConstructAccelerators(const GPUTransformI** dTransforms,
                                                  uint32_t identityTransformIndex)
{
    TracerError e = TracerError::OK;

    // Attach Transform gpu pointer to the Accelerator Batches
    // Get OptiX Base from the class and set the OptixSystem
    for(auto& [_, acc] : accelBatches)
    {
        acc->AttachGlobalTransformArray(dTransforms, identityTransformIndex);

        auto accOptiX = dynamic_cast<GPUAccGroupOptiXI*>(acc);
        accOptiX->SetOptiXSystem(&optixSystem);
    }

    // Construct Accelerators
    for(const auto& accBatch : accelBatches)
    {
        GPUAcceleratorGroupI* acc = accBatch.second;
        if((e = acc->ConstructAccelerators(cudaSystem)) != TracerError::OK)
            return e;

    }
    // Construct Base accelerator using aabb list
    if((e = baseAccelerator.Constrcut(cudaSystem, SurfaceAABBList())) != TracerError::OK)
        return e;

    // We constructed Accelerator
    // Now do OptiX boilerplate
    // =============================== //
    //       MODULE GENERATION         //
    // =============================== //
    OptixModuleCompileOptions moduleCompileOpts = {};
    OptixPipelineCompileOptions pipelineCompileOpts = {};
    moduleCompileOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    if constexpr(METU_DEBUG_BOOL)
    {
        moduleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        pipelineCompileOpts.exceptionFlags = (OPTIX_EXCEPTION_FLAG_DEBUG |
                                              OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                              OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW);
    }
    else
    {
        moduleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

        #if OPTIX_VERSION > 70200
            moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        #else
            moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        #endif
        pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }
    pipelineCompileOpts.usesMotionBlur = false;
    pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOpts.numPayloadValues = 2;
    pipelineCompileOpts.numAttributeValues = (maxHitSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "params";

    if((e = CreateModules(moduleCompileOpts, pipelineCompileOpts,
                          MODULE_BASE_NAME)) != TracerError::OK)
        return e;
    return TracerError::OK;


    // =============================== //
    //    PROGRAM GROUP GENERATION     //
    // =============================== //
    std::vector<HitFunctionNames> hfNames;
    // Query Accelerators & get hf names


    if((e = CreateProgramGroups(RAYGEN_FUNC_NAME, hfNames)) != TracerError::OK)
        return e;

    // =============================== //
    //         SBT GENERATION          //
    // =============================== //


    // =============================== //
    //      PIPELINE GENERATION        //
    // =============================== //
    OptixPipelineLinkOptions pipelineLinkOpts = {};
    pipelineLinkOpts.maxTraceDepth = 1;
    pipelineLinkOpts.debugLevel = moduleCompileOpts.debugLevel;
    if((e = CreatePipelines(pipelineCompileOpts,
                            pipelineLinkOpts)) != TracerError::OK);
        return e;

    // All Done!
    return TracerError::OK;
}

RayPartitions<uint32_t> RayCasterOptiX::HitAndPartitionRays()
{
    return RayPartitions<uint32_t>();
}

void RayCasterOptiX::WorkRays(const WorkBatchMap& workMap,
                         const RayPartitionsMulti<uint32_t>& outPortions,
                         const RayPartitions<uint32_t>& inPartitions,
                         RNGMemory& rngMemory,
                         uint32_t totalRayOut,
                         HitKey baseBoundMatKey)
{

}

void RayCasterOptiX::ResizeRayOut(uint32_t rayCount,
                                  HitKey baseBoundMatKey)
{

}

void RayCasterOptiX::SwapRays()
{

}

RayGMem* RayCasterOptiX::RaysOut()
{
    return nullptr;
}
