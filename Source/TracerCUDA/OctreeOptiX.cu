#include "OctreeOptiX.h"
#include "OptixCheck.h"

#include <optix_stack_size.h>

SVOOptixConeCaster::SVOOptixConeCaster(const OptiXSystem& optixSystem)
    : optixSystem(optixSystem)
    , paramsMemory(&optixSystem.OptixCapableDevices()[0].first,
                   sizeof(OpitXBaseAccelParams))
    , sbtMemory(&optixSystem.OptixCapableDevices()[0].first)
{
    const auto& [gpu, optixContext] = optixSystem.OptixCapableDevices()[0];

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

        #if OPTIX_VERSION > 70300
        moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        #else
        moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        #endif
        pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }
    pipelineCompileOpts.usesMotionBlur = false;
    pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOpts.numPayloadValues = 2;
    pipelineCompileOpts.numAttributeValues = 0;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "params";

    TracerError err = TracerError::OK;
    std::vector<Byte> ptxSource;
    if((err = OptiXSystem::LoadPTXFile(ptxSource, gpu, MODULE_BASE_NAME)) != TracerError::OK)
        throw TracerException(err);

    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOpts, &pipelineCompileOpts,
                                         reinterpret_cast<const char*>(ptxSource.data()),
                                         ptxSource.size(),
                                         nullptr,
                                         nullptr,
                                         &mdl));

    OptixProgramGroupOptions pgOpts = {};
    OptixModule gpuModule = mdl;

    // RADIANCE GEN RAYGEN NAME
    programGroups.emplace_back();
    OptixProgramGroupDesc rgRadGenProgramDesc = {};
    rgRadGenProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgRadGenProgramDesc.raygen.module = gpuModule;
    rgRadGenProgramDesc.raygen.entryFunctionName = RAYGEN_RAD_FUNC_NAME;
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &rgRadGenProgramDesc, 1,
                                        &pgOpts,
                                        nullptr, 0,
                                        &programGroups.back()));

    // CAMERA GEN RAYGEN NAME
    programGroups.emplace_back();
    OptixProgramGroupDesc rgCamGenProgramDesc = {};
    rgCamGenProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgCamGenProgramDesc.raygen.module = gpuModule;
    rgCamGenProgramDesc.raygen.entryFunctionName = RAYGEN_CAM_FUNC_NAME;
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &rgCamGenProgramDesc, 1,
                                        &pgOpts,
                                        nullptr, 0,
                                        &programGroups.back()));

    // EMPTY MISS NAME
    programGroups.emplace_back();
    OptixProgramGroupDesc missProgramDesc = {};
    missProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missProgramDesc.miss.module = gpuModule;
    missProgramDesc.miss.entryFunctionName = MISS_FUNC_NAME;
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &missProgramDesc, 1,
                                        &pgOpts,
                                        nullptr, 0,
                                        &programGroups.back()));

    // HIT GROUP NAME
    programGroups.emplace_back();

    OptixProgramGroupDesc hProgramDesc = {};
    hProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hProgramDesc.hitgroup.moduleCH = gpuModule;
    hProgramDesc.hitgroup.entryFunctionNameCH = CHIT_FUNC_NAME;
    hProgramDesc.hitgroup.moduleAH = nullptr;
    hProgramDesc.hitgroup.entryFunctionNameAH = nullptr;
    // Do we need this?
    //hProgramDesc.hitgroup.moduleIS = gpuModule;
    //hProgramDesc.hitgroup.entryFunctionNameIS = INTERSECT_FUNC_NAME;

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &hProgramDesc, 1,
                                        &pgOpts,
                                        nullptr, 0,
                                        &programGroups.back()));

    // =============================== //
    //      PIPELINE GENERATION        //
    // =============================== //
    OptixPipelineLinkOptions pipelineLinkOpts = {};
    pipelineLinkOpts.maxTraceDepth = 1;
    pipelineLinkOpts.debugLevel = moduleCompileOpts.debugLevel;

    const std::vector<OptixProgramGroup>& pgs = programGroups;

    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOpts, &pipelineLinkOpts,
                                    pgs.data(),
                                    static_cast<uint32_t>(pgs.size()),
                                    nullptr, nullptr,
                                    &pipeline));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    for(const auto& pg : pgs)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));

    uint32_t dcStackSizeTraverse;
    uint32_t dcStackSizeState;
    uint32_t contStackSize;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                           1,   // max trace depth
                                           0, 0,
                                           &dcStackSizeTraverse,
                                           &dcStackSizeState,
                                           &contStackSize));

    const uint32_t maxTraversalDepth = 1; // Single GAS
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
                                          dcStackSizeTraverse,
                                          dcStackSizeState,
                                          contStackSize,
                                          maxTraversalDepth));


    // =============================== //
    //     SHADER BINDING TABLE        //
    // =============================== //
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    // Allocate Record data on device local memory
    // I am allocating multiple here just to be sure (we don't know
    // what is on the header)
    SVOEmptyRecord* dRadRaygenRecord;
    SVOEmptyRecord* dCamRaygenRecord;
    SVOEmptyRecord* dMissRecord;
    SVOEmptyRecord* dHitRecord;
    GPUMemFuncs::AllocateMultiData(std::tie(dCamRaygenRecord,
                                            dRadRaygenRecord,
                                            dMissRecord, dHitRecord),
                                   sbtMemory,
                                   {1, 1, 1},
                                   OPTIX_SBT_RECORD_ALIGNMENT);

    SVOEmptyRecord hRadRGRecord = SVOEmptyRecord{};
    SVOEmptyRecord hCamRGRecord = SVOEmptyRecord{};
    SVOEmptyRecord hMissRecord = SVOEmptyRecord{};
    SVOEmptyRecord hHitRecord = SVOEmptyRecord{};
    // Set Raygen Record
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[0], &hRadRGRecord));
    CUDA_CHECK(cudaMemcpy(dRadRaygenRecord, &hRadRGRecord, sizeof(SVOEmptyRecord),
                          cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[1], &hCamRGRecord));
    CUDA_CHECK(cudaMemcpy(dCamRaygenRecord, &hCamRGRecord, sizeof(SVOEmptyRecord),
                          cudaMemcpyHostToDevice));
    // Set Miss Record
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[2], &hMissRecord));
    CUDA_CHECK(cudaMemcpy(dMissRecord, &hMissRecord, sizeof(SVOEmptyRecord),
                          cudaMemcpyHostToDevice));
    // Set Hit Record
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[3], &hHitRecord));
    CUDA_CHECK(cudaMemcpy(dHitRecord, &hHitRecord, sizeof(SVOEmptyRecord),
                          cudaMemcpyHostToDevice));


    // SBT CAM GEN
    // Although we do not use the miss shader
    // Optix mandates these to be set
    sbtRadGen.raygenRecord = AsOptixPtr(dRadRaygenRecord);
    //
    sbtRadGen.missRecordBase = AsOptixPtr(dMissRecord);
    sbtRadGen.missRecordCount = 1;
    sbtRadGen.missRecordStrideInBytes = sizeof(EmptyRecord);
    //
    sbtRadGen.hitgroupRecordBase = AsOptixPtr(dHitRecord);
    sbtRadGen.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbtRadGen.hitgroupRecordCount = 1;
    // SBT RAD GEN
    sbtCamGen.raygenRecord = AsOptixPtr(dRadRaygenRecord);
    //
    sbtCamGen.missRecordBase = AsOptixPtr(dMissRecord);
    sbtCamGen.missRecordCount = 1;
    sbtCamGen.missRecordStrideInBytes = sizeof(EmptyRecord);
    //
    sbtCamGen.hitgroupRecordBase = AsOptixPtr(dHitRecord);
    sbtCamGen.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbtCamGen.hitgroupRecordCount = 1;
}

void SVOOptixConeCaster::GenerateSVOTraversable(const AnisoSVOctreeCPU& svoCPU)
{


}

void SVOOptixConeCaster::ConeTraceFromCamera(// Output
                                             CamSampleGMem<Vector3f> gSamples,
                                             // Input
                                             const GPUCameraI* gCamera,
                                             WFPGRenderMode mode,
                                             uint32_t maxQueryLevelOffset,
                                             const Vector2i& totalPixelCount)
{

    // TODO:
    hSVOOptixLaunchParams = {};

    CUDA_CHECK(cudaMemcpyAsync(dSVOOptixLaunchParams,
                               &hSVOOptixLaunchParams,
                               sizeof(OctreeAccelParams),
                               cudaMemcpyHostToDevice,
                               (cudaStream_t)0));
    OPTIX_CHECK(optixLaunch(pipeline, (cudaStream_t)0,
                            AsOptixPtr(dSVOOptixLaunchParams),
                            sizeof(OctreeAccelParams),
                            &sbtCamGen,
                            totalPixelCount[0],
                            totalPixelCount[1],
                            1));
}