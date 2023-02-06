#include "OctreeOptiX.h"
#include "OptixCheck.h"

#include "RayLib/CPUTimer.h"

#include <optix_stack_size.h>

#include "TracerDebug.h"
#include "GPUAcceleratorOptiX.cuh"

template<class T>
__global__
void KCGenAABBAndMortonCode(// Ouptuts
                            AABB3f* gAABBs,
                            T* gMortonCodes,
                            // Inputs
                            AnisoSVOctreeGPU svo,
                            uint32_t level,
                            uint32_t nodeCount)
{
    float levelVoxSize = svo.LevelVoxelSize(level);
    uint32_t levelOffset = svo.LevelNodeStart(level);
    bool isLeaf = (level == svo.LeafDepth());

    // Debug
    //nodeCount = (isLeaf) ? 1 : nodeCount;

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < nodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint32_t nodeId = (isLeaf) ? threadId : levelOffset + threadId;
        uint32_t depth;
        Vector3ui voxelId = svo.NodeVoxelId(depth, nodeId, isLeaf);

        // Gen AABB using VoxelId
        Vector3f voxIdF = Vector3f(voxelId);
        Vector3f voxAABBMin = svo.OctreeAABB().Min() + voxIdF * levelVoxSize;
        Vector3f voxAABBMax = voxAABBMin + levelVoxSize;
        //printf("[%u, %u, %u]\n",
        //       voxelId[0], voxelId[1], voxelId[2]);
        // Write
        gMortonCodes[threadId] = MortonCode::Compose3D<T>(voxelId);
        gAABBs[threadId] = AABB3f(voxAABBMin, voxAABBMax);
    }
}

SVOOptixConeCaster::SVOOptixConeCaster(const OptiXSystem& optixSystem,
                                       const GPUBaseAcceleratorI& baseAccelerator,
                                       const AnisoSVOctreeCPU& svoCPU)
    : svoCPU(svoCPU)
    , optixSystem(optixSystem)
    , paramsMemory(&optixSystem.OptixCapableDevices()[0].first)
    , dOptixLaunchParams(nullptr)
    , dOptixTraversables(nullptr)
    , sbtMemory(&optixSystem.OptixCapableDevices()[0].first)
{
    const auto& [gpu, optixContext] = optixSystem.OptixCapableDevices()[0];

    // TODO: Bad design, change this
    const GPUBaseAcceleratorOptiX& baseAccel = static_cast<const GPUBaseAcceleratorOptiX&>(baseAccelerator);
    sceneTraversable = baseAccel.GetBaseTraversable(0);
    sceneSBTCount = baseAccel.TotalHitSBTCount();

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
    //pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

    pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOpts.numPayloadValues = 3;
    pipelineCompileOpts.numAttributeValues = 0;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = PARAMS_BUFFER;

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
    assert(programGroups.size() == (RAD_RAYGEN_PG_INDEX + 1));

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
    assert(programGroups.size() == (CAM_RAYGEN_PG_INDEX + 1));

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
    assert(programGroups.size() == (MISS_PG_INDEX + 1));

    // HIT GROUP NAME (MORTON 32)
    programGroups.emplace_back();
    OptixProgramGroupDesc h32ProgramDesc = {};
    h32ProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    h32ProgramDesc.hitgroup.moduleCH = gpuModule;
    h32ProgramDesc.hitgroup.entryFunctionNameCH = CHIT_FUNC_NAME;
    h32ProgramDesc.hitgroup.moduleAH = nullptr;
    h32ProgramDesc.hitgroup.entryFunctionNameAH = nullptr;
    h32ProgramDesc.hitgroup.moduleIS = gpuModule;
    h32ProgramDesc.hitgroup.entryFunctionNameIS = INTERSECT32_FUNC_NAME;
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &h32ProgramDesc, 1,
                                        &pgOpts,
                                        nullptr, 0,
                                        &programGroups.back()));
    assert(programGroups.size() == (MORTON32_HIT_PG_INDEX + 1));

    // HIT GROUP NAME (MORTON 64)
    programGroups.emplace_back();
    OptixProgramGroupDesc h64ProgramDesc = {};
    h64ProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    h64ProgramDesc.hitgroup.moduleCH = gpuModule;
    h64ProgramDesc.hitgroup.entryFunctionNameCH = CHIT_FUNC_NAME;
    h64ProgramDesc.hitgroup.moduleAH = nullptr;
    h64ProgramDesc.hitgroup.entryFunctionNameAH = nullptr;
    h64ProgramDesc.hitgroup.moduleIS = gpuModule;
    h64ProgramDesc.hitgroup.entryFunctionNameIS = INTERSECT64_FUNC_NAME;
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &h64ProgramDesc, 1,
                                        &pgOpts,
                                        nullptr, 0,
                                        &programGroups.back()));

    // HIT GROUP NAME (SCENE)
    programGroups.emplace_back();
    OptixProgramGroupDesc scenHitProgramDesc = {};
    scenHitProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    scenHitProgramDesc.hitgroup.moduleCH = gpuModule;
    scenHitProgramDesc.hitgroup.entryFunctionNameCH = CHIT_SCENE_FUNC_NAME;
    scenHitProgramDesc.hitgroup.moduleAH = nullptr;
    // TODO: This is not correct
    scenHitProgramDesc.hitgroup.entryFunctionNameAH = nullptr;
    scenHitProgramDesc.hitgroup.moduleIS = nullptr;
    scenHitProgramDesc.hitgroup.entryFunctionNameIS = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &scenHitProgramDesc, 1,
                                        &pgOpts,
                                        nullptr, 0,
                                        &programGroups.back()));
    assert(programGroups.size() == (SCENE_HIT_PG_INDEX + 1));

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
}

SVOOptixConeCaster::~SVOOptixConeCaster()
{
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    for(auto& pg : programGroups)
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
    OPTIX_CHECK(optixModuleDestroy(mdl));
}

void SVOOptixConeCaster::GenerateSVOTraversable()
{
    Utility::CPUTimer t;
    t.Start();

    const auto& [gpu, optixContext] = optixSystem.OptixCapableDevices()[0];

    std::vector<uint32_t> levelNodeOffsets = svoCPU.LevelNodeOffsets();
    AnisoSVOctreeGPU svo = svoCPU.TreeGPU();

    // Allocate mortonCodeMemory
    // For records
    // Slightly improve memory here use 32-bit for 1024 levels
    static constexpr uint32_t VOXEL_MORTON3D_FIT_THRESHOLD = 10;
    static constexpr std::array<size_t, 2> MORTON_SIZE = {sizeof(uint32_t), sizeof(uint64_t)};
    mortonByteOffsets.reserve(svo.LeafDepth() + 2);

    // Record type for the svo level
    std::vector<uint32_t> levelRecordTypeIndex;

    size_t offset = 0;
    // Determine Size & Record Type (32-bit or 64-bit)
    mortonByteOffsets.push_back(0);
    for(uint32_t i = 1; i <= svo.LeafDepth(); i++)
    {
        uint32_t localPrimCount = levelNodeOffsets[i + 1] - levelNodeOffsets[i];
        bool is32BitRecord = (i <= VOXEL_MORTON3D_FIT_THRESHOLD);

        size_t mortonSize = (is32BitRecord) ? MORTON_SIZE[0] : MORTON_SIZE[1];

        levelRecordTypeIndex.push_back((is32BitRecord) ? LEVEL_32_BIT : LEVEL_64_BIT);
        size_t localSize = localPrimCount * mortonSize;
        mortonByteOffsets.push_back(offset);
        offset += localSize;
    }
    mortonByteOffsets.push_back(offset);
    mortonMemory = DeviceMemory(mortonByteOffsets.back());

    std::vector<uint32_t*> dMortonPtrs32(svo.LeafDepth() + 1, nullptr);
    std::vector<uint64_t*> dMortonPtrs64(svo.LeafDepth() + 1, nullptr);
    for(uint32_t i = 1; i <= svo.LeafDepth(); i++)
    {
        uint32_t localPrimCount = levelNodeOffsets[i + 1] - levelNodeOffsets[i];
        Byte* dMortonStart = static_cast<Byte*>(mortonMemory) + mortonByteOffsets[i];
        if(i <= VOXEL_MORTON3D_FIT_THRESHOLD)
            dMortonPtrs32[i] = reinterpret_cast<uint32_t*>(dMortonStart);
        else
            dMortonPtrs64[i] = reinterpret_cast<uint64_t*>(dMortonStart);
    }

    DeviceLocalMemory tempAABB(&gpu, svo.LeafCount() * sizeof(AABB3f));
    AABB3f* dTempAABBs = static_cast<AABB3f*>(tempAABB);

    // Call a kernel for each level to generate AABB
    svoLevelAcceleratorMemory.resize(svo.LeafDepth(), DeviceLocalMemory(&gpu));
    for(uint32_t i = 1; i <= svo.LeafDepth(); i++)
    {
        uint32_t localPrimCount = levelNodeOffsets[i + 1] - levelNodeOffsets[i];

        // Generate AABB
        if(dMortonPtrs32[i] != nullptr)
        {
            gpu.GridStrideKC_X(0, (cudaStream_t)0, localPrimCount,
                               //
                               KCGenAABBAndMortonCode<uint32_t>,
                               // Outputs
                               dTempAABBs,
                               dMortonPtrs32[i],
                               // Inputs
                               svo,
                               i,
                               localPrimCount);
        }
        else
        {
            gpu.GridStrideKC_X(0, (cudaStream_t)0, localPrimCount,
                               //
                               KCGenAABBAndMortonCode<uint64_t>,
                               // Outputs
                               dTempAABBs,
                               dMortonPtrs64[i],
                               // Inputs
                               svo,
                               i,
                               localPrimCount);
        }
        // Debug
        //METU_LOG("-----------------------");
        //Debug::DumpMemToStdOut(dMortonPtrs32[i], localPrimCount);
        //METU_LOG("-----------------------");

        //
        uint32_t flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        CUdeviceptr aabbBuffer = AsOptixPtr(dTempAABBs);

        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        // AABB
        buildInput.customPrimitiveArray.aabbBuffers = &aabbBuffer;
        buildInput.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(localPrimCount);
        buildInput.customPrimitiveArray.strideInBytes = sizeof(AABB3f);
        buildInput.customPrimitiveArray.primitiveIndexOffset = 0;
        // SBT
        buildInput.customPrimitiveArray.flags = &flags;
        buildInput.customPrimitiveArray.numSbtRecords = 1;
        buildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
        buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        buildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                   OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes accelMemorySizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
        (
            optixContext,
            &accelOptions, &buildInput,
            1, &accelMemorySizes
        ));

        // Allocate Temp Buffer for Build
        DeviceLocalMemory buildBuffer(&gpu, accelMemorySizes.outputSizeInBytes);
        Byte* dTempBuild = static_cast<Byte*>(buildBuffer);
        Byte* dTemp;
        uint64_t* dCompactedSize;
        DeviceLocalMemory tempMemory(&gpu);
        GPUMemFuncs::AllocateMultiData(std::tie(dTemp, dCompactedSize), tempMemory,
                                       {accelMemorySizes.tempSizeInBytes, 1},
                                       OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);

        // While building fetch compacted output size
        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = AsOptixPtr(dCompactedSize);

        OptixTraversableHandle traversable;
        OPTIX_CHECK(optixAccelBuild(optixContext, (cudaStream_t)0,
                                    &accelOptions,
                                    // Build Inputs
                                    &buildInput, 1,
                                    // Temp Memory
                                    AsOptixPtr(dTemp), accelMemorySizes.tempSizeInBytes,
                                    // Output Memory
                                    AsOptixPtr(dTempBuild), accelMemorySizes.outputSizeInBytes,
                                    &traversable, &emitProperty, 1));
        CUDA_KERNEL_CHECK();

        // Get compacted size to CPU
        uint64_t hCompactAccelSize;
        CUDA_CHECK(cudaMemcpy(&hCompactAccelSize, dCompactedSize,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));

        if(hCompactAccelSize < buildBuffer.Size())
        {
            DeviceLocalMemory compactedMemory(&gpu, hCompactAccelSize);

            // use handle as input and output
            OptixTraversableHandle compacted;
            OPTIX_CHECK(optixAccelCompact(optixContext, (cudaStream_t)0,
                                          traversable,
                                          AsOptixPtr(compactedMemory),
                                          hCompactAccelSize,
                                          &compacted));
            CUDA_KERNEL_CHECK();
            traversable = compacted;
            svoLevelAcceleratorMemory[i - 1] = std::move(compactedMemory);
        }
        else
        {
            svoLevelAcceleratorMemory[i - 1] = std::move(buildBuffer);
        }

        svoLevelAccelerators.emplace_back(traversable);

        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0));
    }

    t.Split();
    METU_LOG("OptiX SVO GAS hierarchy generated in {:f} ms.",
             t.Elapsed<CPUTimeMillis>());


    // =============================== //
    //     SHADER BINDING TABLES       //
    // =============================== //
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Calculate the HR Amount
    SVOEmptyRecord* dRadRaygenRecord;
    SVOEmptyRecord* dCamRaygenRecord;
    SVOEmptyRecord* dMissRecord;
    SVOHitRecord<uint64_t>* dHitRecords;
    SVOEmptyRecord* dSceneHitRecord;

    // Sanity Check
    static_assert(sizeof(SVOHitRecord<uint64_t>) == sizeof(SVOHitRecord<uint32_t>));
    static constexpr uint32_t HIT_RECORD_STRIDE = static_cast<uint32_t>(std::max(sizeof(SVOHitRecord<uint64_t>),
                                                                                 sizeof(SVOHitRecord<uint32_t>)));

    const uint32_t recordCount = static_cast<uint32_t>(svo.LeafDepth());

    GPUMemFuncs::AllocateMultiData(std::tie(dCamRaygenRecord,
                                            dRadRaygenRecord,
                                            dMissRecord,
                                            dSceneHitRecord,
                                            dHitRecords),
                                   sbtMemory,
                                   {1, 1, 1, sceneSBTCount, svo.LeafDepth()},
                                   OPTIX_SBT_RECORD_ALIGNMENT);

    SVOEmptyRecord hRadRGRecord = SVOEmptyRecord{};
    SVOEmptyRecord hCamRGRecord = SVOEmptyRecord{};
    SVOEmptyRecord hMissRecord = SVOEmptyRecord{};
    SVOHitRecord<uint32_t> hHitRecord32 = SVOHitRecord<uint32_t>{};
    SVOHitRecord<uint64_t> hHitRecord64 = SVOHitRecord<uint64_t>{};
    SVOEmptyRecord hSceneHitRecord = SVOEmptyRecord{};

    // Set Raygen Record
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[RAD_RAYGEN_PG_INDEX],
                                         &hRadRGRecord));
    CUDA_CHECK(cudaMemcpy(dRadRaygenRecord, &hRadRGRecord, sizeof(SVOEmptyRecord),
                          cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[CAM_RAYGEN_PG_INDEX],
                                         &hCamRGRecord));
    CUDA_CHECK(cudaMemcpy(dCamRaygenRecord, &hCamRGRecord, sizeof(SVOEmptyRecord),
                          cudaMemcpyHostToDevice));
    // Set Miss Record
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[MISS_PG_INDEX], &hMissRecord));
    CUDA_CHECK(cudaMemcpy(dMissRecord, &hMissRecord, sizeof(SVOEmptyRecord),
                          cudaMemcpyHostToDevice));
    // Set Hit Record
    // Preset the headers (According to docs it is opaque but copyable if each loc uses same
    // program)
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[MORTON32_HIT_PG_INDEX], &hHitRecord32));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[MORTON64_HIT_PG_INDEX], &hHitRecord64));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[SCENE_HIT_PG_INDEX], &hSceneHitRecord));

    // Memcpys

    // Scene Accelerator requires many SBTs (according to the scene)
    // we need to provide single SBT for all of these
    // so duplicate the common hit shader

    hSceneHitRecord.empty = nullptr;
    std::vector<SVOEmptyRecord> expandedSceneRecords(sceneSBTCount,
                                                     hSceneHitRecord);
    CUDA_CHECK(cudaMemcpy(dSceneHitRecord,
                          expandedSceneRecords.data(),
                          sizeof(SVOEmptyRecord) * sceneSBTCount,
                          cudaMemcpyHostToDevice));

    for(uint32_t i = 0; i < svo.LeafDepth(); i++)
    {
        if(levelRecordTypeIndex[i] == LEVEL_32_BIT)
        {
            hHitRecord32.dMortonCode = dMortonPtrs32[i + 1];
            CUDA_CHECK(cudaMemcpy(dHitRecords + i,
                                  reinterpret_cast<SVOHitRecord<uint64_t>*>(&hHitRecord32),
                                  sizeof(SVOHitRecord<uint32_t>),
                                  cudaMemcpyHostToDevice));
        }
        else
        {
            hHitRecord64.dMortonCode = dMortonPtrs64[i + 1];
            CUDA_CHECK(cudaMemcpy(dHitRecords + i,
                                  &hHitRecord64, sizeof(SVOHitRecord<uint64_t>),
                                  cudaMemcpyHostToDevice));
        }
    }

    // SBT CAM GEN
    // Although we do not use the miss shader
    // Optix mandates these to be set
    // Exception can be null though
    // SBT RAD GEN
    sbtRadGen = {};
    sbtRadGen.raygenRecord = AsOptixPtr(dRadRaygenRecord);
    //
    sbtRadGen.missRecordBase = AsOptixPtr(dMissRecord);
    sbtRadGen.missRecordCount = 1;
    sbtRadGen.missRecordStrideInBytes = sizeof(EmptyRecord);
    //
    sbtRadGen.hitgroupRecordBase = AsOptixPtr(dHitRecords);
    sbtRadGen.hitgroupRecordStrideInBytes = HIT_RECORD_STRIDE;
    sbtRadGen.hitgroupRecordCount = recordCount;
     // SBT RADGEN-SCENE
    sbtRadGenScene = {};
    sbtRadGenScene.raygenRecord = AsOptixPtr(dRadRaygenRecord);
    //
    sbtRadGenScene.missRecordBase = AsOptixPtr(dMissRecord);
    sbtRadGenScene.missRecordCount = 1;
    sbtRadGenScene.missRecordStrideInBytes = sizeof(EmptyRecord);
    //
    sbtRadGenScene.hitgroupRecordBase = AsOptixPtr(dSceneHitRecord);
    sbtRadGenScene.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbtRadGenScene.hitgroupRecordCount = sceneSBTCount;
    // SBT CAM GEN
    sbtCamGen = {};
    sbtCamGen.raygenRecord = AsOptixPtr(dCamRaygenRecord);
    //
    sbtCamGen.missRecordBase = AsOptixPtr(dMissRecord);
    sbtCamGen.missRecordCount = 1;
    sbtCamGen.missRecordStrideInBytes = sizeof(EmptyRecord);
    //
    sbtCamGen.hitgroupRecordBase = AsOptixPtr(dHitRecords);
    sbtCamGen.hitgroupRecordStrideInBytes = HIT_RECORD_STRIDE;
    sbtCamGen.hitgroupRecordCount = recordCount;
    // SBT CAMGEN-SCENE
    sbtCamGenScene = {};
    sbtCamGenScene.raygenRecord = AsOptixPtr(dCamRaygenRecord);
    //
    sbtCamGenScene.missRecordBase = AsOptixPtr(dMissRecord);
    sbtCamGenScene.missRecordCount = 1;
    sbtCamGenScene.missRecordStrideInBytes = sizeof(EmptyRecord);
    //
    sbtCamGenScene.hitgroupRecordBase = AsOptixPtr(dSceneHitRecord);
    sbtCamGenScene.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbtCamGenScene.hitgroupRecordCount = sceneSBTCount;


    // Finally Allocate Traversable GPU Array & Parameter Buffer
    GPUMemFuncs::AllocateMultiData(std::tie(dOptixLaunchParams,
                                            dOptixTraversables),
                                   paramsMemory,
                                   {1, svoLevelAccelerators.size()});
    // And copy the traversables to memory
    CUDA_CHECK(cudaMemcpy(dOptixTraversables,
                          svoLevelAccelerators.data(),
                          sizeof(OptixTraversableHandle) *
                          svoLevelAccelerators.size(),
                          cudaMemcpyHostToDevice));
}

void SVOOptixConeCaster::ConeTraceFromCamera(// Output
                                             CamSampleGMem<Vector4f> gSamples,
                                             // Input
                                             const RayGMem* gRays,
                                             const RayAuxWFPG* gRayAux,
                                             WFPGRenderMode mode,
                                             uint32_t maxQueryLevelOffset,
                                             bool useSceneAccelerator,
                                             float pixelAperture,
                                             const uint32_t totalRayCount)
{
    // Copy Params to GPU
    OctreeAccelParams& params = hOptixLaunchParams;
    // Common
    params.octreeLevelBVHs = dOptixTraversables;
    params.sceneBVH = sceneTraversable;
    params.utilizeSceneAccelerator = useSceneAccelerator;
    params.svo = svoCPU.TreeGPU();
    params.pixelOrConeAperture = pixelAperture;
    // Mode Related
    params.gRays = gRays;
    params.gRayAux = gRayAux;
    params.renderMode = mode;
    params.maxQueryOffset = maxQueryLevelOffset;
    params.gSamples = gSamples;

    //// test large apertures
    //params.pixelOrConeAperture = 4.0f * MathConstants::Pi / 64.0f / 64.0f;

    CUDA_CHECK(cudaMemcpyAsync(dOptixLaunchParams,
                               &params,
                               sizeof(OctreeAccelParams),
                               cudaMemcpyHostToDevice,
                               (cudaStream_t)0));

    OptixShaderBindingTable* sbt = (useSceneAccelerator)
                                    ? &sbtCamGenScene
                                    : &sbtCamGen;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));
    OPTIX_CHECK(optixLaunch(pipeline, (cudaStream_t)0,
                            AsOptixPtr(dOptixLaunchParams),
                            sizeof(OctreeAccelParams),
                            sbt,
                            totalRayCount,
                            1, 1));
    CUDA_CHECK(cudaEventRecord(stop));


    float milliseconds = 0;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    METU_LOG("----{}", milliseconds);


    CUDA_KERNEL_CHECK();
}

void SVOOptixConeCaster::CopyRadianceMapGenParams(const Vector4f* dRadianceFieldRayOrigins,
                                                  const float* dProjJitters,
                                                  SVOOptixRadianceBuffer::SegmentedField<float*> fieldSegments,
                                                  bool useSceneAccelerator,
                                                  float coneAperture)
{
    // Copy Params to GPU
    OctreeAccelParams& params = hOptixLaunchParams;
    // Common
    params.octreeLevelBVHs = dOptixTraversables;
    params.sceneBVH = sceneTraversable;
    params.utilizeSceneAccelerator = useSceneAccelerator;
    params.svo = svoCPU.TreeGPU();
    params.pixelOrConeAperture = coneAperture;
    // Mode Related
    params.fieldSegments = fieldSegments;
    params.dRadianceFieldRayOrigins = dRadianceFieldRayOrigins;
    params.dProjJitters = dProjJitters;
    params.binOffset = 0;

    CUDA_CHECK(cudaMemcpyAsync(dOptixLaunchParams,
                               &params,
                               sizeof(OctreeAccelParams),
                               cudaMemcpyHostToDevice,
                               (cudaStream_t)0));
}

void SVOOptixConeCaster::RadianceMapGen(// Range over this id/offsets
                                        uint32_t segmentOffset,
                                        uint32_t totalRayCount)
{
    // Going full C here
    size_t innerOffset = offsetof(OctreeAccelParams, binOffset);
    Byte* dBinOffsetLoc = (reinterpret_cast<Byte*>(dOptixLaunchParams) +
                           innerOffset);
    CUDA_CHECK(cudaMemcpyAsync(dBinOffsetLoc,
                               &segmentOffset,
                               sizeof(int32_t),
                               cudaMemcpyHostToDevice,
                               (cudaStream_t)0));

    OptixShaderBindingTable* sbt = (hOptixLaunchParams.utilizeSceneAccelerator)
                                        ? &sbtRadGenScene
                                        : &sbtRadGen;

    OPTIX_CHECK(optixLaunch(pipeline, (cudaStream_t)0,
                            AsOptixPtr(dOptixLaunchParams),
                            sizeof(OctreeAccelParams),
                            sbt,
                            totalRayCount,
                            1, 1));
}