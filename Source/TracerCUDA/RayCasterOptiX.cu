#include "RayCasterOptiX.h"
#include "RayLib/GPUSceneI.h"
#include "CudaSystem.h"
#include "GPUAcceleratorOptiX.cuh"
#include "OptixCheck.h"
#include "TracerDebug.h"

#include <optix_stack_size.h>
#include <numeric>
#include <execution>

RayCasterOptiX::RayCasterOptiX(const GPUSceneI& gpuScene,
                               const CudaSystem& system)
    : RayCaster(gpuScene, system)
    , optixSystem(system)
    , dGlobalTransformArray(nullptr)
{
    optixGPUData.reserve(optixSystem.OptixCapableDevices().size());
    for(const auto& [gpu, optixCosntext] : optixSystem.OptixCapableDevices())
    {
        optixGPUData.push_back(
        {
            0,                          // Traversable Handle
            nullptr,                    // Pipeline
            nullptr,                    // Module
            {},                         // Program Groups
            OpitXBaseAccelParams{},     // Host Params
            nullptr,                    // Device Params
            DeviceLocalMemory(&gpu),    // Device Params Memory
            OptixShaderBindingTable{},  // SBT
            DeviceLocalMemory(&gpu)     // SBT Memory
        });
    }
}

RayCasterOptiX::~RayCasterOptiX()
{
    int optixDeviceIndex = 0;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        const auto& optixData = optixGPUData[optixDeviceIndex];

        // Destroy using opposite order
        if(optixData.pipeline) OPTIX_CHECK(optixPipelineDestroy(optixData.pipeline));
        for(auto pg : optixData.programGroups)
            OPTIX_CHECK(optixProgramGroupDestroy(pg));

        if(optixData.mdl) OPTIX_CHECK(optixModuleDestroy(optixData.mdl));
        optixDeviceIndex++;

        baseAccelerator.Destruct(cudaSystem);
        for(const auto& acc : accelBatches)
            acc.second->DestroyAccelerators(cudaSystem);
    }
    // Rest is memory so those will be destroyed automatically
}

TracerError RayCasterOptiX::CreateProgramGroups(const std::string& rgFuncName,
                                                const std::string& missFuncName,
                                                const std::vector<HitFunctionNames>& hitFuncNames)
{
    OptixProgramGroupOptions pgOpts = {};

    TracerError err = TracerError::OK;
    uint32_t i = 0;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        OptixModule gpuModule = optixGPUData[i].mdl;

        optixGPUData[i].programGroups.emplace_back();
        OptixProgramGroupDesc rhProgramDesc = {};
        rhProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rhProgramDesc.raygen.module = gpuModule;
        rhProgramDesc.raygen.entryFunctionName = rgFuncName.c_str();

        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &rhProgramDesc, 1,
                                            &pgOpts,
                                            nullptr, 0,
                                            &optixGPUData[i].programGroups.back()));

        optixGPUData[i].programGroups.emplace_back();
        OptixProgramGroupDesc missProgramDesc = {};
        missProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missProgramDesc.miss.module = gpuModule;
        missProgramDesc.miss.entryFunctionName = missFuncName.c_str();

        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &missProgramDesc, 1,
                                            &pgOpts,
                                            nullptr, 0,
                                            &optixGPUData[i].programGroups.back()));

        for(const auto& [chFuncName, ahFuncName, iFuncName] : hitFuncNames)
        {
            optixGPUData[i].programGroups.emplace_back();

            OptixProgramGroupDesc hProgramDesc = {};
            hProgramDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hProgramDesc.hitgroup.moduleCH = (chFuncName.empty()) ? nullptr : gpuModule;
            hProgramDesc.hitgroup.entryFunctionNameCH = (chFuncName.empty())
                                                            ? nullptr
                                                            : chFuncName.c_str();
            hProgramDesc.hitgroup.moduleAH = (ahFuncName.empty()) ? nullptr : gpuModule;
            hProgramDesc.hitgroup.entryFunctionNameAH = (ahFuncName.empty())
                                                            ? nullptr
                                                            : ahFuncName.c_str();
            hProgramDesc.hitgroup.moduleIS = (iFuncName.empty()) ? nullptr : gpuModule;
            hProgramDesc.hitgroup.entryFunctionNameIS = (iFuncName.empty())
                                                            ? nullptr
                                                            : iFuncName.c_str();
            OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                                &hProgramDesc, 1,
                                                &pgOpts,
                                                nullptr, 0,
                                                &optixGPUData[i].programGroups.back()));
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
    uint32_t i = 0;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        std::vector<Byte> ptxSource;
        if((err = OptiXSystem::LoadPTXFile(ptxSource, gpu, baseFileName)) != TracerError::OK)
            return err;

        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                             &mOpts, &pOpts,
                                             reinterpret_cast<const char*>(ptxSource.data()),
                                             ptxSource.size(),
                                             nullptr,
                                             nullptr,
                                             &optixGPUData[i].mdl));
        i++;
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
        const ProgramGroups& pgs = optixGPUData[i].programGroups;

        OPTIX_CHECK(optixPipelineCreate(optixContext,
                                        &pOpts, &lOpts,
                                        pgs.data(),
                                        static_cast<uint32_t>(pgs.size()),
                                        nullptr, nullptr,
                                        &optixGPUData[i].pipeline));

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

        const uint32_t maxTraversalDepth = 2;
        OPTIX_CHECK(optixPipelineSetStackSize(optixGPUData[i].pipeline,
                                              dcStackSizeTraverse,
                                              dcStackSizeState,
                                              contStackSize,
                                              maxTraversalDepth));
        i++;
    }
    return TracerError::OK;
}

TracerError RayCasterOptiX::CreateSBTs(const std::vector<Record<void,void>>& records,
                                       const std::vector<uint32_t>& programGroupIds)
{
    uint32_t i = 0;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        auto& optixData = optixGPUData[i];
        size_t sbtCount = records.size();

        // I don't know what sbt record pack header writes
        // to the header maybe it is GPU specific
        // set device beforehand
        CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

        // Allocate Record data on device local memory
        EmptyRecord* dRaygenRecord;
        EmptyRecord* dMissRecord;
        HitGroupRecord<void, void>* dHitRecords;
        GPUMemFuncs::AllocateMultiData(std::tie(dRaygenRecord, dMissRecord, dHitRecords),
                                       optixData.sbtMemory,
                                       {1, 1, sbtCount},
                                       OPTIX_SBT_RECORD_ALIGNMENT);


        // Host Data
        EmptyRecord hRGRecord = EmptyRecord{};
        EmptyRecord hMissRecord = EmptyRecord{};
        std::vector<HitGroupRecord<void, void>> hHitRecords(sbtCount,
                                                            HitGroupRecord<void, void>{});

        // Set Raygen Record
        OPTIX_CHECK(optixSbtRecordPackHeader(optixData.programGroups[0], &hRGRecord));
        CUDA_CHECK(cudaMemcpy(dRaygenRecord, &hRGRecord, sizeof(EmptyRecord),
                              cudaMemcpyHostToDevice));
        // Set Miss Record
        OPTIX_CHECK(optixSbtRecordPackHeader(optixData.programGroups[1], &hMissRecord));
        CUDA_CHECK(cudaMemcpy(dMissRecord, &hMissRecord, sizeof(EmptyRecord),
                              cudaMemcpyHostToDevice));

        int recordId = 0;
        for(auto& hitRecord : hHitRecords)
        {
            optixSbtRecordPackHeader(optixData.programGroups[programGroupIds[recordId]],
                                     &hitRecord);
            hitRecord.data = records[recordId];
            recordId++;
        }
        CUDA_CHECK(cudaMemcpy(dHitRecords, hHitRecords.data(),
                              hHitRecords.size() * sizeof(HitGroupRecord<void, void>),
                              cudaMemcpyHostToDevice));

        optixData.sbt.raygenRecord = AsOptixPtr(dRaygenRecord);
        // Although we do not use the miss shader
        // Optix mandates these to be set
        optixData.sbt.missRecordBase = AsOptixPtr(dMissRecord);
        optixData.sbt.missRecordCount = 1;
        optixData.sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
        //
        optixData.sbt.hitgroupRecordBase = AsOptixPtr(dHitRecords);
        optixData.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord<void, void>);
        optixData.sbt.hitgroupRecordCount = static_cast<uint32_t>(sbtCount);

        i++;
    }
    return TracerError::OK;
}

TracerError RayCasterOptiX::AllocateParams()
{
    uint32_t i = 0;
    for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
    {
        optixGPUData[i].paramsMemory = DeviceLocalMemory(&gpu, sizeof(OpitXBaseAccelParams));
        optixGPUData[i].dOptixLaunchParams = static_cast<OpitXBaseAccelParams*>(optixGPUData[i].paramsMemory);
        i++;
    }
    return TracerError::OK;
}

TracerError RayCasterOptiX::ConstructAccelerators(const GPUTransformI** dTransforms,
                                                  uint32_t identityTransformIndex)
{
    dGlobalTransformArray = dTransforms;
    TracerError e = TracerError::OK;

    // Find out total accelerator count
    size_t totalAcceleratorCount = 0;
    for(auto& [_, acc] : accelBatches)
    {
        totalAcceleratorCount += acc->AcceleratorCount();
    }

    // No objects on the scene return early
    if(totalAcceleratorCount == 0)
        return TracerError::OK;

    // Required Data for OptiX
    // List of traversables for each accelerator, (for each optix GPU)
    std::vector<std::vector<OptixTraversableHandle>> traversables;
    // Primitive Transform type of the Accelerator Group's Primitive (for each AcceleratorGroup)
    std::vector<PrimTransformType> hPrimTransformTypes;
    // List of all accelerator's transform ids (for each Accelerator on the scene)
    DeviceMemory allTransformIdMemory(totalAcceleratorCount * sizeof(TransformId));
    TransformId* dAllTransformIds = static_cast<TransformId*>(allTransformIdMemory);
    // Program Group index for each accelerator (for each SBT on the scene)
    std::vector<uint32_t> hProgramGroupIndices;
    // SBT Records (for each SBT on the scene)
    std::vector<Record<void, void>> hRecords;
    // Hit Function Names (for each accelerator group)
    std::vector<HitFunctionNames> hfNames;
    // CullFace flags (for each accelerator in the scene)
    std::vector<bool> hCullFlags;
    // SBT Offsets (for each accelerator in the scene)
    std::vector<uint32_t> hGlobalSBTCounts;
    // Reserve Memory
    traversables.resize(optixSystem.OptixCapableDevices().size());
    hPrimTransformTypes.reserve(accelBatches.size());
    hfNames.reserve(accelBatches.size());
    hCullFlags.reserve(totalAcceleratorCount);
    // Entire scene aabb
    AABB3f sceneAABB = NegativeAABB3f;
    // Attach Transform gpu pointer to the Accelerator Batches
    // Set the OptixSystem
    // Get optix required data from the accelerator groups
    size_t offset = 0;
    // 0th index is for raygen group
    // 1st index is for miss group
    uint32_t programIndex = 2;
    for(auto& [_, acc] : accelBatches)
    {
        acc->AttachGlobalTransformArray(dTransforms, identityTransformIndex);
        size_t acceleratorCount = acc->AcceleratorCount();

        auto accOptiX = dynamic_cast<GPUAccGroupOptiXI*>(acc);
        accOptiX->SetOptiXSystem(&optixSystem);

        // Construct the Accelerator
        if((e = acc->ConstructAccelerators(cudaSystem)) != TracerError::OK)
            return e;

        // Copy required data
        // (for base accelerator module creation sbt etc..)
        auto accTraversables = accOptiX->GetOptixTraversables();
        assert(traversables.size() == accTraversables.size());
        uint32_t i = 0;
        for(auto& t : traversables)
        {
            t.insert(t.end(), accTraversables[i].cbegin(),
                     accTraversables[i].cend());
            i++;
        }
        accTraversables.clear();

        // Copy Primitive Transform Type
        hPrimTransformTypes.insert(hPrimTransformTypes.end(), acceleratorCount,
                                   accOptiX->GetPrimitiveTransformType());


        const auto hAccGroupCullFlags = accOptiX->GetCullFlagPerAccel();
        hCullFlags.insert(hCullFlags.end(),
                          hAccGroupCullFlags.cbegin(),
                          hAccGroupCullFlags.cend());

        // Copy Transform Ids to global linear memory
        CUDA_CHECK(cudaMemcpy(dAllTransformIds + offset,
                              accOptiX->GetDeviceTransformIdsPtr(),
                              acceleratorCount * sizeof(TransformId),
                              cudaMemcpyDeviceToDevice));
        offset += acceleratorCount;

        // Push back the counts currently we will do a prefix sum after
        const auto& localSBTCounts = accOptiX->GetSBTCounts();
        hGlobalSBTCounts.insert(hGlobalSBTCounts.end(),
                                localSBTCounts.cbegin(),
                                localSBTCounts.cend());

        const auto& accRecords = accOptiX->GetRecords();
        hRecords.insert(hRecords.end(),
                        accRecords.cbegin(),
                        accRecords.cend());
        hProgramGroupIndices.insert(hProgramGroupIndices.end(),
                                    accRecords.size(),
                                    programIndex);
        programIndex++;

        const auto& pGroup = acc->PrimitiveGroup();
        std::string primType = pGroup.Type();
        HitFunctionNames names = {};
        std::get<CH_INDEX>(names) = CHIT_FUNC_PREFIX + primType;
        std::get<AH_INDEX>(names) = AHIT_FUNC_PREFIX + primType;
        if(!pGroup.IsTriangle())
            std::get<INTS_INDEX>(names) = INTERSECT_FUNC_PREFIX + primType;
        hfNames.push_back(names);

        // Get the aabb
        std::for_each(acc->AcceleratorAABBs().cbegin(),
                      acc->AcceleratorAABBs().cend(),
                      [&](const std::pair<uint32_t, AABB3f>& aabb)
                      {
                          sceneAABB.UnionSelf(aabb.second);
                      });
    }
    assert(offset == totalAcceleratorCount);

    // Set OptiX system for base accelerator as well
    auto& baseAccOptiX = dynamic_cast<GPUBaseAcceleratorOptiX&>(baseAccelerator);
    baseAccOptiX.SetOptiXSystem(&optixSystem);

    //Generate global SBT offsets from local SBT offsets
    // cpp reference says this can run in-place
    // (d_first can be equal to first)
    // Clang does not like this and fails so i do it over
    // an temp buffer instead (probably a bug on either MSVC or Clang)
    std::vector<uint32_t> hGlobalSBTOffsets(hGlobalSBTCounts.size());
    std::exclusive_scan(std::execution::par_unseq,
                        hGlobalSBTCounts.cbegin(),
                        hGlobalSBTCounts.cend(),
                        hGlobalSBTOffsets.begin(), 0u);
    hGlobalSBTCounts.clear();

    // Construct Base accelerator using the fetched data
    if((e = baseAccOptiX.Construct(traversables, hPrimTransformTypes,
                                   hGlobalSBTOffsets, hCullFlags,
                                   dAllTransformIds, dTransforms,
                                   sceneAABB)) != TracerError::OK)
        return e;

    // Get the traversable handles
    int optixDeviceIndex = 0;
    for(auto& optixData : optixGPUData)
    {
        optixData.baseAccelerator = baseAccOptiX.GetBaseTraversable(optixDeviceIndex);
    }

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

        #if OPTIX_VERSION > 70300
            moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        #else
            moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        #endif
        pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }
    pipelineCompileOpts.usesMotionBlur = false;
    pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOpts.numPayloadValues = 0;
    pipelineCompileOpts.numAttributeValues = (maxHitSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "params";

    if((e = CreateModules(moduleCompileOpts, pipelineCompileOpts,
                          MODULE_BASE_NAME)) != TracerError::OK)
        return e;

    // =============================== //
    //    PROGRAM GROUP GENERATION     //
    // =============================== //
    if((e = CreateProgramGroups(RAYGEN_FUNC_NAME, MISS_FUNC_NAME,
                                hfNames)) != TracerError::OK)
        return e;

    // =============================== //
    //         SBT GENERATION          //
    // =============================== //
    if((e = CreateSBTs(hRecords, hProgramGroupIndices)) != TracerError::OK)
        return e;

    // =============================== //
    //      PIPELINE GENERATION        //
    // =============================== //
    OptixPipelineLinkOptions pipelineLinkOpts = {};
    pipelineLinkOpts.maxTraceDepth = 1;
    pipelineLinkOpts.debugLevel = moduleCompileOpts.debugLevel;
    if((e = CreatePipelines(pipelineCompileOpts,
                            pipelineLinkOpts)) != TracerError::OK)
        return e;

    if((e = AllocateParams()) != TracerError::OK)
        return e;

    // All Done!
    return TracerError::OK;
}

void RayCasterOptiX::HitRays()
{
    // Reset Hit Memory
    rayMemory.ResetHitMemory(boundaryTransformIndex, currentRayCount, maxHitSize);
    // Ray Memory Pointers
    RayGMem* dRays = rayMemory.Rays();
    HitKey* dWorkKeys = rayMemory.WorkKeys();
    TransformId* dTransfomIds = rayMemory.TransformIds();
    PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
    HitStructPtr dHitStructs = rayMemory.HitStructs();

    // Don't bother launching rays if no accel is present on the scene
    // (scene is empty)
    if(accelBatches.size() != 0)
    {
        // =========================//
        //       OPTIX LAUNCH       //
        // =========================//
        size_t optixGPUCount = optixSystem.OptixCapableDevices().size();
        size_t rayPerGPU = (currentRayCount + optixGPUCount - 1) / optixGPUCount;
        // Segment the system
        size_t partitionedRayCount = 0;
        std::vector<ArrayPortion<uint32_t>> portions;
        portions.reserve(optixGPUCount);
        for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
        {
            ArrayPortion<uint32_t> portion;
            portion.offset = partitionedRayCount;
            size_t gpuRayCount = std::min(rayPerGPU,
                                          currentRayCount - partitionedRayCount);

            portion.count = gpuRayCount;
            partitionedRayCount += gpuRayCount;
            portion.portionId = 0;
            portions.push_back(portion);
        }

        // Split the and launch
        uint32_t optixDeviceIndex = 0;
        for(const auto& [gpu, optixContext] : optixSystem.OptixCapableDevices())
        {
            CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
            const auto& portion = portions[optixDeviceIndex];
            const auto& launchData = optixGPUData[optixDeviceIndex];
            const auto paramsPtr = AsOptixPtr(launchData.dOptixLaunchParams);
            auto optixTraverseHandle = optixGPUData[optixDeviceIndex].baseAccelerator;
            auto& hParams = optixGPUData[optixDeviceIndex].hOptixLaunchParams;
            size_t offset = portion.offset;

            hParams = {};
            hParams.gHitStructs = dHitStructs.AdvancedPtr(static_cast<uint32_t>(offset));
            hParams.gPrimitiveIds = dPrimitiveIds + offset;
            hParams.gRays = dRays + offset;
            hParams.gTransformIds = dTransfomIds + offset;
            hParams.gWorkKeys = dWorkKeys + offset;
            hParams.baseAcceleratorOptix = optixTraverseHandle;
            hParams.gGlobalTransformArray = dGlobalTransformArray;

            // Copy portioned pointer to Constant Memory
            CUDA_CHECK(cudaMemcpyAsync(launchData.dOptixLaunchParams,
                                       &hParams,
                                       sizeof(OpitXBaseAccelParams),
                                       cudaMemcpyHostToDevice,
                                       (cudaStream_t)0));

            OPTIX_CHECK(optixLaunch(launchData.pipeline, (cudaStream_t)0,
                                    paramsPtr, sizeof(OpitXBaseAccelParams),
                                    &launchData.sbt,
                                    static_cast<uint32_t>(portion.count),
                                    1, 1));
            CUDA_KERNEL_CHECK();
        }
        // Issue a global sync after optiX calls
        cudaSystem.SyncAllGPUs();
        // =========================//
        //     OPTIX LAUNCH END     //
        // =========================//
    }
    // After launch all rays found out their hit location
    // as primitive id, work key, hit info (barycentrics) etc.
    // All Done!
}

//void RayCasterOptiX::WorkRays(const WorkBatchMap& workMap,
//                              const RayPartitionsMulti<uint32_t>& outPortions,
//                              const RayPartitions<uint32_t>& inPartitions,
//                              RNGeneratorCPUI& rngCPU,
//                              uint32_t totalRayOut,
//                              HitKey baseBoundMatKey)
//{
//    // Sort and Partition happens on leader device
//    CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));
//
//    // Ray Memory Pointers
//    const RayGMem* dRays = rayMemory.Rays();
//    const HitStructPtr dHitStructs = rayMemory.HitStructs();
//    const PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
//    const TransformId* dTransformIds = rayMemory.TransformIds();
//    // These are sorted etc.
//    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
//    RayId* dCurrentRayIds = rayMemory.CurrentIds();
//
//    // Allocate output ray memory
//    rayMemory.ResizeRayOut(totalRayOut, baseBoundMatKey);
//    RayGMem* dRaysOut = rayMemory.RaysOut();
//    HitKey* dBoundKeyOut = rayMemory.WorkKeys();
//
//    // Reorder partitions for efficient calls
//    // (sort by gpu and order for better async access)
//    // ....
//    // TODO:
//
//    // Wait that "ResizeRayOut" is completed on the leader device
//    rayMemory.LeaderDevice().WaitMainStream();
//
//    // For each partition
//    //for(auto pIt = workPartition.crbegin();
//    //    pIt != workPartition.crend(); pIt++)
//    for(const auto& p : inPartitions)
//    {
//        //const auto& p = (*pIt);
//
//        // Skip if null batch or not found material
//        if(p.portionId == HitKey::NullBatch) continue;
//        auto loc = workMap.find(p.portionId);
//        if(loc == workMap.end()) continue;
//
//        // TODO: change this loop to combine iterator instead of find
//        //const auto& pIn = *(workPartition.find<uint32_t>(p.portionId));
//        const auto& pOut = *(outPortions.find(MultiArrayPortion<uint32_t>{p.portionId}));
//
//        // Relativize input & output pointers
//        const RayId* dRayIdStart = dCurrentRayIds + p.offset;
//        const HitKey* dKeyStart = dCurrentKeys + p.offset;
//
//        assert(pOut.counts.size() == pOut.offsets.size());
//        assert(pOut.counts.size() == loc->second.size());
//
//        // Actual Shade Calls
//        int i = 0;
//        for(auto& workBatch : loc->second)
//        {
//            // Output
//            RayGMem* dRayOutStart = dRaysOut + pOut.offsets[i];
//            HitKey* dBoundKeyStart = dBoundKeyOut + pOut.offsets[i];
//
//            workBatch->Work(dBoundKeyStart,
//                            dRayOutStart,
//                            //  Input
//                            dRays,
//                            dPrimitiveIds,
//                            dTransformIds,
//                            dHitStructs,
//                            // Ids
//                            dKeyStart,
//                            dRayIdStart,
//                            //
//                            static_cast<uint32_t>(p.count),
//                            rngCPU);
//
//            i++;
//        }
//        //cudaSystem.SyncGPUAll();
//        //METU_LOG("--------------------------");
//    }
//    currentRayCount = totalRayOut;
//
//    //METU_LOG("Before Sync");
//    // Again wait all of the GPU's since
//    // CUDA functions will be on multiple-gpus
//    cudaSystem.SyncAllGPUs();
//
//    //METU_LOG("After Sync");
//
//    //Debug::DumpMemToFile("workKeyOut", rayMemory.WorkKeys(), totalRayOut);
//    //Debug::DumpMemToFile("dPrimIdsUNsorted", dPrimitiveIds, totalRayOut);
//
//    // Shading complete
//    // Now make "RayOut" to "RayIn"
//    // and continue
//    rayMemory.SwapRays();
//}

size_t RayCasterOptiX::UsedGPUMemory() const
{
    size_t mem = RayCaster::UsedGPUMemory();

    for(const auto& optixData : optixGPUData)
    {
        mem += optixData.paramsMemory.Size();
        mem += optixData.sbtMemory.Size();
    }
    return mem;
}