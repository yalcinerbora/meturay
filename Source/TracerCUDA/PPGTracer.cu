﻿#include "PPGTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/FileUtility.h"
#include "RayLib/VisorTransform.h"

#include "PPGTracerWorks.cuh"
#include "GPULightSamplerUniform.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"
#include "GPUAcceleratorI.h"

#include "RayLib/Options.h"
#include "RayLib/TracerCallbacksI.h"

#include "TracerDebug.h"

std::ostream& operator<<(std::ostream& stream, const RayAuxPPG& v)
{
    stream << std::setw(0)
        << v.sampleIndex << ", "
        << "{" << v.radianceFactor[0]
        << "," << v.radianceFactor[1]
        << "," << v.radianceFactor[2] << "} "
        << v.endpointIndex << ", "
        << v.mediumIndex << " ";
    switch(v.type)
    {
        case RayType::CAMERA_RAY:
            stream << "CAMERA_RAY";
            break;
        case RayType::NEE_RAY:
            stream << "NEE_RAY";
            break;
        case RayType::SPECULAR_PATH_RAY:
            stream << "SPEC_PATH_RAY";
            break;
        case RayType::PATH_RAY:
            stream << "PATH_RAY";
    }
    return stream;
}

void PPGTracer::ResizeAndInitPathMemory()
{
    size_t totalPathNodeCount = TotalPathNodeCount();
    //METU_LOG("Allocating PPGTracer global path buffer: Size {:d} MiB",
    //         totalPathNodeCount * sizeof(PPGPathNode) / 1024 / 1024);

    GPUMemFuncs::EnlargeBuffer(pathMemory, totalPathNodeCount * sizeof(PPGPathNode));
    dPathNodes = static_cast<PPGPathNode*>(pathMemory);

    // Initialize Paths
    const CudaGPU& bestGPU = cudaSystem.BestGPU();
    if(totalPathNodeCount > 0)
        bestGPU.KC_X(0, 0, totalPathNodeCount,
                     //
                     KCInitializePPGPaths,
                     //
                     dPathNodes,
                     static_cast<uint32_t>(totalPathNodeCount));

    //Debug::DumpBatchedMemToFile("PathNodes", dPathNodes,
    //                            MaximumPathNodePerPath(),
    //                            totalPathNodeCount);

}

uint32_t PPGTracer::TotalPathNodeCount() const
{
    return (imgMemory.SegmentSize()[0] * imgMemory.SegmentSize()[1] *
            options.sampleCount * options.sampleCount) * MaximumPathNodePerPath();
}

uint32_t PPGTracer::MaximumPathNodePerPath() const
{
    return (options.maximumDepth == 0) ? 0 : (options.maximumDepth + 1);
}

PPGTracer::PPGTracer(const CudaSystem& s,
                      const GPUSceneI& scene,
                      const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
    , currentTreeIteration(0)
    , nextTreeSwap(1)
{
    // Append Work Types for generation
    boundaryWorkPool.AppendGenerators(PPGBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(PPGPathWorkerList{});
}

TracerError PPGTracer::Initialize()
{
    iterationCount = 0;
    treeDumpCount = 0;

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

    if(options.sdTreeInitPath.empty())
    {
        // Init sTree
        AABB3f worldAABB = scene.BaseAccelerator()->SceneExtents();
        sTree = std::make_unique<STree>(worldAABB, cudaSystem);
    }
    else
    {
        sTree = std::make_unique<STree>(options.sdTreeInitPath, cudaSystem);
    }
    return TracerError::OK;
}

TracerError PPGTracer::SetOptions(const OptionsI& opts)
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

    if((err = opts.GetUInt(options.dTreeMaxDepth, D_TREE_MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.dTreeSplitThreshold, D_TREE_FLUX_RATIO_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.sTreeSplitThreshold, S_TREE_SAMPLE_SPLIT_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.maxSDTreeSizeMB, SD_TREE_MAX_SIZE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.sdTreeInitPath, SD_TREE_PATH_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.skipPG, SKIP_PG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.purePG, PURE_PG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.bxdfPGMisRatio, MIS_RATIO_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.alwaysSendSamples, ALWAYS_SEND_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.pgDumpDebugData, PG_DUMP_DEBUG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.pgDumpInterval, PG_DUMP_INTERVAL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.pgDumpDebugName, PG_DUMP_PATH_NAME)) != TracerError::OK)
        return err;

    return TracerError::OK;
}

bool PPGTracer::Render()
{
    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(rayCaster->CurrentRayCount() == 0 ||
       currentDepth >= options.maximumDepth)
        return false;

    rayCaster->HitRays();
    const auto partitions = rayCaster->PartitionRaysWRTWork();

    //Debug::DumpMemToFile("auxIn",
    //                     static_cast<const RayAuxPPG*>(*dAuxIn),
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
    PPGTracerGlobalState globalData;
    globalData.gSamples = sampleMemory.GMem<Vector4f>();
    globalData.gLightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.gLightSampler = dLightSampler;
    //
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    //
    // Set SD Tree
    const STreeGPU* dSTree;
    const DTreeGPU* dReadDTrees;
    DTreeGPU* dWriteDTrees;
    sTree->TreeGPU(dSTree, dReadDTrees, dWriteDTrees);
    globalData.gStree = dSTree;
    globalData.gReadDTrees = dReadDTrees;
    globalData.gWriteDTrees = dWriteDTrees;
    //
    globalData.gPathNodes = dPathNodes;
    globalData.maximumPathNodePerRay = MaximumPathNodePerPath();

    globalData.skipPG = options.skipPG;
    globalData.purePG = options.purePG;
    globalData.bxdfPGMisRatio = options.bxdfPGMisRatio;
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
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPPG);
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
        const RayAuxPPG* dAuxInLocal = static_cast<const RayAuxPPG*>(*dAuxIn);
        using WorkData = GPUWorkBatchD<PPGTracerGlobalState, RayAuxPPG>;
        int i = 0;
        for(auto& work : loc->second)
        {
            RayAuxPPG* dAuxOutLocal = static_cast<RayAuxPPG*>(*dAuxOut) + p.offsets[i];

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

    //Debug::DumpMemToFile("auxOut",
    //                     static_cast<const RayAuxPPG*>(*dAuxOut),
    //                     totalOutRayCount);
    //// Work rays swapped the ray buffer so read input rays
    //Debug::DumpMemToFile("rayOut", rayMemory.Rays(),
    //                     totalOutRayCount);
    //Debug::DumpMemToFile("rayIdOut", rayMemory.CurrentIds(),
    //                     totalOutRayCount);

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();

    currentDepth++;
    //Debug::DumpBatchedMemToFile("PathNodes", dPathNodes,
    //                            MaximumPathNodePerPath(),
    //                            TotalPathNodeCount());

    //
    //METU_LOG("PASS ENDED=============================================================");
    return true;
}

void PPGTracer::Finalize()
{
    cudaSystem.SyncAllGPUs();

    // Iteration count is used to when dump the entire svo
    // to the disk etc.
    iterationCount++;

    uint32_t totalPathNodeCount = TotalPathNodeCount();

    //Debug::DumpBatchedMemToFile(std::to_string(currentTreeIteration) + "PathNodes",
    //                            dPathNodes,
    //                            MaximumPathNodePerPath(), totalPathNodeCount);

    // Accumulate the finished radiances to the STree
    sTree->AccumulateRaidances(dPathNodes, totalPathNodeCount,
                               MaximumPathNodePerPath(), cudaSystem);
    // We iterated once
    currentTreeIteration += 1;
    if(currentTreeIteration == nextTreeSwap)
    {
        // Double the amount of iterations required for this
        nextTreeSwap <<= 1;

        uint32_t treeSwapIterationCount = Utility::FindLastSet(nextTreeSwap) - 1;

        uint64_t sTreeSplit64 = static_cast<uint64_t>(std::sqrt(std::pow(2.0f, treeSwapIterationCount)));
        sTreeSplit64 *= options.sTreeSplitThreshold;
        // Limit to the 32-bit upper bound
        sTreeSplit64 = std::min<uint64_t>(sTreeSplit64, std::numeric_limits<uint32_t>::max());
        uint32_t currentSTreeSplitThreshold = static_cast<uint32_t>(sTreeSplit64);
        // Split and Swap the trees
        sTree->SplitAndSwapTrees(currentSTreeSplitThreshold,
                                 options.dTreeSplitThreshold,
                                 options.dTreeMaxDepth,
                                 cudaSystem);

        size_t mbSize = sTree->UsedGPUMemory() / 1024 / 1024;
        METU_LOG("I: {:d} S: {:d}, Splitting and Swapping => Split: {:d}, Trees Size: {:d} MiB, Trees: {:d}",
                 treeSwapIterationCount,
                 currentTreeIteration,
                 currentSTreeSplitThreshold,
                 mbSize,
                 sTree->TotalTreeCount());
        // Completely Reset the Image
        // This is done to eliminate variance from prev samples
        ResetImage();
    }

    // Debug Dump
    uint32_t dumpInterval = static_cast<uint32_t>(std::pow(options.pgDumpInterval, treeDumpCount));
    if(options.pgDumpDebugData && iterationCount == dumpInterval)
    {
        // Write SD Tree File
        std::vector<Byte> sdTree;
        sTree->DumpSDTreeAsBinary(sdTree, true);
        std::string iterAsString = std::to_string(nextTreeSwap >> 1);
        std::string name = iterAsString + "_ppg_sdTree";
        Utility::DumpStdVectorToFile(sdTree, name);
        METU_LOG("Dumping {:s}", name);
        treeDumpCount++;
    }

    cudaSystem.SyncAllGPUs();
    frameTimer.Split();
    UpdateFrameAnalytics("paths / sec", options.sampleCount * options.sampleCount);

    uint32_t prevTreeSwap = (nextTreeSwap >> 1);
    if(options.alwaysSendSamples ||
       // Do not send samples until we exceed prev iteration samples
       (currentTreeIteration - prevTreeSwap) >= prevTreeSwap)
    {
        // Base class finalize directly sends the image
        RayTracer::Finalize();
    }
}

void PPGTracer::GenerateWork(uint32_t cameraIndex)
{
    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    GenerateRays<RayAuxPPG, RayAuxInitPPG, RNGIndependentGPU, Vector4f>
    (
        cameraIndex,
        options.sampleCount,
        RayAuxInitPPG(InitialPPGAux,
                        options.sampleCount *
                        options.sampleCount),
        true
    );

    ResizeAndInitPathMemory();
    currentDepth = 0;
}

void PPGTracer::GenerateWork(const VisorTransform& t, uint32_t cameraIndex)
{
    GenerateRays<RayAuxPPG, RayAuxInitPPG, RNGIndependentGPU, Vector4f>
    (
        t, cameraIndex, options.sampleCount,
        RayAuxInitPPG(InitialPPGAux,
                        options.sampleCount *
                        options.sampleCount),
        true
    );
    ResizeAndInitPathMemory();
    currentDepth = 0;
}

void PPGTracer::GenerateWork(const GPUCameraI& dCam)
{
    GenerateRays<RayAuxPPG, RayAuxInitPPG, RNGIndependentGPU, Vector4f>
    (
        dCam, options.sampleCount,
        RayAuxInitPPG(InitialPPGAux,
                        options.sampleCount *
                        options.sampleCount),
        true
    );
    ResizeAndInitPathMemory();
    currentDepth = 0;
}

size_t PPGTracer::TotalGPUMemoryUsed() const
{
    return (RayTracer::TotalGPUMemoryUsed() +
            sTree->UsedGPUMemory() +
            lightSamplerMemory.Size() + pathMemory.Size());
}

void PPGTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(MAX_DEPTH_NAME, OptionVariable(static_cast<int64_t>(options.maximumDepth)));
    list.emplace(SAMPLE_NAME, OptionVariable(static_cast<int64_t>(options.sampleCount)));
    list.emplace(RR_START_NAME, OptionVariable(static_cast<int64_t>(options.rrStart)));
    list.emplace(LIGHT_SAMPLER_TYPE_NAME, OptionVariable(LightSamplerCommon::LightSamplerTypeToString(options.lightSamplerType)));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));
    list.emplace(DIRECT_LIGHT_MIS_NAME, OptionVariable(options.directLightMIS));

    list.emplace(D_TREE_MAX_DEPTH_NAME, OptionVariable(static_cast<int64_t>(options.dTreeMaxDepth)));
    list.emplace(D_TREE_FLUX_RATIO_NAME, OptionVariable(options.dTreeSplitThreshold));
    list.emplace(S_TREE_SAMPLE_SPLIT_NAME, OptionVariable(static_cast<int64_t>(options.sTreeSplitThreshold)));
    list.emplace(SD_TREE_MAX_SIZE_NAME, OptionVariable(static_cast<int64_t>(options.maxSDTreeSizeMB)));

    list.emplace(ALWAYS_SEND_NAME, OptionVariable(options.alwaysSendSamples));
    list.emplace(SKIP_PG_NAME, OptionVariable(options.skipPG));
    list.emplace(PURE_PG_NAME, OptionVariable(options.purePG));
    list.emplace(MIS_RATIO_NAME, OptionVariable(options.bxdfPGMisRatio));

    list.emplace(PG_DUMP_DEBUG_NAME, OptionVariable(options.pgDumpDebugData));
    list.emplace(PG_DUMP_INTERVAL_NAME, OptionVariable(static_cast<int64_t>(options.pgDumpInterval)));
    list.emplace(PG_DUMP_PATH_NAME, OptionVariable(options.pgDumpDebugName));

    if(callbacks) callbacks->SendCurrentOptions(::Options(std::move(list)));
}