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

// Currently These are compile time constants
// sine most of the internal call rely on compile time constants
static constexpr uint32_t PG_KERNEL_TPB = 512;
static constexpr uint32_t PG_KERNEL_X = 32;
static constexpr uint32_t PG_KERNEL_Y = 32;

struct NodeIdFetchFunctor
{
    __device__ inline
    uint32_t operator()(const RayAuxWFPG& aux) const
    {
        return aux.binId;
    }
};

void WFPGTracer::ResizeAndInitPathMemory()
{
    size_t totalPathNodeCount = TotalPathNodeCount();
    //METU_LOG("Allocating WFPGTracer global path buffer: Size {:d} MiB",
    //         totalPathNodeCount * sizeof(PathGuidingNode) / 1024 / 1024);

    GPUMemFuncs::EnlargeBuffer(pathMemory, totalPathNodeCount * sizeof(PathGuidingNode));
    dPathNodes = static_cast<PathGuidingNode*>(pathMemory);

    // Initialize Paths
    const CudaGPU& bestGPU = cudaSystem.BestGPU();
    if(totalPathNodeCount > 0)
        bestGPU.KC_X(0, 0, totalPathNodeCount,
                     //
                     KCInitializePGPaths,
                     //
                     dPathNodes,
                     static_cast<uint32_t>(totalPathNodeCount));

    //Debug::DumpBatchedMemToFile("PathNodes", dPathNodes,
    //                            MaximumPathNodePerPath(),
    //                            totalPathNodeCount);

}

uint32_t WFPGTracer::TotalPathNodeCount() const
{
    return (imgMemory.SegmentSize()[0] * imgMemory.SegmentSize()[1] *
            options.sampleCount * options.sampleCount) * MaximumPathNodePerPath();
}

uint32_t WFPGTracer::MaximumPathNodePerPath() const
{
    return (options.maximumDepth == 0) ? 0 : (options.maximumDepth + 1);
}

void WFPGTracer::GenerateGuidedDirections()
{
    const CudaGPU& gpu = cudaSystem.BestGPU();
    // Cluster the rays according to their svo location
    const RayGMem* dRays = rayCaster->RaysIn();
    RayAuxWFPG* dRayAux = static_cast<RayAuxWFPG*>(*dAuxIn);
    uint32_t rayCount = rayCaster->CurrentRayCount();

    // Zero out the ray counts from the previous iteration
    svo.ClearRayCounts(cudaSystem);

    // Init ray bins
    gpu.GridStrideKC_X(0, (cudaStream_t)0, rayCount,
                       //
                       KCInitializeSVOBins,
                       //
                       dRayAux,
                       dRays,
                       rayCaster->WorkKeys(),
                       scene.BaseBoundaryMaterial(),
                       svo.TreeGPU(),
                       rayCount);

    // Then call SVO to reduce the bins
    svo.CollapseRayCounts(options.minRayBinLevel,
                          options.binRayCount,
                          cudaSystem);

    // Then rays check if their initial node is reduced
    gpu.GridStrideKC_X(0, (cudaStream_t)0, rayCount,
                       //
                       KCCheckReducedSVOBins,
                       //
                       dRayAux,
                       svo.TreeGPU(),
                       rayCount);

    // Partition the generated rays wrt. to the SVO nodeId
    uint32_t hPartitionCount;
    uint32_t* dPartitionOffsets;
    uint32_t* dPartitionBinIds;
    DeviceMemory partitionMemory;
    // Custom Ray Partition
    rayCaster->PartitionRaysWRTCustomData(hPartitionCount,
                                          partitionMemory,
                                          dPartitionOffsets,
                                          dPartitionBinIds,
                                          dRayAux,
                                          NodeIdFetchFunctor(),
                                          rayCount,
                                          cudaSystem);

    // Call the Trace and Sample Kernel
    constexpr auto KCSampleKernel = KCGenAndSampleDistribution<RNGIndependentGPU,
                                                               PG_KERNEL_TPB,
                                                               PG_KERNEL_X,
                                                               PG_KERNEL_Y>;
    RNGeneratorGPUI** gpuGenerators = pgSampleRNG.GetGPUGenerators(gpu);

    auto data = gpu.GetKernelAttributes(KCSampleKernel);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gpu.ExactKC_X(0, (cudaStream_t)0,
                  PG_KERNEL_TPB, pgKernelBlockCount,
                  //
                  KCSampleKernel,
                  // Output
                  dRayAux,
                  // I-O
                  gpuGenerators,
                  // Input
                  // Per-ray
                  dRays,
                  rayCaster->RayIds(),
                  // Per bin
                  dPartitionOffsets,
                  dPartitionBinIds,
                  // Constants
                  svo.TreeGPU(),
                  hPartitionCount);
    CUDA_CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    METU_LOG("Depth {:d} -> PartitionCount {:d} KernelTime {:f}ms",
             currentDepth, hPartitionCount, milliseconds);
}

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

    // Generate Light Sampler (for NEE)
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
    if((err = svo.Constrcut(scene.BaseAccelerator()->SceneExtents(),
                            (1 << options.octreeLevel),
                            scene.AcceleratorBatchMappings(),
                            dLights, lightCount,
                            scene.BaseBoundaryMaterial(),
                            cudaSystem)) != TracerError::OK)
        return err;

    // Generate a Scrambled Sobol Sampler for the
    // Path Guide Sampling
    const auto& gpu = cudaSystem.BestGPU();

    uint32_t rngCount = (gpu.MaxActiveBlockPerSM(PG_KERNEL_TPB) *
                         gpu.SMCount() * PG_KERNEL_TPB);
    pgKernelBlockCount = rngCount / PG_KERNEL_TPB;
    pgSampleRNG = RNGIndependentCPU(params.seed, gpu, rngCount);

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
    if((err = opts.GetBool(options.voxTrace, VOX_TRACE_NAME)) != TracerError::OK)
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
    list.emplace(VOX_TRACE_NAME, OptionVariable(options.voxTrace));
    list.emplace(DEBUG_RENDER_NAME, OptionVariable(options.debugRender));
    list.emplace(DUMP_DEBUG_NAME, OptionVariable(options.dumpDebugData));
    list.emplace(DUMP_INTERVAL_NAME, OptionVariable(options.svoDumpInterval));
    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}

void WFPGTracer::GenerateWork(uint32_t cameraIndex)
{
    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
    (
        cameraIndex,
        options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        !options.debugRender
    );

    // On voxel trace mode we don't need paths
    if(!(options.debugRender && options.voxTrace))
        ResizeAndInitPathMemory();

    currentDepth = 0;
}

void WFPGTracer::GenerateWork(const VisorTransform& t, uint32_t cameraIndex)
{
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
    (
        t, cameraIndex, options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        !options.debugRender
    );
    // On voxel trace mode we don't need paths
    if(!(options.debugRender && options.voxTrace))
        ResizeAndInitPathMemory();
    currentDepth = 0;
}

void WFPGTracer::GenerateWork(const GPUCameraI& dCam)
{
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
    (
        dCam, options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        !options.debugRender
    );
    // On voxel trace mode we don't need paths
    if(!(options.debugRender && options.voxTrace))
        ResizeAndInitPathMemory();
    currentDepth = 0;
}

bool WFPGTracer::Render()
{
    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(rayCaster->CurrentRayCount() == 0 ||
       currentDepth >= options.maximumDepth)
        return false;

    // Generate Global Data Struct
    WFPGTracerGlobalState globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.gLightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.gLightSampler = dLightSampler;
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    //
    globalData.svo = svo.TreeGPU();
    globalData.gPathNodes = dPathNodes;
    globalData.maximumPathNodePerRay = MaximumPathNodePerPath();
    globalData.rawPathGuiding = false;
    //
    globalData.directLightMIS = options.directLightMIS;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;

    // On voxel trace mode we just trace the rays without any material
    if(options.debugRender && options.voxTrace)
    {
        // Just call the voxel trace kernel on a GPU and call it a day
        const auto& gpu = cudaSystem.BestGPU();
        uint32_t totalRayCount = rayCaster->CurrentRayCount();
        gpu.GridStrideKC_X(0, (cudaStream_t)0, totalRayCount,
                            //
                            KCTraceSVO,
                            //
                            globalData,
                            rayCaster->RaysIn(),
                            static_cast<RayAuxWFPG*>(*dAuxIn),
                            totalRayCount);
        // Signal as if we finished processing
        return false;
    }

    // Hit Rays
    rayCaster->HitRays();

    // Before Material Evaluation
    // Generate guideDirection and PDF
    GenerateGuidedDirections();

    // Generate output partitions wrt. materials
    const auto partitions = rayCaster->PartitionRaysWRTWork();

    uint32_t totalOutRayCount = 0;
    auto outPartitions = RayCasterI::PartitionOutputRays(totalOutRayCount,
                                                         partitions,
                                                         workMap);
    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxWFPG);
    GPUMemFuncs::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    for(auto p : outPartitions)
    {
        // Skip if null batch or not found material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers
        const RayAuxWFPG* dAuxInLocal = static_cast<const RayAuxWFPG*>(*dAuxIn);
        using WorkData = GPUWorkBatchD<WFPGTracerGlobalState, RayAuxWFPG>;
        int i = 0;
        for(auto& work : loc->second)
        {
            RayAuxWFPG* dAuxOutLocal = static_cast<RayAuxWFPG*>(*dAuxOut) + p.offsets[i];

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

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    // Increase Depth
    currentDepth++;
    return true;
}

void WFPGTracer::Finalize()
{
    METU_LOG("----------------");
    cudaSystem.SyncAllGPUs();
    frameTimer.Stop();
    UpdateFrameAnalytics("paths / sec", options.sampleCount * options.sampleCount);
    GPUTracer::Finalize();
}

size_t WFPGTracer::TotalGPUMemoryUsed() const
{
    return (RayTracer::TotalGPUMemoryUsed() +
            svo.UsedGPUMemory() +
            lightSamplerMemory.Size() + pathMemory.Size());
}