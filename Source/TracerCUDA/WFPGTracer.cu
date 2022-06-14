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
#include "ParallelReduction.cuh"

#include <array>


template <class RNG>
__global__
void GeneratePhotons(// Output
                     RayGMem* gRays,
                     RayAuxPhotonWFPG* gAuxiliary,
                     // I-O
                     RNGeneratorGPUI** gRNGs,
                     // Input
                     const GPULightI** gLightList,
                     uint32_t totalLightCount,
                     // Constants
                     uint32_t totalPhotonCount)
{
    // Get this Threads RNG
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalPhotonCount;
        threadId += (blockDim.x * gridDim.x))
    {
        // TODO: generate power CDF and use it instead
        // Uniformly Sample Light
        uint32_t index = static_cast<uint32_t>(rng.Uniform() * totalLightCount);
        float pdf = 1.0f / static_cast<float>(totalLightCount);

        float posPDF = 0.0f;
        float dirPDF = 0.0f;
        RayReg ray; Vector3f normal;
        Vector3f power = gLightList[index]->GeneratePhoton(ray, normal,
                                                           posPDF, dirPDF,
                                                           rng);
        uint16_t mediumIndex = gLightList[index]->MediumIndex();

        // Divide by the sampling pdf
        if(posPDF != 0 && dirPDF != 0 && pdf != 0)
            power /= (posPDF * dirPDF * pdf);
        else
            power = Zero3f;

        RayAuxPhotonWFPG aux;
        aux.power = power;
        aux.depth = 1;
        aux.mediumIndex = mediumIndex;

        // Store
        ray.Update(gRays, threadId);
        gAuxiliary[threadId] = aux;
    }
}

__global__
void KCNormalizeImage(// I-O
                      Vector4f* dPixels,
                      //
                      Vector4f& dMax,
                      Vector4f& dMin,
                      //
                      Vector2i imgRes)
{
    int totalPixSize = imgRes.Multiply();

    Vector4f rangeRecip = Vector4f(1.0f) / (dMax - dMin);

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalPixSize;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector4f pix = dPixels[threadId];
        pix = (pix - dMin) * rangeRecip;

        // Do not change the alpha
        pix[3] = dMax[3];

        dPixels[threadId] = pix;
    }
}

__global__
void KCSetCamPosToPathChain(// Output
                            PathGuidingNode* gPathNodes,
                            // Input
                            const RayGMem* gRays,
                            // Constants
                            uint32_t maxPathNodePerRay,
                            uint32_t totalNodeCount,
                            uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        RayReg ray = RayReg(gRays, threadId);

        const uint32_t pathStartIndex = threadId * maxPathNodePerRay;
        gPathNodes[pathStartIndex].worldPosition = ray.ray.getPosition();
    }
}

// Currently These are compile time constants
// since most of the internal call rely on compile time constants
static constexpr uint32_t PG_KERNEL_TYPE_COUNT = 5;

using PathGuideKernelFunction = void (*)(// Output
                                         RayAuxWFPG*,
                                         // I-O
                                         RNGeneratorGPUI**,
                                         // Input
                                         // Per-ray
                                         const RayGMem*,
                                         const RayId*,
                                         // Per bin
                                         const uint32_t*,
                                         const uint32_t*,
                                         // Constants
                                         const AnisoSVOctreeGPU,
                                         uint32_t);

static constexpr std::array<uint32_t, PG_KERNEL_TYPE_COUNT> PG_KERNEL_TPB =
{
    512,
    512,
    512,
    256,
    256,
};

static constexpr uint32_t KERNEL_TBP_MAX = *std::max_element(PG_KERNEL_TPB.cbegin(),
                                                             PG_KERNEL_TPB.cend());

static constexpr std::array<PathGuideKernelFunction, PG_KERNEL_TYPE_COUNT> PG_KERNELS =
{
    KCGenAndSampleDistribution<RNGIndependentGPU, PG_KERNEL_TPB[0], 64, 64>,    // First bounce good approximation
    KCGenAndSampleDistribution<RNGIndependentGPU, PG_KERNEL_TPB[1], 64, 32>,    // Second bounce as well
    KCGenAndSampleDistribution<RNGIndependentGPU, PG_KERNEL_TPB[2], 32, 16>,    // Third bounce not so much
    KCGenAndSampleDistribution<RNGIndependentGPU, PG_KERNEL_TPB[3], 16, 8>,     // Fourth bounce bad
    KCGenAndSampleDistribution<RNGIndependentGPU, PG_KERNEL_TPB[4], 8, 4>       // Fifth is bad as well
};

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

    // Allocate the initial camera position in path chain
    // Path chain does not store direction in order to calculate Wo
    // wee need it
    const RayGMem* dRays = rayCaster->RaysIn();
    uint32_t rayCount = rayCaster->CurrentRayCount();
    bestGPU.GridStrideKC_X(0, (cudaStream_t)0, rayCount,
                           //
                           KCSetCamPosToPathChain,
                           //
                           // Output
                           dPathNodes,
                           // Input
                           dRays,
                           MaximumPathNodePerPath(),
                           static_cast<uint32_t>(totalPathNodeCount),
                           rayCount);

    //Debug::DumpBatchedMemToFile("__PathNodes", dPathNodes,
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
// Select the kernel depending on the depth
uint32_t kernelIndex = std::min(currentDepth, PG_KERNEL_TYPE_COUNT - 1);
auto KCSampleKernel = PG_KERNELS[kernelIndex];
RNGeneratorGPUI** gpuGenerators = pgSampleRNG.GetGPUGenerators(gpu);

auto data = gpu.GetKernelAttributes(reinterpret_cast<const void*>(KCSampleKernel));

cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start));
gpu.ExactKC_X(0, (cudaStream_t)0,
              PG_KERNEL_TPB[kernelIndex], pgKernelBlockCount,
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
float avgRayPerBin = static_cast<float>(rayCount) /
static_cast<float>(pgKernelBlockCount);

CUDA_CHECK(cudaEventSynchronize(stop));
CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
METU_LOG("Depth {:d} -> PartitionCount {:d}, AvgRayPerBin {:f}, KernelTime {:f}ms",
         currentDepth, hPartitionCount, avgRayPerBin, milliseconds);
}

void WFPGTracer::TraceAndStorePhotons()
{
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
    globalData.sketchGPU = sketch.SketchGPU();
    //
    globalData.directLightMIS = options.directLightMIS;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;

    // Generate Photons
    const CudaGPU& gpu = cudaSystem.BestGPU();
    // Allocate Ray & Aux Memory
    size_t auxBufferSize = options.photonPerPass * sizeof(RayAuxPhotonWFPG);
    rayCaster->ResizeRayOut(options.photonPerPass, scene.BaseBoundaryMaterial());
    GPUMemFuncs::EnlargeBuffer(*dAuxOut, auxBufferSize);
    RayAuxPhotonWFPG* dRayAux = static_cast<RayAuxPhotonWFPG*>(*dAuxOut);

    // Photon Generation Kernel Call
    gpu.GridStrideKC_X(0, (cudaStream_t)0,
                       options.photonPerPass,
                       //
                       GeneratePhotons<RNGIndependentGPU>,
                       // Output
                       rayCaster->RaysOut(),
                       dRayAux,
                       // I-O
                       rngCPU->GetGPUGenerators(gpu),
                       // Input
                       dLights,
                       lightCount,
                       // Constants
                       options.photonPerPass);

    // Swap Auxiliary buffers and start photon tracing
    SwapAuxBuffers();
    rayCaster->SwapRays();

    // Photon Bounce Loop
    uint32_t depth = 0;
    uint32_t rayCount = rayCaster->CurrentRayCount();
    while(rayCaster->CurrentRayCount() != 0 &&
          depth < options.maximumDepth)
    {
        // Hit Rays
        rayCaster->HitRays();

         // Generate output partitions wrt. materials
        const auto partitions = rayCaster->PartitionRaysWRTWork();

        uint32_t totalOutRayCount = 0;
        auto outPartitions = RayCasterI::PartitionOutputRays(totalOutRayCount,
                                                             partitions,
                                                             photonWorkMap);
        // Allocate new auxiliary buffer
        // to fit all potential ray outputs
        size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPhotonWFPG);
        GPUMemFuncs::EnlargeBuffer(*dAuxOut, auxOutSize);

        // Set Auxiliary Pointers
        for(auto p : outPartitions)
        {
            // Skip if null batch or not found material
            if(p.portionId == HitKey::NullBatch) continue;
            auto loc = photonWorkMap.find(p.portionId);
            if(loc == photonWorkMap.end()) continue;

            // Set pointers
            const RayAuxPhotonWFPG* dAuxInLocal = static_cast<const RayAuxPhotonWFPG*>(*dAuxIn);
            using WorkData = GPUWorkBatchD<WFPGTracerGlobalState, RayAuxPhotonWFPG>;
            int i = 0;
            for(auto& work : loc->second)
            {
                RayAuxPhotonWFPG* dAuxOutLocal = static_cast<RayAuxPhotonWFPG*>(*dAuxOut) + p.offsets[i];

                auto& wData = static_cast<WorkData&>(*work);
                wData.SetGlobalData(globalData);
                wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);
                i++;
            }
        }

        // Launch Kernels
        rayCaster->WorkRays(photonWorkMap, outPartitions,
                            partitions,
                            *rngCPU.get(),
                            totalOutRayCount,
                            scene.BaseBoundaryMaterial());

        // Swap auxiliary buffers since output rays are now input rays
        // for the next iteration
        SwapAuxBuffers();
        // Increase Depth
        depth++;
    }

    // Now Photons are stored in the global sketch
}

WFPGTracer::WFPGTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
    , sketch(512, 8192 * 8, 0.0004f, 0)
{
    // Append Work Types for generation
    boundaryWorkPool.AppendGenerators(WFPGBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(WFPGPathWorkerList{});

    // Photon mapping pool
    photonWorkPool.AppendGenerators(WFPGPhotonWorkerList{});

    debugBoundaryWorkPool.AppendGenerators(WFPGDebugBoundaryWorkerList{});
    debugPathWorkPool.AppendGenerators(WFPGDebugPathWorkerList{});
}

TracerError WFPGTracer::Initialize()
{
    iterationCount = 0;
    treeDumpCount = 0;

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
        WorkBatchArray workPhotonBatchList;
        uint32_t batchId = std::get<0>(wInfo);
        const GPUPrimitiveGroupI& pg = *std::get<1>(wInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(wInfo);

        // Generic Path work
        GPUWorkBatchI* batch = nullptr;
        GPUWorkBatchI* photonBatch = nullptr;
        if(options.renderMode == WFPGRenderMode::SVO_INITIAL_HIT_QUERY)
        {
            WorkPool<>& wp = debugPathWorkPool;
            if((err = wp.GenerateWorkBatch(batch, mg, pg,
                                            dTransforms)) != TracerError::OK)
                return err;
        }
        else
        {
            WorkPool<bool, bool>& wpCombo = pathWorkPool;
            if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                                dTransforms,
                                                options.nextEventEstimation,
                                                options.directLightMIS)) != TracerError::OK)
                return err;

            // Generate Photon Work as well
            if((err = photonWorkPool.GenerateWorkBatch(photonBatch, mg, pg,
                                                       dTransforms)) != TracerError::OK)
                return err;
        }

        workPhotonBatchList.push_back(photonBatch);
        photonWorkMap.emplace(batchId, workPhotonBatchList);

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
        if(options.renderMode == WFPGRenderMode::SVO_INITIAL_HIT_QUERY)
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

    // Set Range for Sketch
    sketch.SetSceneExtent(scene.BaseAccelerator()->SceneExtents());

    // Init SVO
    if((err = svo.Constrcut(scene.BaseAccelerator()->SceneExtents(),
                            (1 << options.octreeLevel),
                            scene.AcceleratorBatchMappings(),
                            dLights, lightCount,
                            scene.BaseBoundaryMaterial(),
                            cudaSystem)) != TracerError::OK)
        return err;

    // Generate a Sampler for the
    // Path Guide Sampling (Conservatively generate maximum amount of RNGs)
    const auto& gpu = cudaSystem.BestGPU();

    uint32_t rngCount = (gpu.MaxActiveBlockPerSM(KERNEL_TBP_MAX) *
                         gpu.SMCount() * KERNEL_TBP_MAX);
    pgKernelBlockCount = rngCount / KERNEL_TBP_MAX;
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
    if((err = opts.GetUInt(options.svoDumpInterval, DUMP_INTERVAL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.dumpDebugData, DUMP_DEBUG_NAME)) != TracerError::OK)
        return err;

    std::string renderModeString;
    if((err = opts.GetString(renderModeString, RENDER_MODE_NAME)) != TracerError::OK)
        return err;
    if((err = StringToWFPGRenderMode(options.renderMode, renderModeString)) != TracerError::OK)
        return err;

    if((err = opts.GetUInt(options.svoRadRenderIter, SVO_DEBUG_ITER_NAME)) != TracerError::OK)
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
    list.emplace(RENDER_MODE_NAME, OptionVariable(WFPGRenderModeToString(options.renderMode)));
    list.emplace(DUMP_DEBUG_NAME, OptionVariable(options.dumpDebugData));
    list.emplace(DUMP_INTERVAL_NAME, OptionVariable(options.svoDumpInterval));
    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}

void WFPGTracer::GenerateWork(uint32_t cameraIndex)
{
    //// TEST !!!
    //// Equally partition the samples to all cameras
    //if(iterationCount <= options.svoRadRenderIter)
    //{
    //    uint32_t cameraCount = scene.CameraCount();
    //    int samplePerCam = options.svoRadRenderIter / cameraCount;
    //    int i = iterationCount / samplePerCam;
    //    cameraIndex += i;
    //    cameraIndex %= cameraCount;
    //}
    //// TEST END !!!

    // Before Generating Camera Rays do photon pass
    //if(iterationCount < options.svoRadRenderIter &&
    //   options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    //    TraceAndStorePhotons();

    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    bool enableAA = (options.renderMode == WFPGRenderMode::NORMAL ||
                     options.renderMode == WFPGRenderMode::SVO_RADIANCE);
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
    (
        cameraIndex,
        options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        enableAA
    );
    // Save the camera if SVO RADIANCE Mode
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    {
        currentCamera.type = CameraType::SCENE_CAMERA;
        currentCamera.nonTransformedCamIndex = cameraIndex;
    }

    // On voxel trace mode we don't need paths
    if(options.renderMode == WFPGRenderMode::NORMAL ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
        ResizeAndInitPathMemory();

    currentDepth = 0;
}

void WFPGTracer::GenerateWork(const VisorTransform& t, uint32_t cameraIndex)
{
    // Before Generating Camera Rays do photon pass
    //if(iterationCount < options.svoRadRenderIter &&
    //   options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    //    TraceAndStorePhotons();

    bool enableAA = (options.renderMode == WFPGRenderMode::NORMAL ||
                     options.renderMode == WFPGRenderMode::SVO_RADIANCE);
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
    (
        t, cameraIndex, options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        enableAA
    );
    // Save the camera if SVO RADIANCE Mode
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    {
        currentCamera.type = CameraType::TRANSFORMED_SCENE_CAMERA;
        currentCamera.transformedSceneCam.cameraIndex = cameraIndex;
        currentCamera.transformedSceneCam.transform = t;
    }

    // On voxel trace mode we don't need paths
    if(options.renderMode == WFPGRenderMode::NORMAL ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
        ResizeAndInitPathMemory();
    currentDepth = 0;
}

void WFPGTracer::GenerateWork(const GPUCameraI& dCam)
{
    // Before Generating Camera Rays do photon pass
    //if(iterationCount < options.svoRadRenderIter &&
    //   options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    //    TraceAndStorePhotons();

    bool enableAA = (options.renderMode == WFPGRenderMode::NORMAL ||
                     options.renderMode == WFPGRenderMode::SVO_RADIANCE);
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
    (
        dCam, options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        enableAA
    );
        // Save the camera if SVO RADIANCE Mode
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    {
        currentCamera.type = CameraType::CUSTOM_CAMERA;
        currentCamera.dCustomCamera = &dCam;
    }

    // On voxel trace mode we don't need paths
    if(options.renderMode == WFPGRenderMode::NORMAL ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
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

    // Don't do path tracing if debug render iteration is set
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE &&
       iterationCount >= options.svoRadRenderIter)
        return false;

    //// TODO: CHANGE
    //// Do not do path trace at all if svo radiance mode is set
    //if(options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    //    return false;

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
    globalData.sketchGPU = sketch.SketchGPU();
    //
    globalData.directLightMIS = options.directLightMIS;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;

    // On voxel trace mode we just trace the rays without any material
    if(options.renderMode == WFPGRenderMode::SVO_FALSE_COLOR)
    {
        // Just call the voxel trace kernel on a GPU and call it a day
        const auto& gpu = cudaSystem.BestGPU();
        uint32_t totalRayCount = rayCaster->CurrentRayCount();
        gpu.GridStrideKC_X(0, (cudaStream_t)0, totalRayCount,
                            //
                           KCTraceSVO,
                           //
                           imgMemory.GMem<Vector4>(),
                           svo.TreeGPU(),
                           rayCaster->RaysIn(),
                           static_cast<RayAuxWFPG*>(*dAuxIn),
                           WFPGRenderMode::SVO_FALSE_COLOR,
                           totalRayCount);
        // Signal as if we finished processing
        return false;
    }

    // Hit Rays
    rayCaster->HitRays();

    // Before Material Evaluation
    // Generate guideDirection and PDF
    if(options.renderMode != WFPGRenderMode::SVO_INITIAL_HIT_QUERY)
    // Change this
    if(options.renderMode != WFPGRenderMode::SVO_RADIANCE)
    {
        GenerateGuidedDirections();
    }

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
    // Iteration count is used to when dump the entire svo
    // to the disk etc.
    iterationCount++;

    //Debug::DumpBatchedMemToFile("PathNodes", dPathNodes,
    //                            MaximumPathNodePerPath(),
    //                            TotalPathNodeCount());

    // Deposit the radiances on the path chains
    if(options.renderMode == WFPGRenderMode::NORMAL ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    {
        uint32_t totalPathNodeCount = TotalPathNodeCount();
        svo.AccumulateRaidances(dPathNodes, totalPathNodeCount,
                                MaximumPathNodePerPath(), cudaSystem);
        svo.NormalizeAndFilterRadiance(cudaSystem);

        // Dump the SVO tree if requested
        uint32_t dumpInterval = static_cast<uint32_t>(std::pow(options.svoDumpInterval, treeDumpCount));
        if(options.dumpDebugData && iterationCount == dumpInterval)
        {
            std::vector<Byte> svoData;
            svo.DumpSVOAsBinary(svoData, cudaSystem);
            std::string fName = fmt::format("{:d}_svoTree", iterationCount);
            Utility::DumpStdVectorToFile(svoData, fName);
            treeDumpCount++;
        }

        //if(iterationCount <= options.svoRadRenderIter)
        //    sketch.HashRadianceAsPhotonDensity(dPathNodes, totalPathNodeCount,
        //                                       MaximumPathNodePerPath(), cudaSystem);
    }

    // On SVO_Radiance mode clear the image memory
    // And trace the SVO from the camera and send the results
    // On voxel trace mode we just trace the rays without any material
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    {
        // Clear the image buffer
        imgMemory.Reset(cudaSystem);

        // Generate rays appropriate to the camera type
        switch(currentCamera.type)
        {
            case SCENE_CAMERA:
            {
                GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
                (
                    currentCamera.nonTransformedCamIndex, options.sampleCount,
                    RayAuxInitWFPG(InitialWFPGAux,
                                   options.sampleCount *
                                   options.sampleCount),
                    true,
                    false
                );
                break;
            }
            case CUSTOM_CAMERA:
            {
                GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
                (
                    *currentCamera.dCustomCamera, options.sampleCount,
                    RayAuxInitWFPG(InitialWFPGAux,
                                   options.sampleCount *
                                   options.sampleCount),
                    true,
                    false
                );
                break;
            }
            case TRANSFORMED_SCENE_CAMERA:
            {
                GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU>
                (
                    currentCamera.transformedSceneCam.transform,
                    currentCamera.transformedSceneCam.cameraIndex,
                    options.sampleCount,
                    RayAuxInitWFPG(InitialWFPGAux,
                                   options.sampleCount *
                                   options.sampleCount),
                    true,
                    false
                );
                break;
            }

        }

        // Now call the voxel trace kernel
        const auto& gpu = cudaSystem.BestGPU();
        uint32_t totalRayCount = rayCaster->CurrentRayCount();


        gpu.GridStrideKC_X(0, (cudaStream_t)0, totalRayCount,
                            //
                           KCTraceSVO,
                           //
                           imgMemory.GMem<Vector4>(),
                           svo.TreeGPU(),
                           rayCaster->RaysIn(),
                           static_cast<RayAuxWFPG*>(*dAuxIn),
                           WFPGRenderMode::SVO_RADIANCE,
                           totalRayCount);


        //gpu.GridStrideKC_X(0, (cudaStream_t)0, totalRayCount,
        //                    //
        //                   KCQuerySketch,
        //                   //
        //                   imgMemory.GMem<Vector4>(),
        //                   svo.TreeGPU(),
        //                   sketch.SketchGPU(),
        //                   rayCaster->RaysIn(),
        //                   static_cast<RayAuxWFPG*>(*dAuxIn),
        //                   totalRayCount);

        //// Normalize the sketch query
        //Vector4f* dMax;
        //Vector4f* dMin;
        //DeviceMemory minMaxMem;
        //GPUMemFuncs::AllocateMultiData(std::tie(dMax, dMin),
        //                               minMaxMem,
        //                               {1, 1});

        //ReduceArrayGPU<Vector4f, ReduceMax<Vector4f>>
        //(
        //    *dMax,
        //    imgMemory.GMem<Vector4f>().gPixels,
        //    imgMemory.SegmentSize().Multiply(),
        //    Vector4f(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX)
        //);
        //ReduceArrayGPU<Vector4f, ReduceMin<Vector4f>>
        //(
        //    *dMin,
        //    imgMemory.GMem<Vector4>().gPixels,
        //    imgMemory.SegmentSize().Multiply(),
        //    Vector4f(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX)
        //);

        //gpu.GridStrideKC_X(0, (cudaStream_t)0, totalRayCount,
        //                   //
        //                   KCNormalizeImage,
        //                   //
        //                   imgMemory.GMem<Vector4>().gPixels,
        //                   *dMax,
        //                   *dMin,
        //                   //
        //                   imgMemory.SegmentSize());

        // Completely Reset the Image
        // This is done to eliminate variance from prev samples
        if(callbacks)
        {
            Vector2i start = imgMemory.SegmentOffset();
            Vector2i end = start + imgMemory.SegmentSize();
            callbacks->SendImageSectionReset(start, end);
        }
    }

    //METU_LOG("----------------");
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