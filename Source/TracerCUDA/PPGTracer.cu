#include "PPGTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"

#include "PPGTracerWork.cuh"
#include "GPULightSamplerUniform.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"
#include "GPUAcceleratorI.h"

#include "TracerDebug.h"
//std::ostream& operator<<(std::ostream& stream, const RayAuxPPG& v)
//{
//    stream << std::setw(0)
//        << v.pixelIndex << ", "
//        << "{" << v.radianceFactor[0]
//        << "," << v.radianceFactor[1]
//        << "," << v.radianceFactor[2] << "} "
//        << v.endPointIndex << ", "
//        << v.mediumIndex << " ";
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

static std::ostream& operator<<(std::ostream& s, const PathGuidingNode& n)
{
    s << "W:{" << n.worldPosition[0] << ", " 
               << n.worldPosition[1] << ", " 
               << n.worldPosition[2] << "} PN:{"
      << static_cast<uint32_t>(n.prevNext[0]) << ", "
      << static_cast<uint32_t>(n.prevNext[1]) << "} R: {";
    s << n.totalRadiance[0] << ", "
      << n.totalRadiance[1] << ", "
      << n.totalRadiance[2] << "} RF: {";
    s << n.radFactor[0] << ", "
      << n.radFactor[1] << ", "
      << n.radFactor[2] << "}";
    s << " Tree: ";
    if(n.nearestDTreeIndex == STree::InvalidDTreeIndex)
        s << "-";
    else
        s << n.nearestDTreeIndex;
    return s;
}

static std::ostream& operator<<(std::ostream& s, const DTreeNode& n)
{
    constexpr uint32_t UINT32_T_MAX = std::numeric_limits<uint32_t>::max();
    constexpr uint16_t UINT16_T_MAX = std::numeric_limits<uint16_t>::max();

    s << "P{";
    if(n.parentIndex == UINT16_T_MAX) s << "-";
    else s << n.parentIndex;
    s << "} ";
    s << "C{";
    if(n.childIndices[0] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[0];
    s << ", ";
    if(n.childIndices[1] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[1];
    s << ", ";
    if(n.childIndices[2] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[2];
    s << ", ";
    if(n.childIndices[3] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[3];
    s << "} ";
    s << "I{"
        << n.irradianceEstimates[0] << ", "
        << n.irradianceEstimates[1] << ", "
        << n.irradianceEstimates[2] << ", "
        << n.irradianceEstimates[3] << "}";
    return s;
}

static std::ostream& operator<<(std::ostream& s, const DTreeGPU& n)
{
    s << "Irradiane  : " << n.irradiance << std::endl;
    s << "NodeCount  : " << n.nodeCount << std::endl;
    s << "SampleCount: " << n.totalSamples << std::endl;
    return s;
}

__global__
static void KCInitializePaths(PathGuidingNode* gPathNodes,
                              uint32_t totalNodeCount)
{
    uint32_t globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId < totalNodeCount)
    {
        PathGuidingNode node;
        node.nearestDTreeIndex = STree::InvalidDTreeIndex;
        node.radFactor = Vector3f(1.0f);
        node.prevNext = Vector<2, PathGuidingNode::IndexType>(PathGuidingNode::InvalidIndex);
        node.totalRadiance = Zero3;
        node.worldPosition = Zero3;

        gPathNodes[globalId] = node;
    }
}

template <class T>
__global__ void KCConstructLightSampler(T* loc,
                                        const GPULightI** gLights,
                                        const uint32_t lightCount)
{
    uint32_t globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId == 0)
    {
        T* lightSampler = new (loc) T(gLights, lightCount);
    }
}

TracerError PPGTracer::LightSamplerNameToEnum(PPGTracer::LightSamplerType& ls,
                                              const std::string& lsName)
{
    const std::array<std::string, LightSamplerType::END> samplerNames =
    {
        "Uniform"
    };

    uint32_t i = 0;
    for(const std::string s : samplerNames)
    {
        if(lsName == s)
        {
            ls = static_cast<LightSamplerType>(i);
            return TracerError::OK;
        }
        i++;
    }
    return TracerError::UNABLE_TO_INITIALIZE;
}

TracerError PPGTracer::ConstructLightSampler()
{
    LightSamplerType lst;
    TracerError e = LightSamplerNameToEnum(lst, options.lightSamplerType);

    if(e != TracerError::OK) 
        return e;

    switch(lst)
    {
        case LightSamplerType::UNIFORM:
        {
            DeviceMemory::EnlargeBuffer(pathMemory, sizeof(GPULightSamplerUniform));
            dLightSampler = static_cast<const GPUDirectLightSamplerI*>(pathMemory);

            const auto& gpu = cudaSystem.BestGPU();
            gpu.KC_X(0, (cudaStream_t)0, 1,
                     // Kernel
                     KCConstructLightSampler<GPULightSamplerUniform>,
                     // Args
                     static_cast<GPULightSamplerUniform*>(pathMemory),
                     dLights, 
                     lightCount);

            return TracerError::OK;
        }
    }
    return TracerError::UNABLE_TO_INITIALIZE;
}

void PPGTracer::ResizeAndInitPathMemory()
{
    size_t totalPathNodeCount = TotalPathNodeCount();
    METU_LOG("Allocating PPGTracer global path buffer: Size %llu MiB", 
             totalPathNodeCount * sizeof(PathGuidingNode) / 1024 / 1024);

    DeviceMemory::EnlargeBuffer(pathMemory, totalPathNodeCount * sizeof(PathGuidingNode));
    dPathNodes = static_cast<PathGuidingNode*>(pathMemory);

    // Initialize Paths
    const CudaGPU& bestGPU = cudaSystem.BestGPU();
    bestGPU.KC_X(0, 0, totalPathNodeCount,
                 //
                 KCInitializePaths,
                 //
                 dPathNodes,
                 static_cast<uint32_t>(totalPathNodeCount));

}

uint32_t PPGTracer::TotalPathNodeCount() const
{
    return (imgMemory.SegmentSize()[0] * imgMemory.SegmentSize()[1] *
            options.sampleCount * options.sampleCount) * options.maximumDepth;
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
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // Generate Light Sampler (for nee)
    ConstructLightSampler();

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);

        // Generate work batch from appropirate work pool
        WorkBatchArray workBatchList;
        if(mg.IsBoundary())
        {
            bool emptyPrim = (std::string(pg.Type()) ==
                              std::string(BaseConstants::EMPTY_PRIMITIVE_NAME));

            WorkPool<bool, bool, bool>& wp = boundaryWorkPool;
            GPUWorkBatchI* batch = nullptr;
            if((err = wp.GenerateWorkBatch(batch, mg, pg, dTransforms,
                                           options.nextEventEstimation,
                                           options.directLightMIS,
                                           emptyPrim)) != TracerError::OK)
                return err;
            workBatchList.push_back(batch);
        }
        else
        {            
            WorkPool<bool, bool>& wpCombo = pathWorkPool;
            GPUWorkBatchI* batch = nullptr;
            if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                                dTransforms,
                                                options.nextEventEstimation,
                                                options.directLightMIS)) != TracerError::OK)
                return err;
            workBatchList.push_back(batch);            
        }
        workMap.emplace(batchId, workBatchList);
    }    

    // Init sTree
    AABB3f worldAABB = scene.BaseAccelerator()->SceneExtents();
    sTree = std::make_unique<STree>(worldAABB);

    return TracerError::OK;
}

TracerError PPGTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetUInt(options.maximumDepth, MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.rrStart, RR_START_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.lightSamplerType, LIGHT_SAMPLER_TYPE_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.nextEventEstimation, NEE_NAME)) != TracerError::OK)
        return err;    
    if((err = opts.GetBool(options.directLightMIS, DIRECT_LIGHT_MIS_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.rawPathGuiding, RAW_PG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.alwaysSendSamples, ALWAYS_SEND_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetUInt(options.maxDTreeDepth, D_TREE_MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.dTreeSplitThreshold, D_TREE_FLUX_RATIO_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.sTreeSplitThreshold, S_TREE_SAMPLE_SPLIT_NAME)) != TracerError::OK)
        return err;
    
    return TracerError::OK;
}

bool PPGTracer::Render()
{
    HitAndPartitionRays();

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
    globalData.gImage = imgMemory.GMem<Vector4>();
    globalData.lightList = dLights;    
    globalData.totalLightCount = lightCount;
    globalData.lightSampler = dLightSampler;
    //
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    //
    // Set SD Tree
    const STreeGPU* dSTree;
    const DTreeGPU** dDTrees;
    sTree->TreeGPU(dSTree, dDTrees);
    globalData.gStree = dSTree;
    globalData.gDTrees = dDTrees;    
    //
    globalData.gPathNodes = dPathNodes;
    globalData.maximumPathNodePerRay = options.maximumDepth;
    // Todo change these later
    globalData.rawPathGuiding = true;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;    
    
    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = PartitionOutputRays(totalOutRayCount, workMap);

    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPPG);
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    //for(auto pIt = workPartition.crbegin();
    //    pIt != workPartition.crend(); pIt++)
    for(auto p : outPartitions)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers        
        const RayAuxPPG* dAuxInLocal = static_cast<const RayAuxPPG*>(*dAuxIn);
        using WorkData = typename GPUWorkBatchD<PPGTracerGlobalState, RayAuxPPG>;
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
    WorkRays(workMap, outPartitions,
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

    //
    METU_LOG("PASS ENDED=============================================================");

    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(totalOutRayCount == 0 || currentDepth >= options.maximumDepth)
        return false;
    return true;
}

void PPGTracer::Finalize()
{
    uint32_t totalPathNodeCount = TotalPathNodeCount();

    Debug::DumpMemToFile("PathNodes", dPathNodes, totalPathNodeCount);

    // Accumulate the finished radiances to the STree
    sTree->AccumulateRaidances(dPathNodes, totalPathNodeCount,
                               options.maximumDepth, cudaSystem);

    // We iterated once
    currentTreeIteration += options.sampleCount * options.sampleCount;
    // Swap the trees if we achieved treshold
    if(currentTreeIteration == nextTreeSwap)
    {
        // Double the amount of iterations required for this
        nextTreeSwap <<= 1;
     
        uint32_t treeSwapIterationCount = Utility::FindLastSet32(nextTreeSwap) - 1;
        uint32_t currentSTreeSplitThreshold = static_cast<uint32_t>((std::pow(2.0f, treeSwapIterationCount) *
                                                                     options.sTreeSplitThreshold));

        // Split and Swap the trees
        sTree->SplitAndSwapTrees(options.sTreeSplitThreshold,
                                 options.dTreeSplitThreshold,
                                 options.maxDTreeDepth,
                                 cudaSystem);

        printf("Splitting and Swapping Trees\n");
        CUDA_CHECK(cudaDeviceSynchronize());
        // PrintEveryDTree
        std::vector<DTreeGPU> structs;
        std::vector<std::vector<DTreeNode>> nodes;
        sTree->GetAllDTreesToCPU(structs, nodes, false);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            Debug::DumpMemToFile("dTreeN" + std::to_string(i), nodes[i].data(), nodes[i].size());
            Debug::DumpMemToFile("dTree" + std::to_string(i), &structs[i], 1);
        }

        // Completely Reset the Image
        // This is done to eliminate variance from prev samples
        ResetImage();
    }

    uint32_t prevTreeSwap = (nextTreeSwap >> 1);
    if(options.alwaysSendSamples ||
       // Do not send samples untill we exceed prev iteration samples
       (currentTreeIteration - prevTreeSwap) >= prevTreeSwap)
    {
        // Base class finalize directly sends the image
        GPUTracer::Finalize();
    }
}

void PPGTracer::GenerateWork(int cameraId)
{
    if(callbacks)
        callbacks->SendCurrentCamera(SceneCamToVisorCam(cameraId));

    GenerateRays<RayAuxPPG, RayAuxInitPPG>(dCameras[cameraId],
                                           options.sampleCount,
                                           RayAuxInitPPG(InitialPPGAux,
                                                         options.sampleCount *
                                                         options.sampleCount));

    ResizeAndInitPathMemory();
    currentDepth = 0;
}

void PPGTracer::GenerateWork(const VisorCamera& cam)
{
    GenerateRays<RayAuxPPG, RayAuxInitPPG>(cam, options.sampleCount,
                                           RayAuxInitPPG(InitialPPGAux,
                                                         options.sampleCount *
                                                         options.sampleCount));
    ResizeAndInitPathMemory();
    currentDepth = 0;
}