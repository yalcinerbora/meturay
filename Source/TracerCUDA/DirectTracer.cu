#include "DirectTracer.h"
#include "RayTracer.hpp"

#include "GPUWork.cuh"
#include "DirectTracerWorks.cuh"

#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"

//#include "TracerLib/TracerDebug.h"
//std::ostream& operator<<(std::ostream& stream, const RayAuxBasic& v)
//{
//    stream << std::setw(0)
//        << v.pixelId << ", "
//        << "{" << v.radianceFactor[0]
//        << "," << v.radianceFactor[1]
//        << "," << v.radianceFactor[2] << "}";
//    return stream;
//}

const std::array<std::string, DirectTracer::RenderType::END> DirectTracer::RenderTypeNames =
{
    "Furnace",
    "Position",
    "WorldNormal",
    "LinearDepth",
    "LogDepth"
};

DirectTracer::DirectTracer(const CudaSystem& s,
                           const GPUSceneI& scene,
                           const TracerParameters& p)
    : RayTracer(s, scene, p)
    , emptyMat(s.BestGPU())
{
    furnaceWorkPool.AppendGenerators(DirectTracerFurnaceWorkerList{});
    normalWorkPool.AppendGenerators(DirectTracerNormalWorkerList{});
    positionWorkPool.AppendGenerators(DirectTracerPositionWorkerList{});
}

TracerError DirectTracer::StringToRenderType(RenderType& rt, const std::string& s)
{
    uint32_t i = 0;
    for(const std::string name : RenderTypeNames)
    {
        if(name == s)
        {
            rt = static_cast<RenderType>(i);
            return TracerError::OK;
        }
        i++;
    }
    return TracerError::UNABLE_TO_INITIALIZE;
}

std::string DirectTracer::RenderTypeToString(RenderType rt)
{
    return RenderTypeNames[static_cast<int>(rt)];
}

TracerError DirectTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
       return err;

    std::string renderTypeString;
    if((err = opts.GetString(renderTypeString, RENDER_TYPE_NAME)) != TracerError::OK)
        return err;
    if((err = StringToRenderType(options.renderType, renderTypeString)) != TracerError::OK)
       return err;
   return TracerError::OK;
}

TracerError DirectTracer::Initialize()
{
    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& wInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(wInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(wInfo);
        uint32_t batchId = std::get<0>(wInfo);

        GPUWorkBatchI* batch = nullptr;
        switch(options.renderType)
        {
            case RenderType::RENDER_FURNACE:
            {
                if((err = furnaceWorkPool.GenerateWorkBatch(batch, mg, pg,
                                                     dTransforms)) != TracerError::OK)
                    return err;
                break;
            }
            case RenderType::RENDER_POSITION:
            case RenderType::RENDER_LIN_DEPTH:
            case RenderType::RENDER_LOG_DEPTH:
            {
                // Skip empty primitives since those wont have any normal info
                if(std::string(pg.Type()) == std::string(BaseConstants::EMPTY_PRIMITIVE_NAME))
                    continue;

                if((err = positionWorkPool.GenerateWorkBatch(batch, DirectTracerPositionWork::TypeName(),
                                                             emptyMat, emptyPrim,
                                                             dTransforms)) != TracerError::OK)
                    return err;
                break;
            }
            case RenderType::RENDER_WORLD_NORMAL:
            {
                const std::string workTypeName = MangledNames::WorkBatch(pg.Type(), "DirectNormal");

                // Generate work batch from appropirate work pool
                if((err = normalWorkPool.GenerateWorkBatch(batch, workTypeName.c_str(),
                                                           emptyMat, pg, dTransforms)) != TracerError::OK)
                    return err;
                break;
            }
        }
        
        workMap.emplace(batchId, WorkBatchArray{batch});
    }
    return err;
}

bool DirectTracer::Render()
{
    // Do Hit Loop
    HitAndPartitionRays();

    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = PartitionOutputRays(totalOutRayCount, workMap);

    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPath);
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    //for(auto pIt = work+Partition.crbegin();
    //    pIt != workPartition.crend(); pIt++)
    for(auto p : outPartitions)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers
        const RayAuxBasic* dAuxInGlobal = static_cast<const RayAuxBasic*>(*dAuxIn);

        if(options.renderType == RenderType::RENDER_POSITION ||
           options.renderType == RenderType::RENDER_LOG_DEPTH ||
           options.renderType == RenderType::RENDER_LIN_DEPTH)
        {
            using WorkData = GPUWorkBatchD<DirectTracerPositionGlobalState, RayAuxBasic>;

            // Generate Global Data Struct
            DirectTracerPositionGlobalState globalData;
            globalData.gImage = imgMemory.GMem<Vector4>();
            globalData.gCurrentCam = dCameras[currentCameraId];
            if(options.renderType == RenderType::RENDER_POSITION)
                globalData.posRenderType = PositionRenderType::VECTOR3;
            else if(options.renderType == RenderType::RENDER_LOG_DEPTH)
                globalData.posRenderType = PositionRenderType::LOG_DEPTH;
            else
                globalData.posRenderType = PositionRenderType::LINEAR_DEPTH;

            int i = 0;
            for(auto& work : loc->second)
            {
                RayAuxBasic* dAuxOutLocal = static_cast<RayAuxBasic*>(*dAuxOut) + p.offsets[i];

                auto& wData = static_cast<WorkData&>(*work);
                wData.SetGlobalData(globalData);
                wData.SetRayDataPtrs(dAuxOutLocal, dAuxInGlobal);
                i++;
            }
        }
        else
        {
            using WorkData = GPUWorkBatchD<DirectTracerGlobalState, RayAuxBasic>;

            // Generate Global Data Struct
            DirectTracerGlobalState globalData;
            globalData.gImage = imgMemory.GMem<Vector4>();

            int i = 0;
            for(auto& work : loc->second)
            {
                RayAuxBasic* dAuxOutLocal = static_cast<RayAuxBasic*>(*dAuxOut) + p.offsets[i];

                auto& wData = static_cast<WorkData&>(*work);
                wData.SetGlobalData(globalData);
                wData.SetRayDataPtrs(dAuxOutLocal, dAuxInGlobal);
                i++;
            }
        }
    }

    // Launch Kernels
    WorkRays(workMap,
             outPartitions,
             totalOutRayCount,
             scene.BaseBoundaryMaterial());
    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    // Check tracer termination conditions
    if(totalOutRayCount == 0) return false;
    return true;
}

void DirectTracer::GenerateWork(int cameraId)
{
    if(callbacks)
        callbacks->SendCurrentCamera(SceneCamToVisorCam(cameraId));
    // Save Camera Id for potential depth generation
    currentCameraId = cameraId;
    // Only use anti-alias when furnace mode is on
    bool antiAlias = (options.renderType == RenderType::RENDER_FURNACE) ? true : false;
    // Generate Rays
    GenerateRays<RayAuxBasic, RayAuxInitBasic>(cameraId,
                                               options.sampleCount,
                                               RayAuxInitBasic(InitialBasicAux),
                                               antiAlias);
}

void DirectTracer::GenerateWork(const VisorCamera& c)
{
    // TODO: Save Visor Camera to GPU memory for depth map generation;
    currentCameraId = 0;
    // Only use anti-alias when furnace mode is on
    bool antiAlias = (options.renderType == RenderType::RENDER_FURNACE) ? true : false;
    GenerateRays<RayAuxBasic, RayAuxInitBasic>(c, options.sampleCount,
                                               RayAuxInitBasic(InitialBasicAux),
                                               antiAlias);
}

void DirectTracer::GenerateWork(const GPUCameraI& dCam)
{
    // TODO: Save This GPU Camera to GPU memory for depth map generation;
    currentCameraId = 0;
    // Only use anti-alias when furnace mode is on
    bool antiAlias = (options.renderType == RenderType::RENDER_FURNACE) ? true : false;
    // Generate Rays
    GenerateRays<RayAuxBasic, RayAuxInitBasic>(dCam,
                                               options.sampleCount,
                                               RayAuxInitBasic(InitialBasicAux),
                                               antiAlias);
}