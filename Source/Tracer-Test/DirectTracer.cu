#include "DirectTracer.h"
#include "TracerWorks.cuh"

#include "TracerLib/GPUWork.cuh"

#include "RayLib/GPUSceneI.h"

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

DirectTracer::DirectTracer(const CudaSystem& s,
                           const GPUSceneI& scene,
                           const TracerParameters& p)
    : RayTracer(s, scene, p)
{
    workPool.AppendGenerators(DirectTracerWorkerList{});
}

TracerError DirectTracer::SetOptions(const TracerOptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
       return err;
   return TracerError::OK;
}

TracerError DirectTracer::Initialize()
{
    TracerError err = TracerError::OK;
    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);
        
        GPUWorkBatchI* batch = nullptr;
        if((err = workPool.GenerateWorkBatch(batch, mg, pg)) != TracerError::OK)
            return err;        
        workMap.emplace(batchId, batch);
    }
    return RayTracer::Initialize();
}

bool DirectTracer::Render()
{
    // Do Hit Loop
    HitAndPartitionRays();

    // Generate Global Data Struct
    DirectTracerGlobal globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();

    // After Hit Determine Ray Aux Output size    
    uint32_t totalOutRayCount = 0;
    for(const auto& p : workPartition)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        totalOutRayCount += (static_cast<uint32_t>(p.count)*
                             loc->second->OutRayCount());
    }
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxBasic);
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    size_t auxOutOffset = 0;
    for(auto pIt = workPartition.crbegin();
        pIt != workPartition.crend(); pIt++)
    {
        const auto& p = (*pIt);

        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers
        RayAuxBasic* dAuxOutLocal = static_cast<RayAuxBasic*>(*dAuxOut) + auxOutOffset;
        const RayAuxBasic* dAuxInLocal = static_cast<const RayAuxBasic*>(*dAuxIn) + p.offset;
        //auto& wBatch = static_cast<GPUWorkBatchI&>(*(loc->second));

        using WorkData = typename GPUWorkBatchD<DirectTracerGlobal, RayAuxBasic>;
        auto& wData = static_cast<WorkData&>(*loc->second);
        wData.SetGlobalData(globalData);
        wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);

        auxOutOffset += (static_cast<uint32_t>(p.count) *
                         loc->second->OutRayCount());
    }
    assert(auxOutOffset == totalOutRayCount);


    // Hit System Generated bunch of hit pairs
    WorkRays(workMap, scene.BaseBoundaryMaterial());
    SwapAuxBuffers();

    if(totalOutRayCount == 0) return false;
    return true;
}

void DirectTracer::GenerateWork(int cameraId)
{
    // Generate Rays
    GenerateRays(dSceneCameras[cameraId], options.sampleCount);
}

void DirectTracer::GenerateWork(const CPUCamera& c)
{
    LoadCameraToGPU(c);
    GenerateRays(*dCustomCamera, options.sampleCount);
}