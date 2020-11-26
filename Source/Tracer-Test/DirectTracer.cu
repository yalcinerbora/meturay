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
    if((err = RayTracer::Initialize()) != TracerError::OK) 
        return err;

    // Generate your worklist
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& workInfo : infoList)
    {
        const GPUPrimitiveGroupI& pg = *std::get<1>(workInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(workInfo);
        uint32_t batchId = std::get<0>(workInfo);
        
        GPUWorkBatchI* batch = nullptr;
        if((err = workPool.GenerateWorkBatch(batch, mg, pg,
                                             dTransforms)) != TracerError::OK)
            return err;        
        workMap.emplace(batchId, batch);
    }
    return err;
}

bool DirectTracer::Render()
{
    // Do Hit Loop
    HitAndPartitionRays();

    // Generate Global Data Struct
    DirectTracerGlobal globalData;
    globalData.gImage = imgMemory.GMem<Vector4>();

    // Generate output partitions
    uint32_t totalOutRayCount = 0;
    auto outPartitions = PartitionOutputRays(totalOutRayCount, workMap);

    // Allocate new auxiliary buffer 
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxPath);
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
        RayAuxBasic* dAuxOutLocal = static_cast<RayAuxBasic*>(*dAuxOut) + p.offset;
        const RayAuxBasic* dAuxInGlobal = static_cast<const RayAuxBasic*>(*dAuxIn);
        //auto& wBatch = static_cast<GPUWorkBatchI&>(*(loc->second));

        using WorkData = typename GPUWorkBatchD<DirectTracerGlobal, RayAuxBasic>;
        auto& wData = static_cast<WorkData&>(*loc->second);
        wData.SetGlobalData(globalData);
        wData.SetRayDataPtrs(dAuxOutLocal, dAuxInGlobal);

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
    // Generate Rays
    GenerateRays(dCameras[cameraId], options.sampleCount);
}

void DirectTracer::GenerateWork(const VisorCamera& c)
{
    GenerateRays(c, options.sampleCount);
}