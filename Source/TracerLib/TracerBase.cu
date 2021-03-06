#include "TracerBase.h"

#include "RayLib/Camera.h"
#include "RayLib/Log.h"
#include "RayLib/TracerError.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/GPUSceneI.h"

#include "TracerDebug.h"
#include "GPUAcceleratorI.h"
#include "GPUMaterialI.h"
#include "TracerLogicI.h"
#include "GPUEventEstimatorI.h"

//struct RayAuxBasic
//{
//    Vector3f        totalRadiance;
//    uint32_t        pixelId;
//    uint32_t        pixelSampleId;
//};
//
//std::ostream& operator<<(std::ostream& stream, const RayAuxBasic& v)
//{
//    stream << std::setw(0)
//            << v.pixelId << ", "
//            << v.pixelSampleId << ", "
//            << "{" << v.totalRadiance[0]
//            << "," << v.totalRadiance[0]
//            << "," << v.totalRadiance[0] << "}";
//    return stream;
//}

template <class... Args>
inline void TracerBase::SendLog(const char* format, Args... args)
{
    if(!options.verbose) return;

    size_t size = snprintf(nullptr, 0, format, args...);
    std::string s(size, '\0');
    snprintf(&s[0], size, format, args...);
    if(callbacks) callbacks->SendLog(s);
}

void TracerBase::SendError(TracerError e, bool isFatal)
{
    if(callbacks) callbacks->SendError(e);
    healthy = isFatal;
}

void TracerBase::HitRays()
{
    // Sort and Partition happens on the leader device
    CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

    // Tracer Logic interface
    const Vector2i& accBitCounts = currentLogic->SceneAcceleratorMaxBits();
    GPUBaseAcceleratorI& baseAccelerator = currentLogic->BaseAcelerator();
    const AcceleratorBatchMappings& subAccelerators = currentLogic->AcceleratorBatches();
    // Reset Hit Memory for hit loop
    rayMemory.ResetHitMemory(currentRayCount, currentLogic->HitStructSize());
    // Make Base Accelerator to get ready for hitting
    baseAccelerator.GetReady(cudaSystem, currentRayCount);
    // Ray Memory Pointers
    RayGMem* dRays = rayMemory.Rays();
    HitKey* dMaterialKeys = rayMemory.MaterialKeys();
    TransformId* dTransfomIds = rayMemory.TransformIds();
    PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
    HitStructPtr dHitStructs = rayMemory.HitStructs();
    // These are sorted etc.
    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
    RayId*  dCurrentRayIds = rayMemory.CurrentIds();

    // Try to hit rays until no ray is left
    // (these rays will be assigned with a material)
    // outside rays are also assigned with a material (which is special)
    uint32_t rayCount = currentRayCount;
    // At start all rays are valid
    uint32_t validRayOffset = 0;
    while(rayCount > 0)
    {
        //Debug::DumpMemToFile("dAccKeys", dCurrentKeys, currentRayCount);
        //Debug::DumpMemToFile("dRayIds", dCurrentRayIds, currentRayCount);

        // Traverse accelerator
        // Base accelerator provides potential hits
        // Cannot provide an absolute hit (its not its job)
        baseAccelerator.Hit(cudaSystem,
                            dTransfomIds, 
                            dCurrentKeys + validRayOffset,
                            dRays,
                            dCurrentRayIds + validRayOffset,
                            rayCount);

        // Wait all GPUs to finish...
        cudaSystem.SyncGPUMainStreamAll();

        //Debug::DumpMemToFile("dAccKeys", dCurrentKeys, currentRayCount);
        //Debug::DumpMemToFile("dRayIds", dCurrentRayIds, currentRayCount);

        // Base accelerator traverses the data partially
        // Updates current key (which represents inner accelerator batch and id)

        // After that, system sorts rays according to the keys
        // and partitions the array according to batches

        // Sort and Partition happens on the leader device
        CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

        // Sort initial results (in order to partition and launch kernels accordingly)
        // Sort is radix sort.
        // We sort inner indices in addition to batches results for better data locality
        // We only sort up-to a certain bit (radix sort) which is tied to
        // accelerator count
        // Move offset to skip null bathces
        rayMemory.SortKeys(dCurrentRayIds, dCurrentKeys, rayCount, accBitCounts);
        // Parition to sub accelerators
        //
        // There may be invalid rays sprinkled along the array.
        // Holes occur in the structure since in previous iteration,
        // a material may required to write N rays for its output (which is defined
        // by the material) but it wrote < N rays.
        //
        // One of the main examples for such behaviour can be transparent objects
        // where ray may be only reflected (instead of refrating and reflecting) because
        // of the total internal reflection phenomena.
        auto portions = rayMemory.Partition(rayCount);

        //Debug::DumpMemToFile("dAccKeys", dCurrentKeys, currentRayCount);
        //Debug::DumpMemToFile("dRayIds", dCurrentRayIds, currentRayCount);

        // Reorder partitions for efficient calls
        // (group partitions into gpus and order for better async access)
        // ....
        // TODO:
        const int totalGPU = static_cast<int>(cudaSystem.GPUList().size());
        const auto& gpus = cudaSystem.GPUList();
        auto currentGPU = gpus.begin();

        // For each partition
        for(const auto& p : portions)
        {
            // Find Accelerator
            // Since there is no batch for invalid keys
            // that partition will be automatically be skipped
            auto loc = subAccelerators.find(p.portionId);
            if(loc == subAccelerators.end()) continue;

            RayId* dRayIdStart = dCurrentRayIds + validRayOffset + p.offset;
            HitKey* dCurrentKeyStart = dCurrentKeys + validRayOffset + p.offset;

            // Run local hit kernels
            // Local hit kernels returns a material key
            // and primitive inner id.
            // Since materials are batched for both material and
            loc->second->Hit(*currentGPU,
                             // O
                             dMaterialKeys,
                             dPrimitiveIds,
                             dHitStructs,
                             // I-O
                             dRays,
                             // Input
                             dTransfomIds,
                             dRayIdStart,
                             dCurrentKeyStart,
                             static_cast<uint32_t>(p.count));

            // Split to GPUs
            currentGPU++;
            if(currentGPU == gpus.end()) currentGPU = gpus.begin();

            // Hit function updates material key,
            // primitive id and struct if this hit is accepted
        }
        //printf("=====================================================\n");

        // Update new ray count
        // On partition array check first partition
        // it may contain invalid key meaning
        // those rays are totally processed
        // change the offset so that
        // we skip those rays
        auto nullPortion = portions.begin();
        if(nullPortion->portionId == HitKey::NullBatch)
            rayCount = static_cast<uint32_t>(nullPortion->offset);

        // Iteration is done
        // We cant continue loop untill these kernels are finished
        // on gpu(s)
        //
        // Tracer logic mostly utilizies mutiple GPUs so we need to
        // wait all GPUs to finish
        cudaSystem.SyncGPUAll();
    }

    // At the end of iteration all rays found a material, primitive
    // and interpolation weights (which should be on hitStruct)
    //printf("FRAME END\n");
}

void TracerBase::ShadeRays()
{
    // Sort and Partition happens on leader device
    CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

    const Vector2i matMaxBits = currentLogic->SceneMaterialMaxBits();

    // Ray Memory Pointers
    const RayGMem* dRays = rayMemory.Rays();
    const void* dRayAux = rayMemory.RayAux<void>();
    const HitStructPtr dHitStructs = rayMemory.HitStructs();
    const PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
    // These are sorted etc.
    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
    RayId* dCurrentRayIds = rayMemory.CurrentIds();

    // Material Interfaces
    const MaterialBatchMappings& materials = currentLogic->MaterialBatches();
    uint32_t rayCount = currentRayCount;

    // Copy materialKeys to currentKeys
    // to make it ready for sorting
    rayMemory.FillMatIdsForSort(rayCount);

    // Sort with respect to the materials keys
    rayMemory.SortKeys(dCurrentRayIds, dCurrentKeys, rayCount, matMaxBits);

    // Parition w.r.t. material batch
    auto portions = rayMemory.Partition(rayCount);
    // Use partition lis to find out
    // total potential output ray count
    uint32_t totalOutRayCount = 0;
    for(const auto& p : portions)
    {
        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = materials.find(p.portionId);
        if(loc == materials.end()) continue;

        totalOutRayCount += static_cast<uint32_t>(p.count) *
                            loc->second->OutRayCount();
    }

    // Allocate output ray memory
    rayMemory.ResizeRayOut(totalOutRayCount, currentLogic->PerRayAuxDataSize(),
                           currentLogic->SceneBaseBoundMatKey());
    unsigned char* dAuxOut = rayMemory.RayAuxOut<unsigned char>();
    RayGMem* dRaysOut = rayMemory.RaysOut();
    HitKey* dBoundKeyOut = rayMemory.MaterialKeys();

    // Reorder partitions for efficient calls
    // (sort by gpu and order for better async access)
    // ....
    // TODO:

    // For each partition
    size_t outOffset = 0;
    for(auto pIt = portions.crbegin();
        pIt != portions.crend(); pIt++)
    {
        const auto& p = (*pIt);

        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = materials.find(p.portionId);
        if(loc == materials.end()) continue;

        // Relativize input & output pointers
        const RayId* dRayIdStart = dCurrentRayIds + p.offset;
        const HitKey* dKeyStart = dCurrentKeys + p.offset;
        // Output
        RayGMem* dRayOutStart = dRaysOut + outOffset;
        void* dAuxOutStart = dAuxOut + (outOffset * currentLogic->PerRayAuxDataSize());
        HitKey* dBoundKeyStart = dBoundKeyOut + outOffset;

        // Actual Shade Call
        loc->second->ShadeRays(// Output
                               outputImage,
                               //
                               dBoundKeyStart,
                               dRayOutStart,
                               dAuxOutStart,
                               //  Input
                               dRays,
                               dRayAux,
                               dPrimitiveIds,
                               dHitStructs,
                               //
                               dKeyStart,
                               dRayIdStart,

                               static_cast<uint32_t>(p.count),
                               rngMemory);

        // Since output is dynamic (each material may write multiple rays)
        // add offsets to find proper count
        outOffset += p.count * loc->second->OutRayCount();
    }
    assert(totalOutRayCount == outOffset);
    currentRayCount = totalOutRayCount;

    // Again wait all of the GPU's since
    // CUDA functions will be on multiple-gpus
    cudaSystem.SyncGPUAll();

    // Shading complete
    // Now make "RayOut" to "RayIn"
    // and continue
    rayMemory.SwapRays();
}

TracerBase::TracerBase(CudaSystem& s)
    : cudaSystem(s)
    , callbacks(nullptr)
    , currentRayCount(0)
    , currentLogic(nullptr)
    , healthy(false)
    , sampleCountPerRay(0)
    , options(TracerConstants::DefaultTracerOptions)
    , rayMemory(*(s.GPUList().begin()))
{}

TracerError TracerBase::Initialize()
{
    TracerError e = TracerError::OK;

    // No logic set for initalization
    if(currentLogic == nullptr) return TracerError::NO_LOGIC_SET;

    // Construct Accelerators
    GPUBaseAcceleratorI& baseAccelerator = currentLogic->BaseAcelerator();
    const AcceleratorGroupList& acceleratorGroups = currentLogic->AcceleratorGroups();
    GPUEventEstimatorI& estimator = currentLogic->EventEstimator();

    if((e = baseAccelerator.Constrcut(cudaSystem)) != TracerError::OK)
       return e;

    for(const auto& accel : acceleratorGroups)
    {
        if((e = accel->ConstructAccelerators(cudaSystem)) != TracerError::OK)
            return e;        
    }
    // Construct Estimator
    if((e = estimator.Construct(cudaSystem)) != TracerError::OK)
        return e;
    cudaSystem.SyncGPUAll();
    // All seems fine mark tracer as healthy
    healthy = true;
    return TracerError::OK;
}

void TracerBase::SetOptions(const TracerOptions& opts)
{
    options = opts;
}

void TracerBase::RequestBaseAccelerator()
{}

void TracerBase::RequestAccelerator(HitKey key)
{}

void TracerBase::AttachLogic(TracerBaseLogicI& logic)
{
    // Init and set Tracer System
    TracerError e = TracerError::OK;
    if((e = logic.Initialize()) != TracerError::OK)
    {
        if(callbacks) callbacks->SendError(e);
    }
    currentLogic = &logic;

    // Initialize RNG Memory
    rngMemory = RNGMemory(logic.Seed(), cudaSystem);
}

void TracerBase::GenerateInitialRays(const GPUSceneI& scene,
                                     int cameraId,
                                     int samplePerLocation)
{

    const CameraPerspective& cam = scene.CamerasCPU()[cameraId];
    GenerateInitialRays(scene, cam, samplePerLocation);

}

void TracerBase::GenerateInitialRays(const GPUSceneI& scene,
                                     const CameraPerspective& cam,
                                     int samplePerLocation)
{
    if(!healthy) return;
    // Delegate camera ray generation to tracer system
    currentRayCount = currentLogic->GenerateRays(cudaSystem,
                                                 //
                                                 outputImage,
                                                 rayMemory, rngMemory,
                                                 scene, cam, samplePerLocation,
                                                 outputImage.Resolution(),
                                                 outputImage.SegmentOffset(),
                                                 outputImage.SegmentSize());

    // You can only write to out buffer of the ray memory
    // Make that memory in rays for hit/shade system
    rayMemory.SwapRays();
    sampleCountPerRay = samplePerLocation * samplePerLocation;
}

bool TracerBase::Continue()
{
    return (currentRayCount > 0) && healthy;
}

void TracerBase::Render()
{
    if(!healthy) return;
    if(currentRayCount == 0) return;

    SendLog(" Starting Hits: %d rays...", currentRayCount);
    HitRays();
    SendLog(" Hits Complete, Shading...");
    ShadeRays();
    SendLog(" Shading Complete!");

    // Force Nsight Flush
    CUDA_CHECK(cudaDeviceSynchronize());

    //printf("\n-------\n");
}

void TracerBase::FinishSamples()
{
    if(!healthy) return;

    // Normally if ray reaches to boundary material
    // its result should be on image but now always
    SendLog("Finishing Samples: %d rays are left...");
   
    // Determine Size
    Vector2i pixelCount = outputImage.SegmentSize();
    Vector2i start = outputImage.SegmentOffset();
    Vector2i end = start + outputImage.SegmentSize();
    size_t offset = (static_cast<size_t>(pixelCount[0])* pixelCount[1] *
                     outputImage.PixelSize());

    // Flush Devices and Get the Image
    cudaSystem.SyncGPUAll();
    std::vector<Byte> imageData = outputImage.GetImageToCPU(cudaSystem);

    size_t pixelCount1D = static_cast<size_t>(pixelCount[0]) * pixelCount[1];

    // Launch finished image
    if(callbacks) callbacks->SendImage(std::move(imageData),
                                       outputImage.Format(),
                                       offset,
                                       start, end);
    SendLog("Samples Finished!");
}

void TracerBase::SetImagePixelFormat(PixelFormat f)
{
    outputImage.SetPixelFormat(f, cudaSystem);
}

void TracerBase::ReportionImage(Vector2i start,
                                Vector2i end)
{
    outputImage.Reportion(start, end, cudaSystem);
}

void TracerBase::ResizeImage(Vector2i resolution)
{
    outputImage.Resize(resolution);
}

void TracerBase::ResetImage()
{
    outputImage.Reset(cudaSystem);
}
