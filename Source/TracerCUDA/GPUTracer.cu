#include "GPUTracer.h"

#include "RayLib/Log.h"
#include "RayLib/TracerError.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/GPUSceneI.h"
#include "RayLib/MemoryAlignment.h"
#include "RayLib/SceneStructs.h"

#include "CudaConstants.h"
#include "GPUAcceleratorI.h"
#include "GPUWorkI.h"
#include "GPUTransformI.h"
#include "GPUMediumI.h"
#include "GPUMaterialI.h"
#include "GPUTransformI.h"
#include "GPULightI.h"
#include "GPUCameraI.h"

#include "TracerDebug.h"

TracerError GPUTracer::LoadCameras(std::vector<const GPUCameraI*>& dGPUCameras)
{
    TracerError e = TracerError::OK;
    for(auto& camera : cameras)
    {
        CPUCameraGroupI& c = *(camera.second);
        if((e = c.ConstructCameras(cudaSystem, dTransforms)) != TracerError::OK)
            return e;
        const auto& dCList = c.GPUCameras();
        dGPUCameras.insert(dGPUCameras.end(), dCList.begin(), dCList.end());
    }
    cameraCount = static_cast<uint32_t>(dGPUCameras.size());
    return TracerError::OK;
}

TracerError GPUTracer::LoadLights(std::vector<const GPULightI*>& dGPULights)
{
    TracerError e = TracerError::OK;
    for(auto& light : lights)
    {
        CPULightGroupI& l = *(light.second);
        if((e = l.ConstructLights(cudaSystem, dTransforms)) != TracerError::OK)
            return e;
        const auto& dLList = l.GPULights();
        dGPULights.insert(dGPULights.end(), dLList.begin(), dLList.end());
    }
    lightCount = static_cast<uint32_t>(dGPULights.size());
    return TracerError::OK;
}

TracerError GPUTracer::LoadTransforms(std::vector<const GPUTransformI*>& dGPUTransforms)
{
    TracerError e = TracerError::OK;
    for(auto& transform : transforms)
    {
        CPUTransformGroupI& t = *(transform.second);
        if((e = t.ConstructTransforms(cudaSystem)) != TracerError::OK)
            return e;
        const auto& dTList = t.GPUTransforms();
        dGPUTransforms.insert(dGPUTransforms.end(), dTList.begin(), dTList.end());
    }
    transformCount = static_cast<uint32_t>(dGPUTransforms.size());
    return TracerError::OK;
}

TracerError GPUTracer::LoadMediums(std::vector<const GPUMediumI*>& dGPUMediums)
{
    TracerError e = TracerError::OK;
    uint32_t indexOffset = 0;
    for(auto& medium : mediums)
    {
        CPUMediumGroupI& m = *(medium.second);
        if((e = m.ConstructMediums(cudaSystem, indexOffset)) != TracerError::OK)
            return e;
        const auto& dMList = m.GPUMediums();
        dGPUMediums.insert(dGPUMediums.end(), dMList.begin(), dMList.end());
        indexOffset += m.MediumCount();
    }
    mediumCount = static_cast<uint32_t>(dGPUMediums.size());
    return TracerError::OK;
}

GPUTracer::GPUTracer(const CudaSystem& system, 
                     const GPUSceneI& scene,
                     const TracerParameters& p)
    : cudaSystem(system)
    , baseAccelerator(*scene.BaseAccelerator())
    , accelBatches(scene.AcceleratorBatchMappings())
    , materialGroups(scene.MaterialGroups())
    , transforms(scene.Transforms())
    , mediums(scene.Mediums())
    , cameras(scene.Cameras())
    , lights(scene.Lights())
    , baseMediumIndex(scene.BaseMediumIndex())
    , identityTransformIndex(scene.IdentityTransformIndex())
    , maxAccelBits(Vector2i(Utility::FindFirstSet32(scene.MaxAccelIds()[0]) + 1,
                            Utility::FindFirstSet32(scene.MaxAccelIds()[1]) + 1))
    , maxWorkBits(Vector2i(Utility::FindFirstSet32(scene.MaxMatIds()[0]) + 1,
                           Utility::FindFirstSet32(scene.MaxMatIds()[1]) + 1))
    , params(p)
    , maxHitSize(scene.HitStructUnionSize())
    , rayMemory(system.BestGPU())
    , callbacks(nullptr)
    , crashed(false)
    , currentRayCount(0)
{}

TracerError GPUTracer::Initialize()
{
    // Init RNGs for each block
    TracerError e = TracerError::OK;
    rngMemory = RNGMemory(params.seed, cudaSystem);

    std::vector<const GPUTransformI*> dGPUTransforms;
    std::vector<const GPUMediumI*> dGPUMediums;
    std::vector<const GPULightI*> dGPULights;
    std::vector<const GPUCameraI*> dGPUCameras;

    // Calculate Total Sizes
    size_t tCount = 0;
    size_t mCount = 0;
    size_t lCount = 0;
    size_t cCount = 0;
    std::for_each(transforms.cbegin(), transforms.cend(),
                  [&tCount](const auto& transform)
                  {
                      tCount += transform.second->TransformCount();
                  });
    std::for_each(mediums.cbegin(), mediums.cend(),
                  [&mCount](const auto& medium)
                  {
                      mCount += medium.second->MediumCount();
                  });
    std::for_each(lights.cbegin(), lights.cend(),
                  [&lCount](const auto& light)
                  {
                      lCount += light.second->LightCount();
                  });
    std::for_each(cameras.cbegin(), cameras.cend(),
                  [&cCount](const auto& camera)
                  {
                      cCount += camera.second->CameraCount();
                  });
    transformCount = static_cast<uint32_t>(tCount);
    mediumCount = static_cast<uint32_t>(mCount);
    lightCount = static_cast<uint32_t>(lCount);
    cameraCount = static_cast<uint32_t>(cCount);

    // Allocate
    size_t transformSize = transformCount * sizeof(GPUTransformI*);
    transformSize = Memory::AlignSize(transformSize, AlignByteCount);
    size_t mediumSize = mediumCount * sizeof(GPUMediumI*);
    mediumSize = Memory::AlignSize(mediumSize, AlignByteCount);
    size_t lightSize = lightCount * sizeof(GPULightI*);
    lightSize = Memory::AlignSize(lightSize, AlignByteCount);
    size_t cameraSize = cameraCount * sizeof(GPUCameraI*);
    cameraSize = Memory::AlignSize(cameraSize, AlignByteCount);

    size_t totalSize = (transformSize +
                        mediumSize +
                        lightSize +
                        cameraSize);

    DeviceMemory::EnlargeBuffer(commonTypeMemory, totalSize);

    // Determine pointers from allocation
    size_t offset = 0;
    Byte* memory = static_cast<Byte*>(commonTypeMemory);
    dTransforms = reinterpret_cast<const GPUTransformI**>(memory + offset);
    offset += transformSize;
    dMediums = reinterpret_cast<const GPUMediumI**>(memory + offset);
    offset += mediumSize;
    dCameras = reinterpret_cast<const GPUCameraI**>(memory + offset);
    offset += cameraSize;
    dLights = reinterpret_cast<const GPULightI**>(memory + offset);
    offset += lightSize;
    assert(offset == totalSize);

    // Transforms
    if((e = LoadTransforms(dGPUTransforms)) != TracerError::OK)
        return e;
    CUDA_CHECK(cudaMemcpy(const_cast<GPUTransformI**>(dTransforms),
                          dGPUTransforms.data(),
                          dGPUTransforms.size() * sizeof(GPUTransformI*),
                          cudaMemcpyHostToDevice));
    // Mediums
    if((e = LoadMediums(dGPUMediums)) != TracerError::OK)
        return e;
    CUDA_CHECK(cudaMemcpy(const_cast<GPUMediumI**>(dMediums),
                          dGPUMediums.data(),
                          dGPUMediums.size() * sizeof(GPUMediumI*),
                          cudaMemcpyHostToDevice));
    // Lights
    if((e = LoadLights(dGPULights)) != TracerError::OK)
        return e;
    CUDA_CHECK(cudaMemcpy(const_cast<GPULightI**>(dLights),
                          dGPULights.data(),
                          dGPULights.size() * sizeof(GPULightI*),
                          cudaMemcpyHostToDevice));
    // Cameras
    if((e = LoadCameras(dGPUCameras)) != TracerError::OK)
        return e;
    CUDA_CHECK(cudaMemcpy(const_cast<GPUCameraI**>(dCameras),
                          dGPUCameras.data(),
                          dGPUCameras.size() * sizeof(GPUCameraI*),
                          cudaMemcpyHostToDevice));

    // Attach Medium gpu pointer to Material Groups
    for(const auto& mg : materialGroups)
        mg.second->AttachGlobalMediumArray(dMediums, baseMediumIndex);
        
    // Attach Transform gpu pointer to the Accelerator Batches
    for(const auto& acc : accelBatches)
        acc.second->AttachGlobalTransformArray(dTransforms, identityTransformIndex);

    // Construct Accelerators
    SurfaceAABBList allSurfaceAABBs;
    for(const auto& accBatch : accelBatches)
    {
        GPUAcceleratorGroupI* acc = accBatch.second;
        if((e = acc->ConstructAccelerators(cudaSystem)) != TracerError::OK)
            return e;
        // Acquire surface aabb listings for base accelerator consrtuction  
        allSurfaceAABBs.insert(acc->AcceleratorAABBs().cbegin(), 
                               acc->AcceleratorAABBs().cend());

    }

    // Construct Base accelerator using aabb list
    if((e = baseAccelerator.Constrcut(cudaSystem, allSurfaceAABBs)) != TracerError::OK)
        return e;

    cudaSystem.SyncGPUAll();
    return TracerError::OK;
}

void GPUTracer::ResetHitMemory(uint32_t rayCount, HitKey baseBoundMatKey)
{
    currentRayCount = rayCount;
    rayMemory.ResizeRayOut(rayCount, baseBoundMatKey);
}

void GPUTracer::HitAndPartitionRays()
{   
    if(crashed) return;

    // Sort and Partition happens on the leader device
    CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

    // Tracer Logic interface
    const Vector2i& accBitCounts = maxAccelBits;
    const AcceleratorBatchMap& subAccelerators = accelBatches;
    // Reset Hit Memory for hit loop
    rayMemory.ResetHitMemory(identityTransformIndex, currentRayCount, maxHitSize);
    // Make Base Accelerator to get ready for hitting
    baseAccelerator.GetReady(cudaSystem, currentRayCount);
    // Ray Memory Pointers
    RayGMem* dRays = rayMemory.Rays();
    HitKey* dWorkKeys = rayMemory.WorkKeys();
    TransformId* dTransfomIds = rayMemory.TransformIds();
    PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
    HitStructPtr dHitStructs = rayMemory.HitStructs();
    // These are sorted etc.
    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
    RayId*  dCurrentRayIds = rayMemory.CurrentIds();

    //CUDA_CHECK(cudaMemset(dTransfomIds, 0xFF, currentRayCount * sizeof(TransformId)));
    //Debug::DumpMemToFile("dTransforms", dTransfomIds, currentRayCount);


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
                             dWorkKeys,
                             dTransfomIds,
                             dPrimitiveIds,
                             dHitStructs,
                             // I-O
                             dRays,
                             // Input
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

    // Partition rays for work kernel calls
    // Copy materialKeys to currentKeys
    // to make it ready for sorting
    rayMemory.FillMatIdsForSort(currentRayCount);
    // Sort with respect to the materials keys
    rayMemory.SortKeys(dCurrentRayIds, dCurrentKeys, currentRayCount, maxWorkBits);

    //Debug::DumpMemToFile("MatKeysIn", dCurrentKeys, currentRayCount);
    //Debug::DumpMemToFile("workKeyIn", rayMemory.WorkKeys(), currentRayCount);

    // Parition w.r.t. material batch
    workPartition.clear();
    workPartition = rayMemory.Partition(currentRayCount);


    //Debug::DumpMemToFile("dTransforms", dTransfomIds, currentRayCount);
    //printf("HIT PORTION END\n");
}

void GPUTracer::WorkRays(const WorkBatchMap& workMap, 
                         const RayPartitions<uint32_t>& outPortions,
                         uint32_t totalRayOut,
                         HitKey baseBoundMatKey)
{
    // Sort and Partition happens on leader device
    CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice().DeviceId()));

    // Ray Memory Pointers
    const RayGMem* dRays = rayMemory.Rays();
    const HitStructPtr dHitStructs = rayMemory.HitStructs();
    const PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
    const TransformId* dTransformIds = rayMemory.TransformIds();
    // These are sorted etc.
    HitKey* dCurrentKeys = rayMemory.CurrentKeys();
    RayId* dCurrentRayIds = rayMemory.CurrentIds();

    // Allocate output ray memory
    rayMemory.ResizeRayOut(totalRayOut, baseBoundMatKey);
    RayGMem* dRaysOut = rayMemory.RaysOut();
    HitKey* dBoundKeyOut = rayMemory.WorkKeys();

    // Reorder partitions for efficient calls
    // (sort by gpu and order for better async access)
    // ....
    // TODO:

    // For each partition
    //for(auto pIt = workPartition.crbegin();
    //    pIt != workPartition.crend(); pIt++)
    for(const auto& p : workPartition)
    {
        //const auto& p = (*pIt);

        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // TODO: change this loop to combine iterator instead of find
        //const auto& pIn = *(workPartition.find<uint32_t>(p.portionId));
        const auto& pOut = *(outPortions.find(ArrayPortion<uint32_t>{p.portionId}));

        // Relativize input & output pointers
        const RayId* dRayIdStart = dCurrentRayIds + p.offset;
        const HitKey* dKeyStart = dCurrentKeys + p.offset;
        // Output
        RayGMem* dRayOutStart = dRaysOut + pOut.offset;
        HitKey* dBoundKeyStart = dBoundKeyOut + pOut.offset;

        // Actual Shade Call
        loc->second->Work(dBoundKeyStart,
                          dRayOutStart,
                          //  Input
                          dRays,
                          dPrimitiveIds,
                          dTransformIds,
                          dHitStructs,
                          // Ids
                          dKeyStart,
                          dRayIdStart,
                          //
                          static_cast<uint32_t>(p.count),
                          rngMemory);

        //cudaSystem.SyncGPUAll();
        //METU_LOG("--------------------------");
    }
    currentRayCount = totalRayOut;

    // Again wait all of the GPU's since
    // CUDA functions will be on multiple-gpus
    cudaSystem.SyncGPUAll();

    //Debug::DumpMemToFile("workKeyOut", rayMemory.WorkKeys(), totalRayOut);

    // Shading complete
    // Now make "RayOut" to "RayIn"
    // and continue
    rayMemory.SwapRays();
}

void GPUTracer::SetParameters(const TracerParameters& p)
{
    if(params.seed != p.seed)
        rngMemory = std::move(RNGMemory(p.seed, cudaSystem));
    params = p;
}

void GPUTracer::SetImagePixelFormat(PixelFormat f)
{
    imgMemory.SetPixelFormat(f, cudaSystem);
}

void GPUTracer::ReportionImage(Vector2i start,
                                Vector2i end)
{
    imgMemory.Reportion(start, end, cudaSystem);
}

void GPUTracer::ResizeImage(Vector2i resolution)
{
    imgMemory.Resize(resolution);
}

void GPUTracer::ResetImage()
{
    imgMemory.Reset(cudaSystem);
}

template <class... Args>
inline void GPUTracer::SendLog(const char* format, Args... args)
{
    if(!params.verbose) return;

    size_t size = snprintf(nullptr, 0, format, args...);
    std::string s(size, '\0');
    snprintf(&s[0], size, format, args...);
    if(callbacks) callbacks->SendLog(s);
}

void GPUTracer::SendError(TracerError e, bool isFatal)
{
    if(callbacks) callbacks->SendError(e);
    crashed = isFatal;
}

RayPartitions<uint32_t> GPUTracer::PartitionOutputRays(uint32_t& totalOutRay,
                                                       const WorkBatchMap& workMap) const
{
    RayPartitions<uint32_t> outPartitions;

    // Find total ray out
    totalOutRay = 0;
    for(auto pIt = workPartition.crbegin();
        pIt != workPartition.crend(); pIt++)
    {
        const auto& p = (*pIt);

        // Skip if null batch or unfound material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        uint32_t count = (static_cast<uint32_t>(p.count) * 
                          loc->second->OutRayCount());

        outPartitions.emplace(ArrayPortion<uint32_t>
        {
            p.portionId,
                totalOutRay,
            count
        });
        totalOutRay += count;
    }
    return outPartitions;
}

void GPUTracer::Finalize()
{
    if(crashed) return;
    SendLog("Finalizing...");
   
    // Determine Size
    Vector2i pixelCount = imgMemory.SegmentSize();
    Vector2i start = imgMemory.SegmentOffset();
    Vector2i end = start + imgMemory.SegmentSize();
    size_t offset = (static_cast<size_t>(pixelCount[0])* pixelCount[1] *
                     imgMemory.PixelSize());

    // Flush Devices and Get the Image
    cudaSystem.SyncGPUAll();
    std::vector<Byte> imageData = imgMemory.GetImageToCPU(cudaSystem);

    size_t pixelCount1D = static_cast<size_t>(pixelCount[0]) * pixelCount[1];

    // Launch finished image
    if(callbacks) callbacks->SendImage(std::move(imageData),
                                       imgMemory.Format(),
                                       offset,
                                       start, end);
    SendLog("Image sent!");
}

void GPUTracer::AskParameters()
{
    if(callbacks) callbacks->SendCurrentParameters(params);
}
