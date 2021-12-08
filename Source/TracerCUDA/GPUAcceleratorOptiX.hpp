#pragma once

#include "TracerDebug.h"

template <class PGroup>
GPUAccOptiXGroup<PGroup>::GPUAccOptiXGroup(const GPUPrimitiveGroupI& pGroup)
    : GPUAcceleratorGroup<PGroup>(pGroup)
    , optixSystem(nullptr)
{}

template <class PGroup>
const char* GPUAccOptiXGroup<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::FillLeaves(const CudaSystem& system,
                                                 uint32_t surfaceId)
{
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    const uint32_t index = idLookup.at(surfaceId);
    //const Vector2ul& accelRange = acceleratorRanges[index];
    const PrimitiveRangeList& rangeList = primitiveRanges[index];
    const HitKeyList& hitList = primitiveMaterialKeys[index];

    // Copy Locally to stack to send it to const memory of the GPU
    HKList hkList = {{}};
    PRList prList = {{}};
    std::memcpy(const_cast<HitKey*>(hkList.materialKeys), hitList.data(),
                sizeof(HitKey) * SceneConstants::MaxPrimitivePerSurface);
    std::memcpy(const_cast<Vector2ul*>(prList.primRanges), rangeList.data(),
                sizeof(Vector2ul) * SceneConstants::MaxPrimitivePerSurface);

    size_t workCount = accRanges[index][1] - accRanges[index][0];

    // TODO: Select a GPU
    const CudaGPU& gpu = system.BestGPU();
    // KC
    gpu.AsyncGridStrideKC_X
    (
        0,
        workCount,
        //
        KCInitializeLeafs<PGroup>,
        // Args
        // O
        dLeafList,
        // Input
        dAccRanges,
        hkList,
        prList,
        primData,
        index,
        keyExpandOption[index]
    );

    //Debug::DumpMemToFile("leafList" + std::to_string(index), dLeafList, workCount);

    return TracerError::OK;
}

template <class PGroup>
SceneError GPUAccOptiXGroup<PGroup>::InitializeGroup(// Accelerator Option Node
                                                     const SceneNodePtr&,
                                                     // List of surface/material
                                                     // pairings that uses this accelerator type
                                                     // and primitive type
                                                     const std::map<uint32_t, SurfaceDefinition>& surfaceList,
                                                     double)
{
    accRanges.clear();
    primitiveRanges.clear();
    primitiveMaterialKeys.clear();
    idLookup.clear();
    // PG Type
    //const char* primGroupTypeName = this->primitiveGroup.Type();
    // Transform Indices
    std::vector<uint32_t> hTransformIndices;
    hTransformIndices.reserve(surfaceList.size());
    // List of Leaf Offsets
    std::vector<size_t> sbtLeafOffsets;
    // SBT offset for each accelerator
    sbtCounts.reserve(surfaceList.size());

    // Iterate over pairings
    size_t sbtLeafOffset = 0;
    int surfaceInnerId = 0;
    size_t totalSize = 0;
    for(const auto& surface : surfaceList)
    {
        PrimitiveIdList primIdList;
        PrimitiveRangeList primRangeList;
        HitKeyList hitKeyList;
        primRangeList.fill(Vector2ul(std::numeric_limits<uint64_t>::max()));
        primIdList.fill(std::numeric_limits<uint32_t>::max());
        hitKeyList.fill(HitKey::InvalidKey);
        Vector2ul range = Vector2ul(totalSize, 0);

        uint32_t primBatchCount = 0;
        size_t localSize = 0;
        const IdKeyPairs& pList = surface.second.primIdWorkKeyPairs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const IdKeyPair& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max()) break;

            primIdList[i] = p.first;
            primRangeList[i] = this->primitiveGroup.PrimitiveBatchRange(p.first);
            hitKeyList[i] = p.second;

            size_t subLocalSize = primRangeList[i][1] - primRangeList[i][0];
            localSize += subLocalSize;
            // Save the SBT leaf offset as well
            sbtLeafOffsets.push_back(sbtLeafOffset);
            sbtLeafOffset += subLocalSize;

            sbtRecords.push_back(Record<void, void>
            {
                nullptr,
                surface.second.globalTransformIndex,
                nullptr
            });
            primBatchCount++;
        }
        range[1] = range[0] + localSize;
        totalSize += localSize;

        sbtCounts.push_back(primBatchCount);
        hTransformIndices.push_back(surface.second.globalTransformIndex);
        primitiveIds.push_back(primIdList);
        primitiveRanges.push_back(primRangeList);
        primitiveMaterialKeys.push_back(hitKeyList);
        accRanges.push_back(range);
        keyExpandOption.push_back(surface.second.doKeyExpansion);
        idLookup.emplace(surface.first, surfaceInnerId);
        surfaceInnerId++;
    }
    assert(primitiveRanges.size() == primitiveMaterialKeys.size());
    assert(primitiveMaterialKeys.size() == idLookup.size());
    assert(idLookup.size() == accRanges.size());
    assert(keyExpandOption.size() == idLookup.size());
    assert(surfaceInnerId == hTransformIndices.size());
    assert(sbtLeafOffset == totalSize);

    uint32_t accelCount = surfaceInnerId;
    leafCount = totalSize;
    GPUMemFuncs::AllocateMultiData(std::tie(dAccRanges, dAccTransformIds,
                                            dLeafList, dPrimData), memory,
                                   {accelCount, accelCount,
                                   leafCount, 1});

    // Copy Leaf counts to cpu memory
    CUDA_CHECK(cudaMemcpy(dAccRanges, accRanges.data(),
                          accelCount * sizeof(Vector2ul),
                          cudaMemcpyHostToDevice));
    // Copy Transforms
    CUDA_CHECK(cudaMemcpy(dAccTransformIds,
                          hTransformIndices.data(),
                          accelCount * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    // Copy Prim Data
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);
    CUDA_CHECK(cudaMemcpy(dPrimData, &primData,
                          sizeof(PrimitiveData),
                          cudaMemcpyHostToDevice));

    // After Ptrs Are Determined
    // Set pointers on the records
    uint32_t i = 0;
    for(auto& record : sbtRecords)
    {
        record.gLeafs = dLeafList + sbtLeafOffsets[i];
        record.gPrimData = dPrimData;
        i++;
    }

    return SceneError::OK;
}

template <class PGroup>
uint32_t GPUAccOptiXGroup<PGroup>::InnerId(uint32_t surfaceId) const
{
    return idLookup.at(surfaceId);
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::ConstructAccelerators(const CudaSystem& system)
{
    // Copy Prim Data
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);
    CUDA_CHECK(cudaMemcpy(dPrimData, &primData,
                          sizeof(PrimitiveData),
                          cudaMemcpyHostToDevice));

    // TODO: make this a single KC
    TracerError e = TracerError::OK;
    for(const auto& id : idLookup)
    {
        if((e = ConstructAccelerator(id.first, system)) != TracerError::OK)
            return e;
    }
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::ConstructAccelerator(uint32_t surface,
                                                           const CudaSystem& system)
{
    Utility::CPUTimer t;
    t.Start();

    TracerError err = TracerError::OK;
    if((err = FillLeaves(system, surface)) != TracerError::OK)
        return err;

    // Use bestGPU for temp data creation
    const CudaGPU& gpu = system.BestGPU();

    uint64_t totalPrimitiveCount = this->primitiveGroup.TotalPrimitiveCount();
    // Currently optix only supports 32-bit indices
    if(totalPrimitiveCount >= std::numeric_limits<uint32_t>::max())
        return TracerError::UNABLE_TO_CONSTRUCT_ACCELERATOR;

    using LeafData = typename PGroup::LeafData;
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    uint32_t innerIndex = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[innerIndex];
    const PrimitiveIdList& primIdList = primitiveIds[innerIndex];
    const PrimTransformType tType = this->primitiveGroup.TransformType();

    // Select Transform for construction
    const GPUTransformI* worldTransform = nullptr;
    AcquireAcceleratorGPUTransform(worldTransform,
                                   dAccTransformIds,
                                   this->dTransforms,
                                   innerIndex,
                                   gpu);
    const GPUTransformI* transform = worldTransform;
    if(tType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
    {
        AcquireIdentityTransform(transform,
                                 this->dTransforms,
                                 this->identityTransformIndex,
                                 gpu);
    }

    // Use our AABB if it matches
    static_assert(sizeof(AABB3f) == sizeof(OptixAabb), "OptixAabb != AABB3f");
    static_assert((sizeof(AABB3f) % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT) == 0, "AABB3f is not aligned");

    //===============================//
    //  ACTUAL TRAVERSAL GENERATION  //
    //===============================//
    uint32_t deviceIndex = 0;
    for(const auto& [optixGPU, optixContext] : optixSystem->OptixCapableDevices())
    {
        CUDA_CHECK(cudaSetDevice(optixGPU.DeviceId()));
        DeviceTraversables& gpuTraverseData = optixDataPerGPU[deviceIndex];

        // IMPORTANT
        // Optix wants the triangles & Indices to be in a memory
        // that is allocated with a "cudaMalloc" function
        // Current METUray manages its memory using "cudaMallocManaged"
        // for common memory except textures (which uses cudaArray allocation and
        // it is not multi-device capable anyway)
        // We need to copy data to device local memory for construction
        // after that traversable does not refer to these memory so we can
        // delete
        size_t surfacePrimCount = 0;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;
            // Use primitive count as vertex count
            surfacePrimCount += range[1] - range[0];
        }

        DeviceLocalMemory aabbTempBuffer(&gpu, sizeof(AABB3f) * surfacePrimCount);
        AABB3f* dPrimAABBs = static_cast<AABB3f*>(aabbTempBuffer);

        size_t offset = 0;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;

            uint32_t primCount = static_cast<uint32_t>(range[1] - range[0]);
             // Generate AABBs
            optixGPU.GridStrideKC_X(0, 0, primCount,
                                    //
                                    KCGenAABBs<PGroup>,
                                    //
                                    dPrimAABBs + offset,
                                    //
                                    range,
                                    //
                                    AABBGen<PGroup>(primData, *transform),
                                    static_cast<uint32_t>(primCount));

            offset += surfacePrimCount;
        }
        assert(offset == surfacePrimCount);

        // Generate Build Params
        offset = 0;
        uint32_t buildInputCount = 0;
        std::array<uint32_t, SceneConstants::MaxPrimitivePerSurface> geometryFlags;
        std::array<CUdeviceptr, SceneConstants::MaxPrimitivePerSurface> aabbPtrs;
        std::array<OptixBuildInput, SceneConstants::MaxPrimitivePerSurface> buildInputs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;
            buildInputCount++;

            // Gen AABB Ptr
            aabbPtrs[i] = AsOptixPtr(dPrimAABBs + offset);
            size_t localPrimCount = range[1] - range[0];
            offset += localPrimCount;

            // Enable/Disable Any hit if batch has alpha map
            geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            if(this->primitiveGroup.PrimitiveBatchHasAlphaMap(primIdList[i]))
                geometryFlags[i] = OPTIX_GEOMETRY_FLAG_NONE;

            OptixBuildInput& buildInput = buildInputs[i];
            buildInput = {};
            buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            // AABB
            buildInput.customPrimitiveArray.aabbBuffers = aabbPtrs.data() + i;
            buildInput.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(localPrimCount);
            buildInput.customPrimitiveArray.strideInBytes = sizeof(AABB3f);
            buildInput.customPrimitiveArray.primitiveIndexOffset = 0;
            // SBT
            buildInput.customPrimitiveArray.flags = geometryFlags.data() + i;
            buildInput.customPrimitiveArray.numSbtRecords = 1;
            buildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
            buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
            buildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
        }

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes accelMemorySizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
        (
            optixContext,
            &accelOptions, buildInputs.data(),
            buildInputCount, &accelMemorySizes
        ));

        // Allocate Temp Buffer for Build
        DeviceLocalMemory buildBuffer(&optixGPU, accelMemorySizes.outputSizeInBytes);
        Byte* dTempBuild = static_cast<Byte*>(buildBuffer);
        Byte* dTemp;
        uint64_t* dCompactedSize;
        DeviceLocalMemory tempMemory(&optixGPU);
        GPUMemFuncs::AllocateMultiData(std::tie(dTemp, dCompactedSize), tempMemory,
                                       {accelMemorySizes.tempSizeInBytes, 1}, 128);

        // While building fetch compacted output size
        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = AsOptixPtr(dCompactedSize);

        OptixTraversableHandle traversable;
        OPTIX_CHECK(optixAccelBuild(optixContext, (cudaStream_t)0,
                                    &accelOptions,
                                    // Build Inputs
                                    buildInputs.data(), buildInputCount,
                                    // Temp Memory
                                    AsOptixPtr(dTemp), accelMemorySizes.tempSizeInBytes,
                                    // Output Memory
                                    AsOptixPtr(dTempBuild), accelMemorySizes.outputSizeInBytes,
                                    &traversable, &emitProperty, 1));
        CUDA_KERNEL_CHECK();

        // Get compacted size to CPU
        uint64_t hCompactAccelSize;
        CUDA_CHECK(cudaMemcpy(&hCompactAccelSize, dCompactedSize,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));

        if(hCompactAccelSize < buildBuffer.Size())
        {
            DeviceLocalMemory compactedMemory(&optixGPU, hCompactAccelSize);

            // use handle as input and output
            OPTIX_CHECK(optixAccelCompact(optixContext, (cudaStream_t)0,
                                          traversable,
                                          AsOptixPtr(compactedMemory),
                                          hCompactAccelSize,
                                          &traversable));
            CUDA_KERNEL_CHECK();

            gpuTraverseData.tMemories[innerIndex] = std::move(compactedMemory);
        }
        else
            gpuTraverseData.tMemories[innerIndex] = std::move(buildBuffer);

        gpuTraverseData.traversables[innerIndex] = traversable;
        deviceIndex++;

        // Before ending the scope, wait for the build and compact operations
        // Because we will de-alloc the unused memories
        // (temp vertex buffer & buildBuffer(if traversal is compacted))
        gpu.WaitMainStream();

    }
    t.Stop();
    METU_LOG("Surface{:d} OptiX GAS generated in {:f} seconds.",
             surface, t.Elapsed<CPUTimeSeconds>());
    // All Done!
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                            const CudaSystem& system)
{
    TracerError e = TracerError::OK;
    for(const uint32_t& id : surfaces)
    {
        auto it = idLookup.cend();
        if((it = idLookup.find(id)) == idLookup.cend()) continue;

        if((e = ConstructAccelerator(it->second, system)) != TracerError::OK)
            return e;
    }
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerators(const CudaSystem& system)
{
    TracerError e = TracerError::OK;
    for(const auto& id : idLookup)
    {
        if((e = DestroyAccelerator(id.first, system)) != TracerError::OK)
            return e;
    }
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerator(uint32_t, const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>&,
                                                          const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
size_t GPUAccOptiXGroup<PGroup>::UsedGPUMemory() const
{
    size_t mem = 0;
    for(const auto& tData : optixDataPerGPU)
        for(const auto& tMemory : tData.tMemories)
            mem += tMemory.Size();

    mem += memory.Size();
    return mem;
}

template <class PGroup>
size_t GPUAccOptiXGroup<PGroup>::UsedCPUMemory() const
{
    return (primitiveIds.size() * sizeof(PrimitiveIdList) +
            primitiveRanges.size() * sizeof(PrimitiveRangeList) +
            primitiveMaterialKeys.size() * sizeof(HitKeyList) +
            accRanges.size() * sizeof(Vector2ul) +
            keyExpandOption.size() * sizeof(bool) +
            sbtRecords.size() * sizeof(Record<void, void>) +
            sbtCounts.size() * sizeof(uint32_t) +
            idLookup.size() * sizeof(std::pair<uint32_t, uint32_t>) +
            surfaceAABBs.size() * sizeof(std::pair<uint32_t, AABB3f>) +
            sizeof(GPUAccOptiXGroup<PGroup>));
}

template <class PGroup>
void GPUAccOptiXGroup<PGroup>::Hit(const CudaGPU&,
                                   // O
                                   HitKey*,
                                   TransformId*,
                                   PrimitiveId*,
                                   HitStructPtr,
                                   // I-O
                                   RayGMem*,
                                   // Input
                                   const RayId*,
                                   const HitKey*,
                                   const uint32_t) const
{
    // We do nothing on hit this function should not be called
    // Since base accelerator will handle entire hit operation
}

template <class PGroup>
const SurfaceAABBList& GPUAccOptiXGroup<PGroup>::AcceleratorAABBs() const
{
    return surfaceAABBs;
}

template <class PGroup>
size_t GPUAccOptiXGroup<PGroup>::AcceleratorCount() const
{
    return idLookup.size();
}

template <class PGroup>
DeviceMemory GPUAccOptiXGroup<PGroup>::GetHitRecords() const
{
    return DeviceMemory();
}

template <class PGroup>
void GPUAccOptiXGroup<PGroup>::SetOptiXSystem(const OptiXSystem* sys)
{
    optixSystem = sys;

    auto optixGPUCount = sys->OptixCapableDevices().size();
    auto acceleratorCount = idLookup.size();
    optixDataPerGPU.resize(optixGPUCount);

    uint32_t i = 0;
    for(auto& gpuData : optixDataPerGPU)
    {
        const auto& gpu = sys->OptixCapableDevices()[i].first;
        gpuData.tMemories.resize(acceleratorCount, DeviceLocalMemory(&gpu));
        gpuData.traversables.resize(acceleratorCount);
        i++;
    }
}

template <class PGroup>
GPUAccGroupOptiXI::OptixTraversableList GPUAccOptiXGroup<PGroup>::GetOptixTraversables() const
{
    OptixTraversableList result(optixDataPerGPU.size());
    uint32_t i = 0;
    for(const auto& optixData : optixDataPerGPU)
    {
        result[i] = optixData.traversables;
        i++;
    }
    return result;
}

template <class PGroup>
TransformId* GPUAccOptiXGroup<PGroup>::GetDeviceTransformIdsPtr() const
{
    return dAccTransformIds;
}

template <class PGroup>
const GPUAccGroupOptiXI::SBTCountList& GPUAccOptiXGroup<PGroup>::GetSBTCounts() const
{
    return sbtCounts;
}

template <class PGroup>
PrimTransformType GPUAccOptiXGroup<PGroup>::GetPrimitiveTransformType() const
{
    return this->primitiveGroup.TransformType();
}

template <class PGroup>
const GPUAccGroupOptiXI::RecordList& GPUAccOptiXGroup<PGroup>::GetRecords() const
{
    return sbtRecords;
}

template <class PGroup>
const std::vector<bool> GPUAccOptiXGroup<PGroup>::GetCullFlagPerAccel() const
{
    std::vector<bool> result;
    result.reserve(idLookup.size());
    for(const auto& id : idLookup)
    {
        // Here OptiX only let us to define a single cull option
        // per accelerator. METUray has definitions for each primitive batch
        // in the scene.

        // So we return "cull-face = true" iff all batches wants to be
        // culled (most of the time this should be fine, since only
        // enclosed objects probably wants their back-face culled)
        bool doCullFace = true;
        const PrimitiveIdList& primBatchIdList = primitiveIds[id.second];
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& primId = primBatchIdList[i];
            if(primId == std::numeric_limits<uint32_t>::max())
                break;
            doCullFace &= this->primitiveGroup.PrimitiveBatchBackFaceCulled(primId);
        }
        result.push_back(doCullFace);
    }
    return result;
}