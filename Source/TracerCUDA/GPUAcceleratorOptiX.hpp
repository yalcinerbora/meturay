#pragma once

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
SceneError GPUAccOptiXGroup<PGroup>::InitializeGroup(// Accelerator Option Node
                                                     const SceneNodePtr& node,
                                                     // List of surface/material
                                                     // pairings that uses this accelerator type
                                                     // and primitive type
                                                     const std::map<uint32_t, SurfaceDefinition>& surfaceList,
                                                     double time)
{
    const char* primGroupTypeName = this->primitiveGroup.Type();

    std::vector<uint32_t> hTransformIndices;
    hTransformIndices.reserve(surfaceList.size());

    // Iterate over pairings
    int surfaceInnerId = 0;
    for(const auto& surface : surfaceList)
    {
        PrimitiveRangeList primRangeList;
        HitKeyList hitKeyList;
        primRangeList.fill(Vector2ul(std::numeric_limits<uint64_t>::max()));
        hitKeyList.fill(HitKey::InvalidKey);

        const IdKeyPairs& pList = surface.second.primIdWorkKeyPairs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const IdKeyPair& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max())
                break;

            primRangeList[i] = this->primitiveGroup.PrimitiveBatchRange(p.first);
            hitKeyList[i] = p.second;
        }

        hTransformIndices.push_back(surface.second.globalTransformIndex);
        primitiveRanges.push_back(primRangeList);
        primitiveMaterialKeys.push_back(hitKeyList);
        idLookup.emplace(surface.first, surfaceInnerId);
        keyExpandOption.push_back(surface.second.doKeyExpansion);
        surfaceInnerId++;
    }
    assert(surfaceInnerId == hTransformIndices.size());

    // Copy transform ids to GPU Memory
    transformIdMemory = DeviceMemory(sizeof(TransformId) * hTransformIndices.size());
    dAccTransformIds = static_cast<TransformId*>(transformIdMemory);
    CUDA_CHECK(cudaMemcpy(dAccTransformIds,
                          hTransformIndices.data(),
                          transformIdMemory.Size(),
                          cudaMemcpyHostToDevice));

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
    // Use bestGPU for temp data creation
    const CudaGPU& gpu = system.BestGPU();

    uint64_t totalPrimitiveCount = this->primitiveGroup.TotalPrimitiveCount();
    // Currently optix only supports 32-bit indices
    if(totalPrimitiveCount >= std::numeric_limits<uint32_t>::max())
        return TracerError::UNABLE_TO_CONSTRUCT_ACCELERATOR;

    using LeafData = typename GPUPrimitiveTriangle::LeafData;
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    uint32_t innerIndex = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[innerIndex];
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

    // For this batch create a temp AABB Buffer
    size_t currentOffset = 0;
    PrimitiveRangeList indexOffsets;
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& range = primRangeList[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        indexOffsets[i][0] = currentOffset;
        currentOffset += (range[1] - range[0]);
        indexOffsets[i][1] = currentOffset;
    }
    size_t totalPrimCount = currentOffset;

    // Allocate & Generate AABBs
    AABB3f* dPrimAABBs;
    uint64_t* dPrimIds;
    DeviceMemory aabbTempBuffer;
    DeviceMemory::AllocateMultiData(std::tie(dPrimAABBs, dPrimIds), aabbTempBuffer,
                                    {totalPrimCount, totalPrimCount});

    size_t indexOffset = 0;
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& range = primRangeList[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        uint32_t indexCount = static_cast<uint32_t>(range[1] - range[0]);
         // Generate AABBs
        gpu.GridStrideKC_X(0, 0, totalPrimCount,
                           //
                           KCGenAABBs<PGroup>,
                           //
                           dPrimAABBs + indexOffset,
                           //
                           range,
                           //
                           AABBGen<PGroup>(primData, *transform),
                           static_cast<uint32_t>(totalPrimCount));

        indexOffset += indexCount;
    }
    gpu.WaitMainStream();

    //===============================//
    //  ACTUAL TRAVERSAL GENERATION  //
    //===============================//
    uint32_t deviceIndex = 0;
    for(auto [gpu, optixContext] : optixSystem->OptixCapableDevices())
    {
        DeviceTraversables& traversableData = optixTraverseMemory[deviceIndex];

        // Constant Params
        const uint32_t geometryFlag = OPTIX_GEOMETRY_FLAG_NONE;
        // Generate Build Params
        uint32_t buildInputCount = 0;
        std::array<OptixBuildInput, SceneConstants::MaxPrimitivePerSurface> buildInputs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            buildInputCount++;

            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;
            uint32_t primCount = static_cast<uint32_t>(range[1] - range[0]);

            CUdeviceptr aabbBuffers[1] = {AsOptixPtr(dPrimAABBs + indexOffsets[i][0])};

            // Set Input Params
            OptixBuildInput& buildInput = buildInputs[i];
            buildInput = OptixBuildInput{};
            buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            //
            buildInput.customPrimitiveArray.aabbBuffers = aabbBuffers;
            buildInput.customPrimitiveArray.numPrimitives = primCount;
            buildInput.customPrimitiveArray.strideInBytes = sizeof(AABB3f);
            buildInput.customPrimitiveArray.primitiveIndexOffset = static_cast<uint32_t>(range[0]);
            // SBT
            buildInput.triangleArray.flags = &geometryFlag;
            buildInput.triangleArray.numSbtRecords = 1;
            buildInput.triangleArray.sbtIndexOffsetBuffer = 0;
            buildInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
            buildInput.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
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

        Byte* dTempBuild;
        uint64_t* dCompactedSize;
        DeviceMemory tempBuildBuffer;
        DeviceMemory::AllocateMultiData(std::tie(dTempBuild, dCompactedSize), tempBuildBuffer,
                                        {accelMemorySizes.outputSizeInBytes, 1}, 128);
        DeviceMemory tempMem(accelMemorySizes.tempSizeInBytes);

        // While building fetch compacted output size
        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = AsOptixPtr(dCompactedSize);

        OptixTraversableHandle traversable;
        OPTIX_CHECK(optixAccelBuild(optixContext, (cudaStream_t)0,
                                    &accelOptions,
                                    buildInputs.data(),
                                    buildInputCount,
                                    AsOptixPtr(tempMem), tempMem.Size(),
                                    AsOptixPtr(dTempBuild),
                                    accelMemorySizes.outputSizeInBytes,
                                    &traversable, &emitProperty, 1));

        // Get compacted size to CPU
        uint64_t hCompactAccelSize;
        CUDA_CHECK(cudaMemcpy(&hCompactAccelSize, dCompactedSize,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));

        if(hCompactAccelSize < accelMemorySizes.outputSizeInBytes)
        {
            DeviceMemory compactedMemory(hCompactAccelSize);

            // use handle as input and output
            OPTIX_CHECK(optixAccelCompact(optixContext, (cudaStream_t)0,
                                          traversable,
                                          AsOptixPtr(compactedMemory),
                                          hCompactAccelSize,
                                          &traversable));

            traversableData.tMemories[innerIndex] = std::move(compactedMemory);
        }
        else
            traversableData.tMemories[innerIndex] = std::move(tempBuildBuffer);

        traversableData.traversables[innerIndex] = traversable;
        deviceIndex++;
    }

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
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerators(const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerator(uint32_t surface,
                                                         const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                          const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
size_t GPUAccOptiXGroup<PGroup>::UsedGPUMemory() const
{
    // TODO:
    return 0;
}

template <class PGroup>
size_t GPUAccOptiXGroup<PGroup>::UsedCPUMemory() const
{
    // TODO:
    // Write allocator wrapper for which keeps track of total bytes allocated
    // and deallocated
    return 0;
}

template <class PGroup>
void GPUAccOptiXGroup<PGroup>::Hit(const CudaGPU& gpu,
                                 // O
                                   HitKey* dMaterialKeys,
                                   TransformId* dTransformIds,
                                   PrimitiveId* dPrimitiveIds,
                                   HitStructPtr dHitStructs,
                                   // I-O
                                   RayGMem* dRays,
                                   // Input
                                   const RayId* dRayIds,
                                   const HitKey* dAcceleratorKeys,
                                   const uint32_t rayCount) const
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
void GPUAccOptiXGroup<PGroup>::SetOptiXSystem(const OptiXSystem* sys)
{
    optixSystem = sys;
}