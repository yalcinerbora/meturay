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
    return TracerError::OK;
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
    accRanges.clear();
    primitiveRanges.clear();
    primitiveMaterialKeys.clear();
    idLookup.clear();

    const char* primGroupTypeName = this->primitiveGroup.Type();

    std::vector<uint32_t> hTransformIndices;
    hTransformIndices.reserve(surfaceList.size());

    // Iterate over pairings
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

        size_t localSize = 0;
        const IdKeyPairs& pList = surface.second.primIdWorkKeyPairs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const IdKeyPair& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max()) break;

            primIdList[i] = p.first;
            primRangeList[i] = this->primitiveGroup.PrimitiveBatchRange(p.first);
            hitKeyList[i] = p.second;
            localSize += primRangeList[i][1] - primRangeList[i][0];
        }
        range[1] = range[0] + localSize;
        totalSize += localSize;


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
    TracerError err = TracerError::OK;
    if((err = FillLeaves(system, surface)) != TracerError::OK)
        return err;

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
    GPUMemFuncs::AllocateMultiData(std::tie(dPrimAABBs, dPrimIds), aabbTempBuffer,
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
    for(const auto& [gpu, optixContext] : optixSystem->OptixCapableDevices())
    {
        DeviceTraversables& gpuTraverseData = optixDataPerGPU[deviceIndex];

        // Constant Params
        const uint32_t geometryFlag = OPTIX_GEOMETRY_FLAG_NONE;
        // Generate Build Params
        uint32_t buildInputCount = 0;
        std::array<OptixBuildInput, SceneConstants::MaxPrimitivePerSurface> buildInputs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;
            buildInputCount++;

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
        Byte* dTempMem;
        DeviceLocalMemory tempBuildBuffer(&gpu);
        GPUMemFuncs::AllocateMultiData(std::tie(dTempBuild, dCompactedSize), tempBuildBuffer,
                                       {accelMemorySizes.outputSizeInBytes, 1}, 128);
        DeviceLocalMemory tempMem(&gpu, accelMemorySizes.tempSizeInBytes);
        dTempMem = static_cast<Byte*>(tempMem);

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
                                    AsOptixPtr(dTempMem), accelMemorySizes.tempSizeInBytes,
                                    // Output Memory
                                    AsOptixPtr(dTempBuild), accelMemorySizes.outputSizeInBytes,
                                    &traversable, &emitProperty, 1));
        CUDA_KERNEL_CHECK();

        // Get compacted size to CPU
        uint64_t hCompactAccelSize;
        CUDA_CHECK(cudaMemcpy(&hCompactAccelSize, dCompactedSize,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));

        if(hCompactAccelSize < accelMemorySizes.outputSizeInBytes)
        {
            DeviceLocalMemory compactedMemory(&gpu, hCompactAccelSize);

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
            gpuTraverseData.tMemories[innerIndex] = std::move(tempBuildBuffer);

        gpuTraverseData.traversables[innerIndex] = traversable;
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