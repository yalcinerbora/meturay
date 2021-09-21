template <class PGroup>
__global__ void KCConstructGPULight(GPULight<PGroup>* gLightLocations,
                                    //
                                    const TransformId* gTransformIds,
                                    const PrimitiveId* gPrimitiveIds,
                                    const uint16_t* gMediumIndices,
                                    const HitKey* gLightMaterialIds,
                                    //
                                    const typename PGroup::PrimitiveData& pData,
                                    const GPUTransformI** gTransforms,
                                    uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gLightLocations + globalId) GPULight<PGroup>(gLightMaterialIds[globalId],
                                                          gMediumIndices[globalId],
                                                          gPrimitiveIds[globalId],
                                                          *gTransforms[gTransformIds[globalId]],
                                                          pData);
    }
}

template <class PGroup>
SceneError CPULightGroup<PGroup>::InitializeGroup(const LightGroupDataList& lightNodes,
                                                  const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                  const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                  const MaterialKeyListing& allMaterialKeys,
                                                  double time,
                                                  const std::string& scenePath)
{
    lightCount = static_cast<uint32_t>(lightNodes.size());
    hHitKeys.reserve(lightCount);
    hMediumIds.reserve(lightCount);
    hPrimitiveIds.reserve(lightCount);
    hTransformIds.reserve(lightCount);

    lightCount = 0;
    for(const auto& node : lightNodes)
    {
        uint32_t primitiveId = node.constructionId;

        // Convert Ids to inner index
        Vector2ul primitiveRange = primGroup.PrimitiveBatchRange(primitiveId);
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        HitKey materialKey = allMaterialKeys.at(std::make_pair(primGroup.Type(), node.materialId));

        lightCount += static_cast<uint32_t>(primitiveRange[1] - primitiveRange[0]);
        for(PrimitiveId primId = primitiveRange[0]; primId < primitiveRange[1]; primId++)
        {
            // Load to host memory
            hHitKeys.push_back(materialKey);
            hMediumIds.push_back(mediumIndex);
            hPrimitiveIds.push_back(primId);
            hTransformIds.push_back(transformIndex);
        }
    }

    // Allocate for GPULight classes
    size_t totalClassSize = sizeof(GPULight<PGroup>) * lightCount;
    totalClassSize = Memory::AlignSize(totalClassSize);
    size_t totalPDataSize = sizeof(PData);
    totalPDataSize = Memory::AlignSize(totalPDataSize);

    DeviceMemory::EnlargeBuffer(memory, (totalClassSize + totalPDataSize));

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dGPULights = reinterpret_cast<const GPULight<PGroup>*>(dBasePtr + offset);
    offset += totalClassSize;
    dPData = reinterpret_cast<const PData*>(dBasePtr + offset);
    offset += totalPDataSize;
    assert((totalClassSize + totalPDataSize) == offset);

    // Get PData from Primitive Group
    const PData primData = PrimDataAccessor::Data(primGroup);
    CUDA_CHECK(cudaMemcpy(const_cast<PData*>(dPData), &primData,
                          sizeof(PData), cudaMemcpyHostToDevice));

    return SceneError::OK;
}

template <class PGroup>
SceneError CPULightGroup<PGroup>::ChangeTime(const NodeListing& lightNodes, double time,
                                             const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

template <class PGroup>
TracerError CPULightGroup<PGroup>::ConstructLights(const CudaSystem& system,
                                                   const GPUTransformI** dGlobalTransformArray,
                                                   const KeyMaterialMap&)
{
    // Gen Temporary Memory
    DeviceMemory tempMemory;
    // Allocate for GPULight classes
    size_t matKeySize = sizeof(HitKey) * lightCount;
    matKeySize = Memory::AlignSize(matKeySize);
    size_t mediumSize = sizeof(uint16_t) * lightCount;
    mediumSize = Memory::AlignSize(mediumSize);
    size_t primIdSize = sizeof(PrimitiveId) * lightCount;
    primIdSize = Memory::AlignSize(primIdSize);
    size_t transformIdSize = sizeof(TransformId) * lightCount;
    transformIdSize = Memory::AlignSize(transformIdSize);

    size_t totalSize = (matKeySize +
                        mediumSize +
                        primIdSize +
                        transformIdSize);
    DeviceMemory::EnlargeBuffer(tempMemory, totalSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(tempMemory);
    const HitKey* dLightMaterialIds = reinterpret_cast<const HitKey*>(dBasePtr + offset);
    offset += matKeySize;
    const uint16_t* dMediumIndices = reinterpret_cast<const uint16_t*>(dBasePtr + offset);
    offset += mediumSize;
    const PrimitiveId* dPrimitiveIds = reinterpret_cast<const PrimitiveId*>(dBasePtr + offset);
    offset += primIdSize;
    const TransformId* dTransformIds = reinterpret_cast<const TransformId*>(dBasePtr + offset);
    offset += transformIdSize;
    assert(totalSize == offset);

    // Set a GPU
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Load Data to Temp Memory
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dLightMaterialIds),
                          hHitKeys.data(),
                          sizeof(HitKey) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
                          hMediumIds.data(),
                          sizeof(uint16_t) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<PrimitiveId*>(dPrimitiveIds),
                          hPrimitiveIds.data(),
                          sizeof(PrimitiveId) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<TransformId*>(dTransformIds),
                          hTransformIds.data(),
                          sizeof(TransformId) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       LightCount(),
                       //
                       KCConstructGPULight<PGroup>,
                       //
                       const_cast<GPULight<PGroup>*>(dGPULights),
                       //
                       dTransformIds,
                       dPrimitiveIds,
                       dMediumIndices,
                       dLightMaterialIds,

                       *dPData,
                       dGlobalTransformArray,
                       LightCount());

    gpu.WaitMainStream();

    // Generate transform list
    for(uint32_t i = 0; i < LightCount(); i++)
    {
        const auto* ptr = static_cast<const GPULightI*>(dGPULights + i);
        gpuLightList.push_back(ptr);
    }
    return TracerError::OK;
}