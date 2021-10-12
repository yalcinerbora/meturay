template <class PGroup>
__global__ void KCConstructGPULight(GPULight<PGroup>* gLightLocations,
                                    const PrimitiveId* gPrimitiveIds,
                                    //
                                    const TextureRefI<2, Vector3f>** gRads,
                                    const TransformId* gTransformIds,
                                    const uint16_t* gMediumIndices,
                                    //
                                    const typename PGroup::PrimitiveData& pData,
                                    const GPUTransformI** gTransforms,
                                    uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        TransformId tId = gTransformIds[globalId];
        new (gLightLocations + globalId) GPULight<PGroup>(pData,
                                                          gPrimitiveIds[globalId],
                                                          // Base Class
                                                          *gRads[globalId],
                                                          gMediumIndices[globalId],
                                                          *gTransforms[tId]);
    }
}

template <class PGroup>
SceneError CPULightGroup<PGroup>::InitializeGroup(const EndpointGroupDataList& lightNodes,
                                                  const TextureNodeMap& textures,
                                                  const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                  const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                  uint32_t batchId, double time,
                                                  const std::string& scenePath)
{
    lightCount = static_cast<uint32_t>(lightNodes.size());
    hMediumIds.reserve(lightCount);
    hPrimitiveIds.reserve(lightCount);
    hTransformIds.reserve(lightCount);

    uint32_t innerId = 0;
    for(const auto& node : lightNodes)
    {
        uint32_t primitiveId = node.constructionId;

        // Convert Ids to inner index
        Vector2ul primitiveRange = primGroup.PrimitiveBatchRange(primitiveId);
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        
        lightCount += static_cast<uint32_t>(primitiveRange[1] - primitiveRange[0]);
        for(PrimitiveId primId = primitiveRange[0]; primId < primitiveRange[1]; primId++)
        {
            HitKey materialKey = HitKey::CombinedKey(batchId, innerId);

            // Load to host memory
            hMediumIds.push_back(mediumIndex);
            hPrimitiveIds.push_back(primId);
            hTransformIds.push_back(transformIndex);

            innerId++;
        }
    }

    // Allocate Data
    DeviceMemory::AllocateMultiData(std::tie(dGPULights, dPData),
                                    gpuMemory, {lightCount, 1});

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
TracerError CPULightGroup<PGroup>::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
                                                      const CudaSystem&)
{
    // Gen Temporary Memory
    DeviceMemory tempMemory;

    const PrimitiveId* dPrimitiveIds;
    const TransformId* dTransformIds;
    const uint16_t* dMediumIndices;
    DeviceMemory::AllocateMultiData(std::tie(dPrimitiveIds,
                                             dTransformIds,
                                             dMediumIndices),
                                    tempMemory,
                                    {lightCount, lightCount, lightCount});
    // Set a GPU    
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Load Data to Temp Memory    
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
                       lightCount,
                       //
                       KCConstructGPULight<PGroup>,
                       //
                       const_cast<GPULight<PGroup>*>(dGPULights),
                       dPrimitiveIds,
                       //
                       dRadiances,
                       dTransformIds,
                       dMediumIndices,

                       *dPData,
                       dGlobalTransformArray,
                       lightCount);

    gpu.WaitMainStream();

    SetLightList();

    return TracerError::OK;
}