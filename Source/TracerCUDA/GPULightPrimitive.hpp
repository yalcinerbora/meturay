template <class PGroup>
__global__ void KCConstructGPULight(GPULight<PGroup>* gLightLocations,
                                    const PrimitiveId* gPrimitiveIds,
                                    //
                                    const TextureRefI<2, Vector3f>** gRads,
                                    const uint16_t* gMediumIndices,
                                    const HitKey* gWorkKeys,
                                    const TransformId* gTransformIds,
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
                                                          gWorkKeys[globalId],
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
    using namespace TextureFunctions;
    SceneError err = SceneError::OK;

    // Determine Light Count
    // we will create light for each endpoint specified
    for(const auto& node : lightNodes)
    {
        uint32_t primitiveId = node.primitiveId;
        Vector2ul primitiveRange = primGroup.PrimitiveBatchRange(primitiveId);

        lightCount += static_cast<uint32_t>(primitiveRange[1] - primitiveRange[0]);
    }

    hPrimitiveIds.reserve(lightCount);
    hMediumIds.reserve(lightCount);
    hTransformIds.reserve(lightCount);
    hWorkKeys.reserve(lightCount);
    hRadianceConstructionInfo.reserve(lightCount);
    textureIdList.reserve(lightCount);

    texDataCount = 0;
    constDataCount = 0;

    uint32_t innerIndex = 0;
    for(const auto& node : lightNodes)
    {
        TexturedDataNode<Vector3> radianceNode = node.node->CommonTexturedDataVector3(RADIANCE_NAME);

        typename Base::ConstructionInfo constructionInfo;
        constructionInfo.isConstData = !radianceNode.isTexture;
        if(radianceNode.isTexture)
        {
            const TextureI<2>* tex;
            if((err = AllocateTexture(tex,
                                      dTextureMemory,
                                      radianceNode.texNode,
                                      textures,
                                      EdgeResolveType::WRAP,
                                      InterpolationType::LINEAR,
                                      true, true,
                                      gpu, scenePath)) != SceneError::OK)
                return err;
            constructionInfo.tex = static_cast<cudaTextureObject_t>(*tex);
            texDataCount++;
        }
        else
        {
            constructionInfo.data = radianceNode.data;
            constructionInfo.tex = 0;
            constDataCount++;
        }

        uint32_t texId = radianceNode.isTexture
                            ? radianceNode.texNode.texId
                            : std::numeric_limits<uint32_t>::max();

        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        uint32_t primitiveId = node.primitiveId;
        Vector2ul primitiveRange = primGroup.PrimitiveBatchRange(primitiveId);

        for(PrimitiveId primId = primitiveRange[0]; primId < primitiveRange[1]; primId++)
        {
            HitKey materialKey = HitKey::CombinedKey(batchId, innerIndex);

            // Load to host memory
            hMediumIds.push_back(mediumIndex);
            hPrimitiveIds.push_back(primId);
            hTransformIds.push_back(transformIndex);
            hWorkKeys.push_back(HitKey::CombinedKey(batchId, innerIndex));
            hRadianceConstructionInfo.push_back(constructionInfo);
            textureIdList.push_back(texId);

            innerIndex++;

            if(innerIndex >= (1 << HitKey::IdBits))
                return SceneError::TOO_MANY_MATERIAL_IN_GROUP;
        }
    }
    // Allocate data for texture references etc...
    DeviceMemory::AllocateMultiData(std::tie(dGPULights, dConstantRadiance,
                                             dTextureRadiance, dRadiances),
                                    gpuLightMemory,
                                    {lightCount, constDataCount,
                                     texDataCount, lightCount});

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
    TracerError e = TracerError::OK;
    // Construct Texture References
    if((e = ConstructTextureReferences()) != TracerError::OK)
        return e;

    // Generate Temp Memory for Light list GPU Construction
    DeviceMemory tempMemory;

    const PrimitiveId*  dPrimitiveIds;
    const TransformId*  dTransformIds;
    const uint16_t*     dMediumIndices;
    const HitKey*       dWorkKeys;

    DeviceMemory::AllocateMultiData(std::tie(dPrimitiveIds,
                                             dTransformIds,
                                             dMediumIndices,
                                             dWorkKeys),
                                    tempMemory,
                                    {lightCount, lightCount, lightCount, lightCount});
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
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dWorkKeys),
                          hWorkKeys.data(),
                          sizeof(HitKey) * lightCount,
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
                       dMediumIndices,
                       dWorkKeys,
                       dTransformIds,
                       //
                       *dPData,
                       dGlobalTransformArray,
                       lightCount);

    gpu.WaitMainStream();

    SetLightLists();

    return TracerError::OK;
}