template <class PGroup>
__global__ void KCConstructGPULight(GPULight<PGroup>* gLightLocations,
                                    const PrimitiveId* gPrimitiveIds,
                                    //
                                    const TextureRefI<2, Vector3f>** gRads,
                                    const uint16_t* gMediumIndices,
                                    const HitKey* gWorkKeys,
                                    const TransformId* gTransformIds,
                                    //
                                    const typename PGroup::PrimitiveData& gPData,
                                    const GPUTransformI** gTransforms,
                                    uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        TransformId tId = gTransformIds[globalId];
        new (gLightLocations + globalId) GPULight<PGroup>(gPData,
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
                                                  uint32_t batchId, double,
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

        this->lightCount += static_cast<uint32_t>(primitiveRange[1] - primitiveRange[0]);
    }

    hPrimitiveIds.reserve(this->lightCount);
    this->hMediumIds.reserve(this->lightCount);
    this->hTransformIds.reserve(this->lightCount);
    this->hWorkKeys.reserve(this->lightCount);
    this->hRadianceConstructionInfo.reserve(this->lightCount);
    this->textureIdList.reserve(this->lightCount);
    hPackedWorkKeys.reserve(lightNodes.size());
    this->texDataCount = 0;
    this->constDataCount = 0;

    uint32_t innerIndex = 0;
    for(const auto& node : lightNodes)
    {
        TexturedDataNode<Vector3> radianceNode = node.node->CommonTexturedDataVector3(this->RADIANCE_NAME);

        typename Base::ConstructionInfo constructionInfo;
        constructionInfo.isConstData = !radianceNode.isTexture;
        if(radianceNode.isTexture)
        {
            const TextureI<2>* tex;
            if((err = AllocateTexture(tex,
                                      this->dTextureMemory,
                                      radianceNode.texNode,
                                      textures,
                                      EdgeResolveType::WRAP,
                                      InterpolationType::LINEAR,
                                      true, true,
                                      this->gpu, scenePath)) != SceneError::OK)
                return err;
            constructionInfo.tex = static_cast<cudaTextureObject_t>(*tex);
        }
        else
        {
            constructionInfo.data = radianceNode.data;
            constructionInfo.tex = 0;
        }

        uint32_t texId = radianceNode.isTexture
                            ? radianceNode.texNode.texId
                            : std::numeric_limits<uint32_t>::max();

        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        uint32_t primitiveId = node.primitiveId;
        Vector2ul primitiveRange = primGroup.PrimitiveBatchRange(primitiveId);

        hPackedWorkKeys.push_back(HitKey::CombinedKey(batchId, innerIndex));
        for(PrimitiveId primId = primitiveRange[0]; primId < primitiveRange[1]; primId++)
        {
            HitKey materialKey = HitKey::CombinedKey(batchId, innerIndex);

            // Load to host memory
            this->hMediumIds.push_back(mediumIndex);
            hPrimitiveIds.push_back(primId);
            this->hTransformIds.push_back(transformIndex);
            this->hWorkKeys.push_back(HitKey::CombinedKey(batchId, innerIndex));
            this->hRadianceConstructionInfo.push_back(constructionInfo);
            this->textureIdList.push_back(texId);

            innerIndex++;
            if(radianceNode.isTexture)
                this->texDataCount++;
            else
                this->constDataCount++;

            if(innerIndex >= (1 << HitKey::IdBits))
                return SceneError::TOO_MANY_MATERIAL_IN_GROUP;
        }
    }
    // Allocate data for texture references etc...
    GPUMemFuncs::AllocateMultiData(std::tie(this->dGPULights, this->dConstantRadiance,
                                            this->dTextureRadiance, this->dRadiances,
                                            dPData),
                                   this->gpuLightMemory,
                                   {this->lightCount, this->constDataCount,
                                    this->texDataCount, this->lightCount, 1});

    return SceneError::OK;
}

template <class PGroup>
SceneError CPULightGroup<PGroup>::ChangeTime(const NodeListing&, double,
                                             const std::string&)
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
    if((e = this->ConstructTextureReferences()) != TracerError::OK)
        return e;

    // Generate Temp Memory for Light list GPU Construction
    DeviceMemory tempMemory;
    const PrimitiveId*  dPrimitiveIds;
    const TransformId*  dTransformIds;
    const uint16_t*     dMediumIndices;
    const HitKey*       dWorkKeys;

    GPUMemFuncs::AllocateMultiData(std::tie(dPrimitiveIds,
                                            dTransformIds,
                                            dMediumIndices,
                                            dWorkKeys),
                                   tempMemory,
                                   {this->lightCount, this->lightCount,
                                    this->lightCount, this->lightCount});
    // Set a GPU
    CUDA_CHECK(cudaSetDevice(this->gpu.DeviceId()));
    // Load Data to Temp Memory
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
                          this->hMediumIds.data(),
                          sizeof(uint16_t) * this->lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<PrimitiveId*>(dPrimitiveIds),
                          hPrimitiveIds.data(),
                          sizeof(PrimitiveId) * this->lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<TransformId*>(dTransformIds),
                          this->hTransformIds.data(),
                          sizeof(TransformId) * this->lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dWorkKeys),
                          this->hWorkKeys.data(),
                          sizeof(HitKey) * this->lightCount,
                          cudaMemcpyHostToDevice));

    // Get the prim data struct to GPU
    const PrimitiveData hPData = PrimDataAccessor::Data(primGroup);
    CUDA_CHECK(cudaMemcpy(const_cast<PrimitiveData*>(dPData),
                          &hPData,
                          sizeof(PrimitiveData),
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    this->gpu.GridStrideKC_X(0, 0,
                             this->lightCount,
                             //
                             KCConstructGPULight<PGroup>,
                             //
                             const_cast<GPULight<PGroup>*>(this->dGPULights),
                             dPrimitiveIds,
                             //
                             this->dRadiances,
                             dMediumIndices,
                             dWorkKeys,
                             dTransformIds,
                             //
                             *dPData,
                             dGlobalTransformArray,
                             this->lightCount);

    this->gpu.WaitMainStream();

    this->SetLightLists();

    return TracerError::OK;
}