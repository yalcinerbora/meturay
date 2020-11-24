
template <class PGroup>
__global__ void KCConstructGPULight(GPULight<PGroup>* gLightLocations,
                                    //
                                    const TransformId* gTransformIds,
                                    const PrimitiveId* gPrimitiveIds,
                                    const uint16_t*  gMediumIndices,
                                    const HitKey*    gLightMaterialIds,
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
SceneError CPULightGroup<PGroup>::InitializeGroup(const NodeListing& lightNodes,
                                                  const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                  const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                  const MaterialKeyListing& allMaterialKeys,
                                                  double time,
                                                  const std::string& scenePath)
{

}

template <class PGroup>
SceneError CPULightGroup<PGroup>::ChangeTime(const NodeListing& lightNodes, double time,
                                             const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

template <class PGroup>
TracerError CPULightGroup<PGroup>::ConstructLights(const CudaSystem& system)
{

    // Gen Temporary Memory

    //

   HitKey*             dLightMaterialIds;
   uint16_t*           dMediumIndices;
   PrimitiveId*        dPrimitiveIds;
   TransformId*        dTransformIds;


    // Call allocation kernel
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    gpu.AsyncGridStrideKC_X(0, LightCount(),
                            //
                            KCConstructGPULight<PGroup>,
                            //
                            const_cast<GPULight<PGroup>*>(dGPULights),
                            //
                            dTransformIds,
                            dPrimitiveIds,
                            dMediumIndices,
                            dLightMaterialIds,

                            dPData,
                            dGPUTransforms,
                            LightCount());

    gpu.WaitAllStreams();

    // Generate transform list
    for(uint32_t i = 0; i < LightCount(); i++)
    {
        const auto* ptr = static_cast<const GPULightI*>(dGPULights + i);
        gpuLightList.push_back(ptr);
    }
    return TracerError::OK;
}