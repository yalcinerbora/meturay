#include "GPULightConstant.cuh"
#include "CudaSystem.hpp"

__global__
void KCConstructGPULightConstant(GPULightConstant* gLightLocations,
                                 //
                                 const TextureRefI<2, Vector3f>** gRads,
                                 const uint16_t* gMediumIndices,
                                 const HitKey* gWorkKeys,
                                 const TransformId* gTransformIds,
                                 //
                                 const GPUTransformI** gTransforms,
                                 uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gLightLocations + globalId) GPULightConstant(*gRads[globalId],
                                                          gMediumIndices[globalId],
                                                          gWorkKeys[globalId],
                                                          *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPULightGroupConstant::InitializeGroup(const EndpointGroupDataList& lightNodes,
                                                  const TextureNodeMap& textures,
                                                  const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                  const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                  uint32_t batchId, double time,
                                                  const std::string& scenePath)
{
    SceneError e = SceneError::OK;

    if((e = InitializeCommon(lightNodes, textures,
                             mediumIdIndexPairs,
                             transformIdIndexPairs,
                             batchId, time,
                             scenePath)) != SceneError::OK)
        return e;

    // This object does not expose GPU Groups to the system
    // since it is not clearly defined (it only returns radiance value)
    // its pdf / sample routines are not valid
    // so NEE estimator etc. should not utilize this type of light
    gpuLightList.clear();
    gpuEndpointList.clear();

    return SceneError::OK;
}

SceneError CPULightGroupConstant::ChangeTime(const NodeListing&, double,
                                                    const std::string&)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupConstant::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
                                                      const AABB3f&,
                                                      const CudaSystem&)
{
    TracerError e = TracerError::OK;
    // Construct Texture References
    if((e = ConstructTextureReferences()) != TracerError::OK)
        return e;

    // Gen Temporary Memory
    DeviceMemory tempMemory;

    const uint16_t* dMediumIndices;
    const TransformId* dTransformIds;
    const HitKey* dWorkKeys;
    GPUMemFuncs::AllocateMultiData(std::tie(dMediumIndices, dTransformIds, dWorkKeys),
                                   tempMemory,
                                   {lightCount, lightCount, lightCount});

    // Set a GPU
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Load Data to Temp Memory
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
                          hMediumIds.data(),
                          sizeof(uint16_t) * lightCount,
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
                       KCConstructGPULightConstant,
                       //
                       const_cast<GPULightConstant*>(dGPULights),
                       //
                       dRadiances,
                       dMediumIndices,
                       dWorkKeys,
                       dTransformIds,
                       //
                       dGlobalTransformArray,
                       lightCount);

    gpu.WaitMainStream();

    SetLightLists();

    return TracerError::OK;
}