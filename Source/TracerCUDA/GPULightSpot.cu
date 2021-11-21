#include "GPULightSpot.cuh"
#include "TypeTraits.h"
#include "RayLib/MemoryAlignment.h"
#include "CudaSystem.hpp"

__global__ void KCConstructGPULightSpot(GPULightSpot* gLightLocations,
                                        //
                                        const Vector3f* gPositions,
                                        const Vector3f* gDirections,
                                        const Vector2f* gApertures,
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
        new (gLightLocations + globalId) GPULightSpot(gPositions[globalId],
                                                      gDirections[globalId],
                                                      gApertures[globalId],
                                                      //
                                                      *gRads[globalId],
                                                      gMediumIndices[globalId],
                                                      gWorkKeys[globalId],
                                                      *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPULightGroupSpot::InitializeGroup(const EndpointGroupDataList& lightNodes,
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

    hPositions.reserve(lightCount);
    hDirections.reserve(lightCount);
    hCosines.reserve(lightCount);
    for(const auto& node : lightNodes)
    {
        const auto position = node.node->CommonVector3(POSITION_NAME);
        const auto direction = node.node->CommonVector3(DIRECTION_NAME);
        const auto aperture = node.node->CommonVector2(CONE_APERTURE_NAME);
        // Load to host memory
        hPositions.push_back(position);
        hDirections.push_back(direction);
        hCosines.push_back(aperture);
    }
    return SceneError::OK;
}

SceneError CPULightGroupSpot::ChangeTime(const NodeListing&, double,
                                         const std::string&)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupSpot::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
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
    const Vector3f* dPositions;
    const Vector3f* dDirections;
    const Vector2f* dApertures;
    GPUMemFuncs::AllocateMultiData(std::tie(dMediumIndices, dTransformIds,
                                            dWorkKeys, dPositions,
                                            dDirections, dApertures),
                                   tempMemory,
                                   {lightCount, lightCount,
                                   lightCount, lightCount,
                                   lightCount, lightCount});

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
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dPositions),
                          hPositions.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dDirections),
                          hDirections.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector2*>(dApertures),
                          hCosines.data(),
                          sizeof(Vector2) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       lightCount,
                       //
                       KCConstructGPULightSpot,
                       //
                       const_cast<GPULightSpot*>(dGPULights),
                       //
                       dPositions,
                       dDirections,
                       dApertures,
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