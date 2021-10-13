#include "GPULightDirectional.cuh"
#include "TypeTraits.h"
#include "RayLib/MemoryAlignment.h"
#include "CudaSystem.hpp"

__global__ void KCConstructGPULightDirectional(GPULightDirectional* gLightLocations,
                                               //
                                               const Vector3f* gDirections,
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
        new (gLightLocations + globalId) GPULightDirectional(gDirections[globalId],
                                                             //
                                                             *gRads[globalId],
                                                             gMediumIndices[globalId],
                                                             gWorkKeys[globalId],
                                                             *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPULightGroupDirectional::InitializeGroup(const EndpointGroupDataList& lightNodes,
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

    hDirections.reserve(lightCount);
    for(const auto& node : lightNodes)
    {
        const auto direction = node.node->CommonVector3(DIRECTION_NAME);
        // Load to host memory
        hDirections.push_back(direction);
    }
    return SceneError::OK;
}

SceneError CPULightGroupDirectional::ChangeTime(const NodeListing& lightNodes, double time,
                                                const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupDirectional::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
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
    const Vector3f* dDirections;
    DeviceMemory::AllocateMultiData(std::tie(dMediumIndices, dTransformIds,
                                             dWorkKeys, dDirections),
                                    tempMemory,
                                    {lightCount, lightCount,
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
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dDirections),
                          hDirections.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       lightCount,
                       //
                       KCConstructGPULightDirectional,
                       //
                       const_cast<GPULightDirectional*>(dGPULights),
                       //
                       dDirections,
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