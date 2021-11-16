#include "GPULightRectangular.cuh"
#include "TypeTraits.h"
#include "RayLib/MemoryAlignment.h"
#include "CudaSystem.hpp"

__global__ void KCConstructGPULightRectangular(GPULightRectangular* gLightLocations,
                                               //
                                               const Vector3f* gTopLefts,
                                               const Vector3f* gRights,
                                               const Vector3f* gDowns,
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
        new (gLightLocations + globalId) GPULightRectangular(gTopLefts[globalId],
                                                             gRights[globalId],
                                                             gDowns[globalId],
                                                             //
                                                             *gRads[globalId],
                                                             gMediumIndices[globalId],
                                                             gWorkKeys[globalId],
                                                             *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPULightGroupRectangular::InitializeGroup(const EndpointGroupDataList& lightNodes,
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

    hTopLefts.reserve(lightCount);
    hRights.reserve(lightCount);
    hDowns.reserve(lightCount);
    for(const auto& node : lightNodes)
    {
        const auto topLeft = node.node->CommonVector3(POSITION_NAME);
        const auto right = node.node->CommonVector3(RECT_V0_NAME);
        const auto down = node.node->CommonVector3(RECT_V1_NAME);

        // Load to host memory
        hTopLefts.push_back(topLeft);
        hRights.push_back(right);
        hDowns.push_back(down);
    }
    return SceneError::OK;
}

SceneError CPULightGroupRectangular::ChangeTime(const NodeListing& lightNodes, double time,
                                          const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupRectangular::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
                                                         const CudaSystem&)
{
    TracerError e = TracerError::OK;
    // Construct Texture References
    if((e = ConstructTextureReferences()) != TracerError::OK)
        return e;

    const uint16_t* dMediumIndices;
    const TransformId* dTransformIds;
    const HitKey* dWorkKeys;
    const Vector3f* dTopLefts;
    const Vector3f* dRights;
    const Vector3f* dDowns;

    DeviceMemory tempMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dMediumIndices,
                                            dTransformIds,
                                            dWorkKeys,
                                            dTopLefts,
                                            dRights,
                                            dDowns),
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
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dTopLefts),
                          hTopLefts.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dRights),
                          hRights.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dDowns),
                          hDowns.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       lightCount,
                       //
                       KCConstructGPULightRectangular,
                       //
                       const_cast<GPULightRectangular*>(dGPULights),
                       //
                       dTopLefts,
                       dRights,
                       dDowns,
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