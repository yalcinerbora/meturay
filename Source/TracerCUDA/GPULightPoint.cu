#include "GPULightPoint.cuh"
#include "TypeTraits.h"
#include "RayLib/MemoryAlignment.h"

#include "CudaSystem.h"
#include "CudaSystem.hpp"

__global__ void KCConstructGPULightPoint(GPULightPoint* gLightLocations,
                                         //
                                         const Vector3f* gPositions,
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
        new (gLightLocations + globalId) GPULightPoint(gPositions[globalId],
                                                       //
                                                       *gRads[globalId],
                                                       gMediumIndices[globalId],
                                                       gWorkKeys[globalId],
                                                       *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPULightGroupPoint::InitializeGroup(const EndpointGroupDataList& lightNodes,
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

    // Copy Data
    hPositions.reserve(lightCount);
    for(const auto& node : lightNodes)
    {
        Vector3 position = node.node->CommonVector3(POSITION_NAME);
        // Load to host memory
        hPositions.push_back(position);
    }
    return SceneError::OK;
}

SceneError CPULightGroupPoint::ChangeTime(const NodeListing& lightNodes, double time,
                                          const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupPoint::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
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
    GPUMemFuncs::AllocateMultiData(std::tie(dMediumIndices, dTransformIds,
                                            dWorkKeys, dPositions),
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
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dPositions),
                          hPositions.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       lightCount,
                       //
                       KCConstructGPULightPoint,
                       //
                       const_cast<GPULightPoint*>(dGPULights),
                       //
                       dPositions,
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