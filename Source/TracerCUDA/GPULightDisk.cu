#include "GPULightDisk.cuh"
#include "TypeTraits.h"
#include "RayLib/MemoryAlignment.h"
#include "CudaSystem.hpp"

__global__ void KCConstructGPULightDisk(GPULightDisk* gLightLocations,
                                        //
                                        const Vector3f* gCenters,
                                        const Vector3f* gNormals,
                                        const float* gRadius,
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
        new (gLightLocations + globalId) GPULightDisk(gCenters[globalId],
                                                      gNormals[globalId],
                                                      gRadius[globalId],
                                                      //
                                                      *gRads[globalId],
                                                      gMediumIndices[globalId],
                                                      gWorkKeys[globalId],
                                                      *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPULightGroupDisk::InitializeGroup(const EndpointGroupDataList& lightNodes,
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

    hCenters.reserve(lightCount);
    hNormals.reserve(lightCount);
    hRadius.reserve(lightCount);
    for(const auto& node : lightNodes)
    {
        const auto center = node.node->CommonVector3(POSITION_NAME);
        const auto normal = node.node->CommonVector3(NORMAL_NAME);
        const auto radius = node.node->CommonFloat(RADIUS_NAME);
        // Load to host memory
        hCenters.push_back(center);
        hNormals.push_back(normal);
        hRadius.push_back(radius);
    }
    return SceneError::OK;
}

SceneError CPULightGroupDisk::ChangeTime(const NodeListing&, double,
                                         const std::string&)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupDisk::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
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
    const Vector3f* dCenters;
    const Vector3f* dNormals;
    const float* dRadius;

    GPUMemFuncs::AllocateMultiData(std::tie(dMediumIndices, dTransformIds,
                                            dWorkKeys, dCenters,
                                            dNormals, dRadius),
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
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dCenters),
                          hCenters.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dNormals),
                          hNormals.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<float*>(dRadius),
                          hRadius.data(),
                          sizeof(float) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       lightCount,
                       //
                       KCConstructGPULightDisk,
                       //
                       const_cast<GPULightDisk*>(dGPULights),
                       //
                       dCenters,
                       dNormals,
                       dRadius,
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