#include "GPULightDirectional.cuh"
#include "TypeTraits.h"
#include "RayLib/MemoryAlignment.h"
#include "CudaSystem.hpp"

__global__ void KCConstructGPULightDirectional(GPULightDirectional* gLightLocations,
                                               //
                                               const Vector3f* gDirections,
                                               //
                                               const TransformId* gTransformIds,
                                               const uint16_t* gMediumIndices,
                                               const HitKey* gLightMaterialIds,
                                               //
                                               const GPUTransformI** gTransforms,
                                               uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gLightLocations + globalId) GPULightDirectional(gDirections[globalId],
                                                             *gTransforms[gTransformIds[globalId]],
                                                             //
                                                             gLightMaterialIds[globalId],
                                                             gMediumIndices[globalId]);
    }
}

SceneError CPULightGroupDirectional::InitializeGroup(const LightGroupDataList& lightNodes,
                                                     const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                     const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                     const MaterialKeyListing& allMaterialKeys,
                                                     double time,
                                                     const std::string& scenePath)
{
    lightCount = static_cast<uint32_t>(lightNodes.size());
    hHitKeys.reserve(lightCount);
    hMediumIds.reserve(lightCount);
    hTransformIds.reserve(lightCount);

    hDirections.reserve(lightCount);

    for(const auto& node : lightNodes)
    {
        // Convert Ids to inner index
        uint32_t mediumIndex = mediumIdIndexPairs.at(node.mediumId);
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        HitKey materialKey = allMaterialKeys.at(std::make_pair(BaseConstants::EMPTY_PRIMITIVE_NAME,
                                                               node.materialId));

        const auto directions = node.node->AccessVector3(NAME_DIRECTION);

        // Load to host memory
        hHitKeys.push_back(materialKey);
        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);
        hDirections.insert(hDirections.end(), directions.begin(), directions.end());
    }

    // Allocate for GPULight classes
    size_t totalClassSize = sizeof(GPULightDirectional) * lightCount;
    totalClassSize = Memory::AlignSize(totalClassSize);

    DeviceMemory::EnlargeBuffer(memory, totalClassSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dGPULights = reinterpret_cast<const GPULightDirectional*>(dBasePtr + offset);
    offset += totalClassSize;
    assert(totalClassSize == offset);

    return SceneError::OK;
}

SceneError CPULightGroupDirectional::ChangeTime(const NodeListing& lightNodes, double time,
                                                const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupDirectional::ConstructLights(const CudaSystem& system,
                                                      const GPUTransformI** dGlobalTransformArray,
                                                      const KeyMaterialMap&)
{
    // Gen Temporary Memory
    DeviceMemory tempMemory;
    // Allocate for GPULight classes
    size_t matKeySize = sizeof(HitKey) * lightCount;
    matKeySize = Memory::AlignSize(matKeySize);
    size_t mediumSize = sizeof(uint16_t) * lightCount;
    mediumSize = Memory::AlignSize(mediumSize);
    size_t transformIdSize = sizeof(TransformId) * lightCount;
    transformIdSize = Memory::AlignSize(transformIdSize);
    size_t directionSize = sizeof(Vector3f) * lightCount;
    directionSize = Memory::AlignSize(directionSize);

    size_t totalSize = (matKeySize +
                        mediumSize +
                        transformIdSize +
                        directionSize);
    DeviceMemory::EnlargeBuffer(tempMemory, totalSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(tempMemory);
    const HitKey* dLightMaterialIds = reinterpret_cast<const HitKey*>(dBasePtr + offset);
    offset += matKeySize;
    const uint16_t* dMediumIndices = reinterpret_cast<const uint16_t*>(dBasePtr + offset);
    offset += mediumSize;
    const TransformId* dTransformIds = reinterpret_cast<const TransformId*>(dBasePtr + offset);
    offset += transformIdSize;
    const Vector3f* dDirections = reinterpret_cast<const Vector3f*>(dBasePtr + offset);
    offset += directionSize;
    assert(totalSize == offset);

    // Set a GPU
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Load Data to Temp Memory
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dLightMaterialIds),
                          hHitKeys.data(),
                          sizeof(HitKey) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
                          hMediumIds.data(),
                          sizeof(uint16_t) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<TransformId*>(dTransformIds),
                          hTransformIds.data(),
                          sizeof(TransformId) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3*>(dDirections),
                          hDirections.data(),
                          sizeof(Vector3) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       LightCount(),
                       //
                       KCConstructGPULightDirectional,
                       //
                       const_cast<GPULightDirectional*>(dGPULights),
                       //
                       dDirections,
                       //
                       dTransformIds,
                       dMediumIndices,
                       dLightMaterialIds,
                       //
                       dGlobalTransformArray,
                       LightCount());

    gpu.WaitMainStream();

    // Generate transform list
    for(uint32_t i = 0; i < LightCount(); i++)
    {
        const auto* ptr = static_cast<const GPULightI*>(dGPULights + i);
        gpuLightList.push_back(ptr);
    }
    return TracerError::OK;
}