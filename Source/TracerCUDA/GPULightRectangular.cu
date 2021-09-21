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
        new (gLightLocations + globalId) GPULightRectangular(gTopLefts[globalId],
                                                             gRights[globalId],
                                                             gDowns[globalId],
                                                             *gTransforms[gTransformIds[globalId]],
                                                             //
                                                             gLightMaterialIds[globalId],
                                                             gMediumIndices[globalId]);
    }
}

SceneError CPULightGroupRectangular::InitializeGroup(const LightGroupDataList& lightNodes,
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

    hTopLefts.reserve(lightCount);
    hRights.reserve(lightCount);
    hDowns.reserve(lightCount);

    for(const auto& node : lightNodes)
    {
        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        HitKey materialKey = allMaterialKeys.at(std::make_pair(BaseConstants::EMPTY_PRIMITIVE_NAME,
                                                               node.materialId));

        const auto topLefts = node.node->AccessVector3(NAME_POSITION);
        const auto rights = node.node->AccessVector3(NAME_RECT_V0);
        const auto downs = node.node->AccessVector3(NAME_RECT_V1);

        // Load to host memory
        hHitKeys.push_back(materialKey);
        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);
        hTopLefts.insert(hTopLefts.end(), topLefts.begin(), topLefts.end());
        hRights.insert(hRights.end(), rights.begin(), rights.end());
        hDowns.insert(hDowns.end(), downs.begin(), downs.end());
    }

    // Allocate for GPULight classes
    size_t totalClassSize = sizeof(GPULightRectangular) * lightCount;
    totalClassSize = Memory::AlignSize(totalClassSize);

    DeviceMemory::EnlargeBuffer(memory, totalClassSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dGPULights = reinterpret_cast<const GPULightRectangular*>(dBasePtr + offset);
    offset += totalClassSize;
    assert(totalClassSize == offset);

    return SceneError::OK;
}

SceneError CPULightGroupRectangular::ChangeTime(const NodeListing& lightNodes, double time,
                                          const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupRectangular::ConstructLights(const CudaSystem& system,
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
    size_t topLeftSize = sizeof(Vector3f) * lightCount;
    topLeftSize = Memory::AlignSize(topLeftSize);
    size_t rightSize = sizeof(Vector3f) * lightCount;
    rightSize = Memory::AlignSize(rightSize);
    size_t downSize = sizeof(Vector3f) * lightCount;
    downSize = Memory::AlignSize(downSize);

    size_t totalSize = (matKeySize +
                        mediumSize +
                        transformIdSize +
                        topLeftSize +
                        rightSize +
                        downSize);
    DeviceMemory::EnlargeBuffer(tempMemory, totalSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(tempMemory);
    const HitKey* dLightMaterialIds = reinterpret_cast<const HitKey*>(dBasePtr + offset);
    offset += matKeySize;
    const uint16_t* dMediumIndices = reinterpret_cast<const uint16_t*>(dBasePtr + offset);
    offset += mediumSize;
    const TransformId* dTransformIds = reinterpret_cast<const TransformId*>(dBasePtr + offset);
    offset += transformIdSize;
    const Vector3f* dTopLefts = reinterpret_cast<const Vector3f*>(dBasePtr + offset);
    offset += topLeftSize;
    const Vector3f* dRights = reinterpret_cast<const Vector3f*>(dBasePtr + offset);
    offset += rightSize;
    const Vector3f* dDowns = reinterpret_cast<const Vector3f*>(dBasePtr + offset);
    offset += downSize;
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
                       LightCount(),
                       //
                       KCConstructGPULightRectangular,
                       //
                       const_cast<GPULightRectangular*>(dGPULights),
                       //
                       dTopLefts,
                       dRights,
                       dDowns,
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