#include "GPULightSkySphere.cuh"
#include "CudaSystem.hpp"

#include "RayLib/MemoryAlignment.h"
#include "GPUMaterialI.h"

__global__ void KCConstructGPULightSkySphere(GPULightSkySphere* gLightLocations,
                                             //
                                             const GPUDistPiecewiseConst2D* gLuminanceDistributions,
                                             const bool* gIsHemispherical,
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
        new (gLightLocations + globalId) GPULightSkySphere(gLuminanceDistributions[globalId],
                                                           gIsHemispherical[globalId],
                                                           *gTransforms[gTransformIds[globalId]],
                                                           //
                                                           gLightMaterialIds[globalId],
                                                           gMediumIndices[globalId]);
    }
}

SceneError CPULightGroupSkySphere::InitializeGroup(const LightGroupDataList& lightNodes,
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
    hIsHemiOptions.reserve(lightCount);

    for(const auto& node : lightNodes)
    {
        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        HitKey materialKey = allMaterialKeys.at(std::make_pair(BaseConstants::EMPTY_PRIMITIVE_NAME,
                                                node.materialId));

        // Fetch Options from the Node
        std::vector<bool> isHemiOptions = node.node->AccessBool(IS_HEMI_NAME);

        // Load to host memory
        hHitKeys.push_back(materialKey);
        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);

        hIsHemiOptions.insert(hIsHemiOptions.end(), isHemiOptions.cbegin(), isHemiOptions.cend());        
    }

    // Allocate for GPULight classes
    size_t totalClassSize = sizeof(GPULightSkySphere) * lightCount;
    totalClassSize = Memory::AlignSize(totalClassSize);
    size_t totalDistSize = sizeof(GPUDistPiecewiseConst2D) * lightCount;
    totalDistSize = Memory::AlignSize(totalDistSize);
    size_t totalSize = totalDistSize + totalClassSize;
    DeviceMemory::EnlargeBuffer(memory, totalSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dGPULights = reinterpret_cast<const GPULightSkySphere*>(dBasePtr + offset);
    offset += totalClassSize;
    dLuminanceDistributions = reinterpret_cast<const GPUDistPiecewiseConst2D*>(dBasePtr + offset);
    offset += totalDistSize;
    assert(totalSize == offset);

    return SceneError::OK;
}

SceneError CPULightGroupSkySphere::ChangeTime(const NodeListing& lightNodes,
                                              double time,
                                              const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupSkySphere::ConstructLights(const CudaSystem& system,
                                                    const GPUTransformI** dGlobalTransformArray,
                                                    const KeyMaterialMap& materialMap)
{
    std::vector<std::vector<float>> hLuminances;
    std::vector<Vector2ui> hLuminanceSizes;

    // Acquire Luminance Information for each light
    TracerError err = TracerError::OK;
    for(HitKey& key : hHitKeys)
    {     
        const auto loc = materialMap.find(HitKey::FetchBatchPortion(key));
        if(loc == materialMap.cend())
            return TracerError::UNABLE_TO_CONSTRUCT_LIGHT;

        // Materials of light has to be LightMaterial
        // it should be safe to cast here
        const GPUBoundaryMaterialGroupI* matGroup = loc->second;

        Vector2ui dimension;
        std::vector<float> lumData;
        if((err = matGroup->LuminanceData(lumData,
                                          dimension, 
                                          HitKey::FetchIdPortion(key))) != TracerError::OK)
            return err;

        hLuminances.push_back(std::move(lumData));
        hLuminanceSizes.push_back(dimension);

    }

    // Construct Distribution Data
    std::vector<bool> factorInSpherical(hLuminances.size(), false);
    hLuminanceDistributions = CPUDistGroupPiecewiseConst2D(hLuminances, hLuminanceSizes,
                                                           factorInSpherical, system);

    // Copy Generated GPU Distributions
    CUDA_CHECK(cudaMemcpy(const_cast<GPUDistPiecewiseConst2D*>(dLuminanceDistributions),
               hLuminanceDistributions.DistributionGPU().data(),
               lightCount * sizeof(GPUDistPiecewiseConst2D),
               cudaMemcpyHostToDevice));

    // Gen Temporary Memory
    DeviceMemory tempMemory;
    // Allocate for GPULight classes
    size_t matKeySize = sizeof(HitKey) * lightCount;
    matKeySize = Memory::AlignSize(matKeySize);
    size_t mediumSize = sizeof(uint16_t) * lightCount;
    mediumSize = Memory::AlignSize(mediumSize);
    size_t transformIdSize = sizeof(TransformId) * lightCount;
    transformIdSize = Memory::AlignSize(transformIdSize);
    static_assert(sizeof(bool) == sizeof(Byte), "sizeof(bool) != sizeof(Byte)!");
    size_t isHemiSize = sizeof(bool) * lightCount;
    isHemiSize = Memory::AlignSize(isHemiSize);
    size_t totalSize = (matKeySize +
                        mediumSize +
                        transformIdSize + 
                        isHemiSize);
    DeviceMemory::EnlargeBuffer(tempMemory, totalSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(tempMemory);
    const HitKey* dLightMaterialIds = reinterpret_cast<const HitKey*>(dBasePtr + offset);
    offset += matKeySize;
    const uint16_t* dMediumIndices = reinterpret_cast<const uint16_t*>(dBasePtr + offset);
    offset += mediumSize;
    const TransformId* dTransformIds = reinterpret_cast<const TransformId*>(dBasePtr + offset);
    offset += transformIdSize;
    const bool* dIsHemiOptions = reinterpret_cast<const bool*>(dBasePtr + offset);
    offset += isHemiSize;
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
    CUDA_CHECK(cudaMemcpy(const_cast<bool*>(dIsHemiOptions),
               hIsHemiOptions.data(),
               sizeof(bool) * lightCount,
               cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       LightCount(),
                       //
                       KCConstructGPULightSkySphere,
                       //
                       const_cast<GPULightSkySphere*>(dGPULights),
                       //
                       dLuminanceDistributions,
                       dIsHemiOptions,
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