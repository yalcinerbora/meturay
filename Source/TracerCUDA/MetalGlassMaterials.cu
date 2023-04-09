#include "MetalGlassMaterials.cuh"
#include "TextureFunctions.h"
#include "TextureReferenceGenerators.cuh"
#include "CudaSystem.hpp"

SceneError MetalMat::InitializeGroup(const NodeListing& materialNodes,
                                     const TextureNodeMap& textureNodes,
                                     const std::map<uint32_t, uint32_t>&,
                                     double time, const std::string& scenePath)
{
    constexpr const char* ETA = "eta";
    constexpr const char* K = "k";
    constexpr const char* ROUGHNESS = "roughness";
    constexpr const char* SPECULAR = "specular";

    SceneError err = SceneError::OK;


    std::vector<Vector3> etaCPU;
    std::vector<Vector3> kCPU;
    std::vector<Vector3> specularCPU;
    std::vector<float> rougnessCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        // Load Textured Data
        std::vector<Vector3> etaValues = sceneNode->AccessVector3(ETA, time);
        std::vector<Vector3> kValues = sceneNode->AccessVector3(K, time);
        std::vector<Vector3> specularValues = sceneNode->AccessVector3(SPECULAR, time);
        std::vector<float> roughnessValues = sceneNode->AccessFloat(ROUGHNESS, time);

        assert(etaValues.size() == kValues.size());
        assert(kValues.size() == roughnessValues.size());
        assert(roughnessValues.size() == specularValues.size());

        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }

        etaCPU.insert(etaCPU.end(), etaValues.cbegin(), etaValues.cend());
        kCPU.insert(kCPU.end(), kValues.cbegin(), kValues.cend());
        specularCPU.insert(specularCPU.end(), specularValues.cbegin(), specularValues.cend());
        rougnessCPU.insert(rougnessCPU.end(), roughnessValues.cbegin(), roughnessValues.cend());
    }

    // Allocation of pointers etc.
    size_t totalMatCount = innerIds.size();
    // Allocation
    GPUMemFuncs::AllocateMultiData(std::tie(hData.dEta, hData.dK,
                                            hData.dRoughness,
                                            hData.dSpecular),
                                   memory,
                                   {totalMatCount, totalMatCount,
                                    totalMatCount, totalMatCount});

    CUDA_CHECK(cudaMemcpy(const_cast<Vector3f*>(hData.dEta), etaCPU.data(), sizeof(Vector3f) * totalMatCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3f*>(hData.dK), kCPU.data(), sizeof(Vector3f) * totalMatCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Vector3f*>(hData.dSpecular), specularCPU.data(), sizeof(Vector3f) * totalMatCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<float*>(hData.dRoughness), rougnessCPU.data(), sizeof(float) * totalMatCount,
                          cudaMemcpyHostToDevice));
    // All Done!
    return SceneError::OK;
}

TracerError MetalMat::ConstructTextureReferences()
{
    return TracerError::OK;
}

SceneError MetalMat::ChangeTime(const NodeListing&, double,
                                const std::string&)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

size_t MetalMat::UsedGPUMemory() const
{
    return memory.Size();
}

size_t MetalMat::UsedCPUMemory() const
{
    return 0;
}

size_t MetalMat::UsedGPUMemory(uint32_t materialId) const
{
    return 2 * sizeof(Vector3f) + sizeof(float);
}

size_t MetalMat::UsedCPUMemory(uint32_t) const
{
    return 0;
}

uint8_t MetalMat::UsedTextureCount() const
{
    return 0;
}

std::vector<uint32_t> MetalMat::UsedTextureIds() const
{
    return std::vector<uint32_t>();
}