#include "SampleMaterials.cuh"
#include "RayLib/MemoryAlignment.h"

SceneError EmissiveMat::InitializeGroup(const NodeListing& materialNodes, 
                                        std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                        double time, const std::string& scenePath)
{
    constexpr const char* IRRADIANCE = "radiance";

    std::vector<Vector3> irradianceCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> irradiances = sceneNode->AccessVector3(IRRADIANCE);
        irradianceCPU.insert(irradianceCPU.end(), irradiances.begin(), irradiances.end());

        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    // Generate Id List
    SceneError e = SceneError::OK;
    if((e = GenerateInnerIds(materialNodes)) != SceneError::OK)
        return e;

    // Alloc etc
    size_t dIrradianceSize = irradianceCPU.size() * sizeof(Vector3);
    memory = std::move(DeviceMemory(dIrradianceSize));
    Vector3f* dIrradiance = static_cast<Vector3f*>(memory);
    CUDA_CHECK(cudaMemcpy(dIrradiance, irradianceCPU.data(), dIrradianceSize,
               cudaMemcpyHostToDevice));

    dData = EmissiveMatData{dIrradiance};
    return SceneError::OK;
}

SceneError EmissiveMat::ChangeTime(const NodeListing& materialNodes, double time,
                                   const std::string& scenePath)
{
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

// -------------
SceneError LambertMat::InitializeGroup(const NodeListing& materialNodes, 
                                       std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                       double time, const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";

    std::vector<Vector3> albedoCPU;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
        albedoCPU.insert(albedoCPU.end(), albedos.begin(), albedos.end());
    }

    // Generate Id List
    SceneError e = SceneError::OK;
    if((e = GenerateInnerIds(materialNodes)) != SceneError::OK)
        return e;

    // Alloc etc
    size_t dAlbedoSize = albedoCPU.size() * sizeof(Vector3);
    memory = std::move(DeviceMemory(dAlbedoSize));
    Vector3f* dAlbedo = static_cast<Vector3f*>(memory);
    CUDA_CHECK(cudaMemcpy(dAlbedo, albedoCPU.data(), dAlbedoSize,
                          cudaMemcpyHostToDevice));

    dData = AlbedoMatData{dAlbedo};
    return SceneError::OK;
}

SceneError LambertMat::ChangeTime(const NodeListing& materialNodes, double time,
                                  const std::string& scenePath)
{
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

// -------------
SceneError ReflectMat::InitializeGroup(const NodeListing& materialNodes, 
                                       std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                       double time, const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";
    constexpr const char* ROUGHNESS = "roughness";

    std::vector<Vector4> matDataCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
        std::vector<float> rougnessList = sceneNode->AccessFloat(ROUGHNESS);

        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            Vector4 data = Vector4(albedos[i], rougnessList[i]);
            matDataCPU.push_back(data);
            
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    // Alloc etc
    size_t dMatDataSize = matDataCPU.size() * sizeof(Vector4);
    memory = std::move(DeviceMemory(dMatDataSize));
    Vector4f* dMemory = static_cast<Vector4f*>(memory);
    CUDA_CHECK(cudaMemcpy(dMemory, matDataCPU.data(), dMatDataSize,
                          cudaMemcpyHostToDevice));

    dData = ReflectMatData{dMemory};
    return SceneError::OK;
}

SceneError ReflectMat::ChangeTime(const NodeListing& materialNodes, double time,
                                  const std::string& scenePath)
{
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

// -------------
SceneError RefractMat::InitializeGroup(const NodeListing& materialNodes, 
                                       std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                       double time, const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";
    constexpr const char* MEDIUM = "medium";

    std::vector<Vector3> albedosCPU;
    std::vector<uint32_t> mediumIndicesCPU;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
        std::vector<uint32_t> mediumIds = sceneNode->AccessUInt(MEDIUM);

        for(uint32_t& i : mediumIds)
        {
            mediumIndicesCPU.push_back(mediumIdIndexPairs.at(i));
        }        
        albedosCPU.insert(albedosCPU.end(), albedos.begin(), albedos.end());
    }

    // Generate Id List
    SceneError e = SceneError::OK;
    if((e = GenerateInnerIds(materialNodes)) != SceneError::OK)
        return e;

    // Alloc etc
    size_t albedoSize = albedosCPU.size() * sizeof(Vector3);
    albedoSize = Memory::AlignSize(albedoSize);
    size_t mediumIndicesSize = mediumIndicesCPU.size() * sizeof(uint32_t);
    mediumIndicesSize = Memory::AlignSize(mediumIndicesSize);
    memory = std::move(DeviceMemory(albedoSize + mediumIndicesSize));

    size_t offset = 0;
    Byte* dMemory = static_cast<Byte*>(memory);
    Vector3* dAlbedos = reinterpret_cast<Vector3*>(dMemory + offset);
    offset += albedoSize;
    uint32_t* dMedIndices = reinterpret_cast<uint32_t*>(dMemory + offset);
    offset += mediumIndicesSize;
    assert(offset == (albedoSize + mediumIndicesSize));

    CUDA_CHECK(cudaMemcpy(dAlbedos, albedosCPU.data(), 
                          albedosCPU.size() * sizeof(Vector3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMedIndices, mediumIndicesCPU.data(), 
                          mediumIndicesCPU.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    dData = RefractMatData{dAlbedos, dMedIndices, nullptr};
    return SceneError::OK;
}

SceneError RefractMat::ChangeTime(const NodeListing& materialNodes, double time,
                                  const std::string& scenePath)
{
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

void RefractMat::AttachGlobalMediumArray(const GPUMediumI* const* dMediumList,
                                         uint32_t baseMIndex)
{
    dData.dMediums = dMediumList;
    dData.baseMediumIndex = baseMIndex;
}