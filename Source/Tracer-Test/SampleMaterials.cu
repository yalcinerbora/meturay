#include "SampleMaterials.cuh"

SceneError EmissiveMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                        const std::string& scenePath)
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

int EmissiveMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

// -------------
SceneError LambertMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                       const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";

    std::vector<Vector3> albedoCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
        albedoCPU.insert(albedoCPU.end(), albedos.begin(), albedos.end());

        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

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

int LambertMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

// -------------
SceneError ReflectMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                       const std::string& scenePath)
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

int ReflectMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

// -------------
SceneError RefractMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                       const std::string& scenePath)
{
    //constexpr const char* ALBEDO = "albedo";
    //constexpr const char* ROUGHNESS = "roughness";

    //std::vector<Vector4> matDataCPU;
    //uint32_t i = 0;
    //for(const auto& sceneNode : materialNodes)
    //{
    //    std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
    //    std::vector<float> rougnessList = sceneNode->AccessFloat(ROUGHNESS);

    //    const auto& ids = sceneNode->Ids();
    //    for(IdPair id : ids)
    //    {
    //        Vector4 data = Vector4(albedos[i], rougnessList[i]);
    //        matDataCPU.push_back(data);

    //        innerIds.emplace(std::make_pair(id.first, i));
    //        i++;
    //    }
    //}

    //// Alloc etc
    //size_t dMatDataSize = matDataCPU.size() * sizeof(Vector4);
    //memory = std::move(DeviceMemory(dMatDataSize));
    //Vector4f* dMemory = static_cast<Vector4f*>(memory);
    //CUDA_CHECK(cudaMemcpy(dMemory, matDataCPU.data(), dMatDataSize,
    //                      cudaMemcpyHostToDevice));

    //dData = ReflectMatData{dMemory};
    //return SceneError::OK;
}

SceneError RefractMat::ChangeTime(const NodeListing& materialNodes, double time,
                                  const std::string& scenePath)
{
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

int RefractMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}