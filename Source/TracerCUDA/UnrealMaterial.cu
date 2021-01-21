#include "UnrealMaterial.cuh"

SceneError UnrealMat::InitializeGroup(const NodeListing& materialNodes,
                                      const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                      double time, const std::string& scenePath)
{    

    std::vector<Vector3> albedoCPU;
    std::vector<Vector3> rougnessMetallicSheenCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {        
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
    //size_t dIrradianceSize = irradianceCPU.size() * sizeof(Vector3);
    //memory = std::move(DeviceMemory(dIrradianceSize));
    //Vector3f* dIrradiance = static_cast<Vector3f*>(memory);
    //CUDA_CHECK(cudaMemcpy(dIrradiance, irradianceCPU.data(), dIrradianceSize,
    //                      cudaMemcpyHostToDevice));

    //dData = EmissiveMatData{dIrradiance};
    return SceneError::OK;

    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

SceneError UnrealMat::ChangeTime(const NodeListing& materialNodes, double time,
                                 const std::string& scenePath)
{
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}