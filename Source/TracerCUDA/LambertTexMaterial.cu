#include "LambertTexMaterial.cuh"


SceneError LambertTexMat::InitializeGroup(const NodeListing& materialNodes,
                                          const TextureNodeMap& textureNodes,
                                          const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,                                          
                                          double time, const std::string& scenePath)
{
    //constexpr const char* ALBEDO = "albedo";
    //constexpr const char* NORMAL = "normal";

    //std::vector<Vector3> albedoCPU;
    //uint32_t i = 0;
    //for(const auto& sceneNode : materialNodes)
    //{
    //    TexturedDataNodeList<Vector3> matAlbedoNodes = sceneNode->AccessTexturedDataVector3(ALBEDO);

    //    for(const auto& albedoNode : matAlbedoNodes)
    //    
    //    {
    //        // Check if this is texture node
    //        if(albedoNode.isTexture)
    //        {
    //            uint32_t textureId = 0;
    //            // Find the texture node
    //            auto loc = textureNodes.cend();
    //            if((loc = textureNodes.find(textureId)) != textureNodes.cend())
    //            {
    //                std::string path = loc->second->CommonString(NodeNames::TEXTURE_FILE);
    //            }
    //            else return SceneError::TEXTURE_ID_NOT_FOUND;
    //        }
    //        else
    //        {
    //            // Just access the node
    //        }
    //    }
    //    


    //    irradianceCPU.insert(irradianceCPU.end(), irradiances.begin(), irradiances.end());

    //    const auto& ids = sceneNode->Ids();
    //    for(IdPair id : ids)
    //    {
    //        innerIds.emplace(std::make_pair(id.first, i));
    //        i++;
    //    }
    //}

    ////// Generate Id List
    ////SceneError e = SceneError::OK;
    ////if((e = GenerateInnerIds(materialNodes)) != SceneError::OK)
    ////    return e;

    //// Alloc etc
    //size_t dIrradianceSize = irradianceCPU.size() * sizeof(Vector3);
    //memory = std::move(DeviceMemory(dIrradianceSize));
    //Vector3f* dIrradiance = static_cast<Vector3f*>(memory);
    //CUDA_CHECK(cudaMemcpy(dIrradiance, irradianceCPU.data(), dIrradianceSize,
    //                      cudaMemcpyHostToDevice));

    //dData = EmissiveMatData{dIrradiance};
    //return SceneError::OK;
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

SceneError LambertTexMat::ChangeTime(const NodeListing& materialNodes, double time,
                                     const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

size_t LambertTexMat::UsedGPUMemory() const
{
    // TODO: Implement
    return 0;
}

size_t LambertTexMat::UsedCPUMemory() const 
{ 
    return sizeof(LambertTMatData); 
}

size_t LambertTexMat::UsedGPUMemory(uint32_t materialId) const
{
    // TODO: Implement
    return 0;
}

size_t LambertTexMat::UsedCPUMemory(uint32_t materialId) const
{
    // TODO: Implement
    return 0;
}

uint8_t LambertTexMat::UsedTextureCount() const
{
    // TODO: Implement
    return 0;
}

std::vector<uint32_t> LambertTexMat::UsedTextureIds() const
{
    // TODO: Implement
    return std::vector<uint32_t>();
}