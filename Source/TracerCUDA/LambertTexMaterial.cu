#include "LambertTexMaterial.cuh"
#include "RayLib/MemoryAlignment.h"
#include "TextureFunctions.h"

SceneError LambertTexMat::InitializeGroup(const NodeListing& materialNodes,
                                          const TextureNodeMap& textureNodes,
                                          const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,                                          
                                          double time, const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";
    constexpr const char* NORMAL = "normal";
    SceneError err = SceneError::OK;

    uint32_t constAlbedoCount = 0;
    uint32_t texAlbedoCount = 0;
    uint32_t texNormalCount = 0;

    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        // Load Textured Albedo Data
        TexturedDataNodeList<Vector3> matAlbedoNodes = sceneNode->AccessTexturedDataVector3(ALBEDO);
        OptionalNodeList<MaterialTextureStruct> matNormalNodes = sceneNode->AccessOptionalTextureNode(NORMAL);        
        assert(matAlbedoNodes.size() == matNormalNodes.size());

        

        std::vector<ConstructionInfo> localInfo;
        // Iterate over these nodes one by one to find textures
        for(int j = 0; j < matAlbedoNodes.size(); j++)
        {
            const TexturedDataNode<Vector3>& albedoNode = matAlbedoNodes[j];
            const OptionalNode<MaterialTextureStruct>& normalNode = matNormalNodes[j];

            // Check if this is texture node
            ConstructionInfo constructionInfo;
            if(albedoNode.isTexture)
            {
                const MaterialTextureStruct& texInfo = albedoNode.texNode;

                const TextureI<2, 4>* texture;
                if((err = TextureFunctions::AllocateTexture(texture,
                                                            dTextureMemory, texInfo,
                                                            textureNodes,
                                                            EdgeResolveType::WRAP,
                                                            InterpolationType::LINEAR,
                                                            true,
                                                            gpu, scenePath)) != SceneError::OK)
                    return err;

                constructionInfo.isConstantAlbedo = false;
                constructionInfo.albedoTexture = static_cast<cudaTextureObject_t>(*texture);

                //// Find the texture node
                //auto loc = textureNodes.cend();
                //if((loc = textureNodes.find(textureId)) != textureNodes.cend())
                //{
                //    const TextureStruct& texInfo = loc->second;
                //    
                //}
                //else return SceneError::TEXTURE_ID_NOT_FOUND;

                texAlbedoCount++;
            }
            else
            {
                constructionInfo.constantAlbedo = albedoNode.data;
                constAlbedoCount++;
            }

            // Check NormalMap
            if(normalNode.first)
            {
                const MaterialTextureStruct& texInfo = normalNode.second;
                const TextureI<2, 4>* texture;
                if((err = TextureFunctions::AllocateTexture(texture,
                                                            dTextureMemory, texInfo,
                                                            textureNodes,
                                                            EdgeResolveType::WRAP,
                                                            InterpolationType::LINEAR,
                                                            true,
                                                            gpu, scenePath)) != SceneError::OK)
                    return err;

                constructionInfo.hasNormalMap = true;
                constructionInfo.normalTexture = static_cast<cudaTextureObject_t>(*texture);
            }


            localInfo.push_back(constructionInfo);
        }
        matConstructionInfo.insert(matConstructionInfo.end(), 
                                   localInfo.begin(), 
                                   localInfo.end());

        // Generate Id lookup
        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    size_t totalMatCount = matConstructionInfo.size();
    // Allocation
    size_t albedoSize = constAlbedoCount * sizeof(Vector3);
    albedoSize = Memory::AlignSize(albedoSize);
    size_t albedoTexRefSize = texAlbedoCount * sizeof(Texture2DRef);
    albedoTexRefSize = Memory::AlignSize(albedoTexRefSize);
    size_t normalTexRefSize = texNormalCount * sizeof(Texture2DRef);
    normalTexRefSize = Memory::AlignSize(normalTexRefSize);
    size_t albedoPtrSize = totalMatCount * sizeof(Texture2DRefI*);
    albedoPtrSize = Memory::AlignSize(albedoPtrSize);
    size_t normalPtrSize = totalMatCount * sizeof(Texture2DRefI*);
    normalPtrSize = Memory::AlignSize(normalPtrSize);
    //
    size_t totalSize = (albedoSize + albedoTexRefSize +
                        normalTexRefSize + albedoPtrSize +
                        normalPtrSize);


    memory = std::move(DeviceMemory(totalSize));
    Byte* memPtr = static_cast<Byte*>(memory);
    size_t offset = 0;
    dConstAlbedo = reinterpret_cast<ConstantAlbedoRef*>(memPtr + offset);
    offset += albedoSize;
    dTextureAlbedoRef = reinterpret_cast<Texture2DRef*>(memPtr + offset);
    offset += albedoTexRefSize;
    dTextureNormalRef = reinterpret_cast<Texture2DRef*>(memPtr + offset);
    offset += normalTexRefSize;
    dAlbedo = reinterpret_cast<const Texture2DRefI**>(memPtr + offset);
    offset += albedoPtrSize;
    dNormal = reinterpret_cast<const Texture2DRefI**>(memPtr + offset);
    offset += normalPtrSize;
    assert(totalSize == offset);

    // Finally Initialize Struct
    dData = LambertTMatData{dAlbedo, dNormal};
    return SceneError::OK;
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