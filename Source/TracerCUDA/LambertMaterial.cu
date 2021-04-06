#include "LambertMaterial.cuh"
#include "RayLib/MemoryAlignment.h"
#include "TextureFunctions.h"
#include "TextureReferenceGenerators.cuh"
#include "CudaConstants.hpp"

SceneError LambertMat::InitializeGroup(const NodeListing& materialNodes,
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
        OptionalNodeList<NodelTextureStruct> matNormalNodes = sceneNode->AccessOptionalTextureNode(NORMAL);
        assert(matAlbedoNodes.size() == matNormalNodes.size());

        std::vector<ConstructionInfo> localInfo;
        // Iterate over these nodes one by one to find textures
        for(int j = 0; j < matAlbedoNodes.size(); j++)
        {
            const TexturedDataNode<Vector3>& albedoNode = matAlbedoNodes[j];
            const OptionalNode<NodelTextureStruct>& normalNode = matNormalNodes[j];

            // Check if this is texture node
            ConstructionInfo constructionInfo;
            if(albedoNode.isTexture)
            {
                const NodelTextureStruct& texInfo = albedoNode.texNode;

                const TextureI<2, 4>* texture;
                if((err = TextureFunctions::AllocateTexture(texture,
                                                            dTextureMemory, texInfo,
                                                            textureNodes,
                                                            EdgeResolveType::WRAP,
                                                            InterpolationType::LINEAR,
                                                            true, true,
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
                const NodelTextureStruct& texInfo = normalNode.second;
                const TextureI<2, 4>* texture;
                if((err = TextureFunctions::AllocateTexture(texture,
                                                            dTextureMemory, texInfo,
                                                            textureNodes,
                                                            EdgeResolveType::WRAP,
                                                            InterpolationType::LINEAR,
                                                            true, true,
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
    dData = LambertMatData{dAlbedo, dNormal};
    return SceneError::OK;
}

SceneError LambertMat::ChangeTime(const NodeListing& materialNodes, double time,
                                  const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

TracerError LambertMat::ConstructTextureReferences()
{
    size_t materialCount = matConstructionInfo.size();

    // CPU Split Data
    std::vector<cudaTextureObject_t> hNormalTextureObjects(materialCount, 0);
    std::vector<TextureOrConstReferenceData<Vector3>> hAlbedoData(materialCount);

    // Split Mat Construction Info
    uint32_t i = 0;
    for(const auto& mInfo : matConstructionInfo)
    {
        if(mInfo.hasNormalMap)
            hNormalTextureObjects[i] = mInfo.normalTexture;
        hAlbedoData[i].isConstData = mInfo.isConstantAlbedo;
        if(mInfo.isConstantAlbedo)
            hAlbedoData[i].data = mInfo.constantAlbedo;
        else
            hAlbedoData[i].tex = mInfo.albedoTexture;

        i++;
    }

    // Allocate Temp GPU Memory
    // Size Determination

    size_t counterSize = sizeof(uint32_t) * 3;
    counterSize = Memory::AlignSize(counterSize);
    size_t albedoConstructionSize = sizeof(TextureOrConstReferenceData<Vector3>) * materialCount;
    albedoConstructionSize = Memory::AlignSize(albedoConstructionSize);
    size_t normalConstructionSize = sizeof(cudaTextureObject_t) * materialCount;
    normalConstructionSize = Memory::AlignSize(normalConstructionSize);

    //
    size_t totalSize = (counterSize +
                        albedoConstructionSize +
                        normalConstructionSize);
    DeviceMemory tempMemory(totalSize);

    size_t offset = 0;
    Byte* tempMemPtr = static_cast<Byte*>(tempMemory);

    TextureOrConstReferenceData<Vector3>* dConstructionData = reinterpret_cast<TextureOrConstReferenceData<Vector3>*>(tempMemPtr + offset);
    offset += albedoConstructionSize;
    cudaTextureObject_t* dNormalTextures = reinterpret_cast<cudaTextureObject_t*>(tempMemPtr + offset);
    offset += normalConstructionSize;
    uint32_t* dAlbedoTexCounter = reinterpret_cast<uint32_t*>(tempMemPtr + offset);
    offset += sizeof(uint32_t);
    uint32_t* dConstantRefCounter = reinterpret_cast<uint32_t*>(tempMemPtr + offset);
    offset += sizeof(uint32_t);
    uint32_t* dNormalTexCounter = reinterpret_cast<uint32_t*>(tempMemPtr + offset);
    offset += sizeof(uint32_t);
    offset += counterSize - (sizeof(uint32_t) * 3);
    assert(offset == totalSize);

    // Load temp memory with data
    CUDA_CHECK(cudaMemset(dAlbedoTexCounter, 0x00, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dConstantRefCounter, 0x00, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dNormalTexCounter, 0x00, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dConstructionData,
                          hAlbedoData.data(),
                          sizeof(TextureOrConstReferenceData<Vector3>) * materialCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNormalTextures,
                          hNormalTextureObjects.data(),
                          sizeof(cudaTextureObject_t) * materialCount,
                          cudaMemcpyHostToDevice));

    // Kernel Call For Albedo Texture Generation
    gpu.AsyncGridStrideKC_X(0, materialCount,
                            GenerateEitherTexOrConstantReference<2, Vector3>,
                            const_cast<TextureRefI<2, Vector3f>**>(dData.dAlbedo),
                            const_cast<ConstantAlbedoRef*>(dConstAlbedo),
                            const_cast<Texture2DRef*>(dTextureAlbedoRef),
                            //
                            *dAlbedoTexCounter,
                            *dConstantRefCounter,
                            //
                            dConstructionData,
                            static_cast<uint32_t>(materialCount));

    // Kernel Call For Normal Map Texture Generation
    gpu.AsyncGridStrideKC_X(0, materialCount,
                            GenerateOptionalTexReference<2, Vector3>,
                            const_cast<TextureRefI<2, Vector3f>**>(dData.dNormal),
                            const_cast<Texture2DRef*>(dTextureNormalRef),
                            //
                            *dNormalTexCounter,
                            //
                            dNormalTextures,
                            static_cast<uint32_t>(materialCount));

    // Clear temporary CPU data
    matConstructionInfo.clear();
    // All Done!
    return TracerError::OK;
}

size_t LambertMat::UsedGPUMemory() const
{
    // TODO: Implement
    return 0;
}

size_t LambertMat::UsedCPUMemory() const
{
    return sizeof(LambertMatData);
}

size_t LambertMat::UsedGPUMemory(uint32_t materialId) const
{
    // TODO: Implement
    return 0;
}

size_t LambertMat::UsedCPUMemory(uint32_t materialId) const
{
    // TODO: Implement
    return 0;
}

uint8_t LambertMat::UsedTextureCount() const
{
    // TODO: Implement
    return 0;
}

std::vector<uint32_t> LambertMat::UsedTextureIds() const
{
    // TODO: Implement
    return std::vector<uint32_t>();
}