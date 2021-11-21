#include "UnrealMaterial.cuh"
#include "TextureFunctions.h"
#include "TextureReferenceGenerators.cuh"
#include "CudaSystem.hpp"

SceneError UnrealMat::LoadAlbedoTexture(const TextureI<2>*& tex,
                                        Vector3& constData,
                                        uint32_t& texCount, uint32_t& constCount,
                                        const TexturedDataNode<Vector3>& node,
                                        const TextureNodeMap& textureNodes,
                                        const std::string& scenePath)
{
    SceneError err = SceneError::OK;
    if(node.isTexture)
    {
        const NodeTextureStruct& texInfo = node.texNode;
        if((err = TextureFunctions::AllocateTexture(tex,
                                                    dTextureMemory,
                                                    texInfo,
                                                    textureNodes,
                                                    EdgeResolveType::WRAP,
                                                    InterpolationType::LINEAR,
                                                    true, true,
                                                    gpu, scenePath)) != SceneError::OK)
            return err;
        texCount++;
    }
    else
    {
        constData = node.data;
        constCount++;
    }
    return SceneError::OK;
}

SceneError UnrealMat::Load1CTexture(const TextureI<2>*& tex,
                                    float& constData,
                                    uint32_t& texCount, uint32_t& constCount,
                                    const TexturedDataNode<float>& node,
                                    const TextureNodeMap& textureNodes,
                                    const std::string& scenePath)
{
    SceneError err = SceneError::OK;
    if(node.isTexture)
    {
        const NodeTextureStruct& texInfo = node.texNode;
        if((err = TextureFunctions::AllocateTexture(tex,
                                                    dTextureMemory,
                                                    texInfo,
                                                    textureNodes,
                                                    EdgeResolveType::WRAP,
                                                    InterpolationType::LINEAR,
                                                    true, true,
                                                    gpu, scenePath)) != SceneError::OK)
            return err;
        texCount++;
    }
    else
    {
        constData = node.data;
        constCount++;
    }
    return SceneError::OK;
}

SceneError UnrealMat::InitializeGroup(const NodeListing& materialNodes,
                                      const TextureNodeMap& textureNodes,
                                      const std::map<uint32_t, uint32_t>&,
                                      double time, const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";
    constexpr const char* NORMAL = "normal";
    constexpr const char* SPECULAR = "specular";
    constexpr const char* METALLIC = "metallic";
    constexpr const char* ROUGHNESS = "roughness";

    SceneError err = SceneError::OK;

    uint32_t albedoTexCount = 0;
    uint32_t albedoConstCount = 0;
    uint32_t normalTexCount = 0;
    uint32_t metallicTexCount = 0;
    uint32_t metallicConstCount = 0;
    uint32_t specularTexCount = 0;
    uint32_t specularConstCount = 0;
    uint32_t roughnessTexCount = 0;
    uint32_t roughnessConstCount = 0;

    std::vector<Vector3> albedoCPU;
    std::vector<Vector3> rougnessMetallicSheenCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        // Load Textured Data
        TexturedDataNodeList<Vector3> matAlbedoNodes = sceneNode->AccessTexturedDataVector3(ALBEDO, time);
        OptionalNodeList<NodeTextureStruct> matNormalNodes = sceneNode->AccessOptionalTextureNode(NORMAL, time);
        TexturedDataNodeList<float> matMetallicNodes = sceneNode->AccessTexturedDataFloat(METALLIC, time);
        TexturedDataNodeList<float> matRoughnessNodes = sceneNode->AccessTexturedDataFloat(ROUGHNESS, time);
        TexturedDataNodeList<float> matSpecularNodes = sceneNode->AccessTexturedDataFloat(SPECULAR, time);
        assert(matAlbedoNodes.size() == matNormalNodes.size());
        assert(matNormalNodes.size() == matMetallicNodes.size());
        assert(matMetallicNodes.size() == matRoughnessNodes.size());
        assert(matRoughnessNodes.size() == matSpecularNodes.size());

        // Iterate over these nodes one by one to find textures
        for(int j = 0; j < matAlbedoNodes.size(); j++)
        {
            TextureIdList texIdList = {};

            const TexturedDataNode<Vector3>& albedoNode = matAlbedoNodes[j];
            const OptionalNode<NodeTextureStruct>& normalNode = matNormalNodes[j];
            const TexturedDataNode<float>& metallicNode = matMetallicNodes[j];
            const TexturedDataNode<float>& roughnessNode = matRoughnessNodes[j];
            const TexturedDataNode<float>& specularNode = matSpecularNodes[j];

            ConstructionInfo constructionInfo;
            constructionInfo.hasAlbedoMap = albedoNode.isTexture;
            constructionInfo.hasNormalMap = normalNode.first;
            constructionInfo.hasMetallicMap = metallicNode.isTexture;
            constructionInfo.hasRoughnessMap = roughnessNode.isTexture;
            constructionInfo.hasSpecularMap = specularNode.isTexture;

            const TextureI<2>* albTex = nullptr;
            if((err = LoadAlbedoTexture(albTex,
                                        constructionInfo.albedoConst,
                                        albedoTexCount, albedoConstCount,
                                        albedoNode, textureNodes, scenePath)) != SceneError::OK)
                return err;
            constructionInfo.albedoMap = albTex ? static_cast<cudaTextureObject_t>(*albTex) : 0;
            texIdList[TexType::ALBEDO] = (albedoNode.isTexture) ? albedoNode.texNode.texId : std::numeric_limits<uint32_t>::max();

            const TextureI<2>* normT = nullptr;
            if(normalNode.first)
            {
                if((err = TextureFunctions::AllocateTexture(normT,
                                                            dTextureMemory,
                                                            normalNode.second,
                                                            textureNodes,
                                                            EdgeResolveType::WRAP,
                                                            InterpolationType::LINEAR,
                                                            true, true, gpu,
                                                            scenePath)) != SceneError::OK)
                    return err;
                normalTexCount++;
                constructionInfo.normalMap = static_cast<cudaTextureObject_t>(*normT);
                texIdList[TexType::NORMAL] = normalNode.second.texId;
            }
            else texIdList[TexType::NORMAL] = std::numeric_limits<uint32_t>::max();
            // METALLIC TEXTURE
            const TextureI<2>* metT = nullptr;
            if((err = Load1CTexture(metT,
                                    constructionInfo.metallicConst,
                                    metallicTexCount, metallicConstCount,
                                    metallicNode, textureNodes, scenePath)) != SceneError::OK)
                return err;

            constructionInfo.metallicMap = metT ? static_cast<cudaTextureObject_t>(*metT) : 0;
            texIdList[TexType::METALLIC] = (metallicNode.isTexture) ? metallicNode.texNode.texId : std::numeric_limits<uint32_t>::max();
            // SPECULAR TEXTURE
            const TextureI<2>* specT = nullptr;
            if((err = Load1CTexture(specT,
                                    constructionInfo.specularConst,
                                    specularTexCount, specularConstCount,
                                    specularNode, textureNodes, scenePath)) != SceneError::OK)
                return err;

            constructionInfo.specularMap = specT ? static_cast<cudaTextureObject_t>(*specT) : 0;
            texIdList[TexType::SPECULAR] = (specularNode.isTexture) ? specularNode.texNode.texId : std::numeric_limits<uint32_t>::max();
            // ROUGHNESS TEXTURE
            const TextureI<2>* roughT = nullptr;
            if((err = Load1CTexture(roughT,
                                    constructionInfo.roughnessConst,
                                    roughnessTexCount, roughnessConstCount,
                                    roughnessNode, textureNodes, scenePath)) != SceneError::OK)
                return err;

            constructionInfo.roughnessMap = roughT ? static_cast<cudaTextureObject_t>(*roughT) : 0;
            texIdList[TexType::ROUGHNESS] = (roughnessNode.isTexture) ? roughnessNode.texNode.texId : std::numeric_limits<uint32_t>::max();

            // Push Backs
            matConstructionInfo.push_back(constructionInfo);
            matTextureIds.push_back(texIdList);
        }

        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    // Allocation of pointers etc.
    size_t totalMatCount = matConstructionInfo.size();
    // Allocation
    size_t albedoConstSize = albedoConstCount * sizeof(Constant3CRef);
    albedoConstSize = Memory::AlignSize(albedoConstSize);
    size_t albedoTexRefSize = albedoTexCount * sizeof(Texture2D3CRef);
    albedoTexRefSize = Memory::AlignSize(albedoTexRefSize);
    size_t normalTexRefSize = normalTexCount * sizeof(Texture2D3CRef);
    normalTexRefSize = Memory::AlignSize(normalTexRefSize);
    size_t metallicConstSize = metallicConstCount * sizeof(Constant1CRef);
    metallicConstSize = Memory::AlignSize(metallicConstSize);
    size_t metallicTexRefSize = metallicTexCount * sizeof(Texture2D1CRef);
    metallicTexRefSize = Memory::AlignSize(metallicTexRefSize);
    size_t specularConstSize = specularConstCount * sizeof(Constant1CRef);
    specularConstSize = Memory::AlignSize(specularConstSize);
    size_t specularTexRefSize = specularTexCount * sizeof(Texture2D1CRef);
    specularTexRefSize = Memory::AlignSize(specularTexRefSize);
    size_t roughnessConstSize = roughnessConstCount * sizeof(Constant1CRef);
    roughnessConstSize = Memory::AlignSize(roughnessConstSize);
    size_t roughnessTexRefSize = roughnessTexCount * sizeof(Texture2D1CRef);
    roughnessTexRefSize = Memory::AlignSize(roughnessTexRefSize);
    // Main Ptrs
    size_t albedoPtrSize = totalMatCount * sizeof(Texture2D3CRefI*);
    albedoPtrSize = Memory::AlignSize(albedoPtrSize);
    size_t normalPtrSize = totalMatCount * sizeof(Texture2D3CRefI*);
    normalPtrSize = Memory::AlignSize(normalPtrSize);
    size_t metallicPtrSize = totalMatCount * sizeof(Texture2D1CRefI*);
    metallicPtrSize = Memory::AlignSize(metallicPtrSize);
    size_t specularPtrSize = totalMatCount * sizeof(Texture2D1CRefI*);
    specularPtrSize = Memory::AlignSize(specularPtrSize);
    size_t roughnessPtrSize = totalMatCount * sizeof(Texture2D1CRefI*);
    roughnessPtrSize = Memory::AlignSize(roughnessPtrSize);

    size_t totalSize = (albedoConstSize + albedoTexRefSize +
                        normalTexRefSize +
                        metallicConstSize + metallicTexRefSize +
                        specularConstSize + specularTexRefSize +
                        roughnessConstSize + roughnessTexRefSize +
                        albedoPtrSize +
                        normalPtrSize +
                        metallicPtrSize +
                        specularPtrSize +
                        roughnessPtrSize);

    memory = std::move(DeviceMemory(totalSize));

    Byte* memPtr = static_cast<Byte*>(memory);
    size_t offset = 0;
    // Albedo
    dConstAlbedo = reinterpret_cast<Constant3CRef*>(memPtr + offset);
    offset += albedoConstSize;
    dTextureAlbedoRef = reinterpret_cast<Texture2D3CRef*>(memPtr + offset);
    offset += albedoTexRefSize;
    // Normal
    dTextureNormalRef = reinterpret_cast<Texture2D3CRef*>(memPtr + offset);
    offset += normalTexRefSize;
    // Metallic
    dConstMetallic = reinterpret_cast<Constant1CRef*>(memPtr + offset);
    offset += metallicConstSize;
    dTextureMetallicRef = reinterpret_cast<Texture2D1CRef*>(memPtr + offset);
    offset += metallicTexRefSize;
    // Speuclar
    dConstSpecular = reinterpret_cast<Constant1CRef*>(memPtr + offset);
    offset += specularConstSize;
    dTextureSpecularRef = reinterpret_cast<Texture2D1CRef*>(memPtr + offset);
    offset += specularTexRefSize;
    // Roughness
    dConstRoughness = reinterpret_cast<Constant1CRef*>(memPtr + offset);
    offset += roughnessConstSize;
    dTextureRoughnessRef = reinterpret_cast<Texture2D1CRef*>(memPtr + offset);
    offset += roughnessTexRefSize;
    // Ptrs
    dAlbedo = reinterpret_cast<const Texture2D3CRefI**>(memPtr + offset);
    offset += albedoPtrSize;
    dNormal = reinterpret_cast<const Texture2D3CRefI**>(memPtr + offset);
    offset += normalPtrSize;
    dMetallic = reinterpret_cast<const Texture2D1CRefI**>(memPtr + offset);
    offset += metallicPtrSize;
    dSpecular = reinterpret_cast<const Texture2D1CRefI**>(memPtr + offset);
    offset += specularPtrSize;
    dRoughness = reinterpret_cast<const Texture2D1CRefI**>(memPtr + offset);
    offset += roughnessPtrSize;
    assert(totalSize == offset);

    // Finally Initialize Struct
    dData = UnrealMatData
    {
        dNormal, dAlbedo, dRoughness,
        dMetallic, dSpecular
    };
    // All Done!
    return SceneError::OK;
}

SceneError UnrealMat::ChangeTime(const NodeListing&, double,
                                 const std::string&)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

TracerError UnrealMat::ConstructTextureReferences()
{
    size_t materialCount = matConstructionInfo.size();

    // CPU Split Data
    std::vector<cudaTextureObject_t> hNormalTextureObjects(materialCount, 0);
    std::vector<TextureOrConstReferenceData<Vector3>> hAlbedoData(materialCount);
    std::vector<TextureOrConstReferenceData<float>> hMetallicData(materialCount);
    std::vector<TextureOrConstReferenceData<float>> hSpecularData(materialCount);
    std::vector<TextureOrConstReferenceData<float>> hRoughnessData(materialCount);

    // Split Mat Construction Info
    uint32_t i = 0;
    for(const auto& mInfo : matConstructionInfo)
    {
        // Normal
        if(mInfo.hasNormalMap)
            hNormalTextureObjects[i] = mInfo.normalMap;
        // Albedo
        hAlbedoData[i].isConstData = !mInfo.hasAlbedoMap;
        if(mInfo.hasAlbedoMap)
            hAlbedoData[i].tex = mInfo.albedoMap;
        else
            hAlbedoData[i].data = mInfo.albedoConst;
        // Metallic
        hMetallicData[i].isConstData = !mInfo.hasMetallicMap;
        if(mInfo.hasMetallicMap)
            hMetallicData[i].tex = mInfo.metallicMap;
        else
            hMetallicData[i].data = mInfo.metallicConst;
        // Specular
        hSpecularData[i].isConstData = !mInfo.hasSpecularMap;
        if(mInfo.hasSpecularMap)
            hSpecularData[i].tex = mInfo.specularMap;
        else
            hSpecularData[i].data = mInfo.specularConst;
        // Roughness
        hRoughnessData[i].isConstData = !mInfo.hasRoughnessMap;
        if(mInfo.hasRoughnessMap)
            hRoughnessData[i].tex = mInfo.roughnessMap;
        else
            hRoughnessData[i].data = mInfo.roughnessConst;

        i++;
    }
    // Allocate Temp GPU Memory
    // Size Determination
    size_t counterSize = sizeof(uint32_t) * 9;
    counterSize = Memory::AlignSize(counterSize);
    size_t albedoConstructionSize = sizeof(TextureOrConstReferenceData<Vector3>) * materialCount;
    albedoConstructionSize = Memory::AlignSize(albedoConstructionSize);
    size_t normalConstructionSize = sizeof(cudaTextureObject_t) * materialCount;
    normalConstructionSize = Memory::AlignSize(normalConstructionSize);
    size_t metallicConstructionSize = sizeof(TextureOrConstReferenceData<float>) * materialCount;
    metallicConstructionSize = Memory::AlignSize(metallicConstructionSize);
    size_t specularConstructionSize = sizeof(TextureOrConstReferenceData<float>) * materialCount;
    specularConstructionSize = Memory::AlignSize(specularConstructionSize);
    size_t roughnessConstructionSize = sizeof(TextureOrConstReferenceData<float>) * materialCount;
    roughnessConstructionSize = Memory::AlignSize(roughnessConstructionSize);

    size_t totalSize = (counterSize +
                        albedoConstructionSize +
                        metallicConstructionSize +
                        specularConstructionSize +
                        roughnessConstructionSize +
                        normalConstructionSize);
    DeviceMemory tempMemory(totalSize);

    size_t offset = 0;
    Byte* tempMemPtr = static_cast<Byte*>(tempMemory);
    TextureOrConstReferenceData<Vector3>* dAlbedoConstructionData = reinterpret_cast<TextureOrConstReferenceData<Vector3>*>(tempMemPtr + offset);
    offset += albedoConstructionSize;
    cudaTextureObject_t* dNormalTextures = reinterpret_cast<cudaTextureObject_t*>(tempMemPtr + offset);
    offset += normalConstructionSize;
    TextureOrConstReferenceData<float>* dMetallicConstructionData = reinterpret_cast<TextureOrConstReferenceData<float>*>(tempMemPtr + offset);
    offset += metallicConstructionSize;
    TextureOrConstReferenceData<float>* dSpecularConstructionData = reinterpret_cast<TextureOrConstReferenceData<float>*>(tempMemPtr + offset);
    offset += specularConstructionSize;
    TextureOrConstReferenceData<float>* dRoughnessConstructionData = reinterpret_cast<TextureOrConstReferenceData<float>*>(tempMemPtr + offset);
    offset += roughnessConstructionSize;
    uint32_t* dCounters = reinterpret_cast<uint32_t*>(tempMemPtr + offset);
    offset += counterSize;
    assert(offset == totalSize);

    // Load temp memory with data
    CUDA_CHECK(cudaMemset(dCounters, 0x00, sizeof(uint32_t) * 9));
    CUDA_CHECK(cudaMemcpy(dAlbedoConstructionData,
                          hAlbedoData.data(),
                          sizeof(TextureOrConstReferenceData<Vector3>) * materialCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNormalTextures,
                          hNormalTextureObjects.data(),
                          sizeof(cudaTextureObject_t) * materialCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMetallicConstructionData,
                          hMetallicData.data(),
                          sizeof(TextureOrConstReferenceData<float>) * materialCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSpecularConstructionData,
                          hSpecularData.data(),
                          sizeof(TextureOrConstReferenceData<float>) * materialCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRoughnessConstructionData,
                          hRoughnessData.data(),
                          sizeof(TextureOrConstReferenceData<float>) * materialCount,
                          cudaMemcpyHostToDevice));

    // Async Kernel Calls for Texture Reference Generations
    // Albedo
    gpu.AsyncGridStrideKC_X(0, materialCount,
                            GenerateEitherTexOrConstantReference<2, Vector3>,
                            const_cast<TextureRefI<2, Vector3f>**>(dData.dAlbedo),
                            const_cast<Constant3CRef*>(dConstAlbedo),
                            const_cast<Texture2D3CRef*>(dTextureAlbedoRef),
                            //
                            dCounters[0],
                            dCounters[1],
                            //
                            dAlbedoConstructionData,
                            static_cast<uint32_t>(materialCount));
    // Normal
    gpu.AsyncGridStrideKC_X(0, materialCount,
                            GenerateOptionalTexReference<2, Vector3>,
                            const_cast<TextureRefI<2, Vector3f>**>(dData.dNormal),
                            const_cast<Texture2D3CRef*>(dTextureNormalRef),
                            //
                            dCounters[2],
                            //
                            dNormalTextures,
                            static_cast<uint32_t>(materialCount));
    // Metallic
    gpu.AsyncGridStrideKC_X(0, materialCount,
                            GenerateEitherTexOrConstantReference<2, float>,
                            const_cast<TextureRefI<2, float>**>(dData.dMetallic),
                            const_cast<Constant1CRef*>(dConstMetallic),
                            const_cast<Texture2D1CRef*>(dTextureMetallicRef),
                            //
                            dCounters[3],
                            dCounters[4],
                            //
                            dMetallicConstructionData,
                            static_cast<uint32_t>(materialCount));
    // Specular
    gpu.AsyncGridStrideKC_X(0, materialCount,
                            GenerateEitherTexOrConstantReference<2, float>,
                            const_cast<TextureRefI<2, float>**>(dData.dSpecular),
                            const_cast<Constant1CRef*>(dConstSpecular),
                            const_cast<Texture2D1CRef*>(dTextureSpecularRef),
                            //
                            dCounters[5],
                            dCounters[6],
                            //
                            dSpecularConstructionData,
                            static_cast<uint32_t>(materialCount));
    // Roughness
    gpu.AsyncGridStrideKC_X(0, materialCount,
                            GenerateEitherTexOrConstantReference<2, float>,
                            const_cast<TextureRefI<2, float>**>(dData.dRoughness),
                            const_cast<Constant1CRef*>(dConstRoughness),
                            const_cast<Texture2D1CRef*>(dTextureRoughnessRef),
                            //
                            dCounters[7],
                            dCounters[8],
                            //
                            dRoughnessConstructionData,
                            static_cast<uint32_t>(materialCount));

    gpu.WaitAllStreams();
    // Clear temporary CPU data
    matConstructionInfo.clear();
    // All Done!
    return TracerError::OK;
}

size_t UnrealMat::UsedGPUMemory() const
{
    size_t totalSize = 0;
    for(const auto& tex : dTextureMemory)
    {
        totalSize += tex.second->Size();
    }
    totalSize += memory.Size();
    return totalSize;
}

size_t UnrealMat::UsedCPUMemory() const
{
    return matTextureIds.size() * sizeof(TextureIdList);
}

size_t UnrealMat::UsedGPUMemory(uint32_t materialId) const
{
    size_t totalSize = 0;
    auto i = innerIds.cend();
    if((i = innerIds.find(materialId)) == innerIds.cend())
        return 0;

    uint32_t index = i->second;
    uint32_t albedoTexId = matTextureIds[index][ALBEDO];
    uint32_t normalTexId = matTextureIds[index][NORMAL];
    uint32_t metallicTexId = matTextureIds[index][METALLIC];
    uint32_t roughnessTexId = matTextureIds[index][ROUGHNESS];
    uint32_t specularTexId = matTextureIds[index][SPECULAR];

    // Find the textures and size
    // ALBEDO
    auto texLoc = dTextureMemory.cend();
    if(((texLoc = dTextureMemory.find(albedoTexId)) != dTextureMemory.cend()))
    {
        totalSize += texLoc->second->Size();
    }
    else totalSize += sizeof(Vector3);
    // NORMAL
    if(((texLoc = dTextureMemory.find(normalTexId)) != dTextureMemory.cend()))
    {
        totalSize += texLoc->second->Size();
    }
    // METALLIC
    if(((texLoc = dTextureMemory.find(metallicTexId)) != dTextureMemory.cend()))
    {
        totalSize += texLoc->second->Size();
    }
    else totalSize += sizeof(float);
    // ROUGHNESS
    if(((texLoc = dTextureMemory.find(roughnessTexId)) != dTextureMemory.cend()))
    {
        totalSize += texLoc->second->Size();
    }
    else totalSize += sizeof(float);
    // SPECULAR
    if(((texLoc = dTextureMemory.find(specularTexId)) != dTextureMemory.cend()))
    {
        totalSize += texLoc->second->Size();
    }
    else totalSize += sizeof(float);
    return totalSize;
}

size_t UnrealMat::UsedCPUMemory(uint32_t) const
{
    return sizeof(TextureIdList);
}

uint8_t UnrealMat::UsedTextureCount() const
{
    // TODO: Implement
    return 0;
}

std::vector<uint32_t> UnrealMat::UsedTextureIds() const
{
    // TODO: Implement
    return std::vector<uint32_t>();
}