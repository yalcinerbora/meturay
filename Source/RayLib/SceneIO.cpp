#include "SceneIO.h"
#include "Constants.h"
#include "SceneNodeNames.h"

using namespace NodeNames;

TextureAccessLayout SceneIO::LoadTextureAccessLayout(const nlohmann::json& node)
{
    std::string layout = node;
    if(layout == "r")
        return TextureAccessLayout::R;
    else if(layout == "g")
        return TextureAccessLayout::G;
    else if(layout == "b")
        return TextureAccessLayout::B;
    else if(layout == "a")
        return TextureAccessLayout::A;
    else if(layout == "rg")
        return TextureAccessLayout::RG;
    else if(layout == "rgb")
        return TextureAccessLayout::RGB;
    else if(layout == "rgba")
        return TextureAccessLayout::RGBA;
    else throw SceneException(SceneError::UNKNOWN_TEXTURE_ACCESS_LAYOUT);
}

std::vector<TextureStruct> SceneIO::LoadTexture(const nlohmann::json& jsn)
{
    std::vector<TextureStruct> result;

    const nlohmann::json& id = jsn[ID];
    if(id.is_array())
    {
        // Pre-check singed data is avail
        bool hasSignedNode = (jsn.find(TEXTURE_SIGNED) != jsn.cend());

        size_t texCount = id.size();
        for(size_t i = 0; i < texCount; i++)
        {
            TextureStruct tex;
            tex.texId = jsn[ID][i];
            tex.filePath = jsn[TEXTURE_FILE][i];
            // Optional Signed data flag
            if(hasSignedNode)
                tex.isSigned = jsn[TEXTURE_SIGNED][i];
            else
                tex.isSigned = false;

            result.push_back(tex);
        }
    }
    else
    {
        TextureStruct singleTex;
        singleTex.texId = jsn[ID];
        singleTex.filePath = jsn[TEXTURE_FILE];
        // Optional Signed data flag
        auto loc = jsn.cend();
        if((loc = jsn.find(TEXTURE_SIGNED)) != jsn.cend())
            singleTex.isSigned = *loc;
        else singleTex.isSigned = false;

        result.push_back(singleTex);
    }
    return result;
}

SurfaceStruct SceneIO::LoadSurface(const nlohmann::json& jsn)
{
    // Load as array
    if(jsn.is_array())
    {
        // Array type does not have any light info
        // just try to fetch it

        SurfaceStruct s = {};
        s.transformId = jsn[0];
        s.acceleratorId = jsn[1];
        s.matPrimPairs.fill(std::make_pair(std::numeric_limits<uint32_t>::max(),
                                           std::numeric_limits<uint32_t>::max()));

        const auto& material = jsn[2];
        const auto& primitive = jsn[3];
        if(primitive.size() != material.size())
            throw SceneException(SceneError::PRIM_MATERIAL_NOT_SAME_SIZE);

        s.pairCount = static_cast<uint8_t>(material.size());
        if(s.pairCount >= SceneConstants::MaxPrimitivePerSurface)
            throw SceneException(SceneError::TOO_MANY_SURFACE_ON_NODE);

        if(material.size() == 1)
        {
            std::get<SurfaceStruct::MATERIAL_INDEX>(s.matPrimPairs[0]) = material;
            std::get<SurfaceStruct::PRIM_INDEX>(s.matPrimPairs[0]) = primitive;
        }
        else for(int i = 0; i < static_cast<int>(material.size()); i++)
        {
            std::get<SurfaceStruct::MATERIAL_INDEX>(s.matPrimPairs[i]) = material[i];
            std::get<SurfaceStruct::PRIM_INDEX>(s.matPrimPairs[i]) = primitive[i];
        }
        return s;
    }
    else
    {
        SurfaceStruct s = {};
        s.transformId = jsn[TRANSFORM];
        s.acceleratorId = jsn[ACCELERATOR];
        s.matPrimPairs.fill(std::make_pair(std::numeric_limits<uint32_t>::max(),
                                           std::numeric_limits<uint32_t>::max()));

        // Array Like Couples
        const auto primIdArray = jsn[PRIMITIVE];
        const auto materialIdArray = jsn[MATERIAL];
        if(primIdArray.size() != materialIdArray.size())
            throw SceneException(SceneError::PRIM_MATERIAL_NOT_SAME_SIZE);
        if(primIdArray.size() > SceneConstants::MaxPrimitivePerSurface)
            throw SceneException(SceneError::TOO_MANY_SURFACE_ON_NODE);

        if(primIdArray.size() == 1)
        {
            s.matPrimPairs[0] = std::make_pair(materialIdArray, primIdArray);
        }
        else for(int i = 0; i < static_cast<int>(primIdArray.size()); i++)
            s.matPrimPairs[i] = std::make_pair(materialIdArray[i], primIdArray[i]);

        std::sort(s.matPrimPairs.begin(), s.matPrimPairs.end());
        s.pairCount = static_cast<int8_t>(primIdArray.size());
        return s;
    }
}

LightSurfaceStruct SceneIO::LoadLightSurface(uint32_t baseMediumId,
                                             uint32_t identityTransformId,
                                             const nlohmann::json& jsnNode,
                                             const nlohmann::json& jsnLights,
                                             const IndexLookup& lightIndexLookup)
{
    LightSurfaceStruct s;

    // Transform
    auto i = jsnNode.end();
    if((i = jsnNode.find(TRANSFORM)) != jsnNode.end())
        s.transformId = *i;
    else s.transformId = identityTransformId;

    // Medium
    i = jsnNode.end();
    if((i = jsnNode.find(MEDIUM)) != jsnNode.end())
        s.mediumId = *i;
    else s.mediumId = baseMediumId;

    // Light Id
    if((i = jsnNode.find(LIGHT)) != jsnNode.end())
    {
        s.lightId = *i;
    }
    else throw SceneException(SceneError::TYPE_MISMATCH);

    // Check if this light is primitive
    if(auto loc = lightIndexLookup.find(s.lightId); loc != lightIndexLookup.end())
    {
        const NodeIndex nIndex = loc->second.first;
        //const InnerIndex iIndex = loc->second.second;

        const auto& lightJSNNode = jsnLights[nIndex];
        std::string lightTypeName = lightJSNNode[NodeNames::TYPE];

        s.isPrimitive = (lightTypeName == NodeNames::LIGHT_TYPE_PRIMITIVE);
        if(s.isPrimitive)
        {
            s.acceleratorId = lightJSNNode[NodeNames::ACCELERATOR];
            s.primId = lightJSNNode[NodeNames::PRIMITIVE];
        }
        else
        {
            s.acceleratorId = std::numeric_limits<uint32_t>::max();
            s.primId = std::numeric_limits<uint32_t>::max();
        }
    }
    else throw SceneException(SceneError::LIGHT_ID_NOT_FOUND);

    return s;
}

CameraSurfaceStruct SceneIO::LoadCameraSurface(uint32_t baseMediumId,
                                               uint32_t identityTransformId,
                                               const nlohmann::json& jsn)
{
    CameraSurfaceStruct s;

    // Transform
    auto i = jsn.end();
    if((i = jsn.find(TRANSFORM)) != jsn.end())
        s.transformId = *i;
    else s.transformId = identityTransformId;

    // Medium
    i = jsn.end();
    if((i = jsn.find(MEDIUM)) != jsn.end())
        s.mediumId = *i;
    else s.mediumId = baseMediumId;

    // CamId
    s.cameraId = jsn[CAMERA];

    return s;
}

NodeTextureStruct SceneIO::LoadNodeTextureStruct(const nlohmann::json& node,
                                                 double)
{
    NodeTextureStruct s;
    s.texId = LoadNumber<uint32_t>(node[TEXTURE_NAME]);
    s.channelLayout = LoadTextureAccessLayout(node[TEXTURE_CHANNEL]);
    return s;
}
