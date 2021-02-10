#pragma once

#include "RayLib/SceneError.h"
#include "RayLib/SceneStructs.h"
#include "Texture.cuh"

#include <string>
#include <FreeImage.h>

class TextureLoader
{
    public:
        static std::unique_ptr<TextureLoader>& Instance();       

    private:
    protected:
    public:
        // Constructors & Destructor
                                    TextureLoader();
                                    TextureLoader(const TextureLoader&) = delete;
                                    TextureLoader(TextureLoader&&) = delete;
        TextureLoader&              operator=(const TextureLoader&) = delete;
        TextureLoader&              operator=(TextureLoader&&) = delete;
                                    ~TextureLoader();

        // Functionality
        SceneError                  LoadTexture(const std::string& filePath);
};

inline std::unique_ptr<TextureLoader>& TextureLoader::Instance()
{
    // Singleton using unique_ptr
    static std::unique_ptr<TextureLoader> tMan = nullptr;
    if(tMan) return tMan;
    tMan = std::make_unique<TextureLoader>();
    return tMan;
}

namespace TextureFunctions
{
    template <int D, class T>
    SceneError AllocateTexture(const Texture<D, T>*& tex,
                               std::map<uint32_t, Texture<2, T>>& textureAllocations,
                               const MaterialTextureStruct& requestedTextureInfo,
                               const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                               const std::string& scenePath);
}

template <int D, class T>
SceneError TextureFunctions::AllocateTexture(const Texture<D, T>*& tex,
                                             std::map<uint32_t, Texture<2, T>>& textureAllocations,
                                             const MaterialTextureStruct& requestedTextureInfo,
                                             const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                                             const std::string& scenePath)
{
    uint32_t textureId = requestedTextureInfo.texId;

    // Check if the tex is already loaded
    auto i = textureAllocations.cend();
    if((i = textureAllocations.find(textureId)) != textureAllocations.cend())
    {
        // Just return that texture
        tex = &i->second;
        return SceneError::OK;
    }

    // Load the texture to the CPU first


    return SceneError::TEXTURE_NOT_FOUND;
}
