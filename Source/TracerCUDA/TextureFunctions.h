#pragma once

#include "RayLib/SceneError.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/Log.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/FileSystemUtility.h"

#include "Texture.cuh"
#include "CudaSystem.h"
#include "GPUBitmap.h"

#include <string>
#include <cstring>

template <int D>
using TextureAllocationMap = std::map<uint32_t, std::unique_ptr<TextureI<D>>>;

template<int D, class RType = SceneError>
using Tex2DEnable = typename std::enable_if<D == 2, RType>::type;

template<int D, class RType = SceneError>
using Tex3DEnable = typename std::enable_if<D == 3, RType>::type;

template<int D, class RType = SceneError>
using Tex1DEnable = typename std::enable_if<D == 1, RType>::type;

class TextureLoader
{
    private:
        static SceneError           LoadTexture2D(std::unique_ptr<TextureI<2>>&,
                                                  EdgeResolveType, InterpolationType,
                                                  bool normalizeIntegers,
                                                  bool normalizeCoordinates,
                                                  bool asSigned,
                                                  bool generateMipmaps,
                                                  const CudaGPU& gpu,
                                                  const std::string& filePath);

    protected:
    public:
        // Functionality
        template <int D>
        static SceneError           LoadTexture(std::unique_ptr<TextureI<D>>&,
                                                EdgeResolveType, InterpolationType,
                                                bool normalizeIntegers,
                                                bool normalizeCoordinates,
                                                bool asSigned,
                                                bool generateMipmaps,
                                                const CudaGPU& gpu,
                                                const std::string& filePath);
        //template <int D, int C>
        //static SceneError         LoadTextureArray(std::unique_ptr<TextureArrayI<D, C>>&,
        //                                             EdgeResolveType, InterpolationType,
        //                                             bool normalizeIntegers,
        //                                             const CudaGPU& gpu,
        //                                             const std::string& filePath);
        //template <int C>
        //static SceneError         LoadTextureCube(std::unique_ptr<TextureCubeI<C>>&,
        //                                            EdgeResolveType, InterpolationType,
        //                                            bool normalizeIntegers,
        //                                            const CudaGPU& gpu,
        //                                            const std::string& filePath);
};

namespace TextureFunctions
{
    // Firstly bitmaps require only one channel check if
    // access layout is single channel
    std::vector<TextureChannelType> TextureAccessLayoutToTextureChannels(TextureAccessLayout);
    uint32_t                        ConvertChannelTypeToChannelIndex(TextureChannelType);

    template <int D>
    SceneError AllocateTexture(// Returned Texture Ptr
                               const TextureI<D>*& tex,
                               // Allocation Data structure
                               TextureAllocationMap<D>& textureAllocations,
                               // Information about the texture (name and channels)
                               const NodeTextureStruct& requestedTextureInfo,
                               // Available textures that are defined on the scene file
                               const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                               // Texture Access Specifiers
                               EdgeResolveType,
                               InterpolationType,
                               bool normalizeIntegers,
                               bool normalizeCoordinates,
                               // GPU info
                               const CudaGPU& gpu,
                               // Scene file path (all file specifiers are relative to this path)
                               const std::string& scenePath);

    SceneError LoadBitMap(// Returned Bitmap Data
                          std::vector<Byte>& bits,
                          Vector2ui& dimension,
                          // Requested Texture Information
                          uint32_t textureId,
                          TextureChannelType  channel,
                          // Available textures that are defined on the scene file
                          const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                          // Scene file path (all file specifiers are relative to this path)
                          const std::string& scenePath);
}

template <int D>
SceneError TextureFunctions::AllocateTexture(// Returned Texture Ptr
                                             const TextureI<D>*& tex,
                                             // Allocation Data structure
                                             TextureAllocationMap<D>& textureAllocations,
                                             // Information about the texture (name and channels)
                                             const NodeTextureStruct& requestedTextureInfo,
                                             // Available textures that are defined on the scene file
                                             const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                                             // Texture Access Specifiers
                                             EdgeResolveType edgeResolve,
                                             InterpolationType interp,
                                             bool normalizeIntegers,
                                             bool normalizeCoordinates,
                                             // GPU info
                                             const CudaGPU& gpu,
                                             // Scene file path (all file specifiers are relative to this path)
                                             const std::string& scenePath)
{
    uint32_t textureId = requestedTextureInfo.texId;

    // Check if the tex is already loaded
    auto i = textureAllocations.cend();
    if((i = textureAllocations.find(textureId)) != textureAllocations.cend())
    {
        // Just return that texture
        tex = i->second.get();
        return SceneError::OK;
    }

    // Texture is not loaded load
    TextureStruct s;
    auto j = loadableTextureInfo.cend();
    if((j = loadableTextureInfo.find(textureId)) != loadableTextureInfo.cend())
    {
        s = j->second;
    }
    else return SceneError::TEXTURE_ID_NOT_FOUND;

    // Combine file name and scene path for combined path
    std::string combinedPath = Utility::MergeFileFolder(scenePath, s.filePath);

    // Load the texture to the CPU first
    SceneError e = SceneError::OK;
    std::unique_ptr<TextureI<D>> ptr;
    if((e = TextureLoader::LoadTexture(ptr,
                                       edgeResolve, interp,
                                       normalizeIntegers,
                                       normalizeCoordinates,
                                       s.isSigned,
                                       s.generateMipmaps,
                                       gpu,
                                       combinedPath)) != SceneError::OK)
        return e;

    // Emplace the loaded texture to the memory
    auto loc = textureAllocations.emplace(textureId, std::move(ptr));
    tex = loc.first->second.get();

    // All done
    return SceneError::OK;
}

template <int D>
SceneError TextureLoader::LoadTexture(std::unique_ptr<TextureI<D>>& t,
                                      EdgeResolveType edgeR, InterpolationType interp,
                                      bool normalizeIntegers,
                                      bool normalizeCoordinates,
                                      bool asSigned,
                                      bool generateMipmaps,
                                      const CudaGPU& gpu,
                                      const std::string& filePath)
{
    if constexpr(D == 2)
        // Delegate to the FreeImagLib Loading function
        return LoadTexture2D(t, edgeR, interp,
                             normalizeIntegers,
                             normalizeCoordinates,
                             asSigned,
                             generateMipmaps,
                             gpu, filePath);
    // TODO: add more texture loading functions
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;
}