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
#include <FreeImage.h>

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
    public:
        static TextureLoader&       Instance();

    private:
        static uint32_t             ColorTypeToChannelCount(FREE_IMAGE_COLOR_TYPE cType);

    protected:
    public:
        // Constructors & Destructor
                                    TextureLoader();
                                    TextureLoader(const TextureLoader&) = delete;
                                    TextureLoader(TextureLoader&&) = delete;
        TextureLoader&              operator=(const TextureLoader&) = delete;
        TextureLoader&              operator=(TextureLoader&&) = delete;
                                    ~TextureLoader();

        template<class T>
        static void                 ConvertData(std::vector<Byte>& expandedData,
                                                const Byte* packedData,
                                                const Vector2ui& dim,
                                                uint32_t sourcePitch,
                                                bool doSingedConversion,
                                                bool expand3DDataTo4D,
                                                uint32_t inChannelCount);

        template<class T>
        static void                 PackImageChannelToBits(std::vector<Byte>& bitmap,
                                                           const Byte* imgPixels,
                                                           const Vector2ui& dimension,
                                                           uint32_t pitch,
                                                           uint32_t channelIndex,
                                                           uint32_t imgChannelCount);


        // Functionality
        template <int D>
        SceneError                  LoadTexture(std::unique_ptr<TextureI<D>>&,
                                                EdgeResolveType, InterpolationType,
                                                bool normalizeIntegers,
                                                bool normalizeCoordinates,
                                                bool asSigned,
                                                const CudaGPU& gpu,
                                                const std::string& filePath);

        SceneError                  LoadTexture2D(std::unique_ptr<TextureI<2>>&,
                                                  EdgeResolveType, InterpolationType,
                                                  bool normalizeIntegers,
                                                  bool normalizeCoordinates,
                                                  bool asSigned,
                                                  const CudaGPU& gpu,
                                                  const std::string& filePath);
        //template <int D, int C>
        //SceneError                  LoadTextureArray(std::unique_ptr<TextureArrayI<D, C>>&,
        //                                             EdgeResolveType, InterpolationType,
        //                                             bool normalizeIntegers,
        //                                             const CudaGPU& gpu,
        //                                             const std::string& filePath);
        //template <int C>
        //SceneError                  LoadTextureCube(std::unique_ptr<TextureCubeI<C>>&,
        //                                            EdgeResolveType, InterpolationType,
        //                                            bool normalizeIntegers,
        //                                            const CudaGPU& gpu,
        //                                            const std::string& filePath);

        SceneError                  LoadBitMapFromTexture(std::vector<Byte>& bits,
                                                          Vector2ui& dimension,
                                                          TextureChannelType channel,
                                                          const std::string& filePath);
};

template<class T>
void TextureLoader::ConvertData(std::vector<Byte>& expandedData,
                                const Byte* packedData,
                                const Vector2ui& dim,
                                uint32_t sourcePitch,
                                bool doSingedConversion,
                                bool expand3DDataTo4D,
                                uint32_t inChannelCount)
{
    // Skip if no conversion
    if(!doSingedConversion && !expand3DDataTo4D)
        return;

    const uint32_t outChannelCount = expand3DDataTo4D ? 4: inChannelCount;

    expandedData.resize(sizeof(T) * outChannelCount * dim[0] * dim[1]);
    for(uint32_t y = 0; y < dim[1]; y++)
    {
        const Byte* srcRowPtr = packedData + sourcePitch * y;
        for(uint32_t x = 0; x < dim[0]; x++)
        {
            ptrdiff_t dstOffset = (dim[0] * y * sizeof(T) * outChannelCount +
                                            x * sizeof(T) * outChannelCount);
            ptrdiff_t srcRowOffset = (x * sizeof(T) * inChannelCount);

            Byte* destPixel = expandedData.data() + dstOffset;
            const Byte* srcPixel = srcRowPtr + srcRowOffset;

            std::memcpy(destPixel, srcPixel, sizeof(T) * inChannelCount);

            // Convert unsigned data to signed
            if(doSingedConversion)
            for(uint32_t i = 0; i < inChannelCount; i++)
            {
                T& t = *reinterpret_cast<T*>(destPixel + sizeof(T) * i);
                constexpr T mid = static_cast<T>(0x1 << ((sizeof(T) * 8) - 1));
                t -= mid;
            }
        }
    }
}

template<class T>
void TextureLoader::PackImageChannelToBits(std::vector<Byte>& bitmap,
                                           const Byte* imgPixels,
                                           const Vector2ui& dim,
                                           uint32_t sourcePitch,
                                           uint32_t channelIndex,
                                           uint32_t imgChannelCount)
{
    size_t bitmapByteSize = (dim[0] * dim[1] + BYTE_BITS - 1) / BYTE_BITS;
    bitmap.resize(bitmapByteSize, 0);
    for(uint32_t y = 0; y < dim[1]; y++)
    {
        const Byte* srcRowPtr = imgPixels + sourcePitch * y;
        for(uint32_t x = 0; x < dim[0]; x++)
        {
            // Pixel by pixel copy
            ptrdiff_t srcRowOffset = (x * sizeof(T) * imgChannelCount + channelIndex);
            T srcPixel = *reinterpret_cast<const T*>(srcRowPtr + srcRowOffset);
            bool srcBit = srcPixel;

            // Find Destination Bit
            if(srcBit)
            {
                size_t pixelLinearIndex = (y * dim[0] + x);
                size_t byteIndex = pixelLinearIndex / BYTE_BITS;
                size_t byteInnerIndex = pixelLinearIndex % BYTE_BITS;
                bitmap[byteIndex] |= (0x1 << byteInnerIndex);
            }
        }
    }
}

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
    if((e = TextureLoader::Instance()->LoadTexture(ptr,
                                                   edgeResolve, interp,
                                                   normalizeIntegers,
                                                   normalizeCoordinates,
                                                   s.isSigned,
                                                   gpu,
                                                   combinedPath)) != SceneError::OK)
        return e;

    // Emplace the loaded texture to the memory
    auto loc = textureAllocations.emplace(textureId, std::move(ptr));
    tex = loc.first->second.get();

    // All done
    return SceneError::OK;
}

inline uint32_t TextureLoader::ColorTypeToChannelCount(FREE_IMAGE_COLOR_TYPE cType)
{
    switch(cType)
    {
        case FREE_IMAGE_COLOR_TYPE::FIC_RGB:
        case FREE_IMAGE_COLOR_TYPE::FIC_CMYK:
            return 3;
        case FREE_IMAGE_COLOR_TYPE::FIC_RGBALPHA:
            return 4;
        case FREE_IMAGE_COLOR_TYPE::FIC_MINISBLACK:
            return 1;
        default:
            return 0;
    }
}

template <int D>
SceneError TextureLoader::LoadTexture(std::unique_ptr<TextureI<D>>& t,
                                      EdgeResolveType edgeR, InterpolationType interp,
                                      bool normalizeIntegers,
                                      bool normalizeCoordinates,
                                      bool asSigned,
                                      const CudaGPU& gpu,
                                      const std::string& filePath)
{
    if constexpr(D == 2)
        // Delegate to the FreeImagLib Loading function
        return LoadTexture2D(t, edgeR, interp,
                             normalizeIntegers,
                             normalizeCoordinates,
                             asSigned, gpu, filePath);
    // TODO: add more texture loading functions
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;

    return SceneError::OK;
}