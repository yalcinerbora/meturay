#pragma once

#include "RayLib/SceneError.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/Log.h"
#include "RayLib/BitManipulation.h"
#include "Texture.cuh"
#include "CudaConstants.h"

#include <string>
#include <FreeImage.h>

template <int D, int C>
using TextureAllocationMap = std::map<uint32_t, std::unique_ptr<TextureI<D, C>>>;

template<int D, class RType = SceneError>
using Tex2DEnable = typename std::enable_if<D==2, RType>::type;

template<int D, class RType = SceneError>
using Tex3DEnable = typename std::enable_if<D == 3, RType>::type;

template<int D, class RType = SceneError>
using Tex1DEnable = typename std::enable_if<D == 1, RType>::type;

class TextureLoader
{
    public:
        static std::unique_ptr<TextureLoader>& Instance();

    private:
        static int                  ColorTypeToChannelCount(FREE_IMAGE_COLOR_TYPE cType);

        template<class T>
        static void                 Expand2DData3ChannelTo4Channel(std::vector<Byte>& expandedData,
                                                                   const Byte* packedData,
                                                                   const Vector2ui& dim,
                                                                   uint32_t sourcePitch);

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
        template <int D, int C>
        SceneError                  LoadTexture(std::unique_ptr<TextureI<D, C>>&,
                                                EdgeResolveType, InterpolationType,
                                                bool normalizeIntegers,
                                                bool normalizeCoordinates,
                                                const CudaGPU& gpu,
                                                const std::string& filePath);

        template <int C>
        SceneError                  LoadTexture2D(std::unique_ptr<TextureI<2, C>>&,
                                                  EdgeResolveType, InterpolationType,
                                                  bool normalizeIntegers,
                                                  bool normalizeCoordinates,
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
};

template<class T>
void TextureLoader::Expand2DData3ChannelTo4Channel(std::vector<Byte>& expandedData,
                                                   const Byte* packedData,
                                                   const Vector2ui& dim,
                                                   uint32_t sourcePitch)
{
    expandedData.resize(sizeof(T) * 4 * dim[0] * dim[1]);
    for(uint32_t y = 0; y < dim[1]; y++)
    {
        const Byte* srcRowPtr = packedData + sourcePitch * y;
        for(uint32_t x = 0; x < dim[0]; x++)
        {
            // Pixel by pixel copy
            ptrdiff_t dstOffset = (dim[0] * y * sizeof(T) * 4 +
                                            x * sizeof(T) * 4);
            ptrdiff_t srcRowOffset = (x * sizeof(T) * 3);

            Byte* destPixel = expandedData.data() + dstOffset;
            const Byte* srcPixel = srcRowPtr + srcRowOffset;
            std::memcpy(destPixel, srcPixel, sizeof(T) * 3);
        }
    }
}

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
    template <int D, int C>
    SceneError AllocateTexture(// Returned Texture Ptr
                               const TextureI<D, C>*& tex,
                               // Allocation Data structure
                               TextureAllocationMap<D, C>& textureAllocations,
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
}

template <int D, int C>
SceneError TextureFunctions::AllocateTexture(// Returned Texture Ptr
                                             const TextureI<D, C>*& tex,
                                             // Allocation Data structure
                                             TextureAllocationMap<D, C>& textureAllocations,
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
    std::string filePath;
    auto j = loadableTextureInfo.cend();
    if((j = loadableTextureInfo.find(textureId)) != loadableTextureInfo.cend())
    {
        filePath = j->second.filePath;
    }
    else return SceneError::TEXTURE_ID_NOT_FOUND;

    // Combine file name and scene path for combined path
    std::string combinedPath = scenePath + "/" + filePath;

    // Load the texture to the CPU first
    SceneError e = SceneError::OK;
    std::unique_ptr<TextureI<D, C>> ptr;
    if((e = TextureLoader::Instance()->LoadTexture(ptr,
                                                   edgeResolve, interp,
                                                   normalizeIntegers,
                                                   normalizeCoordinates,
                                                   gpu,
                                                   combinedPath)) != SceneError::OK)
        return e;

    // Emplace the loaded texture to the memory
    auto loc = textureAllocations.emplace(textureId, std::move(ptr));
    tex = loc.first->second.get();

    // All done
    return SceneError::OK;
}

inline int TextureLoader::ColorTypeToChannelCount(FREE_IMAGE_COLOR_TYPE cType)
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

template <int D, int C>
SceneError TextureLoader::LoadTexture(std::unique_ptr<TextureI<D, C>>& t,
                                      EdgeResolveType edgeR, InterpolationType interp,
                                      bool normalizeIntegers,
                                      bool normalizeCoordinates,
                                      const CudaGPU& gpu,
                                      const std::string& filePath)
{
    if constexpr(D == 2)
        // Delegate to the FreeImagLib Loading function
        return LoadTexture2D(t, edgeR, interp,
                             normalizeIntegers,
                             normalizeCoordinates,
                             gpu, filePath);
    // TODO: add more texture loading functions
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;

    return SceneError::OK;
}

template <int C>
SceneError TextureLoader::LoadTexture2D(std::unique_ptr<TextureI<2, C>>& tex,
                                        EdgeResolveType edgeR, InterpolationType interp,
                                        bool normalizeIntegers,
                                        bool normalizeCoordinates,
                                        const CudaGPU& gpu,
                                        const std::string& filePath)
{
    FREE_IMAGE_FORMAT f;
    // Check file to find type
    if((f = FreeImage_GetFileType(filePath.c_str())) == FIF_UNKNOWN)
        // Use file extension to determine type
        f = FreeImage_GetFIFFromFilename(filePath.c_str());
    // If it is still unknown just return error
    if(f == FIF_UNKNOWN)
        return SceneError::UNKNOWN_TEXTURE_TYPE;

    FIBITMAP* imgCPU = FreeImage_Load(f, filePath.c_str(), 0);
    if(imgCPU)
    {
        // Bit per pixel
        uint32_t bpp = FreeImage_GetBPP(imgCPU);
        uint32_t w = FreeImage_GetWidth(imgCPU);
        uint32_t h = FreeImage_GetHeight(imgCPU);
        uint32_t pitch = FreeImage_GetPitch(imgCPU);
        const Vector2ui dimension(w, h);

        FREE_IMAGE_TYPE imgType = FreeImage_GetImageType(imgCPU);
        FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(imgCPU);

        int channels = ColorTypeToChannelCount(colorType);
        if(channels == 0) return SceneError::UNABLE_TO_LOAD_TEXTURE;

        // Cuda Textures does not support 3 channel textures convert accordingly
        int cudaChannels = (channels == 3) ? 4 : channels;
        if(cudaChannels != C)
            return SceneError::TEXTURE_CHANNEL_MISMATCH;

        // It looks ok (channels match etc.)
        // Generate Texutre and Copy Image to GPU
        std::vector<Byte> expandedPixels;
        switch(imgType)
        {
            case FREE_IMAGE_TYPE::FIT_BITMAP:
            {
                // Equal mask is mandatory for bitmap images
                int rMask = FreeImage_GetRedMask(imgCPU);
                int gMask = FreeImage_GetGreenMask(imgCPU);
                int bMask = FreeImage_GetBlueMask(imgCPU);

                if(Utility::BitCount(rMask) != Utility::BitCount(gMask) ||
                   Utility::BitCount(rMask) != Utility::BitCount(bMask))
                    return SceneError::UNABLE_TO_LOAD_TEXTURE;

                // Only two bpp are supported
                if(bpp == 24 ||
                   bpp == 32)
                {
                    // std::unique_ptr<Texture2D<uchar4>>
                    auto texPtr = std::make_unique<Texture2D<uchar4>>(gpu.DeviceId(),
                                                                      interp,
                                                                      edgeR,
                                                                      normalizeIntegers,
                                                                      normalizeCoordinates,
                                                                      false,
                                                                      dimension,
                                                                      1);

                    BYTE* imgPixels = FreeImage_GetBits(imgCPU);
                    Byte* srcPixels;
                    if(bpp == 24)
                    {
                        Expand2DData3ChannelTo4Channel<unsigned char>(expandedPixels,
                                                                      imgPixels,
                                                                      dimension,
                                                                      pitch);
                        srcPixels = expandedPixels.data();
                    }
                    else srcPixels = imgPixels;
                    texPtr->Copy(srcPixels, dimension);

                    // Transfer to the Interface ptr
                    tex = std::move(texPtr);
                }
                // Skip low bitrate bitmaps
                else return SceneError::UNABLE_TO_LOAD_TEXTURE;
                break;
            }
            // TODO: Add other tpyes of textures (16bit 32bit HDR etc.)
            default:
                return SceneError::UNABLE_TO_LOAD_TEXTURE;
        }
        // All done!
        // Unallocate cpu image
        FreeImage_Unload(imgCPU);
    }
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;

    return SceneError::OK;
}