#pragma once

#include "RayLib/SceneError.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/Log.h"
#include "Texture.cuh"

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
                                                const CudaGPU& gpu,
                                                const std::string& filePath);


        template <int C>
        SceneError                  LoadTexture2D(std::unique_ptr<TextureI<2, C>>&,
                                                  EdgeResolveType, InterpolationType,
                                                  bool normalizeIntegers,
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
                               const MaterialTextureStruct& requestedTextureInfo,
                               // Available textures that are defined on the scene file
                               const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                               // Texture Access Specifiers
                               EdgeResolveType, 
                               InterpolationType,
                               bool normalizeIntegers,
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
                                             const MaterialTextureStruct& requestedTextureInfo,
                                             // Available textures that are defined on the scene file
                                             const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                                             // Texture Access Specifiers
                                             EdgeResolveType edgeResolve,
                                             InterpolationType interp,
                                             bool normalizeIntegers,
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
                                      const CudaGPU& gpu,
                                      const std::string& filePath)
{
    if constexpr(D == 2)
        // Delegate to the FreeImagLib Loading function
        return LoadTexture2D(t, edgeR, interp, normalizeIntegers,
                             gpu, filePath);
    // TODO: add more texture loading functions
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;
    
    return SceneError::OK;
}


template <int C>
SceneError TextureLoader::LoadTexture2D(std::unique_ptr<TextureI<2, C>>&,
                                        EdgeResolveType, InterpolationType,
                                        bool normalizeIntegers,
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

        FREE_IMAGE_TYPE imgType = FreeImage_GetImageType(imgCPU);
        FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(imgCPU);

        int channels = ColorTypeToChannelCount(colorType);
        if(channels == 0) return SceneError::UNABLE_TO_LOAD_TEXTURE;

        // Cuda Textures does not support 3 channel textures convert accordingly
        int cudaChannels = (channels == 3) ? 4 : channels;
        if(cudaChannels != C) 
            return SceneError::TEXTURE_CHANNEL_MISMATCH;



        BYTE* pixels = FreeImage_GetBits(imgCPU);
        for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++)
        {
            RGBQUAD c;
            bool b = FreeImage_GetPixelColor(imgCPU, x, y, c)

            METU_DEBUG_LOG(c)
        }

        // It looks ok
        std::unique_ptr<TextureI<2, C>> tex;
        switch(imgType)
        {
            case FREE_IMAGE_TYPE::FIT_BITMAP:
            {
                // Equal mask is mandatory for bitmap images
                int rMask = FreeImage_GetRedMask(imgCPU);
                int gMask = FreeImage_GetGreenMask(imgCPU);
                int bMask = FreeImage_GetBlueMask(imgCPU);

                // Classic bitmap determine channel from
                // bpp
                if(bpp == 16)
                {
                    
                }
                else if(bpp == 24)
                {

                }
                else if(bpp == 32)
                {

                }
                // Skip low bitrate bitmaps
                else return SceneError::UNABLE_TO_LOAD_TEXTURE;
                break;
            }
        }


        // Do some checks for compatilbility
        

        

        //if((e = ConstructTexture(Channel,...)) != SceneError::OK)
        //    return e;

        //tex = std::make_unique(new Texture<D, >(
        //    gpu.DeviceId(),
        //    interp, edgeR, normalizeIntegers,
        //    
        //    ));



        //FreeImage_GetBits()




        FreeImage_Unload(imgCPU);
    }
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;


    return SceneError::OK;
}