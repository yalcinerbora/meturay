#include "TextureFunctions.h"

#include "ImageIO/EntryPoint.h"

#include <cuda_fp16.h>

//#include "TracerDebug.h"

SceneError ConvertImageIOErrorToSceneError(ImageIOError e)
{
    switch(e)
    {
        case ImageIOError::IMAGE_NOT_FOUND:
            return SceneError::TEXTURE_NOT_FOUND;
        case ImageIOError::UNKNOWN_IMAGE_TYPE:
            return SceneError::UNKNOWN_TEXTURE_TYPE;

        // TODO More Specific Texture Errors
        case ImageIOError::TYPE_IS_NOT_SIGN_CONVERTIBLE:
        case ImageIOError::READ_INTERNAL_ERROR:
        case ImageIOError::UNKNOWN_PIXEL_FORMAT:
        default:
            return SceneError::UNABLE_TO_LOAD_TEXTURE;
    }
}

SceneError TextureLoader::LoadTexture2D(std::unique_ptr<TextureI<2>>& tex,
                                        EdgeResolveType edgeR, InterpolationType interp,
                                        bool normalizeIntegers,
                                        bool normalizeCoordinates,
                                        bool asSigned,
                                        const CudaGPU& gpu,
                                        const std::string& filePath)
{
    // Always try 3 Channel -> 4 Channel conversion
    // since CUDA does not support 3 Channel textures
    ImageIOFlags flags = ImageIOI::TRY_3C_4C_CONVERSION;
    // If requested load as signed
    if(asSigned) flags |= ImageIOI::LOAD_AS_SIGNED;

    // Load Image
    Vector2ui dim;
    PixelFormat pf;
    std::vector<Byte> textureData;
    ImageIOError e = ImageIOInstance()->ReadImage(textureData,
                                                  pf, dim,
                                                  filePath,
                                                  flags);
    // Check ImageIOError and convert it to scene error
    if(e != ImageIOError::OK)
        return ConvertImageIOErrorToSceneError(e);

    // Convenience Lambda ("Construct Texture")
    // Also loads the pixel data
    auto CT = [&]<class T>(bool falseNorm = false) -> auto
    {
        bool normInts = (falseNorm) ? false : normalizeIntegers;
        std::unique_ptr<T> ptr = std::make_unique<T>(&gpu,
                                                     interp,
                                                     edgeR,
                                                     normInts,
                                                     normalizeCoordinates,
                                                     false,
                                                     dim,
                                                     1);
        ptr->Copy(textureData.data(), dim);
        return ptr;
    };

    // According to the pixel format allocate texture
    switch(pf)
    {
        case PixelFormat::R8_UNORM:     tex = CT.operator()<Texture2D<uint8_t>>(); break;
        case PixelFormat::RG8_UNORM:    tex = CT.operator()<Texture2D<uchar2>>(); break;
        case PixelFormat::RGBA8_UNORM:  tex = CT.operator()<Texture2D<uchar4>>(); break;

        case PixelFormat::R16_UNORM:    tex = CT.operator()<Texture2D<uint16_t>>(); break;
        case PixelFormat::RG16_UNORM:   tex = CT.operator()<Texture2D<ushort2>>(); break;
        case PixelFormat::RGBA16_UNORM: tex = CT.operator()<Texture2D<ushort4>>(); break;

        case PixelFormat::R8_SNORM:     tex = CT.operator()<Texture2D<char>>(); break;
        case PixelFormat::RGB8_SNORM:   tex = CT.operator()<Texture2D<char2>>(); break;
        case PixelFormat::RGBA8_SNORM:  tex = CT.operator()<Texture2D<char4>>(); break;

        case PixelFormat::R16_SNORM:    tex = CT.operator()<Texture2D<int16_t>>(); break;
        case PixelFormat::RG16_SNORM:   tex = CT.operator()<Texture2D<short2>>(); break;
        case PixelFormat::RGBA16_SNORM: tex = CT.operator()<Texture2D<short4>>(); break;

        case PixelFormat::R_HALF:       tex = CT.operator()<Texture2D<half>>(true); break;
        case PixelFormat::RG_HALF:      tex = CT.operator()<Texture2D<half2>>(true); break;
        case PixelFormat::RGBA_HALF:
            // TODO: cuda does not have half4 type
            //tex = CT.operator()<Texture2D<half4>>(); break;
            return SceneError::UNABLE_TO_LOAD_TEXTURE;
        case PixelFormat::R_FLOAT:      tex = CT.operator()<Texture2D<float>>(true); break;
        case PixelFormat::RG_FLOAT:     tex = CT.operator()<Texture2D<Vector2>>(true); break;
        case PixelFormat::RGBA_FLOAT:   tex = CT.operator()<Texture2D<Vector4>>(true); break;
        // BC Compression data not yet supported (although CUDA supports it)
        case PixelFormat::BC1_U:
        case PixelFormat::BC2_U:
        case PixelFormat::BC3_U:
        case PixelFormat::BC4_U:
        case PixelFormat::BC4_S:
        case PixelFormat::BC5_U:
        case PixelFormat::BC5_S:
        case PixelFormat::BC6H_U:
        case PixelFormat::BC6H_S:
        case PixelFormat::BC7_U:
            return SceneError::UNKNOWN_TEXTURE_TYPE;
        // Cuda Does not Support 3 Channel Images
        // Also we did convert so code should not come here
        case PixelFormat::RGB8_UNORM:
        case PixelFormat::RG8_SNORM:
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGB_HALF:
        case PixelFormat::RGB_FLOAT:
            return SceneError::UNKNOWN_TEXTURE_TYPE;
        //
        case PixelFormat::END:
        default:
            return SceneError::UNKNOWN_TEXTURE_TYPE;
    }

    return SceneError::OK;
}

std::vector<TextureChannelType> TextureFunctions::TextureAccessLayoutToTextureChannels(TextureAccessLayout layout)
{
    switch(layout)
    {
        case TextureAccessLayout::R:
            return {TextureChannelType::R};
        case TextureAccessLayout::G:
            return {TextureChannelType::G};
        case TextureAccessLayout::B:
            return {TextureChannelType::B};
        case TextureAccessLayout::A:
            return {TextureChannelType::A};
        case TextureAccessLayout::RG:
            return {TextureChannelType::R,
                    TextureChannelType::G};
        case TextureAccessLayout::RGB:
            return {TextureChannelType::R,
                    TextureChannelType::G,
                    TextureChannelType::B};
        case TextureAccessLayout::RGBA:
            return {TextureChannelType::R,
                    TextureChannelType::G,
                    TextureChannelType::B,
                    TextureChannelType::A};
        default: return {};
    }
}

SceneError TextureFunctions::LoadBitMap(// Returned Bitmap Data
                                        std::vector<Byte>& bits,
                                        Vector2ui& dimension,
                                        // Requested Texture Information
                                        uint32_t textureId,
                                        TextureChannelType channel,
                                        // Available textures that are defined on the scene file
                                        const std::map<uint32_t, TextureStruct>& loadableTextureInfo,
                                        // Scene file path (all file specifiers are relative to this path)
                                        const std::string& scenePath)
{
    std::string filePath;
    auto j = loadableTextureInfo.cend();
    if((j = loadableTextureInfo.find(textureId)) != loadableTextureInfo.cend())
    {
        filePath = j->second.filePath;
    }
    else return SceneError::TEXTURE_ID_NOT_FOUND;

    // Combine file name and scene path for combined path
    std::string combinedPath = Utility::MergeFileFolder(scenePath, filePath);

    // Load the texture to the CPU first
    auto TexCTypeToImgageCType = [] (TextureChannelType t)
    {
        return static_cast<ImageChannelType>(t);
    };

    ImageIOError e = ImageIOError::OK;
    if((e = ImageIOInstance()->ReadImageChannelAsBitMap(bits, dimension,
                                                        TexCTypeToImgageCType(channel),
                                                        combinedPath)) != ImageIOError::OK)
        return ConvertImageIOErrorToSceneError(e);
    // All done
    return SceneError::OK;
}