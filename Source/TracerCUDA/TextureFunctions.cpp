#include "TextureFunctions.h"
//#include "TracerDebug.h"

TextureLoader::TextureLoader()
{
    FreeImage_Initialise();
}

TextureLoader::~TextureLoader()
{
    FreeImage_DeInitialise();
}

SceneError TextureLoader::LoadTexture2D(std::unique_ptr<TextureI<2>>& tex,
                                        EdgeResolveType edgeR, InterpolationType interp,
                                        bool normalizeIntegers,
                                        bool normalizeCoordinates,
                                        bool asSigned,
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

        uint32_t channels = ColorTypeToChannelCount(colorType);
        if(channels == 0) return SceneError::UNABLE_TO_LOAD_TEXTURE;
        // It looks ok (channels match etc.)
        // Generate Texutre and Copy Image to GPU
        std::vector<Byte> pixelBuffer;
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
                    BYTE* imgPixels = FreeImage_GetBits(imgCPU);
                    Byte* srcPixels;
                    if(bpp == 24)
                    {
                        Expand2DData3ChannelTo4Channel<unsigned char>(pixelBuffer,
                                                                      imgPixels,
                                                                      dimension,
                                                                      pitch,
                                                                      asSigned);
                        srcPixels = pixelBuffer.data();
                    }
                    else srcPixels = imgPixels;

                    if(asSigned)
                    {
                        auto texPtr = std::make_unique<Texture2D<char4>>(gpu.DeviceId(),
                                                                         interp,
                                                                         edgeR,
                                                                         normalizeIntegers,
                                                                         normalizeCoordinates,
                                                                         false,
                                                                         dimension,
                                                                         1);
                        texPtr->Copy(srcPixels, dimension);
                        // Transfer to the Interface ptr
                        tex = std::move(texPtr);
                    }                       
                    else
                    {
                        auto texPtr = std::make_unique<Texture2D<uchar4>>(gpu.DeviceId(),
                                                                          interp,
                                                                          edgeR,
                                                                          normalizeIntegers,
                                                                          normalizeCoordinates,
                                                                          false,
                                                                          dimension,
                                                                          1);
                        texPtr->Copy(srcPixels, dimension);
                        // Transfer to the Interface ptr
                        tex = std::move(texPtr);
                    }
                }
                // Skip low bitrate bitmaps
                else return SceneError::UNABLE_TO_LOAD_TEXTURE;
                break;
            }
            case FREE_IMAGE_TYPE::FIT_RGBF:
            {
                // Probably HDRI Image
                // Kinda dull but if but float image should be 32-bit
                if((bpp / channels) != sizeof(float) * BYTE_BITS)
                    return SceneError::UNABLE_TO_LOAD_TEXTURE;

                // Float textures cannot be normalized so set this as false
                normalizeIntegers = false;                
                // Allocate
                auto texPtr = std::make_unique<Texture2D<Vector4>>(gpu.DeviceId(),
                                                                   interp,
                                                                   edgeR,
                                                                   normalizeIntegers,
                                                                   normalizeCoordinates,
                                                                   false,
                                                                   dimension,
                                                                   1);

                BYTE* imgPixels = FreeImage_GetBits(imgCPU);
                Expand2DData3ChannelTo4Channel<float>(pixelBuffer,
                                                      FreeImage_GetBits(imgCPU),
                                                      dimension,
                                                      pitch, false);
                Byte* srcPixels = pixelBuffer.data();
                texPtr->Copy(srcPixels, dimension);

                // Transfer to the Interface ptr
                tex = std::move(texPtr);
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

SceneError TextureLoader::LoadBitMapFromTexture(std::vector<Byte>& bits,
                                                Vector2ui& dimension,
                                                TextureChannelType channel,
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
        dimension = Vector2ui(w, h);

        FREE_IMAGE_TYPE imgType = FreeImage_GetImageType(imgCPU);
        FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(imgCPU);

        // Check channel count
        uint32_t channels = ColorTypeToChannelCount(colorType);
        if(channels == 0) return SceneError::UNABLE_TO_LOAD_TEXTURE;

        // Check if requested channel is available on this texture
        uint32_t channelIndex = TextureFunctions::ConvertChannelTypeToChannelIndex(channel);
        if(channelIndex >= channels)
            return SceneError::TEXTURE_CHANNEL_MISMATCH;

        // It looks ok (channels match etc.)
        // Generate Texutre and Copy Image to GPU  
        switch(imgType)
        {
            case FREE_IMAGE_TYPE::FIT_BITMAP:
            {
                // Equal mask is mandatory for bitmap images
                int rMask = FreeImage_GetRedMask(imgCPU);
                int gMask = FreeImage_GetGreenMask(imgCPU);
                int bMask = FreeImage_GetBlueMask(imgCPU);

                // Only two bpp are supported
                if(bpp == 24 || bpp == 32)
                {
                    BYTE* imgPixels = FreeImage_GetBits(imgCPU);

                    PackImageChannelToBits<unsigned char>(bits,
                                                          imgPixels,
                                                          dimension,
                                                          pitch,
                                                          channelIndex,
                                                          channels);
                    break;
                }
                else return SceneError::UNABLE_TO_LOAD_TEXTURE;
                
            }
            // TODO: Add more later
            default:
                return SceneError::UNABLE_TO_LOAD_TEXTURE;
        }
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

uint32_t TextureFunctions::ConvertChannelTypeToChannelIndex(TextureChannelType channel)
{
    return static_cast<uint32_t>(channel);
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
    SceneError e = SceneError::OK;
    if((e = TextureLoader::Instance()->LoadBitMapFromTexture(bits, dimension, channel,
                                                             combinedPath)) != SceneError::OK)
        return e;

    // All done
    return SceneError::OK;
}