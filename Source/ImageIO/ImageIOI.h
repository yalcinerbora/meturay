#pragma once

#pragma once

#include <vector>
#include <string>
#include <memory>

#include "RayLib/Vector.h"
#include "RayLib/Types.h"
#include "RayLib/ImageIOError.h"

enum class ImageType
{
    PNG,
    JPG,
    BMP,
    HDR,
    EXR
};

class ImageIOI
{
    private:
        template <class T>
        bool                    IsCompatibleType(PixelFormat);

    protected:
        virtual bool            IsConvertible(PixelFormat toFormat, PixelFormat fromFormat) const = 0;
        virtual void            CopyPixels(Byte* toData, PixelFormat toFormat,
                                           const Byte* fromData, PixelFormat fromFormat,
                                           const Vector2ui& dimension) const = 0;

    public:        
        virtual                 ~ImageIOI() = default;

        // Read Functions
        // Try to read any image as is
        // Always pack the data (no scanline stuff)
        virtual ImageIOError   ReadImage(std::vector<Byte>&,
                                         PixelFormat&, Vector2ui& dimensions,
                                         const std::string& filePath) const = 0;        
        // Read a channel as alpha bit map
        virtual ImageIOError   ReadImageAlphaChannelAsBitMap(std::vector<Byte>&,
                                                             Vector2ui& dimension,
                                                             const std::string& filePath) const = 0;
        // Read and Try to Convert Image
        template<class T>
        ImageIOError            ReadAndConvertImage(std::vector<T>&, Vector2ui& dimension,
                                                    PixelFormat requestedFormat,
                                                    const std::string& filePath);

        // Write Functions
        virtual ImageIOError    WriteImage(const Byte* data,
                                           const Vector2ui& dimension,
                                           PixelFormat, ImageType,
                                           const std::string& filePath) const = 0;
        virtual ImageIOError    WriteBitmap(const Byte* bits,
                                            const Vector2ui& dimension,
                                            const std::string& filePath) const = 0;
        template<class T>
        ImageIOError            WriteImage(const std::vector<T>& data,                                           
                                           const Vector2ui& dimension,
                                           PixelFormat, ImageType,
                                           const std::string& filePath) const;
};

template <class T>
bool ImageIOI::IsCompatibleType(PixelFormat pf)
{
    // If Byte it is ok
    if(std::is_same_v<T, Byte>) return true;

    // Force some Types here
    switch(pf)
    {
        case PixelFormat::R8_UNORM:     return std::is_same_v<T, uint8_t>;        
        case PixelFormat::RG8_UNORM:    return std::is_same_v<T, Vector2uc>;
        case PixelFormat::RGB8_UNORM:   return std::is_same_v<T, Vector3uc>;
        case PixelFormat::RGBA8_UNORM:  return std::is_same_v<T, Vector4uc>;

        case PixelFormat::R16_UNORM:    return std::is_same_v<T, uint16_t>;
        case PixelFormat::RG16_UNORM:   return std::is_same_v<T, Vector2us>;
        case PixelFormat::RGB16_UNORM:  return std::is_same_v<T, Vector3us>;
        case PixelFormat::RGBA16_UNORM: return std::is_same_v<T, Vector4us>;
        // We dont have half type in the project so return false
        case PixelFormat::R_HALF:
        case PixelFormat::RG_HALF:
        case PixelFormat::RGB_HALF:
        case PixelFormat::RGBA_HALF:
            return false;
        case PixelFormat::R_FLOAT:      return std::is_same_v<T, float>;
        case PixelFormat::RG_FLOAT:     return std::is_same_v<T, Vector2f>;
        case PixelFormat::RGB_FLOAT:    return std::is_same_v<T, Vector3f>;
        case PixelFormat::RGBA_FLOAT:   return std::is_same_v<T, Vector4f>;
        // We dont have BC conversion utiltiy so skip
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
        case PixelFormat::END:
        default:
            return false;
    }
    return false;
}

template<class T>
ImageIOError ImageIOI::WriteImage(const std::vector<T>& data,
                                  const Vector2ui& dimension,
                                  PixelFormat pf, ImageType it,
                                  const std::string& filePath) const
{
    const Byte* dataPtr = reinterpret_cast<const Byte*>(data.data());
    return WriteImage(dataPtr, dimension, pf, it, filePath);
}

template<class T>
ImageIOError ImageIOI::ReadAndConvertImage(std::vector<T>& data, Vector2ui& dimension,
                                           PixelFormat requestedFormat,
                                           const std::string& filePath)
{
    std::vector<Byte> imgData;
    PixelFormat imgPixFormat;

    // Read the Image
    ImageIOError e = ImageIOError::OK;
    if((e = ReadImage(imgData, imgPixFormat, dimension, filePath)) != ImageIOError::OK)
        return ImageIOError(e, filePath);

    // Check if the template data type is compatible for the requested format
    if(!IsCompatibleType<T>(requestedFormat))
        return ImageIOError(ImageIOError::TEMPLATE_TYPE_IS_NOT_COMPATIBLE, filePath);
    // Check if formats are convertible
    if(!IsConvertible(requestedFormat, imgPixFormat))
        return ImageIOError(ImageIOError::FORMATS_ARE_NOT_CONVERTIBLE, filePath);

    // All looks fine allocate and copy
    data.resize(dimension[0] * dimension[1]);
    CopyPixels(reinterpret_cast<Byte*>(data.data()), requestedFormat,
               imgData.data(), imgPixFormat, dimension);
    
    return ImageIOError::OK;
}