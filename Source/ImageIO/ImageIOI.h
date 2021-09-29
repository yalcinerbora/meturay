#pragma once

#pragma once

#include <vector>
#include <string>
#include <memory>

#include "RayLib/Vector.h"
#include "RayLib/Types.h"

#include "RayLib/ImageIOError.h"
#include "RayLib/Flag.h"

enum class ImageType
{
    PNG,
    JPG,
    BMP,
    HDR,
    EXR
};

enum class ChannelType
{
    R, G, B, A
};

class ImageIOI
{
    public:
        enum FlagTypes
        {
            LOAD_AS_SIGNED          = 0,
            TRY_3C_4C_CONVERSION    = 1,

            END
        };

        using ImageIOFlags = Flags<FlagTypes>;

    private:
    protected:        
        //virtual void            ConvertPixels(Byte* toData, PixelFormat toFormat,
        //                                      const Byte* fromData, PixelFormat fromFormat,
        //                                      const Vector2ui& dimension) const = 0;


    public:        
        virtual                 ~ImageIOI() = default;

        // Statics
        static size_t           FormatToPixelSize(PixelFormat);
        static int8_t           FormatToChannelCount(PixelFormat);
        // Sign Conversion Related
        static PixelFormat      SignConvertedFormat(PixelFormat);
        static bool             IsSignConvertible(PixelFormat);
        static bool             HasSignConversion(PixelFormat, PixelFormat);
        // Expansion Related
        static PixelFormat      Expanded4CFormat(PixelFormat);
        static bool             Is4CExpandable(PixelFormat);

        // Read Functions
        // Try to read any image as is
        // Always pack the data (no scanline stuff)
        virtual ImageIOError    ReadImage(std::vector<Byte>&,
                                          PixelFormat&, Vector2ui& dimensions,
                                          const std::string& filePath,
                                          const ImageIOFlags = ImageIOFlags()) const = 0;
        // Read Image but convert it to the non-byte vector
        template<class T, typename = std::enable_if_t<!std::is_same_v<T, Byte>>>
        ImageIOError            ReadImage(std::vector<T>&, 
                                          PixelFormat&, Vector2ui& dimension,
                                          const std::string& filePath,
                                          const ImageIOFlags = ImageIOFlags());
        // Read a channel as alpha bit map
        virtual ImageIOError   ReadImageChannelAsBitMap(std::vector<Byte>&,
                                                        Vector2ui& dimension,
                                                        ChannelType,
                                                        const std::string& filePath,
                                                        const ImageIOFlags = ImageIOFlags()) const = 0;
        

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

// TODO: is this a good pattern???
using ImageIOFlags = ImageIOI::ImageIOFlags;

inline int8_t ImageIOI::FormatToChannelCount(PixelFormat pf)
{
    switch(pf)
    {
        case PixelFormat::R8_UNORM:
        case PixelFormat::R16_UNORM:
        case PixelFormat::R8_SNORM:
        case PixelFormat::R16_SNORM:
        case PixelFormat::R_HALF:
        case PixelFormat::R_FLOAT:
            return 1;
        case PixelFormat::RG8_UNORM:
        case PixelFormat::RG16_UNORM:
        case PixelFormat::RG8_SNORM:
        case PixelFormat::RG16_SNORM:
        case PixelFormat::RG_HALF:
        case PixelFormat::RG_FLOAT:
            return 2;
        case PixelFormat::RGB8_UNORM:
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGB8_SNORM:
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGB_HALF:
        case PixelFormat::RGB_FLOAT:
            return 3;
        case PixelFormat::RGBA8_UNORM:
        case PixelFormat::RGBA16_UNORM:
        case PixelFormat::RGBA8_SNORM:            
        case PixelFormat::RGBA16_SNORM:
        case PixelFormat::RGBA_HALF:
        case PixelFormat::RGBA_FLOAT:
            return 4;
        default:
            // TODO: Add bc compression channels
            return std::numeric_limits<int8_t>::max();
    }
}

inline size_t ImageIOI::FormatToPixelSize(PixelFormat pf)
{
    // Yolo switch
    switch(pf)
    {
        // UNORM_INT8 Types
        case PixelFormat::R8_UNORM:     return sizeof(uint8_t) * 1;
        case PixelFormat::RG8_UNORM:    return sizeof(uint8_t) * 2;
        case PixelFormat::RGB8_UNORM:   return sizeof(uint8_t) * 3;
        case PixelFormat::RGBA8_UNORM:  return sizeof(uint8_t) * 4;
        // UNORM_INT16 Types
        case PixelFormat::R16_UNORM:    return sizeof(uint16_t) * 1;
        case PixelFormat::RG16_UNORM:   return sizeof(uint16_t) * 2;
        case PixelFormat::RGB16_UNORM:  return sizeof(uint16_t) * 3;
        case PixelFormat::RGBA16_UNORM: return sizeof(uint16_t) * 4;
        // Half Types
        case PixelFormat::R_HALF:       return sizeof(uint16_t) * 1;
        case PixelFormat::RG_HALF:      return sizeof(uint16_t) * 2;
        case PixelFormat::RGB_HALF:     return sizeof(uint16_t) * 3;
        case PixelFormat::RGBA_HALF:    return sizeof(uint16_t) * 4;

        case PixelFormat::R_FLOAT:      return sizeof(float) * 1;
        case PixelFormat::RG_FLOAT:     return sizeof(float) * 1;
        case PixelFormat::RGB_FLOAT:    return sizeof(float) * 1;
        case PixelFormat::RGBA_FLOAT:   return sizeof(float) * 1;
        // BC Types
        // TODO: Implement these
        case PixelFormat::BC1_U:    return 0;
        case PixelFormat::BC2_U:    return 0;
        case PixelFormat::BC3_U:    return 0;
        case PixelFormat::BC4_U:    return 0;
        case PixelFormat::BC4_S:    return 0;
        case PixelFormat::BC5_U:    return 0;
        case PixelFormat::BC5_S:    return 0;
        case PixelFormat::BC6H_U:   return 0;
        case PixelFormat::BC6H_S:   return 0;
        case PixelFormat::BC7_U:    return 0;
        // Unknown Type
        case PixelFormat::END:
        default: 
            return 0;
            
    }
}

inline PixelFormat ImageIOI::SignConvertedFormat(PixelFormat pf)
{
    switch(pf)
    {
        case PixelFormat::R8_UNORM:     return PixelFormat::R8_SNORM;
        case PixelFormat::RG8_UNORM:    return PixelFormat::RG8_SNORM;
        case PixelFormat::RGB8_UNORM:   return PixelFormat::RGB8_SNORM;
        case PixelFormat::RGBA8_UNORM:  return PixelFormat::RGBA8_SNORM;
        case PixelFormat::R16_UNORM:    return PixelFormat::R16_SNORM;
        case PixelFormat::RG16_UNORM:   return PixelFormat::RG16_SNORM;
        case PixelFormat::RGB16_UNORM:  return PixelFormat::RGB16_SNORM;
        case PixelFormat::RGBA16_UNORM: return PixelFormat::RGBA16_SNORM;

        case PixelFormat::R8_SNORM:     return PixelFormat::R8_UNORM;
        case PixelFormat::RG8_SNORM:    return PixelFormat::RG8_UNORM;
        case PixelFormat::RGB8_SNORM:   return PixelFormat::RGB8_UNORM;
        case PixelFormat::RGBA8_SNORM:  return PixelFormat::RGBA8_UNORM;
        case PixelFormat::R16_SNORM:    return PixelFormat::R16_UNORM;
        case PixelFormat::RG16_SNORM:   return PixelFormat::RG16_UNORM;
        case PixelFormat::RGB16_SNORM:  return PixelFormat::RGB16_UNORM;
        case PixelFormat::RGBA16_SNORM: return PixelFormat::RGBA16_UNORM;

        default: return PixelFormat::END;
    }
}

inline bool ImageIOI::IsSignConvertible(PixelFormat pf)
{
    switch(pf)
    {
        case PixelFormat::R8_UNORM:
        case PixelFormat::RG8_UNORM:
        case PixelFormat::RGB8_UNORM:
        case PixelFormat::RGBA8_UNORM:
        case PixelFormat::R16_UNORM:
        case PixelFormat::RG16_UNORM:
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGBA16_UNORM:
        case PixelFormat::R8_SNORM:
        case PixelFormat::RG8_SNORM:
        case PixelFormat::RGB8_SNORM:
        case PixelFormat::RGBA8_SNORM:
        case PixelFormat::R16_SNORM:
        case PixelFormat::RG16_SNORM:
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGBA16_SNORM:
            return true;
        default: return false;
    }
}

inline bool ImageIOI::HasSignConversion(PixelFormat toFormat, PixelFormat fromFormat)
{
    switch(toFormat)
    {
        case PixelFormat::R8_UNORM:     
        case PixelFormat::RG8_UNORM:    
        case PixelFormat::RGB8_UNORM:   
        case PixelFormat::RGBA8_UNORM:  
        case PixelFormat::R16_UNORM:    
        case PixelFormat::RG16_UNORM:   
        case PixelFormat::RGB16_UNORM:  
        case PixelFormat::RGBA16_UNORM: 
        {
            return (fromFormat == PixelFormat::R8_SNORM     ||
                    fromFormat == PixelFormat::RG8_SNORM    ||
                    fromFormat == PixelFormat::RGB8_SNORM   ||
                    fromFormat == PixelFormat::RGBA8_SNORM  ||
                    fromFormat == PixelFormat::R16_SNORM    ||
                    fromFormat == PixelFormat::RG16_SNORM   ||
                    fromFormat == PixelFormat::RGB16_SNORM  ||
                    fromFormat == PixelFormat::RGBA16_SNORM);
        }

        case PixelFormat::R8_SNORM:     
        case PixelFormat::RG8_SNORM:    
        case PixelFormat::RGB8_SNORM:   
        case PixelFormat::RGBA8_SNORM:  
        case PixelFormat::R16_SNORM:    
        case PixelFormat::RG16_SNORM:   
        case PixelFormat::RGB16_SNORM:  
        case PixelFormat::RGBA16_SNORM: 
        {
            return (fromFormat == PixelFormat::R8_UNORM     ||
                    fromFormat == PixelFormat::RG8_UNORM    ||
                    fromFormat == PixelFormat::RGB8_UNORM   ||
                    fromFormat == PixelFormat::RGBA8_UNORM  ||
                    fromFormat == PixelFormat::R16_UNORM    ||
                    fromFormat == PixelFormat::RG16_UNORM   ||
                    fromFormat == PixelFormat::RGB16_UNORM  ||
                    fromFormat == PixelFormat::RGBA16_UNORM);
        }
        default: return false;
    }
}

inline PixelFormat ImageIOI::Expanded4CFormat(PixelFormat pf)
{
    switch(pf)
    {
        
        case PixelFormat::RGB8_UNORM:     return PixelFormat::RGBA8_UNORM;
        case PixelFormat::RGB16_UNORM:    return PixelFormat::RGBA16_UNORM;
        case PixelFormat::RGB8_SNORM:     return PixelFormat::RGBA8_SNORM;
        case PixelFormat::RGB16_SNORM:    return PixelFormat::RGBA16_SNORM;
        case PixelFormat::RGB_HALF:       return PixelFormat::RGBA_HALF;
        case PixelFormat::RGB_FLOAT:      return PixelFormat::RGBA_FLOAT;
            
        default: return PixelFormat::END;
    }
}

inline bool ImageIOI::Is4CExpandable(PixelFormat pf)
{
    switch(pf)
    {
        case PixelFormat::RGB8_UNORM: 
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGB8_SNORM: 
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGB_HALF:   
        case PixelFormat::RGB_FLOAT:  
            return true;
        default: return false;
    }
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

template<class T, typename>
ImageIOError ImageIOI::ReadImage(std::vector<T>& data, 
                                 PixelFormat& pf, 
                                 Vector2ui& dimension,
                                 const std::string& filePath,
                                 ImageIOFlags flags)
{
    return ImageIOError::IMAGE_NOT_FOUND;
}