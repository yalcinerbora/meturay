#include "ImageIO.h"
#include "FreeImgRAII.h"

#include <execution>
#include <algorithm>
#include <array>

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfArray.h>

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>

#include "RayLib/FileSystemUtility.h"

template <class T>
void ChannelSignConvert(T& t)
{   
    static_assert(std::is_signed_v<T>, "Type should be a signed type");

    constexpr T mid = static_cast<T>(0x1 << ((sizeof(T) * BYTE_BITS) - 1));
    t -= mid;
}

static
void SignConvert(std::array<Byte, 16>& pixel, PixelFormat fmt)
{
    int8_t channelCount = ImageIOI::FormatToChannelCount(fmt);
    
    switch(fmt)
    {
        case PixelFormat::R8_UNORM:
        case PixelFormat::RG8_UNORM:
        case PixelFormat::RGB8_UNORM:
        case PixelFormat::RGBA8_UNORM:
        case PixelFormat::R8_SNORM:
        case PixelFormat::RG8_SNORM:
        case PixelFormat::RGB8_SNORM:
        case PixelFormat::RGBA8_SNORM:
        {
            for(int8_t channel = 0; channel < channelCount; channel++)
            {
                int8_t* data = reinterpret_cast<int8_t*>(pixel.data() + (channel * sizeof(Byte)));
                ChannelSignConvert<int8_t>(*data);
            }
            break;
        }
        case PixelFormat::R16_UNORM:
        case PixelFormat::RG16_UNORM:
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGBA16_UNORM:        
        case PixelFormat::R16_SNORM:
        case PixelFormat::RG16_SNORM:
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGBA16_SNORM:
        {
            for(int8_t channel = 0; channel < channelCount; channel++)
            {
                int16_t* data = reinterpret_cast<int16_t*>(pixel.data() + (channel * sizeof(int16_t)));
                ChannelSignConvert<int16_t>(*data);
            }
            break;
        }
        // Others cannot be sign converted
        default: break;
    }
}

static
ImageIOError PixelFormatFromEXR(PixelFormat& pf, const Imf::Header& header)
{
    // TODO: Implement properly
    int channelCount = 0;
    const Imf::ChannelList& channels = header.channels();

    // Get Pixel Type & Channel Count
    Imf::PixelType consistentType;
    for(auto c = channels.begin(); c != channels.end(); c++)
    {
        channelCount++;
        const Imf::Channel& channel = c.channel();
        if(c == channels.begin())
        {
            consistentType = channel.type;
            continue;
        }

        if(channel.type != consistentType)
            return ImageIOError::READ_INTERNAL_ERROR;
    }
    
    // OpenEXR is quite complex but use it as if single image is present
    // with specific channels
    if(channelCount >= 5 || channelCount == 0)
        return ImageIOError::READ_INTERNAL_ERROR;
    // Also only read Float/Half types
    if(consistentType == Imf::PixelType::UINT)
        return ImageIOError::READ_INTERNAL_ERROR;

    switch(consistentType)
    {
        case Imf::HALF: 
            if(channelCount == 1) { pf = PixelFormat::R_HALF; break;}
            if(channelCount == 2) { pf = PixelFormat::RG_HALF; break; }
            if(channelCount == 3) { pf = PixelFormat::RGB_HALF; break; }
            if(channelCount == 4) { pf = PixelFormat::RGBA_HALF; break; }
        case Imf_3_1::FLOAT:
            if(channelCount == 1) { pf = PixelFormat::R_FLOAT; break; }
            if(channelCount == 2) { pf = PixelFormat::RG_FLOAT; break; }
            if(channelCount == 3) { pf = PixelFormat::RGB_FLOAT; break; }
            if(channelCount == 4) { pf = PixelFormat::RGBA_FLOAT; break; }
    }
    return ImageIOError::OK;
}

ImageIOError ImageIO::ConvertFreeImgFormat(PixelFormat& pf, FREE_IMAGE_TYPE t, uint32_t bpp)
{
    switch(t)
    {
        case FIT_UNKNOWN: return ImageIOError::UNKNOWN_PIXEL_FORMAT;

        case FIT_BITMAP:
        {
            switch(bpp)
            {
                case 8:  pf = PixelFormat::R8_UNORM; break;
                case 24: pf = PixelFormat::RGB8_UNORM; break;
                case 32: pf = PixelFormat::RGBA8_UNORM; break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        case FIT_RGB16:
        case FIT_RGBA16:
        {
            // Only these two are supported
            switch(bpp)
            {
                case 48: pf = PixelFormat::RGB16_UNORM; break;
                case 64: pf = PixelFormat::RGBA16_UNORM; break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        case FIT_FLOAT:
        case FIT_RGBF:
        case FIT_RGBAF:
        {
            // Only these are supported
            switch(bpp)
            {
                case 32:  pf = PixelFormat::R_FLOAT; break;
                case 96:  pf = PixelFormat::RGB_FLOAT; break;
                case 128: pf = PixelFormat::RGBA_FLOAT; break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }

        // Skip these
        case FIT_UINT16:
        case FIT_INT16:
        case FIT_UINT32:
        case FIT_INT32:
        case FIT_DOUBLE:
        case FIT_COMPLEX:
        default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
    }
    return ImageIOError::OK;
}

ImageIOError ImageIO::ConvertImageTypeToFreeImgType(FREE_IMAGE_FORMAT& fit, ImageType it)
{
    switch(it)
    {
        case ImageType::PNG: fit = FIF_PNG;  break;
        case ImageType::JPG: fit = FIF_JPEG; break;
        case ImageType::BMP: fit = FIF_BMP;  break;
        case ImageType::HDR: fit = FIF_HDR;  break;
        // Rest is not used by FreeImage
        default: return ImageIOError::WRITE_INTERNAL_ERROR;
    }
    return ImageIOError::OK;
}

ImageIOError ImageIO::ConvertPixelFormatToFreeImgType(FREE_IMAGE_TYPE& t, PixelFormat pf)
{
    switch(pf)
    {
        case PixelFormat::R8_UNORM:
        case PixelFormat::RG8_UNORM:
        case PixelFormat::RGB8_UNORM:
        case PixelFormat::RGBA8_UNORM:
        case PixelFormat::R8_SNORM:
        case PixelFormat::RG8_SNORM:
        case PixelFormat::RGB8_SNORM:
        case PixelFormat::RGBA8_SNORM:
            t = FIT_BITMAP; break;

        case PixelFormat::RGB_FLOAT:  t = FIT_RGBF; break;
        case PixelFormat::RGBA_FLOAT: t = FIT_RGBAF; break;
            
        // Maybe Implement later
        case PixelFormat::R16_UNORM:
        case PixelFormat::RG16_UNORM:
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGBA16_UNORM:
        case PixelFormat::R16_SNORM:
        case PixelFormat::RG16_SNORM:
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGBA16_SNORM:
        case PixelFormat::R_HALF:
        case PixelFormat::RG_HALF:
        case PixelFormat::RGB_HALF:
        case PixelFormat::RGBA_HALF:
        case PixelFormat::R_FLOAT:
        case PixelFormat::RG_FLOAT:
        default:
            return ImageIOError::READ_INTERNAL_ERROR;
    }
    return ImageIOError::OK;
}

ImageIO::ImageIO()
{
    FreeImage_Initialise();
}

ImageIO::~ImageIO()
{
    FreeImage_DeInitialise();
}

ImageIOError ImageIO::WriteAsEXR(const Byte* pixels,
                                 const Vector2ui& dimension, PixelFormat pf,
                                 const std::string& fileName) const
{
    // TODO: This code is quite awful change it to a more generic version
    const int channelCount = FormatToChannelCount(pf);

    // Always try to write as half
    size_t pixelSizeLinear = dimension[0] * dimension[1];
    std::vector<Imath::half> convertedData(pixelSizeLinear * channelCount);
    
    // Call appropriate function

    auto ConvtForEXRWrap = [&]<class T>()
    {
        ConvertForEXR(convertedData.data(),
                      reinterpret_cast<const T*>(pixels),
                      pf, dimension);
    };

    switch(pf)
    {
        case PixelFormat::R_FLOAT:      
        case PixelFormat::RG_FLOAT:  
        case PixelFormat::RGB_FLOAT: 
        case PixelFormat::RGBA_FLOAT:
            ConvtForEXRWrap.operator()<float>(); 
            break;
        default: return ImageIOError::WRITE_INTERNAL_ERROR;
    }

    // Set Channels
    Imf::Header header(dimension[0], dimension[1]);
    std::array<std::string, 4> channelNames = 
    {
        (pf == PixelFormat::R_FLOAT) ? "Y" : "R", 
        "G", "B", "A"
    };
    for(int i = 0; i < channelCount; i++)
    {
        header.channels().insert(channelNames[i], Imf::Channel(Imf::HALF));
    }

    Imf::OutputFile file(fileName.c_str(), header);
    Imf::FrameBuffer frameBuffer;
    for(int i = 0; i < channelCount; i++)
    {
        Imf::Slice lumSlice = Imf::Slice(Imf::HALF,
                                         reinterpret_cast<char*>(convertedData.data() + i), // base // 8
                                         sizeof(Imath::half) * channelCount,
                                         sizeof(Imath::half) * dimension[0] * channelCount);
        frameBuffer.insert(channelNames[i], lumSlice);
    }
    file.setFrameBuffer(frameBuffer);

    // Finally write    
    file.writePixels(dimension[1]);
    return ImageIOError::OK;
}

ImageIOError ImageIO::WriteUsingFreeImage(const Byte* pixels,
                                          const Vector2ui& dimension, 
                                          PixelFormat pf, ImageType it,
                                          const std::string& fileName) const
{
    FREE_IMAGE_TYPE fit;
    ImageIOError e = ImageIOError::OK;

    if((e = ConvertPixelFormatToFreeImgType(fit, pf)) != ImageIOError::OK)
        return e;

    size_t pixelSize = FormatToPixelSize(pf);
    size_t bpp = pixelSize * BYTE_BITS;

    FreeImgRAII fImg = FreeImgRAII(FreeImage_AllocateT(fit,
                                                       dimension[0],
                                                       dimension[1],
                                                       static_cast<int>(bpp)));

    for(uint32_t j = 0; j < dimension[1]; j++)
    {
        size_t scanLineSize = pixelSize * dimension[0];

        Byte* outScanLine = FreeImage_GetScanLine(fImg, j);
        const Byte* inScanLine = pixels + j * pixelSize * dimension[0];
      
        std::memcpy(outScanLine, inScanLine, scanLineSize);
    }

    FREE_IMAGE_FORMAT fif;
    if((e = ConvertImageTypeToFreeImgType(fif, it)) != ImageIOError::OK)
        return e;
    
    if(!FreeImage_Save(fif, fImg, fileName.c_str()))
        return ImageIOError::WRITE_INTERNAL_ERROR;

    return ImageIOError::OK;
}

bool ImageIO::CheckIfEXR(const std::string& filePath)
{
    // Utilitze FreeImage to find ext
    FREE_IMAGE_FORMAT f;
    // Check file to find type
    if((f = FreeImage_GetFileType(filePath.c_str())) == FIF_UNKNOWN)
        // Use file extension to determine type
        f = FreeImage_GetFIFFromFilename(filePath.c_str());

    if(f == FIF_EXR) 
        return true;
    return false;
}

ImageIOError ImageIO::ReadImage_FreeImage(FreeImgRAII& freeImg,
                                          //std::vector<Byte>& pixels,
                                          PixelFormat& format, Vector2ui& dimension,
                                          const std::string& filePath) const
{
    FREE_IMAGE_FORMAT f;
    // Check file to find type
    if((f = FreeImage_GetFileType(filePath.c_str())) == FIF_UNKNOWN)
        // Use file extension to determine type
        f = FreeImage_GetFIFFromFilename(filePath.c_str());
    // If it is still unknown just return error
    if(f == FIF_UNKNOWN)
        return ImageIOError(ImageIOError::UNKNOWN_IMAGE_TYPE, filePath);

    // Generate Object
    //FreeImgRAII imgCPU(f, filePath.c_str());
    freeImg = FreeImgRAII(f, filePath.c_str());
    // If imgCPU not avail fail
    if(!freeImg) return ImageIOError(ImageIOError::READ_INTERNAL_ERROR, filePath);
   
    // Bit per pixel
    uint32_t bpp = FreeImage_GetBPP(freeImg);
    uint32_t w = FreeImage_GetWidth(freeImg);
    uint32_t h = FreeImage_GetHeight(freeImg);
    uint32_t pitch = FreeImage_GetPitch(freeImg);
    dimension = Vector2ui(w, h);

    FREE_IMAGE_TYPE imgType = FreeImage_GetImageType(freeImg);
    //FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(freeImg);

    ImageIOError e = ImageIOError::OK;
    if((e = ConvertFreeImgFormat(format, imgType, bpp)) != ImageIOError::OK)
        return ImageIOError(e, filePath);

    //// Allocate 
    ////pixels.resize(dimension[0] * dimension[1] * FormatToPixelSize(format));

    //// Do copy over
    //BYTE* imgPixels = FreeImage_GetBits(imgCPU);
    //size_t outPitch = dimension[0] * FormatToPixelSize(format);
    //
    //// TODO: Parallelize this 
    //for(uint32_t y = 0; y < dimension[1]; y++)
    //{
    //    Byte* outRowPtr = pixels.data() + outPitch * y;
    //    const Byte* srcRowPtr = imgPixels + pitch * y;
    //    std::memcpy(outRowPtr, srcRowPtr, outPitch);
    //}
    // All done!
    return ImageIOError::OK;
}

ImageIOError ImageIO::ReadImage_OpenEXR(std::vector<Byte>& pixels,
                                        PixelFormat& pf, Vector2ui& size,
                                        const std::string& filePath) const
{    
    ImageIOError e = ImageIOError::OK;

    //Imf::InputFile file(filePath.c_str());
    Imf::RgbaInputFile file(filePath.c_str());
    if((e = PixelFormatFromEXR(pf, file.header())) != ImageIOError::OK)
        return e;

    // Implement This properly (directly write to vector)
    Imf::Array2D<Imf::Rgba> exrPixels;    
    Imath::Box2i dw = file.dataWindow();
    size[0] = dw.max.x - dw.min.x + 1;
    size[1] = dw.max.y - dw.min.y + 1;
    exrPixels.resizeErase(size[1], size[0]);
    file.setFrameBuffer(&exrPixels[0][0] - dw.min.x - dw.min.y * size[0], 1, size[0]);
    file.readPixels(dw.min.y, dw.max.y);

    // ?
    return ImageIOError::READ_INTERNAL_ERROR;

    // TODO: 
    // Convert to Vector4f and Invert Y Axis
    for(uint32_t y = 0; y < size[1]; y++)
    for(uint32_t x = 0; x < size[0]; x++)
    {
        uint32_t invertexY = size[1] - y - 1;
        uint32_t outIndex = invertexY * size[0] + x;

        const Imf::Rgba& v = pixels[y][x];
        image[outIndex] = Vector4f(static_cast<float>(v.r),
                                   static_cast<float>(v.g),
                                   static_cast<float>(v.b),
                                   static_cast<float>(v.a));
    }

    //int8_t channelCount = FormatToChannelCount(pf);

    //file.setFrameBuffer(framebuffer);
    //file.readPixels(data_window.min.y, data_window.max.y);

    //Imf::FrameBuffer framebuffer;
    //framebuffer.insert();

    // Convert to Vector4f and Invert Y Axis
   /* image.resize(size[0] * size[1]);
    for(uint32_t y = 0; y < size[1]; y++)
    for(uint32_t x = 0; x < size[0]; x++)
    {
        uint32_t invertexY = size[1] - y - 1;
        uint32_t outIndex = invertexY * size[0] + x;

        const Imf::Rgba& v = pixels[y][x];
        image[outIndex] = Vector4f(static_cast<float>(v.r),
                                   static_cast<float>(v.g),
                                   static_cast<float>(v.b),
                                   static_cast<float>(v.a));
    }
    return true;*/
    return ImageIOError::READ_INTERNAL_ERROR;
}

void ImageIO::PackChannelBits(Byte* bits,
                              const Byte* fromData, PixelFormat fromFormat,
                              size_t fromPitch, ImageChannelType type, 
                              const Vector2ui& dimension) const
{    
    int8_t channelIndex = ChannelTypeToChannelIndex(type);
    size_t fromPixelSize = FormatToPixelSize(fromFormat);
    size_t fromChannelSize = FormatToChannelSize(fromFormat);
    
    for(uint32_t y = 0; y < dimension[1]; y++)
    {
        const Byte* fromPixelRow = fromData + fromPitch * y;
        for(uint32_t x = 0; x < dimension[0]; x++)
        {
            // Pixel by pixel copy
            const Byte* fromPixelChannelPtr = (fromPixelRow
                                               + (x * fromPixelSize)
                                               + (channelIndex * fromChannelSize));

            // Copy Pixel Channel to Stack
            assert(fromChannelSize <= sizeof(uint64_t));
            uint64_t fromPixelChannel = 0;
            std::memcpy(&fromPixelChannel, fromPixelChannelPtr, fromChannelSize);
            
            if(fromPixelChannel)
            {
                size_t pixelLinearIndex = (y * dimension[0] + x);
                size_t byteIndex = pixelLinearIndex / BYTE_BITS;
                size_t byteInnerIndex = pixelLinearIndex % BYTE_BITS;
                bits[byteIndex] |= (0x1 << byteInnerIndex);
            }
        }
    }
}

void ImageIO::ConvertPixels(Byte* toData, PixelFormat toFormat,
                            const Byte* fromData, PixelFormat fromFormat, size_t fromPitch,
                            const Vector2ui& dimension) const
{
    // Just to paralleize create a iota
    // TODO: change to parallelizable ranges C++23 or after
    // This is static just to prevent some alloc/unalloc etc 
    static std::vector<uint32_t> indices;
    size_t newSize = (toFormat == fromFormat) 
                        ? dimension[1] 
                        : (dimension[0] * dimension[1]);
    // Do iota only if the size is increased
    // and do the new parts
    if(newSize > indices.size())
    {
        size_t oldSize = indices.size();
        indices.resize(newSize);
        std::iota(std::next(indices.begin(), oldSize), indices.end(), 
                  static_cast<uint32_t>(oldSize));
    }

    // We did check if this can be converted etc
    // Just do it
    const size_t toPixelSize = FormatToPixelSize(toFormat);
    const size_t fromPixelSize = FormatToPixelSize(fromFormat);

    auto ConvertFunc = [&] (const uint32_t pixelId) -> void
    {
        uint32_t x = pixelId % dimension[0];
        uint32_t y = pixelId / dimension[0];

        Byte* toPixelPtr = toData + (y * dimension[0] + x) * toPixelSize;
        const Byte* fromPixelPtr = fromData + (y * fromPitch + (x * fromPixelSize));

        // Copy Pixel to Stack
        std::array<Byte, 16> fromPixel;
        std::memcpy(fromPixel.data(), fromPixelPtr, fromPixelSize);

        if(HasSignConversion(toFormat, fromFormat))
        {
            SignConvert(fromPixel, toFormat);
        }
        // Copy (automatic 3D->4D expansion)
        std::memcpy(toPixelPtr, fromPixel.data(), fromPixelSize);
    };

    auto PackFunc = [&](const uint32_t y) -> void
    {
        Byte* toPixelRowPtr = toData + (y * dimension[0]) * toPixelSize;
        const Byte* fromPixelRowPtr = fromData + (y * fromPitch);
        size_t toPitch = dimension[0] * toPixelSize;
        std::memcpy(toPixelRowPtr, fromPixelRowPtr, toPitch);
    };
   
    // Do sequential if data is not large
    size_t iterationCount = newSize;       
    if(iterationCount <= PARALLEL_EXEC_TRESHOLD)
    {
        if(fromFormat == toFormat)
            std::for_each_n(indices.cbegin(), iterationCount,
                            PackFunc);
        else std::for_each_n(indices.cbegin(), iterationCount,
                             ConvertFunc);
    }
    else
    {
        if(fromFormat == toFormat)
            std::for_each_n(std::execution::par_unseq,
                            indices.cbegin(), iterationCount,
                            PackFunc);
        else std::for_each_n(std::execution::par_unseq,
                             indices.cbegin(), iterationCount,
                             ConvertFunc);
    }
}



ImageIOError ImageIO::ReadImage(std::vector<Byte>& pixels,
                                PixelFormat& pf, Vector2ui& dimension,
                                const std::string& filePath,
                                const ImageIOFlags flags) const
{
    // First check if the file exists
    if(!Utility::CheckFileExistance(filePath))
        return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);

    bool isEXRFile = CheckIfEXR(filePath);


    std::vector<Byte> exrPixels;
    ImageIOError e = ImageIOError::OK;
    FreeImgRAII freeImg;
    if(isEXRFile)
        e = ReadImage_OpenEXR(exrPixels, pf, dimension, filePath);
    else 
        e = ReadImage_FreeImage(freeImg, pf, dimension, filePath);
    
    if(e != ImageIOError::OK)
        return ImageIOError(e, filePath);

    // Check Flags
    if(flags[ImageIOI::LOAD_AS_SIGNED] && !IsSignConvertible(pf))
    { 
        return ImageIOError(ImageIOError::TYPE_IS_NOT_SIGN_CONVERTIBLE, filePath);
    }
    
    PixelFormat convertedFormat = pf;
    size_t newPixelSize = FormatToPixelSize(convertedFormat);
    // Check the conversion
    if(Is4CExpandable(pf) && flags[ImageIOI::TRY_3C_4C_CONVERSION])
    {
        // Determine new format
        convertedFormat = (flags[ImageIOI::LOAD_AS_SIGNED])
                ? Expanded4CFormat(SignConvertedFormat(pf))
                : Expanded4CFormat(pf);

        newPixelSize = FormatToPixelSize(convertedFormat);        
    }
    // Only Signed Conversion
    else if(IsSignConvertible(pf) && flags[ImageIOI::LOAD_AS_SIGNED])
    {    
        convertedFormat = SignConvertedFormat(pf);
        newPixelSize = FormatToPixelSize(convertedFormat);
    }

    // Exr files are tightly packed
    // And if no conversion is requested / required
    // Just move the data
    if(isEXRFile && (convertedFormat == pf))
    {
        pixels = std::move(exrPixels);
        return ImageIOError::OK;
    }

    // Convert the data
    // Resize Output Buffer
    pixels.resize(dimension[0] * dimension[1] * newPixelSize);    
    if(isEXRFile)
    {
        ConvertPixels(pixels.data(), convertedFormat,
                      exrPixels.data(), pf, 
                      // Exr pixels are packed tightly (atleast after load)
                      dimension[0] * newPixelSize,
                      dimension);
    }
    // FreeImg files have pitch so directly convert from its buffer
    // If no conversion occurs ConvertPixels just packs the scanlines
    else
    {
        // Directly Convert from FreeImg Buffer
        ConvertPixels(pixels.data(), convertedFormat, 
                      freeImg.Data(), pf, freeImg.Pitch(),
                      dimension);
    }
    pf = convertedFormat;
    return ImageIOError::OK;
}

ImageIOError ImageIO::ReadImageChannelAsBitMap(std::vector<Byte>& bitMap,
                                               Vector2ui& dimension,
                                               ImageChannelType channel,
                                               const std::string& filePath,
                                               ImageIOFlags) const
{
    PixelFormat pf;
    // First check if the file exists
    if(!Utility::CheckFileExistance(filePath))
        return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);

    bool isEXRFile = CheckIfEXR(filePath);

    std::vector<Byte> exrPixels;
    ImageIOError e = ImageIOError::OK;
    FreeImgRAII freeImg;
    if(isEXRFile)
        e = ReadImage_OpenEXR(exrPixels, pf, dimension, filePath);
    else
        e = ReadImage_FreeImage(freeImg, pf, dimension, filePath);

    if(e != ImageIOError::OK)
        return ImageIOError(e, filePath);

    // Convert it to a bitmap
    size_t bitmapByteSize = (dimension[0] * dimension[1] + BYTE_BITS - 1) / BYTE_BITS;
    bitMap.resize(bitmapByteSize, 0);
    if(isEXRFile)
    {
        PackChannelBits(bitMap.data(),
                        exrPixels.data(), pf,
                        // Exr pixels are packed tightly (atleast after load)
                        dimension[0] * FormatToPixelSize(pf), 
                        channel, dimension);
    }
    else
    {
        PackChannelBits(bitMap.data(),
                        freeImg.Data(), pf,
                        freeImg.Pitch(), channel,
                        dimension);
    }

    return ImageIOError::OK;
}

ImageIOError ImageIO::WriteImage(const Byte* data,
                                 const Vector2ui& dimension,
                                 PixelFormat pf, ImageType it,
                                 const std::string& filePath) const
{
    switch(it)
    {
        case ImageType::PNG:
        case ImageType::JPG:
        case ImageType::BMP:
        case ImageType::HDR:
            return WriteUsingFreeImage(data, dimension, pf, it, filePath);
        case ImageType::EXR:
            return WriteAsEXR(data, dimension, pf, filePath);
        default:
            return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);
    }
}

ImageIOError ImageIO::WriteBitmap(const Byte* bits,
                                  const Vector2ui& size, ImageType it,
                                  const std::string& fileName) const
{
    ImageIOError e = ImageIOError::OK;
    FreeImgRAII fImg(FreeImage_Allocate(size[0], size[1], 8, 255));

    if(fImg == nullptr)
        return ImageIOError::WRITE_INTERNAL_ERROR;

    // Push Bits
    for(uint32_t j = 0; j < size[1]; j++)
    for(uint32_t i = 0; i < size[0]; i++)
    {
        size_t linearByteSize = j * size[0] + i;
        size_t byteIndex = linearByteSize / BYTE_BITS;
        size_t bitIndex = linearByteSize % BYTE_BITS;
        bool bit = (bits[byteIndex] >> static_cast<Byte>(bitIndex) & 0x01);

        RGBQUAD color;
        color.rgbRed = static_cast<BYTE>(bit) * 255;
        //color.rgbGreen = static_cast<BYTE>(bit) * 255;
        //color.rgbBlue = static_cast<BYTE>(bit) * 255;

        bool pixLoaded = FreeImage_SetPixelColor(fImg, i, j, &color);
    }

    FREE_IMAGE_FORMAT fif;
    if((e = ConvertImageTypeToFreeImgType(fif, it)) != ImageIOError::OK)
        return e;

    bool result = FreeImage_Save(fif, fImg, fileName.c_str());
    if(!result) ImageIOError::WRITE_INTERNAL_ERROR;
    return ImageIOError::OK;
}