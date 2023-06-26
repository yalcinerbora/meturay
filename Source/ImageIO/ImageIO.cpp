#include "ImageIO.h"
#include "FreeImgRAII.h"

#include <execution>
#include <algorithm>
#include <array>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include "RayLib/FileSystemUtility.h"

void ImageIO::PackChannelBits(Byte* bits,
                              const Byte* fromData, PixelFormat fromFormat,
                              size_t fromPitch, ImageChannelType type,
                              const Vector2ui& dimension)
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

            // TODO: this is not good bitmap
            // Also change this to UNORM 8-bit and stochastic culling maybe
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

ImageIOError ImageIO::OIIOImageSpecToPixelFormat(PixelFormat& pf, const OIIO::ImageSpec& spec)
{
    // TODO: This is order dependent so be careful
    switch(spec.format.basetype)
    {
        case OIIO::TypeDesc::UINT8:
        {
            switch(spec.nchannels)
            {
                case 1: pf = PixelFormat::R8_UNORM;     break;
                case 2: pf = PixelFormat::RG8_UNORM;    break;
                case 3: pf = PixelFormat::RGB8_UNORM;   break;
                case 4: pf = PixelFormat::RGBA8_UNORM;  break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        case OIIO::TypeDesc::INT8:
        {
            switch(spec.nchannels)
            {
                case 1: pf = PixelFormat::R8_SNORM;     break;
                case 2: pf = PixelFormat::RG8_SNORM;    break;
                case 3: pf = PixelFormat::RGB8_SNORM;   break;
                case 4: pf = PixelFormat::RGBA8_SNORM;  break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        case OIIO::TypeDesc::UINT16:
        {
            switch(spec.nchannels)
            {
                case 1: pf = PixelFormat::R16_UNORM;     break;
                case 2: pf = PixelFormat::RG16_UNORM;    break;
                case 3: pf = PixelFormat::RGB16_UNORM;   break;
                case 4: pf = PixelFormat::RGBA16_UNORM;  break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        case OIIO::TypeDesc::INT16:
        {
            switch(spec.nchannels)
            {
                case 1: pf = PixelFormat::R16_SNORM;     break;
                case 2: pf = PixelFormat::RG16_SNORM;    break;
                case 3: pf = PixelFormat::RGB16_SNORM;   break;
                case 4: pf = PixelFormat::RGBA16_SNORM;  break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }

        // TODO: Support reading these
        case OIIO::TypeDesc::UINT32:
        case OIIO::TypeDesc::INT32:
        case OIIO::TypeDesc::UINT64:
        case OIIO::TypeDesc::INT64:
        case OIIO::TypeDesc::DOUBLE:
            return ImageIOError::UNKNOWN_PIXEL_FORMAT;

        case OIIO::TypeDesc::HALF:
        {
            switch(spec.nchannels)
            {
                case 1: pf = PixelFormat::R_HALF;     break;
                case 2: pf = PixelFormat::RG_HALF;    break;
                case 3: pf = PixelFormat::RGB_HALF;   break;
                case 4: pf = PixelFormat::RGBA_HALF;  break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        case OIIO::TypeDesc::FLOAT:
        {
            switch(spec.nchannels)
            {
                case 1: pf = PixelFormat::R_FLOAT;     break;
                case 2: pf = PixelFormat::RG_FLOAT;    break;
                case 3: pf = PixelFormat::RGB_FLOAT;   break;
                case 4: pf = PixelFormat::RGBA_FLOAT;  break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        default:
            return ImageIOError::UNKNOWN_PIXEL_FORMAT;
    }
    return ImageIOError::OK;
}

ImageIOError ImageIO::PixelFormatToOIIOImageSpec(OIIO::ImageSpec& spec, PixelFormat pf,
                                                 const Vector2ui& resolution)
{
    int nChannels = 0;
    switch(pf)
    {
        case PixelFormat::R8_UNORM:
        case PixelFormat::R16_UNORM:
        case PixelFormat::R8_SNORM:
        case PixelFormat::R16_SNORM:
        case PixelFormat::R_HALF:
        case PixelFormat::R_FLOAT:
        {
            nChannels = 1;
            break;
        }
        case PixelFormat::RG8_UNORM:
        case PixelFormat::RG16_UNORM:
        case PixelFormat::RG8_SNORM:
        case PixelFormat::RG16_SNORM:
        case PixelFormat::RG_HALF:
        case PixelFormat::RG_FLOAT:
        {
            nChannels = 2;
            break;
        }
        case PixelFormat::RGB8_UNORM:
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGB8_SNORM:
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGB_HALF:
        case PixelFormat::RGB_FLOAT:
        {
            nChannels = 3;
            break;
        }
        case PixelFormat::RGBA8_UNORM:
        case PixelFormat::RGBA16_UNORM:
        case PixelFormat::RGBA8_SNORM:
        case PixelFormat::RGBA16_SNORM:
        case PixelFormat::RGBA_HALF:
        case PixelFormat::RGBA_FLOAT:
        {
            nChannels = 4;
            break;
        }
        // TODO: Implement these
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
        default:
            return ImageIOError::UNKNOWN_PIXEL_FORMAT;
    }

    // Find underlying type
    OIIO::TypeDesc td;
    switch(pf)
    {
        case PixelFormat::R8_UNORM:
        case PixelFormat::RG8_UNORM:
        case PixelFormat::RGB8_UNORM:
        case PixelFormat::RGBA8_UNORM:
        {
            td = OIIO::TypeDesc::UINT8;
            break;
        }
        case PixelFormat::R16_UNORM:
        case PixelFormat::RG16_UNORM:
        case PixelFormat::RGB16_UNORM:
        case PixelFormat::RGBA16_UNORM:
        {
            td = OIIO::TypeDesc::UINT16;
            break;
        }


        case PixelFormat::R8_SNORM:
        case PixelFormat::RG8_SNORM:
        case PixelFormat::RGB8_SNORM:
        case PixelFormat::RGBA8_SNORM:
        {
            td = OIIO::TypeDesc::INT8;
            break;
        }
        case PixelFormat::R16_SNORM:
        case PixelFormat::RG16_SNORM:
        case PixelFormat::RGB16_SNORM:
        case PixelFormat::RGBA16_SNORM:
        {
            td = OIIO::TypeDesc::INT16;
            break;
        }
        case PixelFormat::R_HALF:
        case PixelFormat::RG_HALF:
        case PixelFormat::RGB_HALF:
        case PixelFormat::RGBA_HALF:
        {
            td = OIIO::TypeDesc::HALF;
            break;
        }
        case PixelFormat::R_FLOAT:
        case PixelFormat::RG_FLOAT:
        case PixelFormat::RGB_FLOAT:
        case PixelFormat::RGBA_FLOAT:
        {
            td = OIIO::TypeDesc::FLOAT;
            break;
        }
        // TODO: Implement these
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
        default:
            return ImageIOError::UNKNOWN_PIXEL_FORMAT;
    }

    spec = OIIO::ImageSpec(resolution[0], resolution[1], nChannels, td);
    return ImageIOError::OK;
}

ImageIOError ImageIO::ReadImage(std::vector<Byte>& pixels,
                                PixelFormat& pf, Vector2ui& dimension,
                                const std::string& filePath,
                                const ImageIOFlags flags) const
{
    ImageIOError e = ImageIOError::OK;

    // First check if the file exists
    if(!Utility::CheckFileExistance(filePath))
        return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);

    auto inFile = OIIO::ImageInput::open(filePath);

    // Check the spec
    const OIIO::ImageSpec& spec = inFile->spec();
    if((e = OIIOImageSpecToPixelFormat(pf, spec)) != ImageIOError::OK)
        return e;
    // We can safely set the dimension now
    dimension = Vector2ui(spec.width, spec.height);

    // TODO: Do a conversion to an highest precision channel for these
    if(!spec.channelformats.empty())
        return ImageIOError(ImageIOError::READ_INTERNAL_ERROR,
                            "Channel-specific formats are not supported.");

    // Is this for deep images??
    if(spec.format.is_array())
        return ImageIOError(ImageIOError::READ_INTERNAL_ERROR,
                            "Arrayed per-pixel formats are not supported.");

    // Calculate the final spec according to the flags
    // channel expand and sign convert..
    OIIO::ImageSpec finalSpec = spec;

    // Check if sign convertible
    OIIO::TypeDesc readFormat = spec.format;
    if(flags[ImageIOI::LOAD_AS_SIGNED] && !IsSignConvertible(pf))
        return ImageIOError(ImageIOError::TYPE_IS_NOT_SIGN_CONVERTIBLE, filePath);
    else if(flags[ImageIOI::LOAD_AS_SIGNED])
    {
        if(spec.format == OIIO::TypeDesc::UINT32)
            readFormat = OIIO::TypeDesc::INT32;
        else if(spec.format == OIIO::TypeDesc::UINT16)
            readFormat = OIIO::TypeDesc::INT16;
        else if(spec.format == OIIO::TypeDesc::UINT8)
            readFormat = OIIO::TypeDesc::INT16;
        else
            return ImageIOError::READ_INTERNAL_ERROR;

        finalSpec.format = readFormat;
    }

    // Find the x stride to do a channel expand
    bool doChannelExpand = (Is4CExpandable(pf) && flags[ImageIOI::TRY_3C_4C_CONVERSION]);
    int nChannels = (doChannelExpand) ? (spec.nchannels + 1) : (spec.nchannels);
    OIIO::stride_t xStride = (doChannelExpand) ? (nChannels * readFormat.size()) : (OIIO::AutoStride);
    // Change the final spec as well for color convert
    if(doChannelExpand) finalSpec.nchannels = nChannels;

    // Allocate the expanded (or non-expanded) buffer and directly load into it
    pixels.resize(spec.width * spec.height * nChannels * readFormat.size());
    OIIO::stride_t scanLineSize = spec.width * nChannels * readFormat.size();
    Byte* dataLastElement = pixels.data() + (dimension[1] - 1) * scanLineSize;
    // Now we can read the file directly flipped and with proper format etc. etc.
    if(!inFile->read_image(readFormat, dataLastElement, xStride,  -scanLineSize))
        return ImageIOError(ImageIOError::READ_INTERNAL_ERROR, inFile->geterror());

    // Re-adjust the pixelFormat (we may have done channel expand and sign convert
    if((e = OIIOImageSpecToPixelFormat(pf, finalSpec)) != ImageIOError::OK)
        return e;

    return ImageIOError::OK;
}

ImageIOError ImageIO::ReadImageChannelAsBitMap(std::vector<Byte>& bitMap,
                                               Vector2ui& dimension,
                                               ImageChannelType channel,
                                               const std::string& filePath,
                                               ImageIOFlags) const
{
    // TODO: We cna do this more efficiently maybe?
    // by directly reading from the file instead of doing intermediate
    // ImageBufAlgo to packing the channel
    ImageIOError e = ImageIOError::OK;
    // First check if the file exists
    if(!Utility::CheckFileExistance(filePath))
        return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);

    OIIO::ImageBuf imgBuffer(filePath);
    const OIIO::ImageSpec& spec = imgBuffer.spec();
    // We can safely set the dimension now
    dimension = Vector2ui(spec.width, spec.height);

    int channelIndex = static_cast<int>(channel);
    imgBuffer = OIIO::ImageBufAlgo::channels(imgBuffer, 1,
                                             {channelIndex, -1, -1, -1},
                                             {0.0, 0.0, 0.0, 1.0},
                                             {"R", "", "", ""});
    if(imgBuffer.has_error())
        return ImageIOError(ImageIOError::READ_INTERNAL_ERROR, imgBuffer.geterror());

    // And finally flip since MRay uses classic cartesian coordinate system
    imgBuffer = OIIO::ImageBufAlgo::flip(imgBuffer);
    if(imgBuffer.has_error())
        return ImageIOError(ImageIOError::READ_INTERNAL_ERROR, imgBuffer.geterror());

    // Get the final spec of the image
    const OIIO::ImageSpec& finalSpec = imgBuffer.spec();

    // Re-set the format
    PixelFormat pf;
    if((e = OIIOImageSpecToPixelFormat(pf, finalSpec)) != ImageIOError::OK)
    return e;

    // Finally allocate buffer and read
    std::vector<Byte> pixels;
    pixels.resize(finalSpec.width * finalSpec.height *
                  finalSpec.nchannels * finalSpec.format.size());
    if(!imgBuffer.get_pixels(OIIO::ROI(0, finalSpec.width,
                                       0, finalSpec.height),
                             finalSpec.format, pixels.data()))
        return ImageIOError(ImageIOError::READ_INTERNAL_ERROR, imgBuffer.geterror());

    // Pack the bytes to bits
    size_t bitmapByteSize = (dimension[0] * dimension[1] + BYTE_BITS - 1) / BYTE_BITS;
    bitMap.resize(bitmapByteSize, 0);
    PackChannelBits(bitMap.data(),
                    pixels.data(), pf,
                    dimension[0] * FormatToPixelSize(pf),
                    channel, dimension);

    return ImageIOError::OK;
}

ImageIOError ImageIO::WriteImage(const Byte* data,
                                 const Vector2ui& dimension,
                                 PixelFormat pf, ImageType,
                                 const std::string& filePath) const
{
    ImageIOError e = ImageIOError::OK;
    auto out = OIIO::ImageOutput::create(filePath);

    OIIO::ImageSpec spec;
    if((e = PixelFormatToOIIOImageSpec(spec, pf, dimension)) != ImageIOError::OK)
        return e;

    auto scanLineSize = spec.scanline_bytes();
    const Byte* dataLastElement = data + (dimension[1] - 1) * scanLineSize;

    // TODO: properly write an error check/out code for these.
    if(!out->open(filePath, spec))
        return ImageIOError(ImageIOError::WRITE_INTERNAL_ERROR, out->geterror());
    if(!out->write_image(spec.format, dataLastElement, OIIO::AutoStride, -scanLineSize))
        return ImageIOError(ImageIOError::WRITE_INTERNAL_ERROR, out->geterror());
    if(!out->close())
        return ImageIOError(ImageIOError::WRITE_INTERNAL_ERROR, out->geterror());

    return ImageIOError::OK;
}

ImageIOError ImageIO::WriteBitmap(const Byte* bits,
                                  const Vector2ui& size, ImageType it,
                                  const std::string& fileName) const
{
    ImageIOError e = ImageIOError::OK;
    std::vector<Byte> expandedBits(size[0] * size[1]);
    // Push Bits
    for(uint32_t j = 0; j < size[1]; j++)
    for(uint32_t i = 0; i < size[0]; i++)
    {
        size_t linearByteSize = j * size[0] + i;
        size_t byteIndex = linearByteSize / BYTE_BITS;
        size_t bitIndex = linearByteSize % BYTE_BITS;
        bool bit = (bits[byteIndex] >> static_cast<Byte>(bitIndex) & 0x01);

        expandedBits[linearByteSize] = static_cast<Byte>(bit) * 255;
    }

    if((e = WriteImage(expandedBits.data(),
                       size, PixelFormat::R8_UNORM, it,
                       fileName)) != ImageIOError::OK)
        return e;

    return ImageIOError::OK;
}


ImageIOError ImageIO::ConvertPixels(Byte* toData, PixelFormat toFormat,
                                    const Byte* fromData, PixelFormat fromFormat,
                                    const Vector2ui& dimension) const
{
    ImageIOError e = ImageIOError::OK;
    if(toFormat == fromFormat)
    {
        std::memcpy(toData, fromData,
                    FormatToPixelSize(toFormat) *
                    dimension[0] * dimension[1]);
    }

    OIIO::ImageSpec fromSpec;
    if((e = PixelFormatToOIIOImageSpec(fromSpec, fromFormat, dimension)) != ImageIOError::OK)
        return e;

    OIIO::ImageSpec toSpec;
    if((e = PixelFormatToOIIOImageSpec(toSpec, toFormat, dimension)) != ImageIOError::OK)
        return e;

    // Only half to float is supported currently
    else if(toFormat == PixelFormat::R_FLOAT &&
            fromFormat == PixelFormat::R_HALF)
    {
        // Probably in buffer will not be modified.
        OIIO::ImageBuf inBuffer(fromSpec, const_cast<Byte*>(fromData));
        OIIO::ImageBuf outBuffer(toSpec, toData);

        if(!OIIO::ImageBufAlgo::copy(outBuffer, inBuffer, OIIO::TypeDesc::FLOAT))
            return ImageIOError::UNABLE_TO_CONVERT_BETWEEN_FORMATS;

        // Now after scope out all is fine i think.
    }
    // TODO: Implement more
    else return ImageIOError::UNABLE_TO_CONVERT_BETWEEN_FORMATS;
    return ImageIOError::OK;
}