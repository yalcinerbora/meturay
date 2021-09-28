#include "ImageIO.h"
#include "FreeImgRAII.h"

#include <execution>
#include <algorithm>

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfArray.h>

#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>

#include "RayLib/FileSystemUtility.h"


ImageIOError ImageIO::ConvertFreeImgFormat(PixelFormat& pf, FREE_IMAGE_TYPE t, uint32_t bpp)
{
    switch(t)
    {
        case FIT_UNKNOWN: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            
        case FIT_BITMAP:
        {
            switch(bpp)
            {
                case 24: pf = PixelFormat::RGB8_UNORM; break;
                case 32: pf = PixelFormat::RGBA8_UNORM; break;
                default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
            }
            break;
        }
        case FIT_UINT16:
            break;
        case FIT_INT16:
            break;
        case FIT_UINT32:
            break;
        case FIT_INT32:
            break;
        case FIT_FLOAT:
            break;
        case FIT_DOUBLE:
            break;
        case FIT_COMPLEX:
            break;
        case FIT_RGB16:
            break;
        case FIT_RGBA16:
            break;
        case FIT_RGBF:
            break;
        case FIT_RGBAF:
            break;
        default: return ImageIOError::UNKNOWN_PIXEL_FORMAT;
    }

    //switch(t)
    //{
    //    case FREE_IMAGE_TYPE::FIT_BITMAP:
    //    {
    //        
    //    }
    //    case FREE_IMAGE_TYPE::FIT_RGB16: return PixelFormat::RGB16_UNORM;
    //    case FREE_IMAGE_TYPE::FIT_RGBA16: return PixelFormat::RGBA16_UNORM;
    //    case FREE_IMAGE_TYPE::FIT_RGBF: return PixelFormat::RGB_FLOAT;
    //    case FREE_IMAGE_TYPE::FIT_RGBAF: return PixelFormat::RGBA_FLOAT;
    //}
    return ImageIOError::OK;
}

size_t ImageIO::FormatToPixelSize(PixelFormat pf)
{
    // Sanity Check of Imath::half (may be it will have a padding etc.)
    static_assert(sizeof(Imath::half) == sizeof(uint16_t), "Imath size is not 16-bit");
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
        case PixelFormat::R_HALF:       return sizeof(Imath::half) * 1;
        case PixelFormat::RG_HALF:      return sizeof(Imath::half) * 2;
        case PixelFormat::RGB_HALF:     return sizeof(Imath::half) * 3;
        case PixelFormat::RGBA_HALF:    return sizeof(Imath::half) * 4;

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

ImageIO::ImageIO()
{
    FreeImage_Initialise();
}

ImageIO::~ImageIO()
{
    FreeImage_DeInitialise();
}

bool ImageIO::ReadEXR(std::vector<Vector4>& image,
                      Vector2ui& size,
                      const std::string& fileName) const
{
    Imf::Array2D<Imf::Rgba> pixels;

    Imf::RgbaInputFile file(fileName.c_str());
    Imath::Box2i dw = file.dataWindow();
    size[0] = dw.max.x - dw.min.x + 1;
    size[1] = dw.max.y - dw.min.y + 1;
    pixels.resizeErase(size[1], size[0]);
    file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * size[0], 1, size[0]);
    file.readPixels(dw.min.y, dw.max.y);

    // Convert to Vector4f and Invert Y Axis
    image.resize(size[0] * size[1]);
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
    return true;
}

bool ImageIO::ReadHDR(std::vector<Vector4>& image,
                      Vector2ui& size,
                      const std::string& fileName) const
{
    FIBITMAP* dib2 = nullptr;
    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(fileName.c_str());

    dib2 = FreeImage_Load(fif, fileName.c_str());
    if(dib2 == nullptr)
    {
        return false;
    }
    FIBITMAP* dib1 = FreeImage_ConvertToRGBAF(dib2);
    //FIBITMAP *dib1 = FreeImage_TmoReinhard05Ex(dib2);
    FreeImage_Unload(dib2);

    BITMAPINFOHEADER* header = FreeImage_GetInfoHeader(dib1);

    // Size
    size[0] = header->biWidth;
    size[1] = header->biHeight;
    image.resize(size[0] * size[1]);

    for(int j = 0; j < header->biHeight; j++)
    {
        FIRGBAF* bits = (FIRGBAF*)FreeImage_GetScanLine(dib1, j);
        for(int i = 0; i < header->biWidth; i++)
        {
        /*  RGBQUAD rgb;
            bool fetched = FreeImage_GetPixelColor(dib1, i, header->biHeight - j - 1, &rgb);*/

            Vector4 pixel;
            //pixel[0] = static_cast<float>(rgb.rgbRed) / 255.0f;
            //pixel[1] = static_cast<float>(rgb.rgbGreen) / 255.0f;
            //pixel[2] = static_cast<float>(rgb.rgbBlue) / 255.0f;
            //pixel[3] = 0.0f;

            pixel[0] = bits[i].red * 2.5f;
            pixel[1] = bits[i].green * 2.5f;
            pixel[2] = bits[i].blue * 2.5f;
            pixel[3] = 0.0f;

            //if(pixel[0] > 1.0f)
            //  printf("%f ", pixel[0]);
            //if(pixel[1] > 1.0f)
            //  printf("%f ", pixel[1]);
            //if(pixel[2] > 1.0f)
            //  printf("%f ", pixel[2]);
            //printf("\n");

            image[j * header->biWidth + i] = pixel;
        }
    }
    return true;
}

bool ImageIO::WriteAsEXR(const float* image,
                         const Vector2ui& size,
                         const std::string& fileName) const
{
    std::vector<Imath::half> convertedData(size[0] * size[1]);
    // Y Invert data and convert to half
    for(uint32_t y = 0; y < size[1]; y++)
        for(uint32_t x = 0; x < size[0]; x++)
        {
            uint32_t inIndex = y * size[0] + x;
            uint32_t invertexY = size[1] - y - 1;
            uint32_t outIndex = invertexY * size[0] + x;

            convertedData[outIndex] = image[inIndex];
        }

    Imf::Header header(size[0], size[1]);
    header.channels().insert("Y", Imf::Channel(Imf::HALF));

    Imf::OutputFile file(fileName.c_str(), header);
    Imf::FrameBuffer frameBuffer;
    Imf::Slice lumSlice = Imf::Slice(Imf::HALF,
                                     reinterpret_cast<char*>(convertedData.data()), // base // 8
                                     sizeof(Imath::half),
                                     sizeof(Imath::half) * size[0]);
    frameBuffer.insert("Y", lumSlice);

    file.setFrameBuffer(frameBuffer);
    file.writePixels(size[1]);

    return true;
}

bool ImageIO::WriteAsEXR(const Vector4f* image,
                         const Vector2ui& size,
                         const std::string& fileName) const
{
    std::vector<Imf::Rgba> convertedData(size[0] * size[1]);
    // Cant invert the image while using std::transform easily
    //std::transform(std::execution::par_unseq,
    //               image, image + size[0] * size[1],
    //               convertedData.begin(), [] (const Vector4f& v) -> Imf::Rgba
    //               {
    //                   return Imf::Rgba(v[0], v[1], v[2], v[3]);
    //               });

    // Instead using simple loops
    for(uint32_t y = 0; y < size[1]; y++)
        for(uint32_t x = 0; x < size[0]; x++)
        {
            uint32_t inIndex = y * size[0] + x;
            uint32_t invertexY = size[1] - y - 1;
            uint32_t outIndex = invertexY * size[0] + x;

            const Vector4f& v = image[inIndex];
            convertedData[outIndex] = Imf::Rgba(v[0], v[1], v[2], v[3]);
        }

        // In this header file INCREASING_Y does not invert image
        // I think it is only for memory layout
        //Imf::Header header(size[0], size[1], 1.0f, 
        //                   Imath::V2f(0, 0), 1.0f, 
        //                   Imf::DECREASING_Y);
    Imf::RgbaOutputFile file(fileName.c_str(),
                             size[0], size[1],
                             Imf::RgbaChannels::WRITE_RGBA);
    file.setFrameBuffer(convertedData.data(), 1, size[0]);
    file.writePixels(size[1]);
    return true;
}

bool ImageIO::WriteAsPNG(const Vector4f* image,
                         const Vector2ui& size,
                         const std::string& fileName) const
{
    auto* bitmap = FreeImage_Allocate(size[0], size[1], 24);

    for(uint32_t j = 0; j < size[1]; j++)
        for(uint32_t i = 0; i < size[0]; i++)
        {
            RGBQUAD color;
            Vector4f rgbImage = image[j * size[0] + i];

            rgbImage.ClampSelf(0.0f, 1.0f);
            rgbImage *= 255.0f;

            color.rgbRed = static_cast<BYTE>(rgbImage[0]);
            color.rgbGreen = static_cast<BYTE>(rgbImage[1]);
            color.rgbBlue = static_cast<BYTE>(rgbImage[2]);

            FreeImage_SetPixelColor(bitmap, i, j, &color);
        }
    bool result = FreeImage_Save(FIF_PNG, bitmap, fileName.c_str());
    FreeImage_Unload(bitmap);
    return result;
}

bool ImageIO::WriteAsPNG(const Vector4uc* image,
                         const Vector2ui& size,
                         const std::string& fileName) const
{
    auto* bitmap = FreeImage_Allocate(size[0], size[1], 24);

    for(uint32_t j = 0; j < size[1]; j++)
        for(uint32_t i = 0; i < size[0]; i++)
        {
            RGBQUAD color;
            Vector4uc rgbImage = image[j * size[0] + i];
            color.rgbRed = rgbImage[0];
            color.rgbGreen = rgbImage[1];
            color.rgbBlue = rgbImage[2];
            FreeImage_SetPixelColor(bitmap, i, j, &color);
        }
    bool result = FreeImage_Save(FIF_PNG, bitmap, fileName.c_str());
    FreeImage_Unload(bitmap);
    return result;
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

ImageIOError ImageIO::ReadImage_FreeImage(std::vector<Byte>& pixels,
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
    FreeImgRAII imgCPU(f, filePath.c_str());
    // If imgCPU not avail fail
    if(!imgCPU) return ImageIOError(ImageIOError::READ_INTERNAL_ERROR, filePath);
   
    // Bit per pixel
    uint32_t bpp = FreeImage_GetBPP(imgCPU);
    uint32_t w = FreeImage_GetWidth(imgCPU);
    uint32_t h = FreeImage_GetHeight(imgCPU);
    uint32_t pitch = FreeImage_GetPitch(imgCPU);
    dimension = Vector2ui(w, h);

    FREE_IMAGE_TYPE imgType = FreeImage_GetImageType(imgCPU);
    FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(imgCPU);

    ImageIOError e = ImageIOError::OK;
    if((e = ConvertFreeImgFormat(format, imgType, bpp)) != ImageIOError::OK)
    {
        return ImageIOError(e, filePath);
    }

    // Now start to load
    BYTE* pixelsFreeImg = FreeImage_GetBits(imgCPU);
    pixels.resize(bpp * w * h / BYTE_BITS);
    size_t rowSize = bpp / BYTE_BITS * w;
    // Compact the buffer 
    for(uint32_t j = 0; j < h; j++)
    {
        std::memcpy(pixels.data() + j * rowSize,
                    pixelsFreeImg + j * pitch,
                    rowSize);
    }
    // All done!
    return ImageIOError::OK;
}

ImageIOError ImageIO::ReadImage_OpenEXR(std::vector<Byte>& pixels,
                                        PixelFormat&, Vector2ui& size,
                                        const std::string& filePath) const
{
    return ImageIOError::READ_INTERNAL_ERROR;
}

bool ImageIO::IsConvertible(PixelFormat toFormat, PixelFormat fromFormat) const
{
    return false;
}

void ImageIO::CopyPixels(Byte* toData, PixelFormat toFormat,
                         const Byte* fromData, PixelFormat fromFormat,
                         const Vector2ui& dimension) const
{

}

ImageIOError ImageIO::ReadImage(std::vector<Byte>& pixels,
                                PixelFormat& pf, Vector2ui& dimension,
                                const std::string& filePath) const
{
    // First check if the file exists
    if(!Utility::CheckFileExistance(filePath))
        return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);
    if(CheckIfEXR(filePath))
        return ReadImage_FreeImage(pixels, pf, dimension, filePath);
    else 
       return ReadImage_OpenEXR(pixels, pf, dimension, filePath);
}

ImageIOError ImageIO::ReadImageAlphaChannelAsBitMap(std::vector<Byte>&,
                                                    Vector2ui& dimension,
                                                    const std::string& filePath) const
{
    return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);
}

ImageIOError ImageIO::WriteImage(const Byte* data,
                                 const Vector2ui& dimension,
                                 PixelFormat, ImageType,
                                 const std::string& filePath) const
{
    return ImageIOError(ImageIOError::IMAGE_NOT_FOUND, filePath);
}

ImageIOError ImageIO::WriteBitmap(const Byte* bits,
                                  const Vector2ui& size,
                                  const std::string& fileName) const
{
    auto* bitmap = FreeImage_Allocate(size[0], size[1], 24);

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
        color.rgbGreen = static_cast<BYTE>(bit) * 255;
        color.rgbBlue = static_cast<BYTE>(bit) * 255;

        bool pixLoaded = FreeImage_SetPixelColor(bitmap, i, j, &color);
    }
    bool result = FreeImage_Save(FIF_PNG, bitmap, fileName.c_str());
    FreeImage_Unload(bitmap);
    return ImageIOError::OK;
}
