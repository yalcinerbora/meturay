#include "ImageIO.h"
#include "FreeImgRAII.h"

#include <execution>
#include <algorithm>

#include <OpenEXR/ImfRgbaFile.h>
//#include <OpenEXR/ImfConvert.h>
#include <OpenEXR/ImfRgba.h>

ImageIO& ImageIO::Instance()
{
    static std::unique_ptr<ImageIO> instance = nullptr;

    if(instance == nullptr)
        instance = std::make_unique<ImageIO>();
    return *(instance.get());
}

static PixelFormat ConvertFreeImgFormat(FREE_IMAGE_TYPE t, uint32_t bpp)
{
    switch(t)
    {
        case FREE_IMAGE_TYPE::FIT_BITMAP:
        {
            switch(bpp)
            {
                case 24: return PixelFormat::RGB8_UNORM;
                case 32: return PixelFormat::RGBA8_UNORM;
            }
            break;
        }
        case FREE_IMAGE_TYPE::FIT_RGB16: return PixelFormat::RGB16_UNORM;
        case FREE_IMAGE_TYPE::FIT_RGBA16: return PixelFormat::RGBA16_UNORM;
        case FREE_IMAGE_TYPE::FIT_RGBF: return PixelFormat::RGB_FLOAT;
        case FREE_IMAGE_TYPE::FIT_RGBAF: return PixelFormat::RGBA_FLOAT;
    }
    return PixelFormat::END;
}

ImageIO::ImageIO()
{
    FreeImage_Initialise();
}

ImageIO::~ImageIO()
{
    FreeImage_DeInitialise();
}

bool ImageIO::ReadHDR(std::vector<Vector4>& image,
                      Vector2ui& size,
                      const std::string& fileName) const
{
    FIBITMAP *dib2 = nullptr;
    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(fileName.c_str());

    dib2 = FreeImage_Load(fif, fileName.c_str());
    if(dib2 == nullptr)
    {
        return false;
    }
    FIBITMAP *dib1 = FreeImage_ConvertToRGBAF(dib2);
    //FIBITMAP *dib1 = FreeImage_TmoReinhard05Ex(dib2);
    FreeImage_Unload(dib2);

    BITMAPINFOHEADER* header = FreeImage_GetInfoHeader(dib1);

    // Size
    size[0] = header->biWidth;
    size[1] = header->biHeight;
    image.resize(size[0] * size[1]);

    for(int j = 0; j < header->biHeight; j++)
    {
        FIRGBAF *bits = (FIRGBAF *)FreeImage_GetScanLine(dib1, j);
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

bool ImageIO::ReadImage(std::vector<Byte>& pixels,
                        PixelFormat& format, Vector2ui& size,
                        const std::string& filePath) const
{
    FREE_IMAGE_FORMAT f;
    // Check file to find type
    if((f = FreeImage_GetFileType(filePath.c_str())) == FIF_UNKNOWN)
        // Use file extension to determine type
        f = FreeImage_GetFIFFromFilename(filePath.c_str());
    // If it is still unknown just return error
    if(f == FIF_UNKNOWN) return false;

    FreeImgRAII imgCPU(f, filePath.c_str());
    if(!imgCPU) return false;

    // Bit per pixel
    uint32_t bpp = FreeImage_GetBPP(imgCPU);
    uint32_t w = FreeImage_GetWidth(imgCPU);
    uint32_t h = FreeImage_GetHeight(imgCPU);
    uint32_t pitch = FreeImage_GetPitch(imgCPU);
    size = Vector2ui(w, h);

    FREE_IMAGE_TYPE imgType = FreeImage_GetImageType(imgCPU);
    FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(imgCPU);

    format = ConvertFreeImgFormat(imgType, bpp);
    if(format == PixelFormat::END) return false;


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
    return true;
}

bool ImageIO::WriteAsEXR(const Vector4f* image,
                         const Vector2ui& size,
                         const std::string& fileName) const
{
    std::vector<Imf::Rgba> convertedData(size[0] * size[1]);
    std::transform(std::execution::par_unseq,
                   image, image + size[0] * size[1],
                   convertedData.begin(), [] (const Vector4f& v) -> Imf::Rgba
                   {
                       return Imf::Rgba(v[0], v[1], v[2], v[3]);
                   });
    
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

        FreeImage_SetPixelColor(bitmap, i , j, &color);
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

bool ImageIO::WriteBitmap(const Byte* bits,
                          const Vector2ui& size,
                          const std::string& fileName) const
{
    auto* bitmap = FreeImage_Allocate(size[0], size[1], 24);

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
    return result;
}