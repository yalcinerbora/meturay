#pragma once

#include <vector>
#include <string>
#include <memory>

#include <FreeImage.h>
#include <Imath/half.h>

#include "ImageIOI.h"

class FreeImgRAII;

class ImageIO : public ImageIOI
{
    private:
        static constexpr size_t PARALLEL_EXEC_TRESHOLD = 2048;

        // Methods
        // Write
        ImageIOError            WriteAsEXR(const Byte* pixels,
                                   const Vector2ui& dimension, PixelFormat,
                                   const std::string& fileName) const;
        ImageIOError            WriteUsingFreeImage(const Byte* pixels,
                                                    const Vector2ui& dimension,
                                                    PixelFormat, ImageType,
                                                    const std::string& fileName) const;

        static bool             CheckIfEXR(const std::string& fileName);

        static ImageIOError     ConvertFreeImgFormat(PixelFormat&, FREE_IMAGE_TYPE t, uint32_t bpp);
        static ImageIOError     ConvertImageTypeToFreeImgType(FREE_IMAGE_FORMAT&, ImageType);
        static ImageIOError     ConvertPixelFormatToFreeImgType(FREE_IMAGE_TYPE& t, PixelFormat);

        ImageIOError            ReadImage_FreeImage(FreeImgRAII&,
                                                    PixelFormat&, Vector2ui& dimension,
                                                    const std::string& filePath) const;
        ImageIOError            ReadImage_OpenEXR(std::vector<Byte>& pixels,
                                                  PixelFormat&, Vector2ui& size,
                                                  const std::string& filePath) const;

    protected:
        void                PackChannelBits(Byte* bits,
                                            const Byte* fromData, PixelFormat fromFormat,
                                            size_t pitch, ImageChannelType,
                                            const Vector2ui& dimension) const;
        void                ConvertPixelsInternal(Byte* toData, PixelFormat toFormat,
                                                  const Byte* fromData, PixelFormat fromFormat, size_t fromPitch,
                                                  const Vector2ui& dimension) const;

        template <class T>
        static void         ConvertForEXR(Imath::half* toData,
                                          const T* fromData, PixelFormat fromFormat,
                                          const Vector2ui& dimension);

    public:
        // Constructors & Destructor
                            ImageIO();
                            ImageIO(const ImageIO&) = delete;
        ImageIO&            operator=(const ImageIO&) = delete;
                            ~ImageIO();

        // Interface
        ImageIOError        ReadImage(std::vector<Byte>& pixels,
                                      PixelFormat&, Vector2ui& dimension,
                                      const std::string& filePath,
                                      const ImageIOFlags = ImageIOFlags()) const override;
        ImageIOError        ReadImageChannelAsBitMap(std::vector<Byte>&,
                                                     Vector2ui& dimension,
                                                     ImageChannelType,
                                                     const std::string& filePath,
                                                     ImageIOFlags = ImageIOFlags()) const override;

        ImageIOError        WriteImage(const Byte* data,
                                       const Vector2ui& dimension,
                                       PixelFormat, ImageType,
                                       const std::string& filePath) const override;
        ImageIOError        WriteBitmap(const Byte* bits,
                                        const Vector2ui& dimension, ImageType,
                                        const std::string& filePath) const override;

        ImageIOError        ConvertPixels(Byte* toData, PixelFormat toFormat,
                                          const Byte* fromData, PixelFormat fromFormat,
                                          const Vector2ui& dimension) const override;
};

template <class T>
void ImageIO::ConvertForEXR(Imath::half* toData,
                            const T* fromData, PixelFormat fromFormat,
                            const Vector2ui& dimension)
{
    const int channelCount = FormatToChannelCount(fromFormat);

    for(uint32_t j = 0; j < dimension[1]; j++)
    for(uint32_t i = 0; i < dimension[0]; i++)
    {
        // Don't forget to convert pixels
        uint32_t inIndex = j * dimension[0] + i;
        uint32_t invertexY = dimension[1] - j - 1;
        uint32_t outIndex = invertexY * dimension[0] + i;

        const T* inPixel = fromData + inIndex * channelCount;
        Imath::half* outPixel = toData + outIndex * channelCount;

        // Use = operator to convert from half to float
        for(int i = 0; i < channelCount; i++)
        {
            outPixel[i] = inPixel[i];
        }
    }

}