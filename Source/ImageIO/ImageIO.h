#pragma once

#include <vector>
#include <string>
#include <memory>

#include "ImageIOI.h"

#include <OpenImageIO/imageio.h>

class OIIO::ImageSpec;

class ImageIO : public ImageIOI
{
    private:
        static constexpr size_t PARALLEL_EXEC_TRESHOLD = 2048;

        // Conversion Enums
        static ImageIOError OIIOImageSpecToPixelFormat(PixelFormat&, const OIIO::ImageSpec&);
        static ImageIOError PixelFormatToOIIOImageSpec(OIIO::ImageSpec&, PixelFormat,
                                                       const Vector2ui&);
        static void         PackChannelBits(Byte* bits,
                                            const Byte* fromData, PixelFormat fromFormat,
                                            size_t fromPitch, ImageChannelType type,
                                            const Vector2ui& dimension);

    protected:
    public:
        // Constructors & Destructor
                            ImageIO() = default;
                            ImageIO(const ImageIO&) = delete;
        ImageIO&            operator=(const ImageIO&) = delete;
                            ~ImageIO() = default;

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