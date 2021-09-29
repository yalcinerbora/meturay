#pragma once

#include <vector>
#include <string>
#include <memory>
#include <FreeImage.h>

#include "ImageIOI.h"

class FreeImgRAII;

class ImageIO : public ImageIOI
{
    private:
        // Methods
        // Read
        bool        ReadHDR(std::vector<Vector4>& image,
                            Vector2ui& size,
                            const std::string& fileName) const;
        bool        ReadEXR(std::vector<Vector4>& image,
                            Vector2ui& size,
                            const std::string& fileName) const;
        // Write
        bool        WriteAsEXR(const float* image,
                               const Vector2ui& size,
                               const std::string& fileName) const;
        bool        WriteAsEXR(const Vector4f* image,
                               const Vector2ui& size,
                               const std::string& fileName) const;
        bool        WriteAsPNG(const Vector4f* image,
                               const Vector2ui& size,
                               const std::string& fileName) const;
        bool        WriteAsPNG(const Vector4uc* image,
                               const Vector2ui& size,
                               const std::string& fileName) const;

        static bool             CheckIfEXR(const std::string& fileName);
        static ImageIOError     ConvertFreeImgFormat(PixelFormat& pf, FREE_IMAGE_TYPE t, uint32_t bpp);


        ImageIOError        ReadImage_FreeImage(FreeImgRAII&,
                                                PixelFormat&, Vector2ui& dimension,
                                                const std::string& filePath) const;
        ImageIOError        ReadImage_OpenEXR(std::vector<Byte>& pixels,
                                              PixelFormat&, Vector2ui& size,
                                              const std::string& filePath) const;

    protected:        
        void                ConvertPixels(Byte* toData, PixelFormat toFormat,
                                          const Byte* fromData, PixelFormat fromFormat, size_t fromPitch,
                                          const Vector2ui& dimension) const;

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
                                                     ChannelType,
                                                     const std::string& filePath,
                                                     ImageIOFlags = ImageIOFlags()) const override;

        ImageIOError        WriteImage(const Byte* data,
                                       const Vector2ui& dimension,
                                       PixelFormat, ImageType,
                                       const std::string& filePath) const override;
        ImageIOError        WriteBitmap(const Byte* bits,
                                        const Vector2ui& dimension,
                                        const std::string& filePath) const override;
};