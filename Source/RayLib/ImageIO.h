#pragma once

#include <vector>
#include <string>
#include <memory>

#include "Vector.h"
#include "Types.h"

class ImageIO
{
    public:
        static ImageIO&     Instance();

    private:
    protected:
    public:
        // Constructors & Destructor
                             ImageIO();
                             ImageIO(const ImageIO&) = delete;
        ImageIO&             operator=(const ImageIO&) = delete;
                             ~ImageIO();

        // Usage
        // Read
        bool                 ReadHDR(std::vector<Vector4>& image,
                                     Vector2ui& size,
                                     const std::string& fileName) const;
        bool                 ReadImage(std::vector<Byte>& pixels,
                                       PixelFormat&, Vector2ui& size,
                                       const std::string& filePath) const;

        // Write
        bool                 WriteAsEXR(const Vector4f* image,
                                        const Vector2ui& size,
                                        const std::string& fileName) const;
        bool                 WriteAsPNG(const Vector4f* image,
                                        const Vector2ui& size,
                                        const std::string& fileName) const;
        bool                 WriteBitmap(const Byte* bits,
                                        const Vector2ui& size,
                                        const std::string& fileName) const;
};