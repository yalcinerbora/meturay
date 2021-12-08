#include "TextureGL.h"
#include "GLConversionFunctions.h"

#include "RayLib/UTF8StringConversion.h"
#include "RayLib/Log.h"

#include "ImageIO/EntryPoint.h"


TextureGL::TextureGL()
    : texId(0)
    , dimensions(Zero2ui)
    , pixFormat(PixelFormat::END)
{}

TextureGL::TextureGL(const Vector2ui& dim,
                     PixelFormat fmt)
    : texId(0)
    , dimensions(dim)
    , pixFormat(fmt)
{
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexStorage2D(GL_TEXTURE_2D, 1, PixelFormatToSizedGL(pixFormat),
                   static_cast<GLsizei>(dimensions[0]),
                   static_cast<GLsizei>(dimensions[1]));
}

TextureGL::TextureGL(const std::string& filePath)
    : texId(0)
    , dimensions(Zero2ui)
    , pixFormat(PixelFormat::END)
{
    std::vector<Byte> pixels;
    ImageIOError e = ImageIOInstance()->ReadImage(pixels,
                                                  pixFormat,
                                                  dimensions,
                                                  filePath);
    if(e != ImageIOError::OK)
    {
        throw ImageIOException(ImageIOError(e, filePath));
    }

    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexStorage2D(GL_TEXTURE_2D, 1, PixelFormatToSizedGL(pixFormat),
                   static_cast<GLsizei>(dimensions[0]),
                   static_cast<GLsizei>(dimensions[1]));

    // Copy the data to GPU
    //glTexIm
    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    0, 0,
                    dimensions[0], dimensions[1],
                    PixelFormatToGL(pixFormat),
                    PixelFormatToTypeGL(pixFormat),
                    pixels.data());

    // Override Filtering to Nearest
    // Useful for imgui etc.
    glTextureParameteri(texId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(texId, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

TextureGL::TextureGL(TextureGL&& other)
    : texId(other.texId)
    , dimensions(other.dimensions)
    , pixFormat(other.pixFormat)
{
    other.texId = 0;
}

TextureGL& TextureGL::operator=(TextureGL&& other)
{
    assert(this != &other);

    if(texId) glDeleteTextures(1, &texId);
    texId = other.texId;
    dimensions = other.dimensions;
    pixFormat = other.pixFormat;

    other.texId = 0;
    return *this;
}

TextureGL::~TextureGL()
{
    glDeleteTextures(1, &texId);
}

void TextureGL::CopyToImage(const std::vector<Byte>& pixels,
                            const Vector2ui& start,
                            const Vector2ui& end,
                            PixelFormat format)
{
    const Vector2ui subSize = end - start;

    glBindTexture(GL_TEXTURE_2D, texId);

    // If formats ad compatible between these
    // OGL will do the conversion automatically
    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    start[0], start[1],
                    subSize[0], subSize[1],
                    PixelFormatToGL(format),
                    PixelFormatToTypeGL(format),
                    pixels.data());
}

void TextureGL::Bind(GLuint bindingIndex) const
{
    glActiveTexture(GL_TEXTURE0 + bindingIndex);
    glBindTexture(GL_TEXTURE_2D, texId);
}

void SamplerGL::Bind(GLuint bindingIndex) const
{
    glBindSampler(bindingIndex, samplerId);
}