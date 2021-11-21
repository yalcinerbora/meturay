#pragma once

#include <GL/glew.h>
#include <vector>

#include "RayLib/Vector.h"
#include "RayLib/Types.h"

enum class SamplerGLEdgeResolveType
{
    CLAMP,
    REPEAT,
};

enum class SamplerGLInterpType
{
    NEAREST,
    LINEAR
};

// Very simple OpenGL Texture Wrapper with load etc. functionality
// Not performance critical but its not needed
// These objects are using immutable storage (OGL 4.2+)
class TextureGL
{
    private:
        GLuint          texId;
        Vector2ui       dimensions;
        PixelFormat     pixFormat;

    protected:
    public:
        // Constructors & Destructor
                        TextureGL();
                        TextureGL(const Vector2ui& dimensions,
                                  PixelFormat);
                        TextureGL(const std::string& filePath);
                        TextureGL(const TextureGL&) = delete;
                        TextureGL(TextureGL&&);
        TextureGL&      operator=(const TextureGL&) = delete;
        TextureGL&      operator=(TextureGL&&);
                        ~TextureGL();

        void            Bind(GLuint bindingIndex) const;

        GLuint          TexId();
        uint32_t        Width() const;
        uint32_t        Height() const;
        Vector2ui       Size() const;
        PixelFormat     Format() const;

        void            CopyToImage(const std::vector<Byte>& pixels,
                                    const Vector2ui& start,
                                    const Vector2ui& end,
                                    PixelFormat format);

};

class SamplerGL
{
    private:
        GLuint          samplerId;

    protected:
    public:
        // Constructors & Destructor
                        SamplerGL(SamplerGLEdgeResolveType,
                                  SamplerGLInterpType);
                        SamplerGL(const SamplerGL&) = delete;
                        SamplerGL(SamplerGL&&) = default;
        SamplerGL&      operator=(const SamplerGL&) = delete;
        SamplerGL&      operator=(SamplerGL&&) = default;
                        ~SamplerGL();

        GLuint          SamplerId();
        void            Bind(GLuint bindingIndex) const;
};

inline GLuint TextureGL::TexId()
{
    return texId;
}

inline uint32_t TextureGL::Width() const
{
    return dimensions[0];
}

inline uint32_t TextureGL::Height() const
{
    return dimensions[1];
}

inline Vector2ui TextureGL::Size() const
{
    return dimensions;
}

inline PixelFormat TextureGL::Format() const
{
    return pixFormat;
}

inline SamplerGL::SamplerGL(SamplerGLEdgeResolveType edgeResolve,
                            SamplerGLInterpType interp)
{
    glGenSamplers(1, &samplerId);

    if(interp == SamplerGLInterpType::NEAREST)
    {
        glSamplerParameteri(samplerId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glSamplerParameteri(samplerId, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }
    else if(interp == SamplerGLInterpType::LINEAR)
    {
        glSamplerParameteri(samplerId, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glSamplerParameteri(samplerId, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    if(edgeResolve == SamplerGLEdgeResolveType::CLAMP)
    {
        glSamplerParameteri(samplerId, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(samplerId, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(samplerId, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }
    else if(edgeResolve == SamplerGLEdgeResolveType::REPEAT)
    {
        glSamplerParameteri(samplerId, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glSamplerParameteri(samplerId, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glSamplerParameteri(samplerId, GL_TEXTURE_WRAP_R, GL_REPEAT);
    }

}

inline SamplerGL::~SamplerGL()
{
    glDeleteSamplers(1, &samplerId);
}

inline GLuint SamplerGL::SamplerId()
{
    return samplerId;
}