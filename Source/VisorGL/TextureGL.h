#pragma once

#include <glbinding/gl/gl.h>
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
        gl::GLuint      texId;
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

        void            Bind(gl::GLuint bindingIndex) const;

        gl::GLuint      TexId();
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
        gl::GLuint      samplerId;

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

        gl::GLuint      SamplerId();
        void            Bind(gl::GLuint bindingIndex) const;
};

inline gl::GLuint TextureGL::TexId()
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
    gl::glGenSamplers(1, &samplerId);

    if(interp == SamplerGLInterpType::NEAREST)
    {
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_MAG_FILTER, gl::GL_NEAREST);
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_MIN_FILTER, gl::GL_NEAREST);
    }
    else if(interp == SamplerGLInterpType::LINEAR)
    {
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_MAG_FILTER, gl::GL_LINEAR);
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_MIN_FILTER, gl::GL_LINEAR);
    }

    if(edgeResolve == SamplerGLEdgeResolveType::CLAMP)
    {
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_WRAP_S, gl::GL_CLAMP_TO_EDGE);
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_WRAP_T, gl::GL_CLAMP_TO_EDGE);
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_WRAP_R, gl::GL_CLAMP_TO_EDGE);
    }
    else if(edgeResolve == SamplerGLEdgeResolveType::REPEAT)
    {
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_WRAP_S, gl::GL_REPEAT);
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_WRAP_T, gl::GL_REPEAT);
        gl::glSamplerParameteri(samplerId, gl::GL_TEXTURE_WRAP_R, gl::GL_REPEAT);
    }

}

inline SamplerGL::~SamplerGL()
{
    gl::glDeleteSamplers(1, &samplerId);
}

inline gl::GLuint SamplerGL::SamplerId()
{
    return samplerId;
}