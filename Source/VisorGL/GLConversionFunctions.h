#pragma once

#include <glbinding/gl/gl.h>
#include "RayLib/Types.h"

inline gl::GLenum PixelFormatToGL(PixelFormat f)
{
    static constexpr gl::GLenum TypeList[static_cast<int>(PixelFormat::END)] =
    {
        gl::GL_RED,
        gl::GL_RG,
        gl::GL_RGB,
        gl::GL_RGBA,

        gl::GL_RED,
        gl::GL_RG,
        gl::GL_RGB,
        gl::GL_RGBA,

        gl::GL_RED,
        gl::GL_RG,
        gl::GL_RGB,
        gl::GL_RGBA,

        gl::GL_RED,
        gl::GL_RG,
        gl::GL_RGB,
        gl::GL_RGBA,

        gl::GL_RED,
        gl::GL_RG,
        gl::GL_RGB,
        gl::GL_RGBA,

        gl::GL_RED,
        gl::GL_RG,
        gl::GL_RGB,
        gl::GL_RGBA,

        // TODO: Change These
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE
    };
    return TypeList[static_cast<int>(f)];
}

inline gl::GLenum PixelFormatToSizedGL(PixelFormat f)
{
    static constexpr gl::GLenum TypeList[static_cast<int>(PixelFormat::END)] =
    {
        gl::GL_R8,
        gl::GL_RG8,
        gl::GL_RGB8,
        gl::GL_RGBA8,

        gl::GL_R16,
        gl::GL_RG16,
        gl::GL_RGB16,
        gl::GL_RGBA16,

        gl::GL_R8,
        gl::GL_RG8,
        gl::GL_RGB8,
        gl::GL_RGBA8,

        gl::GL_R16,
        gl::GL_RG16,
        gl::GL_RGB16,
        gl::GL_RGBA16,

        gl::GL_R16F,
        gl::GL_RG16F,
        gl::GL_RGB16F,
        gl::GL_RGBA16F,

        gl::GL_R32F,
        gl::GL_RG32F,
        gl::GL_RGB32F,
        gl::GL_RGBA32F,

        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE,
        gl::GL_NONE
    };
    return TypeList[static_cast<int>(f)];
}

inline gl::GLenum PixelFormatToTypeGL(PixelFormat f)
{
    static constexpr gl::GLenum TypeList[static_cast<int>(PixelFormat::END)] =
    {
        gl::GL_UNSIGNED_BYTE,
        gl::GL_UNSIGNED_BYTE,
        gl::GL_UNSIGNED_BYTE,
        gl::GL_UNSIGNED_BYTE,

        gl::GL_UNSIGNED_SHORT,
        gl::GL_UNSIGNED_SHORT,
        gl::GL_UNSIGNED_SHORT,
        gl::GL_UNSIGNED_SHORT,

        gl::GL_BYTE,
        gl::GL_BYTE,
        gl::GL_BYTE,
        gl::GL_BYTE,

        gl::GL_SHORT,
        gl::GL_SHORT,
        gl::GL_SHORT,
        gl::GL_SHORT,

        gl::GL_HALF_FLOAT,  // TODO: Wrong
        gl::GL_HALF_FLOAT,  // TODO: Wrong
        gl::GL_HALF_FLOAT,  // TODO: Wrong
        gl::GL_HALF_FLOAT,  // TODO: Wrong

        gl::GL_FLOAT,
        gl::GL_FLOAT,
        gl::GL_FLOAT,
        gl::GL_FLOAT
    };
    return TypeList[static_cast<int>(f)];
}

inline PixelFormat PixelFormatTo4ChannelPF(PixelFormat f)
{
    static constexpr PixelFormat INVALID_PF = PixelFormat::END;
    switch(f)
    {
        case PixelFormat::RGB8_UNORM: return PixelFormat::RGBA8_UNORM;
        case PixelFormat::RGB8_SNORM: return PixelFormat::RGBA8_SNORM;
        case PixelFormat::RGB16_UNORM: return PixelFormat::RGBA16_UNORM;
        case PixelFormat::RGB16_SNORM: return PixelFormat::RGBA16_SNORM;
        case PixelFormat::RGB_HALF: return PixelFormat::RGBA_HALF;
        case PixelFormat::RGB_FLOAT: return PixelFormat::RGBA_FLOAT;
        // Relay the 4 channel formats directly
        case PixelFormat::RGBA8_UNORM:
        case PixelFormat::RGBA8_SNORM:
        case PixelFormat::RGBA16_UNORM:
        case PixelFormat::RGBA16_SNORM:
        case PixelFormat::RGBA_HALF:
        case PixelFormat::RGBA_FLOAT:
            return f;
        default: return INVALID_PF;
    }
}