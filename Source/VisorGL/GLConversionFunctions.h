#pragma once

#include <GL/glew.h>
#include "RayLib/Types.h"

inline GLenum PixelFormatToGL(PixelFormat f)
{
    static constexpr GLenum TypeList[static_cast<int>(PixelFormat::END)] =
    {
        GL_RED,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_RED,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_RED,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_RED,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_RED,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_RED,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    return TypeList[static_cast<int>(f)];
}

inline GLenum PixelFormatToSizedGL(PixelFormat f)
{
    static constexpr GLenum TypeList[static_cast<int>(PixelFormat::END)] =
    {
        GL_R8,
        GL_RG8,
        GL_RGB8,
        GL_RGBA8,

        GL_R16,
        GL_RG16,
        GL_RGB16,
        GL_RGBA16,

        GL_R8,
        GL_RG8,
        GL_RGB8,
        GL_RGBA8,

        GL_R16,
        GL_RG16,
        GL_RGB16,
        GL_RGBA16,

        GL_R16F,
        GL_RG16F,
        GL_RGB16F,
        GL_RGBA16F,

        GL_R32F,
        GL_RG32F,
        GL_RGB32F,
        GL_RGBA32F,

        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    return TypeList[static_cast<int>(f)];
}

inline GLenum PixelFormatToTypeGL(PixelFormat f)
{
    static constexpr GLenum TypeList[static_cast<int>(PixelFormat::END)] =
    {
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_BYTE,

        GL_UNSIGNED_SHORT,
        GL_UNSIGNED_SHORT,
        GL_UNSIGNED_SHORT,
        GL_UNSIGNED_SHORT,

        GL_BYTE,
        GL_BYTE,
        GL_BYTE,
        GL_BYTE,

        GL_SHORT,
        GL_SHORT,
        GL_SHORT,
        GL_SHORT,

        GL_HALF_FLOAT,  // TODO: Wrong
        GL_HALF_FLOAT,  // TODO: Wrong
        GL_HALF_FLOAT,  // TODO: Wrong
        GL_HALF_FLOAT,  // TODO: Wrong

        GL_FLOAT,
        GL_FLOAT,
        GL_FLOAT,
        GL_FLOAT
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