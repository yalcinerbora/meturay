#pragma once

#include <GL/glew.h>
#include "RayLib/Types.h"

static GLenum PixelFormatToGL(PixelFormat f)
{
    static constexpr GLenum TypeList[static_cast<int>(PixelFormat::END)] =
    {
        GL_R,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_R,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_R,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_R,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_R,
        GL_RG,
        GL_RGB,
        GL_RGBA,

        GL_R,
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

static GLenum PixelFormatToSizedGL(PixelFormat f)
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

static GLenum PixelFormatToTypeGL(PixelFormat f)
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

        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_BYTE,

        GL_UNSIGNED_SHORT,
        GL_UNSIGNED_SHORT,
        GL_UNSIGNED_SHORT,
        GL_UNSIGNED_SHORT,

        GL_SHORT,  // TODO: Wrong
        GL_SHORT,  // TODO: Wrong
        GL_SHORT,  // TODO: Wrong
        GL_SHORT,  // TODO: Wrong

        GL_FLOAT,
        GL_FLOAT,
        GL_FLOAT,
        GL_FLOAT
    };
    return TypeList[static_cast<int>(f)];
}
