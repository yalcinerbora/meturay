#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"

// METURay only supports
// 2D Textures 2^24 different textures layers (per texture) and 255 different mips per texture.
// For research purposes this should be enough.
using TextureId = HitKeyT<uint32_t, 8u, 24u>;
struct UVList
{
    TextureId id;   // 24-bit layer id, 8-bit mip id
    Vector2us uv;   // UNorm 2x16 data
};