#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"

#include <cuda.h>

// METURay only supports
// 2D Textures 2^24 different textures layers (per texture) and 255 different mips per texture.
// For research purposes this should be enough.
using TextureId = HitKeyT<uint32_t, 8u, 24u>;
struct UVList
{
    Vector2us id;   // 16-bit layer id, 16-bit unorm w
    Vector2us uv;   // UNorm 2x16 data
    Vector2us dxy;  // Gradient
};

struct Texture
{
    cudaTextureObject_t tex;
    
};
