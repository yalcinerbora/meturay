#pragma once

//#include "RayLib/Vector.h"
//#include "TextureReference.cuh"

#include "RayLib/Vector.h"
#include "Texture.cuh"

// Generates mipmap of the level (mipLevel)
// from the texture.
//
//
//
// Texture should be float return capable (float, float2, float4)
// either normalized singed unsigned integer types or
// outright float types.
//
// Only 2D, three/four channel float for now
// It gets complicated to run these over host functions
//
// By design textures are immutable wrt. size and currently textures are loaded
// without any mips
// We will create completely new texture instead
__host__
Texture<2, Vector4f> GenerateMipmaps(const Texture<2, Vector4f>&,
                                     uint32_t upToMip = std::numeric_limits<uint32_t>::max());