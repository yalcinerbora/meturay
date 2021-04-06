#pragma once

#include "Vector.h"

namespace Utility
{
    __device__ __host__
    float RGBToLuminance(const Vector3& rgb);
}

__device__ __host__
inline
float Utility::RGBToLuminance(const Vector3& rgb)
{
    // https://en.wikipedia.org/wiki/Relative_luminance
    // RBG should be in linear space
    return (0.2120f * rgb[0] +
            0.7150f * rgb[1] +
            0.0722f * rgb[2]);
}