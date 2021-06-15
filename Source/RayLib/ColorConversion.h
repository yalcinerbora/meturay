#pragma once

#include "Vector.h"

namespace Utility
{
    __device__ __host__
    float RGBToLuminance(const Vector3f& rgb);

    __device__ __host__
    Vector3f HSVToRGB(const Vector3f& hsv);
}

__device__ __host__ HYBRID_INLINE
float Utility::RGBToLuminance(const Vector3f& rgb)
{
    // https://en.wikipedia.org/wiki/Relative_luminance
    // RBG should be in linear space
    return (0.2120f * rgb[0] +
            0.7150f * rgb[1] +
            0.0722f * rgb[2]);
}

__device__ __host__ HYBRID_INLINE
Vector3f Utility::HSVToRGB(const Vector3f& hsv)
{
    // H, S, V both normalized
    // H: [0-1) (meaning 0 is 0, 1 is 360)
    // S: [0-1] (meaning 0 is 0, 1 is 100)
    // V: [0-1] (meaning 0 is 0, 1 is 100)

    float h = hsv[0] * 360.0f;
    static constexpr float o60 = 1.0f / 60.0f;

    float c = hsv[2] * hsv[1];
    float m = hsv[2] - c;
    float x = c * (1 - fabsf(fmodf(h * o60, 2) - 1));

    Vector3f result;
    switch(static_cast<int>(h) / 60)
    {
        case 0: result = Vector3f(c, x, 0); break;
        case 1: result = Vector3f(x, c, 0); break;
        case 2: result = Vector3f(0, c, x); break;
        case 3: result = Vector3f(0, x, c); break;
        case 4: result = Vector3f(x, 0, c); break;
        case 5: result = Vector3f(c, 0, x); break;
    }
    result = result + m;
    return result;
}