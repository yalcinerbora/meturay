#pragma once

#include "ColorConversion.h"

namespace Utility
{
    __device__ __host__
    Vector3f RandomColorRGB(uint32_t index);
}

__device__ __host__ HYBRID_INLINE
Vector3f Utility::RandomColorRGB(uint32_t index)
{
    // https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
    static constexpr float SATURATION = 0.65f;
    static constexpr float VALUE = 0.95f;
    static constexpr float GOLDEN_RATIO_CONJ = 0.618033988749895f;
    // For large numbers use double arithmetic here
    float hue = 0.1f + static_cast<float>(index) * GOLDEN_RATIO_CONJ;
    hue = fmod(hue, 1.0f);

    return HSVToRGB(Vector3f(hue, SATURATION, VALUE));
}