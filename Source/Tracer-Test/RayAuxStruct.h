#pragma once

#include <cstdint>
#include "RayLib/Vector.h"

struct RayAuxBasic
{
    bool            lightRay;
    Vector3f        irradianceFactor;
    uint32_t        pixelId;
    uint32_t        pixelSampleId;
};